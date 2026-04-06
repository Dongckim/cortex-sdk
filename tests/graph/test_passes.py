"""Tests for dead_node_elimination pass.

Key assertions:
  1. saliency_dft is eliminated in POWER_SAVE mode.
  2. execute() produces identical score_map before and after elimination.
  3. Optimized graph is measurably faster.
"""

import time

import numpy as np
import pytest

from cortex.capture.imu_gate import BatteryMode
from cortex.graph import build_l2_graph, dead_node_elimination
from cortex.optimizer.hybrid_roi import RequestType


@pytest.fixture
def frame() -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[160:320, 213:427] = 200
    return img


# ------------------------------------------------------------------
# Correctness: which nodes get eliminated
# ------------------------------------------------------------------


def test_saliency_eliminated_in_power_save(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    assert "saliency_dft" in result.eliminated


def test_nothing_eliminated_in_balanced(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.BALANCED)
    assert result.eliminated == []


def test_nothing_eliminated_in_aggressive(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.AGGRESSIVE)
    assert result.eliminated == []


def test_optimized_graph_does_not_contain_saliency(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    names = result.graph.node_names()
    assert "saliency_dft" not in names


def test_optimized_graph_retains_other_nodes(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    names = result.graph.node_names()
    for expected in ["center_crop", "text_roi_mser", "motion_map", "score_fusion", "ema_smooth"]:
        assert expected in names, f"Expected {expected} to survive DNE"


def test_score_fusion_inputs_cleaned_after_dne(frame: np.ndarray) -> None:
    """score_fusion inputs must not reference the eliminated saliency_dft node."""
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)

    fusion = result.graph.get_node("score_fusion")
    assert fusion is not None
    input_names = [inp.name for inp in fusion.inputs if hasattr(inp, "name")]
    assert "saliency_dft" not in input_names


# ------------------------------------------------------------------
# Semantic equivalence: before vs after elimination
# ------------------------------------------------------------------


def test_execute_identical_before_after_dne(frame: np.ndarray) -> None:
    """Eliminating saliency_dft must not change the score_map result.

    Both paths apply POWER_SAVE semantics (ss=zeros, ws→wc):
      Before DNE: full graph runs saliency_dft, score_fusion zeroes ss.
      After DNE:  saliency_dft absent, score_fusion treats ss as zeros.
    Result must be numerically identical.
    """
    # Before DNE — full graph, POWER_SAVE semantics applied at runtime
    graph_full = build_l2_graph()
    ctx_before = graph_full.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    # After DNE — saliency_dft eliminated
    graph_opt = build_l2_graph()
    result = dead_node_elimination(graph_opt, BatteryMode.POWER_SAVE)
    ctx_after = result.graph.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    np.testing.assert_allclose(
        ctx_before["score_map"],
        ctx_after["score_map"],
        atol=1e-5,
        err_msg="score_map changed after dead_node_elimination — semantics broken",
    )


def test_score_map_shape_unchanged_after_dne(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    ctx = result.graph.execute(frame, battery_mode=BatteryMode.POWER_SAVE)
    assert ctx["score_map"].shape == (6, 8)


# ------------------------------------------------------------------
# Performance: optimized graph should be faster
# ------------------------------------------------------------------


def test_power_save_dne_is_faster(frame: np.ndarray) -> None:
    """Eliminating saliency_dft (FFT pipeline) should reduce per-frame latency.

    Prints measured timings so you can see the speedup directly.
    Run with: pytest tests/graph/test_passes.py::test_power_save_dne_is_faster -s
    """
    N = 40
    WARMUP = 5

    # Full graph — saliency_dft runs even in POWER_SAVE
    graph_full = build_l2_graph()
    for _ in range(WARMUP):
        graph_full.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    t0 = time.perf_counter()
    for _ in range(N):
        graph_full.execute(frame, battery_mode=BatteryMode.POWER_SAVE)
    ms_full = (time.perf_counter() - t0) / N * 1000

    # Optimized graph — saliency_dft eliminated
    graph_opt = build_l2_graph()
    dne = dead_node_elimination(graph_opt, BatteryMode.POWER_SAVE, measure_frame=frame)
    for _ in range(WARMUP):
        dne.graph.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    t0 = time.perf_counter()
    for _ in range(N):
        dne.graph.execute(frame, battery_mode=BatteryMode.POWER_SAVE)
    ms_opt = (time.perf_counter() - t0) / N * 1000

    speedup = ms_full / ms_opt if ms_opt > 0 else float("inf")

    print(f"\n  ┌── dead_node_elimination timing ─────────────────┐")
    print(f"  │  Full graph (saliency runs):    {ms_full:6.2f} ms/frame  │")
    print(f"  │  Optimized  (saliency removed): {ms_opt:6.2f} ms/frame  │")
    print(f"  │  Speedup:                       {speedup:6.1f}x           │")
    if dne.time_saved_ms > 0:
        print(f"  │  Saliency DFT cost:             {dne.time_saved_ms:6.2f} ms/frame  │")
    print(f"  └─────────────────────────────────────────────────┘")

    assert ms_opt < ms_full, (
        f"Optimized graph ({ms_opt:.2f} ms) should be faster than full graph "
        f"({ms_full:.2f} ms) in POWER_SAVE mode"
    )


def test_elimination_result_reports_time_saved(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(
        graph, BatteryMode.POWER_SAVE, measure_frame=frame
    )
    assert result.time_saved_ms > 0, (
        "saliency_dft elimination should report measurable time saved"
    )
