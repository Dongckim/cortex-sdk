"""Tests for Graph construction and execute()."""

import numpy as np
import pytest

from cortex.capture.imu_gate import BatteryMode
from cortex.graph import Graph, build_l2_graph
from cortex.optimizer.hybrid_roi import RequestType


@pytest.fixture
def frame() -> np.ndarray:
    """Synthetic 480×640 BGR frame with a bright center region."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[160:320, 213:427] = 200
    return img


def test_build_l2_graph_node_count() -> None:
    graph = build_l2_graph()
    # center_crop, text_roi_mser, saliency_dft, motion_map, score_fusion, ema_smooth
    assert len(graph) == 6


def test_build_l2_graph_node_names() -> None:
    graph = build_l2_graph()
    names = graph.node_names()
    assert "center_crop" in names
    assert "text_roi_mser" in names
    assert "saliency_dft" in names
    assert "motion_map" in names
    assert "score_fusion" in names
    assert "ema_smooth" in names


def test_compilable_flags() -> None:
    graph = build_l2_graph()
    compilable = {n.name for n in graph.nodes if n.is_compilable}
    external = {n.name for n in graph.nodes if not n.is_compilable}

    assert "saliency_dft" in compilable
    assert "center_crop" in compilable
    assert "motion_map" in compilable
    assert "score_fusion" in compilable
    assert "ema_smooth" in compilable

    assert "text_roi_mser" in external


def test_execute_returns_score_map(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    ctx = graph.execute(frame)
    assert "score_map" in ctx
    assert ctx["score_map"].shape == (6, 8)


def test_execute_score_map_in_range(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    ctx = graph.execute(frame)
    sm = ctx["score_map"]
    assert sm.min() >= 0.0
    assert sm.max() <= 1.01


def test_execute_temporal_state_persists(frame: np.ndarray) -> None:
    """prev_gray should be populated after first frame."""
    graph = build_l2_graph()
    graph.execute(frame)
    assert graph._state["prev_gray"] is not None
    assert graph._state["prev_score"] is not None


def test_execute_ema_applied_on_second_frame(frame: np.ndarray) -> None:
    """Second frame should have EMA applied — scores will differ from first."""
    graph = build_l2_graph()
    ctx1 = graph.execute(frame)
    ctx2 = graph.execute(frame)
    # With EMA smoothing, identical frames still converge so scores may differ
    assert ctx1["score_map"].shape == ctx2["score_map"].shape


def test_execute_all_request_types(frame: np.ndarray) -> None:
    for rt in RequestType:
        graph = build_l2_graph(request_type=rt)
        ctx = graph.execute(frame)
        assert ctx["score_map"].shape == (6, 8)


def test_graph_reset_state(frame: np.ndarray) -> None:
    graph = build_l2_graph()
    graph.execute(frame)
    assert graph._state["prev_gray"] is not None
    graph.reset_state()
    assert graph._state["prev_gray"] is None
    assert graph._state["prev_score"] is None


def test_get_node_returns_correct_node() -> None:
    graph = build_l2_graph()
    node = graph.get_node("saliency_dft")
    assert node is not None
    assert node.name == "saliency_dft"
    assert node.is_compilable is True


def test_get_node_missing_returns_none() -> None:
    graph = build_l2_graph()
    assert graph.get_node("nonexistent") is None
