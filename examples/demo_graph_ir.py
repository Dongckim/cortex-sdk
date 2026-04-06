"""Graph IR demo — dead_node_elimination before/after.

Run:
    python examples/demo_graph_ir.py

Shows:
  1. Full Graph IR printed as ASCII topology
  2. dead_node_elimination pass applied for POWER_SAVE mode
  3. Before/after diff with measured time savings
  4. Semantic equivalence check (score_map identical)
"""

import time

import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.graph import GraphVisualizer, build_l2_graph, dead_node_elimination
from cortex.optimizer.hybrid_roi import RequestType


def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4] = 180
    return img


def benchmark(graph, frame, battery_mode, n: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        graph.execute(frame, battery_mode=battery_mode)
    t0 = time.perf_counter()
    for _ in range(n):
        graph.execute(frame, battery_mode=battery_mode)
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    frame = make_frame()
    viz = GraphVisualizer()

    print("\n" + "=" * 55)
    print("  cortex Graph IR — dead_node_elimination demo")
    print("=" * 55)

    # ----------------------------------------------------------------
    # 1. Build and print the full graph
    # ----------------------------------------------------------------
    graph = build_l2_graph(request_type=RequestType.GENERAL)
    print("\n" + viz.print_graph(graph, title="Full L2 Graph IR"))

    # ----------------------------------------------------------------
    # 2. Apply dead_node_elimination for POWER_SAVE
    # ----------------------------------------------------------------
    print("\nApplying dead_node_elimination pass (POWER_SAVE mode)...\n")
    dne = dead_node_elimination(
        graph,
        battery_mode=BatteryMode.POWER_SAVE,
        measure_frame=frame,
        n_measure=30,
    )

    print(viz.print_before_after(graph, dne))

    # ----------------------------------------------------------------
    # 3. Semantic equivalence check
    # ----------------------------------------------------------------
    print("\n── Semantic equivalence check ───────────────────────")
    g_full = build_l2_graph()
    ctx_full = g_full.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    g_opt = build_l2_graph()
    dne_check = dead_node_elimination(g_opt, BatteryMode.POWER_SAVE)
    ctx_opt = dne_check.graph.execute(frame, battery_mode=BatteryMode.POWER_SAVE)

    max_diff = np.abs(ctx_full["score_map"] - ctx_opt["score_map"]).max()
    ok = "PASS ✓" if max_diff < 1e-5 else f"FAIL ✗  max_diff={max_diff:.2e}"
    print(f"  score_map identical before/after DNE: {ok}")

    # ----------------------------------------------------------------
    # 4. Per-frame latency comparison
    # ----------------------------------------------------------------
    print("\n── Per-frame latency (N=50 frames, 640×480) ─────────")
    g_full2 = build_l2_graph()
    ms_full = benchmark(g_full2, frame, BatteryMode.POWER_SAVE)

    g_opt2 = build_l2_graph()
    dne2 = dead_node_elimination(g_opt2, BatteryMode.POWER_SAVE)
    ms_opt = benchmark(dne2.graph, frame, BatteryMode.POWER_SAVE)

    speedup = ms_full / ms_opt if ms_opt > 0 else float("inf")

    print(f"  Full graph  (saliency_dft runs):    {ms_full:6.2f} ms/frame")
    print(f"  Optimized   (saliency_dft removed): {ms_opt:6.2f} ms/frame")
    print(f"  Speedup:                            {speedup:6.1f}x")
    if dne.time_saved_ms > 0:
        print(f"  Saliency DFT cost (isolated):       {dne.time_saved_ms:6.2f} ms/frame")

    print()


if __name__ == "__main__":
    main()
