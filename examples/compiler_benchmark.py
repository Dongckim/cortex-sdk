"""Phase 3 compiler benchmark — profiling report for saliency DFT.

Run:
    python examples/compiler_benchmark.py

Compilation pipeline stages shown:
  1. Partition Graph IR → compilable vs external_call nodes
  2. Profile per-node latency (100 frames, 640×480)
  3. Benchmark three kernel variants: baseline / vectorized / JIT
  4. Verify numerical equivalence across all three
  5. Write profiling_report.txt

MLIR analogy for each stage is printed inline.
"""

import os
import time

import cv2
import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.compiler import CortexCompiler, saliency_baseline, saliency_jit, saliency_vectorized, warmup_jit
from cortex.graph import build_l2_graph
from cortex.optimizer.hybrid_roi import RequestType


REPORT_PATH = os.path.join(os.path.dirname(__file__), "profiling_report.txt")
N           = 100
WARMUP      = 10
GRID        = (8, 6)


def make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Synthetic frame with text-like regions and edges."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 180
    cv2.putText(img, "CORTEX PHASE 3", (w // 4 + 10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


def _bench(fn, frame, n=N, warmup=WARMUP):
    for _ in range(warmup):
        fn(frame)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(frame)
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    frame = make_frame()

    print("\n" + "=" * 60)
    print("  cortex Phase 3 — Compiler Benchmark")
    print("=" * 60)

    # ── Stage 1: Build Graph IR + Partition ──────────────────────────
    print("\n[Stage 1]  Build Graph IR  →  Partition")
    print("  MLIR analogy: Frontend ingestion → Partitioning pass\n")

    graph    = build_l2_graph(request_type=RequestType.GENERAL)
    compiler = CortexCompiler(graph)
    parts    = compiler.partition()

    print(f"  Compilable nodes  ({len(parts.compilable)}):  "
          + ", ".join(parts.compilable_names))
    print(f"  External nodes    ({len(parts.external)}):  "
          + ", ".join(parts.external_names))
    print()
    print("  Why external_call nodes stay opaque:")
    for n in parts.external:
        lib = n.metadata.get("external_lib", "unknown")
        print(f"    {n.name:<20} op={n.op_type}  lib={lib}")
        print(f"      → TVM BYOC boundary: cannot lower {lib} into compiler IR")

    # ── Stage 2: Per-node profiling ──────────────────────────────────
    print("\n[Stage 2]  Per-node profiling  (N=100 frames, 640×480)")
    print("  MLIR analogy: Profiling pass — identify hot nodes before codegen\n")

    per_node = compiler.profile_nodes(frame, n=N, warmup=WARMUP)

    total_ms = sum(per_node.values())
    for name, ms in per_node.items():
        node = graph.get_node(name)
        tag  = "[compilable]   " if (node and node.is_compilable) else "[external_call]"
        bar  = "█" * max(1, int(ms / max(per_node.values()) * 30))
        print(f"  {name:<20} {tag}  {ms:6.3f} ms  {bar}")
    print(f"\n  {'Total L2':20}               {total_ms:6.3f} ms / frame")

    # target identification
    compilable_ms = {
        name: ms for name, ms in per_node.items()
        if graph.get_node(name) and graph.get_node(name).is_compilable
    }
    if compilable_ms:
        target = max(compilable_ms, key=compilable_ms.get)
        print(f"\n  → Compilation target: {target}  "
              f"({compilable_ms[target]:.3f} ms, largest compilable node)")

    # ── Stage 3: Kernel benchmark ────────────────────────────────────
    print("\n[Stage 3]  Saliency DFT kernel benchmark")
    print("  MLIR analogy: Codegen → Runtime profiling\n")

    print("  Warming up JIT (numba → LLVM compile, one-time cost)...", end=" ", flush=True)
    t_jit_compile = time.perf_counter()
    warmup_jit()
    print(f"done  ({(time.perf_counter() - t_jit_compile)*1000:.0f} ms  ← AOT compile cost)")

    ms_base = _bench(saliency_baseline,   frame)
    ms_vec  = _bench(saliency_vectorized, frame)
    ms_jit  = _bench(saliency_jit,        frame)

    sv = ms_base / ms_vec if ms_vec > 0 else 0
    sj = ms_base / ms_jit if ms_jit > 0 else 0

    print(f"\n  {'Kernel':<16}  {'ms/frame':>10}  {'speedup':>8}  Optimization pass")
    print("  " + "-" * 60)
    print(f"  {'baseline':<16}  {ms_base:10.3f}  {'1.0x':>8}  None (Python loop)")
    print(f"  {'vectorized':<16}  {ms_vec:10.3f}  {sv:7.1f}x  numpy reshape+mean (auto-vec)")
    print(f"  {'jit':<16}  {ms_jit:10.3f}  {sj:7.1f}x  numba @jit → LLVM codegen")

    # ── Stage 4: Numerical equivalence ──────────────────────────────
    print("\n[Stage 4]  Numerical equivalence check")
    print("  All three kernels must produce identical score maps.\n")

    s_base = saliency_baseline(frame, GRID)
    s_vec  = saliency_vectorized(frame, GRID)
    s_jit  = saliency_jit(frame, GRID)

    diff_vec = float(np.abs(s_base - s_vec).max())
    diff_jit = float(np.abs(s_base - s_jit).max())

    ok_vec = "PASS ✓" if diff_vec < 1e-4 else f"FAIL ✗  diff={diff_vec:.2e}"
    ok_jit = "PASS ✓" if diff_jit < 1e-4 else f"FAIL ✗  diff={diff_jit:.2e}"
    print(f"  baseline vs vectorized:  max_diff={diff_vec:.2e}  {ok_vec}")
    print(f"  baseline vs jit:         max_diff={diff_jit:.2e}  {ok_jit}")

    # ── Stage 5: Write report ────────────────────────────────────────
    report = _build_report(
        parts, per_node, ms_base, ms_vec, ms_jit, sv, sj,
        diff_vec, diff_jit, N, frame.shape,
    )
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\n  Report written → {REPORT_PATH}")
    print()


def _build_report(parts, per_node, ms_base, ms_vec, ms_jit, sv, sj,
                  diff_vec, diff_jit, n, shape) -> str:
    lines = [
        "=== cortex Phase 3 — Profiling Report ===",
        f"frames: {n}  |  resolution: {shape[1]}×{shape[0]}",
        "",
        "--- Partition ---",
        f"compilable   : {', '.join(parts.compilable_names)}",
        f"external_call: {', '.join(parts.external_names)}",
        "",
        "--- Per-node latency (L2 Graph IR) ---",
    ]
    for name, ms in per_node.items():
        tag = "[compilable]   " if name in parts.compilable_names else "[external_call]"
        lines.append(f"  {name:<20} {tag}  {ms:.3f} ms")
    lines += [
        "",
        "--- Saliency DFT kernel variants ---",
        f"  baseline    {ms_base:.3f} ms  (1.0x)   Python loop",
        f"  vectorized  {ms_vec:.3f} ms  ({sv:.1f}x)   numpy reshape+mean",
        f"  jit         {ms_jit:.3f} ms  ({sj:.1f}x)   numba @jit → LLVM",
        "",
        "--- Numerical equivalence ---",
        f"  baseline vs vectorized:  max_diff={diff_vec:.2e}",
        f"  baseline vs jit:         max_diff={diff_jit:.2e}",
        "",
        "--- Why score_fusion is NOT the compilation target ---",
        "  score_fusion operates on a 6×8 = 48-cell grid.",
        "  numpy handles 48 float32 multiplications in ~1 μs.",
        "  numba/TVM kernel launch overhead (~5–20 μs) exceeds the work itself.",
        "  Rule: compile only when computation >> kernel launch cost.",
        "",
        "--- Why saliency_dft IS the compilation target ---",
        "  Operates on a 64×64 float32 array (4096 elements).",
        "  Contains a Python-level grid loop (48 iterations) as additional overhead.",
        "  Pure numpy — no cv2 black-box calls in the hot path.",
        "  is_compilable=True in Graph IR → selected by partition() pass.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
