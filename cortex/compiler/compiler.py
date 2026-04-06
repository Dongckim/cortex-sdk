"""CortexCompiler — partitions Graph IR and compiles the compilable subgraph.

Compilation pipeline (MLIR analogy):
  Python ops
      ↓  cortex.graph.build_l2_graph()       # Frontend / Ingestion
  Graph IR  (Node with is_compilable flags)
      ↓  CortexCompiler.partition()           # Partitioning pass
  compilable subgraph  |  external_call nodes
      ↓  CortexCompiler.compile()             # Codegen (numba JIT)
  Compiled kernel  +  external_call stubs
      ↓  CortexCompiler.benchmark()           # Runtime profiling
  Latency report

The partition() step maps directly to TVM's bring_your_own_codegen (BYOC):
  is_compilable=False nodes stay as opaque external function calls.
  is_compilable=True  nodes are handed to the JIT compiler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from cortex.graph.graph import Graph
from cortex.graph.node import Node
from cortex.compiler.saliency_kernel import (
    saliency_baseline,
    saliency_vectorized,
    saliency_jit,
    warmup_jit,
)

logger = logging.getLogger(__name__)


# ── partition result ──────────────────────────────────────────────────

@dataclass
class PartitionResult:
    """Output of CortexCompiler.partition().

    Attributes:
        compilable:   Nodes with is_compilable=True (JIT targets).
        external:     Nodes with is_compilable=False (BYOC boundary).
        compilable_names: Convenience list of node names.
        external_names:   Convenience list of node names.
    """
    compilable: list[Node]
    external:   list[Node]

    @property
    def compilable_names(self) -> list[str]:
        return [n.name for n in self.compilable]

    @property
    def external_names(self) -> list[str]:
        return [n.name for n in self.external]


# ── benchmark result ──────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Per-frame latency measurements for all three kernel variants.

    Attributes:
        baseline_ms:    Avg ms/frame for baseline kernel.
        vectorized_ms:  Avg ms/frame for vectorized kernel.
        jit_ms:         Avg ms/frame for JIT kernel.
        speedup_vec:    vectorized / baseline speedup.
        speedup_jit:    jit / baseline speedup.
        n_frames:       Number of benchmark frames.
    """
    baseline_ms:   float
    vectorized_ms: float
    jit_ms:        float
    speedup_vec:   float
    speedup_jit:   float
    n_frames:      int
    per_node_ms:   dict[str, float] = field(default_factory=dict)


# ── compiler ─────────────────────────────────────────────────────────

class CortexCompiler:
    """Partitions a Graph IR and compiles the compilable subgraph.

    Usage::

        graph    = build_l2_graph()
        compiler = CortexCompiler(graph)
        result   = compiler.partition()
        bench    = compiler.benchmark(n=100)

    Args:
        graph: Graph IR produced by cortex.graph.build_l2_graph().
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    # ── Stage 1: partition ────────────────────────────────────────────

    def partition(self) -> PartitionResult:
        """Walk the Graph IR and split nodes by is_compilable flag.

        Compilation stage analogy: partitioning pass in TVM/MLIR.
          is_compilable=True  → compilable subgraph (JIT target)
          is_compilable=False → external_call boundary (BYOC — stays opaque)

        Returns:
            PartitionResult with compilable and external node lists.
        """
        compilable: list[Node] = []
        external:   list[Node] = []

        for node in self._graph.nodes:
            if node.is_compilable:
                compilable.append(node)
            else:
                external.append(node)

        logger.debug(
            "partition: compilable=%s  external=%s",
            [n.name for n in compilable],
            [n.name for n in external],
        )
        return PartitionResult(compilable=compilable, external=external)

    # ── Stage 2: per-node profiling ───────────────────────────────────

    def profile_nodes(
        self,
        frame: np.ndarray,
        n: int = 100,
        warmup: int = 5,
    ) -> dict[str, float]:
        """Measure per-node latency by running the full graph with profiling.

        Compilation stage analogy: profiling pass — identifies hot nodes
        before committing to a codegen strategy.

        Args:
            frame:  Representative input frame.
            n:      Number of timed iterations.
            warmup: Warmup iterations (excluded from timing).

        Returns:
            Dict mapping node_name → avg ms/frame.
        """
        from cortex.capture.imu_gate import BatteryMode

        for _ in range(warmup):
            self._graph.profile_execute(frame, battery_mode=BatteryMode.BALANCED)

        accum: dict[str, float] = {}
        for _ in range(n):
            _, timings = self._graph.profile_execute(
                frame, battery_mode=BatteryMode.BALANCED
            )
            for name, ms in timings.items():
                accum[name] = accum.get(name, 0.0) + ms

        return {name: ms / n for name, ms in accum.items()}

    # ── Stage 3: kernel benchmark ─────────────────────────────────────

    def benchmark(
        self,
        frame: np.ndarray | None = None,
        n: int = 100,
        warmup: int = 10,
    ) -> BenchmarkResult:
        """Benchmark all three saliency kernel variants.

        Compilation stage analogy: runtime profiling after codegen — measures
        the actual speedup delivered by each optimization pass.

        JIT warmup is performed before timing to exclude LLVM compile cost,
        analogous to TVM's ahead-of-time (AOT) compilation step.

        Args:
            frame:  Input frame. If None, uses a synthetic 640×480 frame.
            n:      Number of timed iterations per kernel.
            warmup: Warmup iterations (JIT compile triggered here).

        Returns:
            BenchmarkResult with ms/frame and speedup ratios.
        """
        if frame is None:
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Per-node profiling
        per_node = self.profile_nodes(frame, n=n, warmup=warmup)

        # JIT warmup — triggers LLVM compilation (AOT analogue)
        warmup_jit()

        def _time_kernel(fn, iters):
            for _ in range(warmup):
                fn(frame)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(frame)
            return (time.perf_counter() - t0) / iters * 1000

        baseline_ms   = _time_kernel(saliency_baseline,   n)
        vectorized_ms = _time_kernel(saliency_vectorized, n)
        jit_ms        = _time_kernel(saliency_jit,        n)

        speedup_vec = baseline_ms / vectorized_ms if vectorized_ms > 0 else 0.0
        speedup_jit = baseline_ms / jit_ms        if jit_ms        > 0 else 0.0

        return BenchmarkResult(
            baseline_ms=baseline_ms,
            vectorized_ms=vectorized_ms,
            jit_ms=jit_ms,
            speedup_vec=speedup_vec,
            speedup_jit=speedup_jit,
            n_frames=n,
            per_node_ms=per_node,
        )
