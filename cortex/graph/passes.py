"""Compiler optimization passes for the cortex Graph IR.

A pass is a transformation over the IR that preserves execution semantics
while improving efficiency. This mirrors MLIR's pass infrastructure
(mlir::Pass) and TVM's relay.transform module.

Currently implemented:
  dead_node_elimination — removes nodes whose outputs are never consumed
                          in the given hardware/battery configuration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.graph.graph import Graph
from cortex.graph.node import Node

logger = logging.getLogger(__name__)

# Dummy context used when timing eliminated nodes in isolation.
_DUMMY_SC = np.zeros((6, 8), dtype=np.float32)


@dataclass
class EliminationResult:
    """Output of the dead_node_elimination pass.

    Attributes:
        graph: Optimized graph with dead nodes removed.
        eliminated: Names of removed nodes.
        time_saved_ms: Measured time saved per frame (ms).
            0.0 if no measure_frame was provided.
    """

    graph: Graph
    eliminated: list[str] = field(default_factory=list)
    time_saved_ms: float = 0.0


def dead_node_elimination(
    graph: Graph,
    battery_mode: BatteryMode,
    measure_frame: np.ndarray | None = None,
    n_measure: int = 30,
) -> EliminationResult:
    """Remove nodes whose outputs are never consumed in this configuration.

    Dead node analysis (POWER_SAVE mode):
      In POWER_SAVE, HybridROI zeroes the saliency map and redistributes
      ws into wc. The output "ss" of saliency_dft is therefore never used
      in score_fusion. saliency_dft is a dead node — it performs expensive
      FFT computation whose result is immediately discarded.

      Eliminating it:
        - Saves the spectral residual DFT cost (~1–5 ms/frame on CPU).
        - Does not change score_fusion output: score_fusion checks ctx["ss"]
          and falls back to zeros when it is absent.

    This is analogous to MLIR's dead code elimination pass:
    operations whose results have no uses are erased from the IR.

    Args:
        graph: Source graph. Not modified in place.
        battery_mode: Current hardware mode. Determines dead set.
        measure_frame: Optional frame to measure actual time saved.
            If None, time_saved_ms = 0.0.
        n_measure: Number of iterations for timing measurement.

    Returns:
        EliminationResult with optimized graph and timing data.
    """
    dead_names: set[str] = set()

    if battery_mode == BatteryMode.POWER_SAVE:
        # saliency_dft output "ss" is zeroed in score_fusion → dead
        dead_names.add("saliency_dft")
        logger.info(
            "[DNE pass] POWER_SAVE: marking saliency_dft as dead "
            "(ws redistributed to wc, ss output unused)"
        )

    # Build optimized graph without dead nodes.
    # Also clean up input references that point to eliminated nodes so
    # that Phase 3 graph partitioning sees a consistent graph.
    new_graph = Graph()
    new_graph._state = dict(graph._state)

    eliminated: list[str] = []
    for node in graph.nodes:
        if node.name in dead_names:
            eliminated.append(node.name)
            logger.info("[DNE pass] eliminated: %s  op=%s", node.name, node.op_type)
            continue

        clean_inputs = [
            inp
            for inp in node.inputs
            if not (isinstance(inp, Node) and inp.name in dead_names)
        ]
        new_node = Node(
            name=node.name,
            op_type=node.op_type,
            inputs=clean_inputs,
            outputs=node.outputs,
            metadata=node.metadata,
            is_compilable=node.is_compilable,
        )
        new_graph.add_node(new_node)

    # Measure actual time saved by timing the eliminated nodes in isolation.
    time_saved_ms = 0.0
    if measure_frame is not None and eliminated:
        dummy_ctx: dict = {
            "frame": measure_frame,
            "prev_gray": None,
            "prev_score": None,
            "sc": _DUMMY_SC.copy(),
        }

        for node_name in eliminated:
            original_node = graph.get_node(node_name)
            if original_node is None:
                continue
            fn = original_node.metadata.get("fn")
            if fn is None:
                continue

            # Warmup
            for _ in range(5):
                fn(dummy_ctx)

            t0 = time.perf_counter()
            for _ in range(n_measure):
                fn(dummy_ctx)
            elapsed_ms = (time.perf_counter() - t0) / n_measure * 1000
            time_saved_ms += elapsed_ms
            logger.info(
                "[DNE pass] %s costs %.2f ms/frame → saved by elimination",
                node_name,
                elapsed_ms,
            )

    return EliminationResult(
        graph=new_graph,
        eliminated=eliminated,
        time_saved_ms=time_saved_ms,
    )
