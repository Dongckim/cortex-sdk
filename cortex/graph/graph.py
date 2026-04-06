"""Graph IR — analogous to an MLIR Module containing a linear Region.

Nodes are stored in topological order. Temporal state (_prev_gray,
_prev_score) is tracked explicitly in the graph's state dict rather
than as hidden instance fields inside HybridROI.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cortex.graph.node import Node

logger = logging.getLogger(__name__)


class Graph:
    """Computational graph of the cortex L2 pipeline.

    Execution model:
      - Nodes run in the order they were added (topological order).
      - Each node's fn receives a context dict and returns a dict of
        its output values, which are merged into the context.
      - Temporal state (prev_gray, prev_score) persists across execute()
        calls, made explicit rather than hidden in object fields.

    In MLIR terms, this is a Module containing one Region with one Block
    of Operations executed sequentially (no control flow).
    """

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        # Temporal state: explicit, not hidden in HybridROI instance vars.
        # In MLIR terms these are block arguments carrying loop-carried values.
        self._state: dict[str, Any] = {
            "prev_gray": None,
            "prev_score": None,
        }

    def add_node(self, node: Node) -> None:
        """Append a node. Caller is responsible for topological ordering.

        Args:
            node: Node to append.
        """
        self._nodes.append(node)

    @property
    def nodes(self) -> list[Node]:
        """All nodes in topological order."""
        return list(self._nodes)

    def node_names(self) -> list[str]:
        """Names of all nodes in order."""
        return [n.name for n in self._nodes]

    def get_node(self, name: str) -> Node | None:
        """Look up a node by name.

        Args:
            name: Node name.

        Returns:
            Node if found, else None.
        """
        for n in self._nodes:
            if n.name == name:
                return n
        return None

    def execute(self, frame: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Execute all nodes in topological order.

        Each node's fn receives the accumulated context dict and returns
        a dict of its outputs, which are merged back into context.
        Nodes with no fn are skipped (useful for input placeholder nodes).

        Temporal state (prev_gray, prev_score) is carried across calls.

        Args:
            frame: Input BGR frame as numpy array.
            **kwargs: Additional context values, e.g. battery_mode.

        Returns:
            Final context dict containing all node outputs.
        """
        context: dict[str, Any] = {
            "frame": frame,
            "prev_gray": self._state["prev_gray"],
            "prev_score": self._state["prev_score"],
            **kwargs,
        }

        for node in self._nodes:
            fn = node.metadata.get("fn")
            if fn is None:
                logger.debug("node %s has no fn — skipping", node.name)
                continue
            outputs = fn(context)
            if outputs:
                context.update(outputs)

        # Persist loop-carried temporal state for next frame
        if context.get("prev_gray") is not None:
            self._state["prev_gray"] = context["prev_gray"]
        if context.get("prev_score") is not None:
            self._state["prev_score"] = context["prev_score"]

        return context

    def profile_execute(
        self, frame: np.ndarray, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Execute the graph and return per-node latencies (ms).

        Same semantics as execute(), but wraps each node's fn with a
        perf_counter timer. Used for live profiling dashboards.

        Args:
            frame: Input BGR frame.
            **kwargs: Additional context values (e.g. battery_mode).

        Returns:
            (context, timings) where timings maps node_name → ms.
        """
        import time as _time

        context: dict[str, Any] = {
            "frame": frame,
            "prev_gray": self._state["prev_gray"],
            "prev_score": self._state["prev_score"],
            **kwargs,
        }
        timings: dict[str, float] = {}

        for node in self._nodes:
            fn = node.metadata.get("fn")
            if fn is None:
                timings[node.name] = 0.0
                continue
            t0 = _time.perf_counter()
            outputs = fn(context)
            timings[node.name] = (_time.perf_counter() - t0) * 1000
            if outputs:
                context.update(outputs)

        if context.get("prev_gray") is not None:
            self._state["prev_gray"] = context["prev_gray"]
        if context.get("prev_score") is not None:
            self._state["prev_score"] = context["prev_score"]

        return context, timings

    def reset_state(self) -> None:
        """Clear temporal state (prev_gray, prev_score).

        Call this when starting a new video sequence.
        """
        self._state = {"prev_gray": None, "prev_score": None}

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"Graph({len(self._nodes)} nodes)"
