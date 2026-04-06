"""ASCII graph visualizer for the cortex Graph IR.

Renders the graph structure and pass diffs as human-readable text —
analogous to mlir::Operation::print() or tvm.relay.pretty_print().
"""

from __future__ import annotations

from cortex.graph.graph import Graph
from cortex.graph.passes import EliminationResult


class GraphVisualizer:
    """Renders a Graph IR as ASCII topology and elimination diffs.

    Designed for terminal output during development and portfolio demos.
    """

    def print_graph(self, graph: Graph, title: str = "Graph IR") -> str:
        """Render the graph topology with compilable/external annotations.

        Args:
            graph: Graph to render.
            title: Section header label.

        Returns:
            Multi-line ASCII string.
        """
        lines: list[str] = [f"╔══ {title} {'═' * max(0, 44 - len(title))}╗"]
        nodes = graph.nodes

        for i, node in enumerate(nodes):
            is_last = i == len(nodes) - 1
            branch = "└─" if is_last else "├─"
            tag = "[ compilable ]" if node.is_compilable else "[external_call]"
            desc = node.metadata.get("description", "")
            lines.append(f"║  {branch} {tag}  {node.name}")
            if desc:
                indent = "║       " if is_last else "║  │    "
                lines.append(f"{indent}  {desc}")
            if not is_last:
                lines.append("║  │")

        lines.append(f"╚{'═' * 49}╝")
        lines.append(f"   {len(nodes)} nodes")
        return "\n".join(lines)

    def print_elimination_diff(self, result: EliminationResult) -> str:
        """Show which nodes were removed and the measured time savings.

        Args:
            result: Output of dead_node_elimination pass.

        Returns:
            Multi-line diff string.
        """
        lines = ["┌── dead_node_elimination pass ─────────────────┐"]

        if not result.eliminated:
            lines.append("│  No nodes eliminated.")
            lines.append("└────────────────────────────────────────────────┘")
            return "\n".join(lines)

        for name in result.eliminated:
            lines.append(f"│  [-] ELIMINATED: {name}")

        if result.time_saved_ms > 0:
            lines.append(f"│")
            lines.append(f"│  Measured time saved: {result.time_saved_ms:.2f} ms/frame")
        lines.append("└────────────────────────────────────────────────┘")
        return "\n".join(lines)

    def print_before_after(
        self,
        original: Graph,
        result: EliminationResult,
    ) -> str:
        """Full before/after comparison with pass diff.

        Args:
            original: Pre-pass graph.
            result: EliminationResult from dead_node_elimination.

        Returns:
            Combined ASCII string for terminal output.
        """
        sections = [
            self.print_graph(original, title="Before: dead_node_elimination"),
            "",
            self.print_elimination_diff(result),
            "",
            self.print_graph(result.graph, title="After:  dead_node_elimination"),
        ]
        return "\n".join(sections)
