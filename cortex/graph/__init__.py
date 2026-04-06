"""cortex.graph — Graph IR and compiler passes for the L2 pipeline.

Public API:
    Node              — IR node (op, operands, is_compilable flag)
    Graph             — container with topological execution
    build_l2_graph()  — frontend: lowers L2 pipeline to Graph IR
    dead_node_elimination() — optimization pass
    EliminationResult — pass output (graph + eliminated names + timing)
    GraphVisualizer   — ASCII renderer
"""

from cortex.graph.node import Node
from cortex.graph.graph import Graph
from cortex.graph.builder import build_l2_graph
from cortex.graph.passes import dead_node_elimination, EliminationResult
from cortex.graph.visualizer import GraphVisualizer

__all__ = [
    "Node",
    "Graph",
    "build_l2_graph",
    "dead_node_elimination",
    "EliminationResult",
    "GraphVisualizer",
]
