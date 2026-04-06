"""Tests for GraphVisualizer ASCII output."""

import numpy as np
import pytest

from cortex.capture.imu_gate import BatteryMode
from cortex.graph import GraphVisualizer, build_l2_graph, dead_node_elimination


@pytest.fixture
def graph():
    return build_l2_graph()


@pytest.fixture
def viz() -> GraphVisualizer:
    return GraphVisualizer()


def test_print_graph_contains_node_names(graph, viz: GraphVisualizer) -> None:
    output = viz.print_graph(graph)
    for name in ["center_crop", "text_roi_mser", "saliency_dft", "score_fusion"]:
        assert name in output


def test_print_graph_annotates_compilable(graph, viz: GraphVisualizer) -> None:
    output = viz.print_graph(graph)
    assert "compilable" in output
    assert "external_call" in output


def test_print_elimination_diff_shows_eliminated(viz: GraphVisualizer) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    output = viz.print_elimination_diff(result)
    assert "saliency_dft" in output
    assert "ELIMINATED" in output or "eliminated" in output.lower()


def test_print_elimination_diff_no_elimination(viz: GraphVisualizer) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.BALANCED)
    output = viz.print_elimination_diff(result)
    assert "No nodes eliminated" in output


def test_print_before_after_contains_both_graphs(viz: GraphVisualizer) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    output = viz.print_before_after(graph, result)
    assert "Before" in output
    assert "After" in output
    assert "saliency_dft" in output


def test_print_before_after_is_string(viz: GraphVisualizer) -> None:
    graph = build_l2_graph()
    result = dead_node_elimination(graph, BatteryMode.POWER_SAVE)
    output = viz.print_before_after(graph, result)
    assert isinstance(output, str)
    assert len(output) > 0
