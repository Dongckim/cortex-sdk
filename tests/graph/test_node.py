"""Tests for the Node IR class."""

import pytest

from cortex.graph.node import Node


def test_node_defaults() -> None:
    n = Node(name="relu", op_type="relu")
    assert n.inputs == []
    assert n.outputs == []
    assert n.metadata == {}
    assert n.is_compilable is False


def test_node_compilable_flag() -> None:
    n = Node(name="saliency_dft", op_type="spectral_residual", is_compilable=True)
    assert n.is_compilable is True


def test_external_call_not_compilable() -> None:
    n = Node(name="text_roi_mser", op_type="external_call", is_compilable=False)
    assert n.is_compilable is False


def test_node_inputs_are_nodes() -> None:
    parent = Node(name="center_crop", op_type="gaussian_window", is_compilable=True)
    child = Node(name="saliency_dft", op_type="spectral_residual", inputs=[parent])
    assert parent in child.inputs


def test_node_repr_contains_name() -> None:
    n = Node(name="score_fusion", op_type="weighted_sum", is_compilable=True)
    assert "score_fusion" in repr(n)
    assert "compilable" in repr(n)


def test_node_repr_external_call() -> None:
    n = Node(name="mser", op_type="external_call", is_compilable=False)
    assert "external_call" in repr(n)
