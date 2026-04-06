"""Tests for CortexCompiler — partition and benchmark."""

import numpy as np
import pytest

from cortex.compiler import CortexCompiler, PartitionResult
from cortex.graph import build_l2_graph


@pytest.fixture
def graph():
    return build_l2_graph()


@pytest.fixture
def compiler(graph):
    return CortexCompiler(graph)


@pytest.fixture
def frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[120:360, 160:480] = 180
    return img


# ── partition ─────────────────────────────────────────────────────────

def test_partition_returns_partition_result(compiler):
    result = compiler.partition()
    assert isinstance(result, PartitionResult)


def test_partition_all_nodes_covered(compiler, graph):
    result = compiler.partition()
    all_names = set(graph.node_names())
    partitioned = set(result.compilable_names) | set(result.external_names)
    assert all_names == partitioned


def test_partition_no_overlap(compiler):
    result = compiler.partition()
    overlap = set(result.compilable_names) & set(result.external_names)
    assert len(overlap) == 0


def test_text_roi_mser_is_external(compiler):
    result = compiler.partition()
    assert "text_roi_mser" in result.external_names


def test_saliency_dft_is_compilable(compiler):
    result = compiler.partition()
    assert "saliency_dft" in result.compilable_names


def test_compilable_nodes_have_flag(compiler, graph):
    result = compiler.partition()
    for node in result.compilable:
        assert node.is_compilable is True


def test_external_nodes_lack_flag(compiler):
    result = compiler.partition()
    for node in result.external:
        assert node.is_compilable is False


# ── profile_nodes ─────────────────────────────────────────────────────

def test_profile_nodes_returns_all_node_names(compiler, graph, frame):
    per_node = compiler.profile_nodes(frame, n=5, warmup=1)
    assert set(per_node.keys()) == set(graph.node_names())


def test_profile_nodes_positive_latencies(compiler, frame):
    per_node = compiler.profile_nodes(frame, n=5, warmup=1)
    for name, ms in per_node.items():
        assert ms >= 0.0, f"{name} has negative latency"


# ── benchmark ─────────────────────────────────────────────────────────

def test_benchmark_returns_benchmark_result(compiler, frame):
    from cortex.compiler import BenchmarkResult
    bench = compiler.benchmark(frame=frame, n=5, warmup=2)
    assert isinstance(bench, BenchmarkResult)


def test_benchmark_positive_ms(compiler, frame):
    bench = compiler.benchmark(frame=frame, n=5, warmup=2)
    assert bench.baseline_ms   > 0
    assert bench.vectorized_ms > 0
    assert bench.jit_ms        > 0


def test_benchmark_speedups_positive(compiler, frame):
    bench = compiler.benchmark(frame=frame, n=5, warmup=2)
    assert bench.speedup_vec > 0
    assert bench.speedup_jit > 0
