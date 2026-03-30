"""Tests for AdaptiveEncoder."""

import numpy as np
import pytest

from cortex.optimizer.encoder import AdaptiveEncoder, NetworkCondition


@pytest.fixture
def encoder() -> AdaptiveEncoder:
    return AdaptiveEncoder()


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_wifi_encoding(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    data = encoder.encode(frame, NetworkCondition.WIFI)
    assert data is not None
    assert len(data) > 0
    assert len(data) < frame.nbytes


def test_lte_encoding(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    data = encoder.encode(frame, NetworkCondition.LTE)
    assert data is not None
    assert len(data) > 0


def test_weak_encoding(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    data = encoder.encode(frame, NetworkCondition.WEAK)
    assert data is not None
    assert len(data) > 0


def test_wifi_larger_than_weak(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    wifi = encoder.encode(frame, NetworkCondition.WIFI)
    weak = encoder.encode(frame, NetworkCondition.WEAK)
    assert len(wifi) > len(weak)


def test_offline_returns_none(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    result = encoder.encode(frame, NetworkCondition.OFFLINE)
    assert result is None


def test_offline_queues_frame(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    encoder.encode(frame, NetworkCondition.OFFLINE)
    assert encoder.compression_stats["queue_size"] == 1


def test_flush_queue(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    encoder.encode(frame, NetworkCondition.OFFLINE)
    encoder.encode(frame, NetworkCondition.OFFLINE)
    results = encoder.flush_queue(NetworkCondition.WIFI)
    assert len(results) == 2
    assert encoder.compression_stats["queue_size"] == 0


def test_estimate_tokens(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    data = encoder.encode(frame, NetworkCondition.WIFI)
    tokens = encoder.estimate_tokens(data)
    assert tokens >= 85


def test_estimate_cost(encoder: AdaptiveEncoder) -> None:
    cost = encoder.estimate_cost(1000, "claude-sonnet-4-20250514")
    assert cost > 0
    assert cost < 1.0


def test_compression_stats(encoder: AdaptiveEncoder, frame: np.ndarray) -> None:
    encoder.encode(frame, NetworkCondition.WIFI)
    stats = encoder.compression_stats
    assert stats["encode_count"] == 1
    assert stats["compression_ratio"] > 0
