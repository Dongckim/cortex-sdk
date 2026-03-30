"""Tests for CenterCropStrategy."""

import numpy as np
import pytest

from cortex.optimizer.center_crop import CenterCropStrategy


@pytest.fixture
def strategy() -> CenterCropStrategy:
    return CenterCropStrategy()


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((200, 300, 3), dtype=np.uint8)


def test_crop_default_ratio(strategy: CenterCropStrategy, frame: np.ndarray) -> None:
    cropped = strategy.crop(frame)
    assert cropped.shape[0] == 140  # 200 * 0.7
    assert cropped.shape[1] == 210  # 300 * 0.7


def test_crop_custom_ratio(strategy: CenterCropStrategy, frame: np.ndarray) -> None:
    cropped = strategy.crop(frame, ratio=0.5)
    assert cropped.shape[0] == 100
    assert cropped.shape[1] == 150


def test_crop_full_ratio(strategy: CenterCropStrategy, frame: np.ndarray) -> None:
    cropped = strategy.crop(frame, ratio=1.0)
    assert cropped.shape[:2] == frame.shape[:2]


def test_score_map_shape(strategy: CenterCropStrategy, frame: np.ndarray) -> None:
    score = strategy.score_map(frame)
    assert score.shape == (6, 8)


def test_score_map_custom_grid(
    strategy: CenterCropStrategy, frame: np.ndarray
) -> None:
    score = strategy.score_map(frame, grid=(4, 3))
    assert score.shape == (3, 4)


def test_score_map_center_is_max(
    strategy: CenterCropStrategy, frame: np.ndarray
) -> None:
    score = strategy.score_map(frame)
    center = score[score.shape[0] // 2, score.shape[1] // 2]
    assert center == pytest.approx(1.0, abs=0.01)


def test_score_map_corners_are_low(
    strategy: CenterCropStrategy, frame: np.ndarray
) -> None:
    score = strategy.score_map(frame)
    corners = [score[0, 0], score[0, -1], score[-1, 0], score[-1, -1]]
    assert all(c < 0.5 for c in corners)


def test_score_map_values_in_range(
    strategy: CenterCropStrategy, frame: np.ndarray
) -> None:
    score = strategy.score_map(frame)
    assert score.min() >= 0.0
    assert score.max() <= 1.0
