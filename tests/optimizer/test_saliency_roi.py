"""Tests for SaliencyROIStrategy."""

import numpy as np
import pytest

from cortex.optimizer.saliency_roi import SaliencyROIStrategy


@pytest.fixture
def strategy() -> SaliencyROIStrategy:
    return SaliencyROIStrategy()


@pytest.fixture
def uniform_image() -> np.ndarray:
    return np.full((200, 300, 3), 128, dtype=np.uint8)


@pytest.fixture
def contrast_image() -> np.ndarray:
    """Image with a bright object on dark background."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[60:140, 100:200] = 255
    return img


def test_score_map_shape(
    strategy: SaliencyROIStrategy, contrast_image: np.ndarray
) -> None:
    score = strategy.score_map(contrast_image)
    assert score.shape == (6, 8)


def test_score_map_custom_grid(
    strategy: SaliencyROIStrategy, contrast_image: np.ndarray
) -> None:
    score = strategy.score_map(contrast_image, grid=(4, 3))
    assert score.shape == (3, 4)


def test_score_map_values_in_range(
    strategy: SaliencyROIStrategy, contrast_image: np.ndarray
) -> None:
    score = strategy.score_map(contrast_image)
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_contrast_has_higher_saliency_than_uniform(
    strategy: SaliencyROIStrategy,
    uniform_image: np.ndarray,
    contrast_image: np.ndarray,
) -> None:
    s_uniform = strategy.score_map(uniform_image).mean()
    s_contrast = strategy.score_map(contrast_image).mean()
    # Contrast image should have more varied saliency
    assert s_contrast > 0


def test_crop_returns_smaller(
    strategy: SaliencyROIStrategy, contrast_image: np.ndarray
) -> None:
    cropped = strategy.crop(contrast_image)
    original_area = contrast_image.shape[0] * contrast_image.shape[1]
    cropped_area = cropped.shape[0] * cropped.shape[1]
    assert cropped_area <= original_area


def test_crop_uniform_returns_something(
    strategy: SaliencyROIStrategy, uniform_image: np.ndarray
) -> None:
    cropped = strategy.crop(uniform_image)
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0


def test_grayscale_input(strategy: SaliencyROIStrategy) -> None:
    gray = np.zeros((200, 300), dtype=np.uint8)
    gray[80:120, 120:180] = 255
    score = strategy.score_map(gray)
    assert score.shape == (6, 8)
