"""Tests for TextROIStrategy."""

import cv2
import numpy as np
import pytest

from cortex.optimizer.text_roi import TextROIStrategy


@pytest.fixture
def strategy() -> TextROIStrategy:
    return TextROIStrategy()


@pytest.fixture
def text_image() -> np.ndarray:
    """Create a synthetic image with text."""
    img = np.full((200, 400, 3), 240, dtype=np.uint8)
    cv2.putText(img, "Hello CORTEX", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "SDK Test", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


@pytest.fixture
def blank_image() -> np.ndarray:
    """Uniform image with no text."""
    return np.full((200, 400, 3), 128, dtype=np.uint8)


def test_detect_text_regions(
    strategy: TextROIStrategy, text_image: np.ndarray
) -> None:
    regions = strategy.detect_regions(text_image)
    assert len(regions) > 0


def test_detect_no_text(strategy: TextROIStrategy, blank_image: np.ndarray) -> None:
    regions = strategy.detect_regions(blank_image)
    assert len(regions) == 0


def test_score_map_shape(strategy: TextROIStrategy, text_image: np.ndarray) -> None:
    score = strategy.score_map(text_image)
    assert score.shape == (6, 8)


def test_score_map_has_nonzero_for_text(
    strategy: TextROIStrategy, text_image: np.ndarray
) -> None:
    score = strategy.score_map(text_image)
    assert score.max() > 0


def test_score_map_blank_is_zero(
    strategy: TextROIStrategy, blank_image: np.ndarray
) -> None:
    score = strategy.score_map(blank_image)
    assert score.max() == 0.0


def test_crop_with_text(strategy: TextROIStrategy, text_image: np.ndarray) -> None:
    cropped = strategy.crop(text_image)
    # Should be smaller than original
    assert cropped.shape[0] <= text_image.shape[0]
    assert cropped.shape[1] <= text_image.shape[1]


def test_crop_no_text_returns_original(
    strategy: TextROIStrategy, blank_image: np.ndarray
) -> None:
    cropped = strategy.crop(blank_image)
    assert cropped.shape == blank_image.shape


def test_grayscale_input(strategy: TextROIStrategy) -> None:
    img = np.full((200, 400), 240, dtype=np.uint8)
    cv2.putText(img, "Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 3)
    regions = strategy.detect_regions(img)
    assert len(regions) >= 0  # Should not crash
