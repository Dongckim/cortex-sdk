"""Tests for BlurDetector."""

import cv2
import numpy as np
import pytest

from cortex.capture.blur_detector import BlurDetector


@pytest.fixture
def detector() -> BlurDetector:
    return BlurDetector()


@pytest.fixture
def sharp_image() -> np.ndarray:
    """Create a synthetic sharp image with strong edges."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Checkerboard pattern creates high-frequency edges
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i : i + 20, j : j + 20] = 255
    return img


@pytest.fixture
def blurry_image(sharp_image: np.ndarray) -> np.ndarray:
    """Create a blurry image by applying heavy Gaussian blur."""
    return cv2.GaussianBlur(sharp_image, (31, 31), 10)


def test_sharp_image_detected(detector: BlurDetector, sharp_image: np.ndarray) -> None:
    assert detector.detect(sharp_image) is True


def test_blurry_image_detected(
    detector: BlurDetector, blurry_image: np.ndarray
) -> None:
    assert detector.detect(blurry_image) is False


def test_bgr_input(detector: BlurDetector, sharp_image: np.ndarray) -> None:
    bgr = cv2.cvtColor(sharp_image, cv2.COLOR_GRAY2BGR)
    assert detector.detect(bgr) is True


def test_threshold_customization() -> None:
    detector = BlurDetector(threshold=10.0)
    assert detector.threshold == 10.0

    # With a very low threshold, even slightly blurry images pass
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 255
    blurred = cv2.GaussianBlur(img, (5, 5), 1)
    assert detector.detect(blurred) is True


def test_score_method(detector: BlurDetector, sharp_image: np.ndarray) -> None:
    score = detector.score(sharp_image)
    assert isinstance(score, float)
    assert score > 0
