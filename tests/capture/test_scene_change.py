"""Tests for SceneChangeDetector."""

import cv2
import numpy as np
import pytest

from cortex.capture.scene_change import SceneChangeDetector


@pytest.fixture
def detector() -> SceneChangeDetector:
    return SceneChangeDetector()


@pytest.fixture
def frame_a() -> np.ndarray:
    """Dark frame with white rectangle on left."""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 20:80] = 255
    return img


@pytest.fixture
def frame_b() -> np.ndarray:
    """Dark frame with white rectangle on right — different scene."""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 120:180] = 255
    return img


def test_first_frame_always_accepted(
    detector: SceneChangeDetector, frame_a: np.ndarray
) -> None:
    assert detector.detect(frame_a) is True


def test_identical_frames_rejected(
    detector: SceneChangeDetector, frame_a: np.ndarray
) -> None:
    detector.detect(frame_a)  # first frame
    assert detector.detect(frame_a) is False


def test_different_frames_accepted(
    detector: SceneChangeDetector,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> None:
    detector.detect(frame_a)  # first frame
    assert detector.detect(frame_b) is True


def test_threshold_customization() -> None:
    # Very low threshold — almost nothing triggers change
    detector = SceneChangeDetector(threshold=0.1)
    assert detector.threshold == 0.1

    img_a = np.full((100, 100), 128, dtype=np.uint8)
    img_b = img_a.copy()
    img_b[45:55, 45:55] = 140  # tiny local change

    detector.detect(img_a)
    # With threshold=0.1, small changes won't be detected as scene change
    assert detector.detect(img_b) is False


def test_reset(
    detector: SceneChangeDetector, frame_a: np.ndarray
) -> None:
    detector.detect(frame_a)
    detector.reset()
    # After reset, next frame should be treated as first
    assert detector.detect(frame_a) is True


def test_bgr_input(detector: SceneChangeDetector) -> None:
    bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    bgr[50:150, 50:150] = [255, 0, 0]
    assert detector.detect(bgr) is True
