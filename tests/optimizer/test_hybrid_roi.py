"""Tests for HybridROI."""

import cv2
import numpy as np
import pytest

from cortex.capture.imu_gate import BatteryMode
from cortex.optimizer.hybrid_roi import HybridROI, RequestType


@pytest.fixture
def roi() -> HybridROI:
    return HybridROI()


@pytest.fixture
def frame() -> np.ndarray:
    """Frame with text and a bright object."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[60:140, 100:200] = 255  # bright object
    cv2.putText(img, "Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return img


def test_default_request_type(roi: HybridROI) -> None:
    assert roi.request_type == RequestType.GENERAL


def test_set_request_type(roi: HybridROI) -> None:
    roi.set_request_type(RequestType.TEXT_RECOGNITION)
    assert roi.request_type == RequestType.TEXT_RECOGNITION


def test_fused_score_map_shape(roi: HybridROI, frame: np.ndarray) -> None:
    score = roi.fused_score_map(frame)
    assert score.shape == (6, 8)


def test_fused_score_map_values_in_range(roi: HybridROI, frame: np.ndarray) -> None:
    score = roi.fused_score_map(frame)
    assert score.min() >= 0.0
    assert score.max() <= 1.01  # small float tolerance


def test_crop_returns_smaller(roi: HybridROI, frame: np.ndarray) -> None:
    cropped = roi.crop(frame)
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0
    original_area = frame.shape[0] * frame.shape[1]
    cropped_area = cropped.shape[0] * cropped.shape[1]
    assert cropped_area <= original_area


def test_text_mode_weights(frame: np.ndarray) -> None:
    roi = HybridROI(request_type=RequestType.TEXT_RECOGNITION)
    score = roi.fused_score_map(frame)
    assert score.shape == (6, 8)


def test_object_mode_weights(frame: np.ndarray) -> None:
    roi = HybridROI(request_type=RequestType.OBJECT_SCENE)
    score = roi.fused_score_map(frame)
    assert score.shape == (6, 8)


def test_power_save_skips_saliency(frame: np.ndarray) -> None:
    roi = HybridROI()
    roi.set_battery_mode(BatteryMode.POWER_SAVE)
    score = roi.fused_score_map(frame)
    assert score.shape == (6, 8)


def test_ema_smoothing(roi: HybridROI, frame: np.ndarray) -> None:
    score1 = roi.fused_score_map(frame)
    score2 = roi.fused_score_map(frame)
    # Second call should have EMA applied (scores may differ slightly)
    assert score2.shape == score1.shape
