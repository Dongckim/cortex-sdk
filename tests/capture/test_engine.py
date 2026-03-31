"""Tests for CaptureEngine."""

import cv2
import numpy as np
import pytest

from cortex.capture.engine import CaptureEngine, CaptureResult
from cortex.capture.imu_gate import BatteryMode


@pytest.fixture
def engine() -> CaptureEngine:
    return CaptureEngine()


@pytest.fixture
def sharp_frame() -> np.ndarray:
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i : i + 20, j : j + 20] = 255
    return img


@pytest.fixture
def blurry_frame(sharp_frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(sharp_frame, (31, 31), 10)


@pytest.fixture
def different_frame() -> np.ndarray:
    img = np.zeros((200, 200), dtype=np.uint8)
    img[80:120, 80:120] = 255
    return img


def test_sharp_changed_frame_accepted(
    engine: CaptureEngine, sharp_frame: np.ndarray
) -> None:
    result = engine.process_frame(sharp_frame)
    assert result.accepted is True
    assert result.reason == "accepted"


def test_blurry_frame_rejected(
    engine: CaptureEngine,
    sharp_frame: np.ndarray,
    blurry_frame: np.ndarray,
) -> None:
    engine.process_frame(sharp_frame)  # first frame
    result = engine.process_frame(blurry_frame)
    assert result.accepted is False
    assert result.reason == "blurry"


def test_unchanged_frame_rejected(
    engine: CaptureEngine, sharp_frame: np.ndarray
) -> None:
    engine.process_frame(sharp_frame)  # first frame
    result = engine.process_frame(sharp_frame)
    assert result.accepted is False
    assert result.reason == "no_change"


def test_no_motion_rejected(engine: CaptureEngine, sharp_frame: np.ndarray) -> None:
    imu = {"accel": (0, 0, 9.8), "gyro": (0, 0, 0)}
    engine.process_frame(sharp_frame, imu_data=imu)  # first (always True)
    result = engine.process_frame(sharp_frame, imu_data=imu)
    assert result.accepted is False
    assert result.reason == "no_motion"


def test_imu_skipped_when_no_data(
    engine: CaptureEngine,
    sharp_frame: np.ndarray,
    different_frame: np.ndarray,
) -> None:
    engine.process_frame(sharp_frame)
    result = engine.process_frame(different_frame)
    # Without imu_data, IMU gate is skipped
    assert result.reason != "no_motion"


def test_battery_mode_switching(engine: CaptureEngine) -> None:
    engine.set_battery_mode(BatteryMode.POWER_SAVE)
    assert engine._blur.threshold == 50.0
    assert engine._scene.threshold == 0.65

    engine.set_battery_mode(BatteryMode.AGGRESSIVE)
    assert engine._blur.threshold == 150.0
    assert engine._scene.threshold == 0.90


def test_stats_tracking(
    engine: CaptureEngine,
    sharp_frame: np.ndarray,
    different_frame: np.ndarray,
) -> None:
    engine.process_frame(sharp_frame)
    engine.process_frame(different_frame)
    engine.process_frame(sharp_frame)  # no_change if similar

    stats = engine.stats
    assert stats["total_frames"] == 3
    assert stats["accepted_frames"] >= 1


def test_callbacks(
    engine: CaptureEngine,
    sharp_frame: np.ndarray,
    blurry_frame: np.ndarray,
) -> None:
    accepted_results: list[CaptureResult] = []
    rejected_results: list[CaptureResult] = []

    engine.on_accepted(lambda r: accepted_results.append(r))
    engine.on_rejected(lambda r: rejected_results.append(r))

    engine.process_frame(sharp_frame)  # accepted (first frame)
    engine.process_frame(blurry_frame)  # rejected

    assert len(accepted_results) == 1
    assert len(rejected_results) == 1
    assert accepted_results[0].reason == "accepted"
    assert rejected_results[0].reason == "blurry"
