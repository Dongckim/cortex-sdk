"""Capture engine integrating blur, scene change, and IMU gate."""

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cortex.capture.blur_detector import BlurDetector
from cortex.capture.imu_gate import BatteryMode, IMUGate
from cortex.capture.scene_change import SceneChangeDetector

logger = logging.getLogger(__name__)

_BLUR_THRESHOLDS: dict[BatteryMode, float] = {
    BatteryMode.AGGRESSIVE: 150.0,
    BatteryMode.BALANCED: 100.0,
    BatteryMode.POWER_SAVE: 50.0,
}

_SCENE_THRESHOLDS: dict[BatteryMode, float] = {
    BatteryMode.AGGRESSIVE: 0.90,
    BatteryMode.BALANCED: 0.80,
    BatteryMode.POWER_SAVE: 0.65,
}


@dataclass
class CaptureResult:
    """Result of processing a single frame.

    Attributes:
        accepted: Whether the frame was accepted.
        reason: Reason for the decision.
        stats: Dictionary of detection scores.
    """

    accepted: bool
    reason: str
    stats: dict = field(default_factory=dict)


class CaptureEngine:
    """Integrates IMU gate, blur detector, and scene change detector.

    Pipeline order: IMU gate -> blur check -> scene change check.

    Args:
        blur_detector: BlurDetector instance. Creates default if None.
        scene_detector: SceneChangeDetector instance. Creates default if None.
        imu_gate: IMUGate instance. Creates default if None.
    """

    def __init__(
        self,
        blur_detector: BlurDetector | None = None,
        scene_detector: SceneChangeDetector | None = None,
        imu_gate: IMUGate | None = None,
    ) -> None:
        self._blur = blur_detector or BlurDetector()
        self._scene = scene_detector or SceneChangeDetector()
        self._imu = imu_gate or IMUGate()
        self._total_frames = 0
        self._accepted_frames = 0
        self._on_accepted: list[Callable] = []
        self._on_rejected: list[Callable] = []

    @property
    def stats(self) -> dict:
        """Capture statistics."""
        return {
            "total_frames": self._total_frames,
            "accepted_frames": self._accepted_frames,
            "acceptance_rate": (
                self._accepted_frames / self._total_frames
                if self._total_frames > 0
                else 0.0
            ),
        }

    def on_accepted(self, callback: Callable) -> None:
        """Register a callback for accepted frames."""
        self._on_accepted.append(callback)

    def on_rejected(self, callback: Callable) -> None:
        """Register a callback for rejected frames."""
        self._on_rejected.append(callback)

    def set_battery_mode(self, mode: BatteryMode) -> None:
        """Update all component thresholds for the given battery mode.

        Args:
            mode: New battery mode.
        """
        self._imu.set_battery_mode(mode)
        self._blur.threshold = _BLUR_THRESHOLDS[mode]
        self._scene.threshold = _SCENE_THRESHOLDS[mode]
        logger.debug("battery_mode=%s", mode.value)

    def process_frame(
        self,
        frame: np.ndarray,
        imu_data: dict | None = None,
    ) -> CaptureResult:
        """Process a frame through the capture pipeline.

        Args:
            frame: Input image as a numpy array.
            imu_data: Optional dict with 'accel' and 'gyro' tuples.
                If None, IMU gate is skipped.

        Returns:
            CaptureResult with accept/reject decision.
        """
        self._total_frames += 1

        blur_score = self._blur.score(frame)
        motion_score = self._imu.motion_score
        battery_mode = self._imu.mode.value

        ssim_score = self._scene.last_score

        base_stats = {
            "blur_score": blur_score,
            "motion_score": motion_score,
            "ssim_score": ssim_score,
            "battery_mode": battery_mode,
        }

        # Step 1: IMU gate
        if imu_data is not None:
            accel = imu_data.get("accel", (0, 0, 9.8))
            gyro = imu_data.get("gyro", (0, 0, 0))
            if not self._imu.update(accel, gyro):
                result = CaptureResult(
                    accepted=False,
                    reason="no_motion",
                    stats={**base_stats, "motion_score": self._imu.motion_score},
                )
                self._notify_rejected(result)
                return result

        # Step 2: Blur check
        if not self._blur.detect(frame):
            result = CaptureResult(
                accepted=False, reason="blurry", stats=base_stats,
            )
            self._notify_rejected(result)
            return result

        # Step 3: Scene change check
        if not self._scene.detect(frame):
            result = CaptureResult(
                accepted=False, reason="no_change", stats=base_stats,
            )
            self._notify_rejected(result)
            return result

        self._accepted_frames += 1
        result = CaptureResult(
            accepted=True, reason="accepted", stats=base_stats,
        )
        self._notify_accepted(result)
        return result

    def _notify_accepted(self, result: CaptureResult) -> None:
        for cb in self._on_accepted:
            cb(result)

    def _notify_rejected(self, result: CaptureResult) -> None:
        for cb in self._on_rejected:
            cb(result)
