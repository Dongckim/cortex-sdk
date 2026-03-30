"""IMU-based motion gating with battery-aware thresholds."""

import enum
import logging
import math

logger = logging.getLogger(__name__)


class BatteryMode(enum.Enum):
    """Battery mode controls capture aggressiveness.

    AGGRESSIVE captures more frames (low threshold),
    POWER_SAVE captures fewer (high threshold).
    """

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"


_THRESHOLDS: dict[BatteryMode, float] = {
    BatteryMode.AGGRESSIVE: 0.5,
    BatteryMode.BALANCED: 1.0,
    BatteryMode.POWER_SAVE: 2.0,
}


class IMUGate:
    """Simulates IMU-based motion detection for wearable devices.

    Computes a motion score from accelerometer and gyroscope deltas,
    and gates frame capture based on whether motion exceeds a threshold.

    Args:
        mode: Initial battery mode. Default is BALANCED.
        accel_weight: Weight for accelerometer delta. Default is 1.0.
        gyro_weight: Weight for gyroscope delta. Default is 0.5.
    """

    def __init__(
        self,
        mode: BatteryMode = BatteryMode.BALANCED,
        accel_weight: float = 1.0,
        gyro_weight: float = 0.5,
    ) -> None:
        self._mode = mode
        self._threshold = _THRESHOLDS[mode]
        self._accel_weight = accel_weight
        self._gyro_weight = gyro_weight
        self._prev_accel: tuple[float, float, float] | None = None
        self._prev_gyro: tuple[float, float, float] | None = None
        self._motion_score: float = 0.0

    @property
    def motion_score(self) -> float:
        """Last computed motion score."""
        return self._motion_score

    @property
    def mode(self) -> BatteryMode:
        """Current battery mode."""
        return self._mode

    @property
    def threshold(self) -> float:
        """Current motion threshold."""
        return self._threshold

    def set_battery_mode(self, mode: BatteryMode) -> None:
        """Update the battery mode and its associated threshold.

        Args:
            mode: New battery mode.
        """
        self._mode = mode
        self._threshold = _THRESHOLDS[mode]
        logger.debug("battery_mode=%s threshold=%.2f", mode.value, self._threshold)

    def update(
        self,
        accel: tuple[float, float, float],
        gyro: tuple[float, float, float],
    ) -> bool:
        """Process new IMU reading and determine if motion is significant.

        Args:
            accel: Accelerometer reading (x, y, z).
            gyro: Gyroscope reading (x, y, z).

        Returns:
            True if motion exceeds threshold, False otherwise.
            First call always returns True (no previous data).
        """
        if self._prev_accel is None or self._prev_gyro is None:
            self._prev_accel = accel
            self._prev_gyro = gyro
            self._motion_score = 0.0
            logger.debug("first imu reading, accepted")
            return True

        accel_delta = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(accel, self._prev_accel))
        )
        gyro_delta = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(gyro, self._prev_gyro))
        )

        self._motion_score = (
            self._accel_weight * accel_delta + self._gyro_weight * gyro_delta
        )

        self._prev_accel = accel
        self._prev_gyro = gyro

        triggered = self._motion_score >= self._threshold
        logger.debug(
            "motion=%.3f threshold=%.2f triggered=%s",
            self._motion_score,
            self._threshold,
            triggered,
        )
        return triggered
