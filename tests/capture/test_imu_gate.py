"""Tests for IMUGate."""

import pytest

from cortex.capture.imu_gate import BatteryMode, IMUGate


@pytest.fixture
def gate() -> IMUGate:
    return IMUGate()


def test_first_call_returns_true(gate: IMUGate) -> None:
    assert gate.update((0, 0, 9.8), (0, 0, 0)) is True


def test_stationary_returns_false(gate: IMUGate) -> None:
    gate.update((0, 0, 9.8), (0, 0, 0))
    assert gate.update((0, 0, 9.8), (0, 0, 0)) is False


def test_motion_returns_true(gate: IMUGate) -> None:
    gate.update((0, 0, 9.8), (0, 0, 0))
    # Large acceleration change
    assert gate.update((5, 5, 9.8), (1, 1, 0)) is True


def test_battery_mode_aggressive() -> None:
    gate = IMUGate(mode=BatteryMode.AGGRESSIVE)
    assert gate.threshold == 0.5
    gate.update((0, 0, 9.8), (0, 0, 0))
    # Small motion — should trigger with low threshold
    assert gate.update((0.5, 0.5, 9.8), (0.3, 0, 0)) is True


def test_battery_mode_power_save() -> None:
    gate = IMUGate(mode=BatteryMode.POWER_SAVE)
    assert gate.threshold == 2.0
    gate.update((0, 0, 9.8), (0, 0, 0))
    # Small motion — should NOT trigger with high threshold
    assert gate.update((0.3, 0.3, 9.8), (0.1, 0, 0)) is False


def test_set_battery_mode(gate: IMUGate) -> None:
    gate.set_battery_mode(BatteryMode.POWER_SAVE)
    assert gate.mode == BatteryMode.POWER_SAVE
    assert gate.threshold == 2.0

    gate.set_battery_mode(BatteryMode.AGGRESSIVE)
    assert gate.mode == BatteryMode.AGGRESSIVE
    assert gate.threshold == 0.5


def test_motion_score_property(gate: IMUGate) -> None:
    gate.update((0, 0, 9.8), (0, 0, 0))
    gate.update((1, 0, 9.8), (0, 0, 0))
    assert gate.motion_score > 0
