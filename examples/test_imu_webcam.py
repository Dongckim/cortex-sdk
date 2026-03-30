"""Webcam test for IMUGate — mouse movement simulates IMU sensor data."""

import cv2

from cortex.capture.imu_gate import BatteryMode, IMUGate

gate = IMUGate(mode=BatteryMode.BALANCED)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

mouse_x, mouse_y = 0, 0
prev_x, prev_y = 0, 0


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


cv2.namedWindow("IMUGate Test")
cv2.setMouseCallback("IMUGate Test", on_mouse)

total = 0
triggered = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total += 1

    # Mouse delta → fake accel/gyro
    dx = (mouse_x - prev_x) / 50.0
    dy = (mouse_y - prev_y) / 50.0
    accel = (dx, dy, 9.8)
    gyro = (dx * 0.3, dy * 0.3, 0.0)
    prev_x, prev_y = mouse_x, mouse_y

    motion = gate.update(accel, gyro)
    if motion:
        triggered += 1

    # Display
    display = frame.copy()
    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    status = "MOTION" if motion else "STILL"
    color = (0, 255, 0) if motion else (100, 100, 100)
    cv2.circle(display, (20, 18), 6, color, -1)
    cv2.putText(
        display, f"{status}  score: {gate.motion_score:.2f}  mode: {gate.mode.value}",
        (36, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    rate = (triggered / total * 100) if total > 0 else 0
    stats = f"triggered {rate:.0f}%  |  move mouse to simulate IMU"
    tw = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
    cv2.putText(
        display, stats,
        (w - tw - 15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
    )

    cv2.imshow("IMUGate Test", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        gate.set_battery_mode(BatteryMode.AGGRESSIVE)
    elif key == ord("2"):
        gate.set_battery_mode(BatteryMode.BALANCED)
    elif key == ord("3"):
        gate.set_battery_mode(BatteryMode.POWER_SAVE)

cap.release()
cv2.destroyAllWindows()
