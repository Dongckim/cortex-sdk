"""Webcam test — BlurDetector + SceneChangeDetector + IMUGate combined.

Frame pixel difference simulates IMU motion data.
Pipeline: IMU gate → blur check → scene change check.
"""

import cv2
import numpy as np

from cortex.capture.blur_detector import BlurDetector
from cortex.capture.imu_gate import BatteryMode, IMUGate
from cortex.capture.scene_change import SceneChangeDetector

blur = BlurDetector(threshold=200.0)
scene = SceneChangeDetector(threshold=0.65)
gate = IMUGate(mode=BatteryMode.BALANCED)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

last_accepted = None
prev_gray = None
total = 0
accepted = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total += 1

    # Estimate motion from frame difference (simulates IMU)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion_val = float(diff.mean()) / 25.0
    else:
        motion_val = 0.0
    prev_gray = gray.copy()

    accel = (motion_val, motion_val * 0.5, 9.8)
    gyro = (motion_val * 0.3, motion_val * 0.2, 0.0)
    has_motion = gate.update(accel, gyro)

    # Pipeline: IMU → blur → scene
    if not has_motion:
        status, color = "NO MOTION", (100, 100, 100)
    elif not blur.detect(frame):
        status, color = "BLURRY", (0, 0, 255)
    elif not scene.detect(frame):
        status, color = "NO CHANGE", (0, 165, 255)
    else:
        status, color = "ACCEPTED", (0, 255, 0)
        last_accepted = frame.copy()
        accepted += 1

    blur_score = blur.score(frame)

    # --- HUD ---
    display = frame.copy()
    h, w = display.shape[:2]

    # Top bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # Status
    cv2.circle(display, (20, 16), 6, color, -1)
    cv2.putText(
        display, status,
        (36, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    # Stats line
    rate = (accepted / total * 100) if total > 0 else 0
    info = (
        f"blur {blur_score:.0f}  |  "
        f"motion {gate.motion_score:.2f}  |  "
        f"mode {gate.mode.value}  |  "
        f"accept {rate:.0f}%"
    )
    cv2.putText(
        display, info,
        (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
    )

    # Last accepted thumbnail
    if last_accepted is not None:
        tw, th = w // 5, h // 5
        small = cv2.resize(last_accepted, (tw, th))
        y1, x1 = h - th - 10, w - tw - 10
        overlay2 = display.copy()
        cv2.rectangle(
            overlay2, (x1 - 1, y1 - 16), (x1 + tw + 1, y1 + th + 1),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay2, 0.5, display, 0.5, 0, display)
        display[y1:y1 + th, x1:x1 + tw] = small
        cv2.putText(
            display, "last accepted",
            (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1,
        )

    cv2.imshow("CORTEX Capture Test", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        gate.set_battery_mode(BatteryMode.AGGRESSIVE)
    elif key == ord("2"):
        gate.set_battery_mode(BatteryMode.BALANCED)
    elif key == ord("3"):
        gate.set_battery_mode(BatteryMode.POWER_SAVE)
    elif key == ord("r"):
        scene.reset()
        total, accepted = 0, 0

cap.release()
cv2.destroyAllWindows()
