"""Webcam test for BlurDetector + SceneChangeDetector."""

import cv2

from cortex.capture.blur_detector import BlurDetector
from cortex.capture.scene_change import SceneChangeDetector

blur = BlurDetector(threshold=200.0)
scene = SceneChangeDetector(threshold=0.65)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

last_accepted = None
accepted_count = 0
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_count += 1
    is_sharp = blur.detect(frame)
    blur_score = blur.score(frame)

    # Decide accept/reject — blur first, skip scene check if blurry
    if not is_sharp:
        status = "REJECTED: blurry"
        color = (0, 0, 255)
    elif not scene.detect(frame):
        status = "REJECTED: no change"
        color = (0, 165, 255)
    else:
        status = "ACCEPTED"
        color = (0, 255, 0)
        last_accepted = frame.copy()
        accepted_count += 1

    # Build display
    display = frame.copy()
    h, w = display.shape[:2]

    # --- Top bar ---
    bar_h = 36
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # Status dot + text
    dot_color = color
    cv2.circle(display, (20, bar_h // 2), 6, dot_color, -1)
    cv2.putText(
        display, status,
        (36, bar_h // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    # Right-aligned stats
    rate = (accepted_count / total_count * 100) if total_count > 0 else 0
    stats = f"blur {blur_score:.0f}  |  accept {rate:.0f}%"
    tw = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.putText(
        display, stats,
        (w - tw - 15, bar_h // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
    )

    # --- Last accepted thumbnail (bottom-right) ---
    if last_accepted is not None:
        tw, th = w // 5, h // 5
        small = cv2.resize(last_accepted, (tw, th))
        y1, x1 = h - th - 10, w - tw - 10
        overlay2 = display.copy()
        cv2.rectangle(overlay2, (x1 - 1, y1 - 18), (x1 + tw + 1, y1 + th + 1), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.5, display, 0.5, 0, display)
        display[y1:y1 + th, x1:x1 + tw] = small
        cv2.putText(
            display, "last accepted",
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1,
        )

    cv2.imshow("Capture Test (q=quit, r=reset)", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        scene.reset()
        accepted_count = 0
        total_count = 0

cap.release()
cv2.destroyAllWindows()
