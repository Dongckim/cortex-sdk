"""Webcam test for BlurDetector — move camera to see blur detection in action."""

import cv2

from cortex.capture.blur_detector import BlurDetector

detector = BlurDetector(threshold=200.0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    is_sharp = detector.detect(frame)
    score = detector.score(frame)

    label = f"SHARP ({score:.1f})" if is_sharp else f"BLURRY ({score:.1f})"
    color = (0, 255, 0) if is_sharp else (0, 0, 255)

    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(
        frame,
        f"Threshold: {detector.threshold:.0f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.imshow("BlurDetector Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
