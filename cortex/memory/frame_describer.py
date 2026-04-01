"""Fake VLM frame describer for demo/testing without API calls.

Generates simple descriptions from image features
(brightness, text presence, motion, color).
"""

import cv2
import numpy as np

from cortex.optimizer.text_roi import TextROIStrategy

_text_detector = TextROIStrategy()


def describe_frame(
    frame: np.ndarray,
    blur_score: float = 0,
    motion_score: float = 0,
) -> tuple[str, str]:
    """Generate a fake VLM description from frame features.

    Args:
        frame: Input BGR image.
        blur_score: Laplacian blur score.
        motion_score: Motion score from HybridROI.

    Returns:
        Tuple of (description, event_type).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # Text detection
    text_regions = _text_detector.detect_regions(frame)
    has_text = len(text_regions) > 0

    # Dominant color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())

    # Build description
    parts: list[str] = []

    # Brightness
    if brightness < 60:
        parts.append("dark scene")
    elif brightness > 180:
        parts.append("bright scene")
    else:
        parts.append("normal lighting")

    # Color
    if s_mean > 80:
        if h_mean < 15 or h_mean > 165:
            parts.append("red tones")
        elif 15 <= h_mean < 45:
            parts.append("warm tones")
        elif 45 <= h_mean < 75:
            parts.append("green tones")
        elif 75 <= h_mean < 105:
            parts.append("blue-green tones")
        elif 105 <= h_mean < 135:
            parts.append("blue tones")
        else:
            parts.append("purple tones")

    # Text
    if has_text:
        parts.append(f"text detected ({len(text_regions)} regions)")
        event_type = "text"
    else:
        event_type = "object"

    # Motion
    if motion_score > 0.5:
        parts.append("movement detected")
        event_type = "motion"

    # Sharpness
    if blur_score > 300:
        parts.append("very sharp")
    elif blur_score > 100:
        parts.append("clear")

    description = ", ".join(parts)
    return description, event_type
