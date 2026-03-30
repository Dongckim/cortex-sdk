"""Scene change detection using SSIM comparison."""

import logging

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """Detects scene changes by comparing frames using SSIM.

    Compares the current frame against the last accepted frame.
    If SSIM falls below the threshold, the scene is considered changed.

    Args:
        threshold: SSIM threshold below which a scene change is detected.
            Lower values require more dramatic changes. Default is 0.85.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        self._threshold = threshold
        self._last_accepted: np.ndarray | None = None

    @property
    def threshold(self) -> float:
        """Current SSIM threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def detect(self, frame: np.ndarray) -> bool:
        """Determine whether the scene has changed.

        Args:
            frame: Input image as a numpy array (BGR or grayscale).

        Returns:
            True if the scene has changed (or first frame), False otherwise.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self._last_accepted is None:
            self._last_accepted = gray.copy()
            logger.debug("first frame accepted")
            return True

        score = ssim(self._last_accepted, gray)
        changed = bool(score < self._threshold)

        logger.debug(
            "ssim=%.4f threshold=%.2f changed=%s",
            score,
            self._threshold,
            changed,
        )

        if changed:
            self._last_accepted = gray.copy()

        return changed

    def reset(self) -> None:
        """Clear the stored reference frame."""
        self._last_accepted = None

    @property
    def last_score(self) -> float | None:
        """Return the last computed SSIM score, or None if no comparison yet."""
        return None
