"""Scene change detection using SSIM comparison."""

import logging
import time as _time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """Detects scene changes by comparing frames using SSIM.

    Compares the current frame against the last accepted frame.
    If SSIM falls below the threshold, the scene is considered changed.

    A cooldown prevents accepting frames too frequently — useful when
    the comparison baseline is a naive timer (e.g. "send every 5s").
    With cooldown_s=5.0 Cortex can send at most once per 5 s window,
    but skips that window entirely if nothing changed.

    Args:
        threshold: SSIM threshold below which a scene change is detected.
            Higher values require less change to trigger. Default 0.85.
        cooldown_s: Minimum seconds between acceptances. 0 disables.
    """

    def __init__(self, threshold: float = 0.85, cooldown_s: float = 0.0) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._last_accepted: np.ndarray | None = None
        self._last_score: float | None = None
        self._last_accepted_time: float = 0.0

    @property
    def threshold(self) -> float:
        """Current SSIM threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def cooldown_s(self) -> float:
        """Minimum seconds between acceptances."""
        return self._cooldown_s

    @cooldown_s.setter
    def cooldown_s(self, value: float) -> None:
        self._cooldown_s = value

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

        now = _time.monotonic()

        if self._last_accepted is None:
            self._last_accepted = gray.copy()
            self._last_accepted_time = now
            logger.debug("first frame accepted")
            return True

        # Honour cooldown: don't even compute SSIM if too soon
        if self._cooldown_s > 0 and (now - self._last_accepted_time) < self._cooldown_s:
            return False

        score = ssim(self._last_accepted, gray)
        self._last_score = float(score)
        changed = bool(score < self._threshold)

        logger.debug(
            "ssim=%.4f threshold=%.2f changed=%s",
            score,
            self._threshold,
            changed,
        )

        if changed:
            self._last_accepted = gray.copy()
            self._last_accepted_time = now

        return changed

    def reset(self) -> None:
        """Clear the stored reference frame."""
        self._last_accepted = None
        self._last_accepted_time = 0.0

    @property
    def last_score(self) -> float | None:
        """Return the last computed SSIM score, or None if no comparison yet."""
        return self._last_score
