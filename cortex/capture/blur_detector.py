"""Blur detection using Laplacian variance method."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BlurDetector:
    """Detects motion blur in frames using Laplacian variance.

    A low Laplacian variance indicates a blurry image, while a high
    variance indicates a sharp image with well-defined edges.

    Args:
        threshold: Minimum Laplacian variance to consider a frame sharp.
            Higher values are more strict. Default is 100.0.
    """

    def __init__(self, threshold: float = 100.0) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Current blur detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def detect(self, frame: np.ndarray) -> bool:
        """Determine whether a frame is sharp.

        Args:
            frame: Input image as a numpy array (BGR or grayscale).

        Returns:
            True if the frame is sharp, False if blurry.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        is_sharp = bool(score >= self._threshold)
        logger.debug(
            "blur_score=%.2f threshold=%.2f sharp=%s",
            score,
            self._threshold,
            is_sharp,
        )
        return is_sharp

    def score(self, frame: np.ndarray) -> float:
        """Compute the Laplacian variance score for a frame.

        Args:
            frame: Input image as a numpy array (BGR or grayscale).

        Returns:
            Laplacian variance score. Higher means sharper.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
