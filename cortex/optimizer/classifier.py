"""Request type classifier for ROI strategy selection."""

import logging
import re

from cortex.optimizer.hybrid_roi import RequestType

logger = logging.getLogger(__name__)

_TEXT_KEYWORDS = [
    "read", "translate", "ocr", "text", "sign", "menu", "label",
    "what does it say", "what is written",
]

_OBJECT_KEYWORDS = [
    "what is", "describe", "identify", "recognize", "look at",
    "what do you see", "tell me about", "explain",
]

_NAVIGATION_KEYWORDS = [
    "where", "navigate", "direction", "turn", "go to",
    "how do i get", "path", "route", "crossing",
]


class RequestClassifier:
    """Classifies requests to determine the appropriate ROI strategy.

    Supports voice command classification (keyword matching) and
    implicit classification based on scene signals.
    """

    def __init__(self) -> None:
        self._confidence: float = 0.0

    @property
    def confidence(self) -> float:
        """Confidence of the last classification (0.0-1.0)."""
        return self._confidence

    def classify_voice(self, text: str) -> RequestType:
        """Classify a voice command into a request type.

        Args:
            text: Voice command text.

        Returns:
            Classified RequestType.
        """
        lower = text.lower().strip()

        text_score = sum(1 for kw in _TEXT_KEYWORDS if kw in lower)
        object_score = sum(1 for kw in _OBJECT_KEYWORDS if kw in lower)
        nav_score = sum(1 for kw in _NAVIGATION_KEYWORDS if kw in lower)

        total = text_score + object_score + nav_score

        if total == 0:
            self._confidence = 0.3
            logger.debug("voice classify=%s confidence=%.2f", RequestType.GENERAL.value, self._confidence)
            return RequestType.GENERAL

        scores = {
            RequestType.TEXT_RECOGNITION: text_score,
            RequestType.OBJECT_SCENE: object_score,
            RequestType.NAVIGATION: nav_score,
        }

        best = max(scores, key=scores.get)
        self._confidence = scores[best] / total

        logger.debug("voice classify=%s confidence=%.2f", best.value, self._confidence)
        return best

    def classify_implicit(
        self, has_text: bool, is_moving: bool
    ) -> RequestType:
        """Classify based on implicit scene signals.

        Args:
            has_text: Whether text was detected in the frame.
            is_moving: Whether the device is in motion.

        Returns:
            Classified RequestType.
        """
        if has_text and not is_moving:
            result = RequestType.TEXT_RECOGNITION
            self._confidence = 0.7
        elif not has_text and is_moving:
            result = RequestType.NAVIGATION
            self._confidence = 0.6
        elif not has_text and not is_moving:
            result = RequestType.OBJECT_SCENE
            self._confidence = 0.5
        else:
            result = RequestType.GENERAL
            self._confidence = 0.4

        logger.debug(
            "implicit classify=%s has_text=%s moving=%s confidence=%.2f",
            result.value, has_text, is_moving, self._confidence,
        )
        return result
