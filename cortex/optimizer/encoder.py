"""Adaptive image encoding based on network conditions."""

import enum
import logging
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NetworkCondition(enum.Enum):
    """Network condition for adaptive encoding."""

    WIFI = "wifi"
    LTE = "lte"
    WEAK = "weak"
    OFFLINE = "offline"


_ENCODE_PARAMS: dict[NetworkCondition, dict] = {
    NetworkCondition.WIFI: {"resolution": 1024, "format": ".webp", "quality": 85},
    NetworkCondition.LTE: {"resolution": 640, "format": ".webp", "quality": 70},
    NetworkCondition.WEAK: {"resolution": 320, "format": ".jpg", "quality": 60},
}

# Approximate per-token cost in USD (input image tokens)
_TOKEN_COSTS: dict[str, float] = {
    "claude-sonnet-4-20250514": 3.0 / 1_000_000,
    "claude-haiku-4-5-20251001": 0.80 / 1_000_000,
    "gpt-4o": 2.5 / 1_000_000,
    "gpt-4o-mini": 0.15 / 1_000_000,
}


class AdaptiveEncoder:
    """Encodes frames based on network conditions.

    Adjusts resolution, format, and quality to balance image quality
    against bandwidth and cost.
    """

    def __init__(self) -> None:
        self._offline_queue: deque[np.ndarray] = deque(maxlen=50)
        self._total_original_bytes = 0
        self._total_encoded_bytes = 0
        self._encode_count = 0

    def encode(
        self, frame: np.ndarray, condition: NetworkCondition
    ) -> bytes | None:
        """Encode a frame according to the network condition.

        Args:
            frame: Input image.
            condition: Current network condition.

        Returns:
            Encoded bytes, or None if offline (frame is queued).
        """
        if condition == NetworkCondition.OFFLINE:
            self._offline_queue.append(frame.copy())
            logger.debug("offline: queued frame (%d in queue)", len(self._offline_queue))
            return None

        params = _ENCODE_PARAMS[condition]
        h, w = frame.shape[:2]
        target = params["resolution"]

        # Resize maintaining aspect ratio
        scale = target / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
        else:
            resized = frame

        # Encode
        fmt = params["format"]
        quality = params["quality"]
        if fmt == ".webp":
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        success, encoded = cv2.imencode(fmt, resized, encode_params)
        if not success:
            logger.error("encoding failed")
            return None

        data = encoded.tobytes()

        # Track stats
        original_bytes = frame.nbytes
        self._total_original_bytes += original_bytes
        self._total_encoded_bytes += len(data)
        self._encode_count += 1

        logger.debug(
            "encoded %s q=%d %dx%d -> %d bytes (%.0f%% reduction)",
            fmt,
            quality,
            resized.shape[1],
            resized.shape[0],
            len(data),
            (1 - len(data) / original_bytes) * 100,
        )
        return data

    def flush_queue(
        self, condition: NetworkCondition
    ) -> list[bytes]:
        """Encode and return all queued offline frames.

        Args:
            condition: Network condition to use for encoding.

        Returns:
            List of encoded frames.
        """
        results = []
        while self._offline_queue:
            frame = self._offline_queue.popleft()
            encoded = self.encode(frame, condition)
            if encoded is not None:
                results.append(encoded)
        return results

    def estimate_tokens(self, encoded: bytes) -> int:
        """Estimate the number of VLM tokens for an encoded image.

        Args:
            encoded: Encoded image bytes.

        Returns:
            Estimated token count.
        """
        # Rough heuristic: ~750 tokens per 100KB
        return max(85, int(len(encoded) / 100_000 * 750))

    def estimate_cost(self, tokens: int, model: str = "claude-sonnet-4-20250514") -> float:
        """Estimate API cost for the given token count.

        Args:
            tokens: Number of tokens.
            model: Model name. Default is claude-sonnet-4.

        Returns:
            Estimated cost in USD.
        """
        cost_per_token = _TOKEN_COSTS.get(model, 3.0 / 1_000_000)
        return tokens * cost_per_token

    @property
    def compression_stats(self) -> dict:
        """Compression statistics."""
        if self._encode_count == 0:
            return {
                "encode_count": 0,
                "total_original_bytes": 0,
                "total_encoded_bytes": 0,
                "compression_ratio": 0.0,
                "queue_size": len(self._offline_queue),
            }
        return {
            "encode_count": self._encode_count,
            "total_original_bytes": self._total_original_bytes,
            "total_encoded_bytes": self._total_encoded_bytes,
            "compression_ratio": (
                1 - self._total_encoded_bytes / self._total_original_bytes
            ),
            "queue_size": len(self._offline_queue),
        }
