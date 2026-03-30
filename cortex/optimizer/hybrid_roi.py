"""Hybrid ROI combining center, text, and saliency strategies."""

import enum
import logging

import cv2
import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.optimizer.center_crop import CenterCropStrategy
from cortex.optimizer.saliency_roi import SaliencyROIStrategy
from cortex.optimizer.text_roi import TextROIStrategy

logger = logging.getLogger(__name__)


class RequestType(enum.Enum):
    """Type of VLM request, determines ROI weight distribution."""

    TEXT_RECOGNITION = "text_recognition"
    OBJECT_SCENE = "object_scene"
    NAVIGATION = "navigation"
    GENERAL = "general"


_WEIGHTS: dict[RequestType, tuple[float, float, float]] = {
    # (center, text, saliency)
    RequestType.TEXT_RECOGNITION: (0.1, 0.6, 0.3),
    RequestType.OBJECT_SCENE: (0.1, 0.1, 0.8),
    RequestType.NAVIGATION: (0.1, 0.1, 0.8),
    RequestType.GENERAL: (0.2, 0.2, 0.6),
}


class HybridROI:
    """Fuses center, text, and saliency score maps for ROI selection.

    Combines three strategies with weighted score map fusion:
    S = wc*Sc + wt*St + ws*Ss

    Applies EMA temporal smoothing on the fused score map.

    Args:
        request_type: Initial request type. Default is GENERAL.
        ema_alpha: EMA smoothing factor. Default is 0.7.
    """

    def __init__(
        self,
        request_type: RequestType = RequestType.GENERAL,
        ema_alpha: float = 0.85,
    ) -> None:
        self._center = CenterCropStrategy()
        self._text = TextROIStrategy()
        self._saliency = SaliencyROIStrategy()
        self._request_type = request_type
        self._ema_alpha = ema_alpha
        self._prev_score: np.ndarray | None = None
        self._battery_mode = BatteryMode.BALANCED

    @property
    def request_type(self) -> RequestType:
        """Current request type."""
        return self._request_type

    def set_request_type(self, request_type: RequestType) -> None:
        """Update the request type and weight distribution.

        Args:
            request_type: New request type.
        """
        self._request_type = request_type
        logger.debug("request_type=%s", request_type.value)

    def set_battery_mode(self, mode: BatteryMode) -> None:
        """Set battery mode. POWER_SAVE skips saliency computation.

        Args:
            mode: Battery mode.
        """
        self._battery_mode = mode

    def fused_score_map(
        self, frame: np.ndarray, grid: tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """Compute the fused, EMA-smoothed score map.

        Args:
            frame: Input image.
            grid: Grid size (columns, rows).

        Returns:
            Fused score map of shape (rows, cols).
        """
        wc, wt, ws = _WEIGHTS[self._request_type]

        sc = self._center.score_map(frame, grid)
        st = self._text.score_map(frame, grid)

        if self._battery_mode == BatteryMode.POWER_SAVE:
            ss = np.zeros_like(sc)
            # Redistribute saliency weight to center
            wc += ws
            ws = 0.0
        else:
            ss = self._saliency.score_map(frame, grid)

        # Apply center weight as multiplicative gate on saliency
        # This suppresses saliency far from center
        ss_adjusted = ss * sc

        fused = wc * sc + wt * st + ws * ss_adjusted

        # Normalize
        f_max = fused.max()
        if f_max > 0:
            fused /= f_max

        # EMA temporal smoothing
        if self._prev_score is not None and self._prev_score.shape == fused.shape:
            fused = self._ema_alpha * fused + (1 - self._ema_alpha) * self._prev_score

        self._prev_score = fused.copy()
        return fused

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop the frame based on the fused score map.

        Selects grid cells above the mean score and crops to their
        bounding box.

        Args:
            frame: Input image.

        Returns:
            Cropped frame.
        """
        grid = (8, 6)
        score = self.fused_score_map(frame, grid)

        h, w = frame.shape[:2]
        cols, rows = grid
        cell_w, cell_h = w / cols, h / rows

        threshold = score.mean() + 0.5 * (score.max() - score.mean())
        mask = score >= threshold

        coords = np.argwhere(mask)
        if len(coords) == 0:
            return frame

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)

        y1 = int(r_min * cell_h)
        y2 = int((r_max + 1) * cell_h)
        x1 = int(c_min * cell_w)
        x2 = int((c_max + 1) * cell_w)

        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(h, y2)
        x2 = min(w, x2)

        logger.debug("hybrid_crop region=(%d,%d,%d,%d)", x1, y1, x2, y2)
        return frame[y1:y2, x1:x2]
