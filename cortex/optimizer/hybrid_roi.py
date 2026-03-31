"""Hybrid ROI combining center, text, saliency, and motion strategies.

Based on CORTEX paper Section III-B:
  S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm

Where Ss_adj applies a soft center gate to saliency,
and Sm captures frame-to-frame motion regions.
"""

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


# Paper Section III-B: weights per request type
# (center, text, saliency, motion)
_WEIGHTS: dict[RequestType, tuple[float, float, float, float]] = {
    RequestType.TEXT_RECOGNITION: (0.2, 0.6, 0.1, 0.1),
    RequestType.OBJECT_SCENE: (0.2, 0.1, 0.5, 0.2),
    RequestType.NAVIGATION: (0.2, 0.1, 0.3, 0.4),
    RequestType.GENERAL: (0.3, 0.2, 0.3, 0.2),
}

# Minimum saliency after center gating (prevents total suppression)
_CENTER_GATE_FLOOR = 0.3


class HybridROI:
    """Fuses center, text, saliency, and motion score maps for ROI.

    Score map fusion (paper Section III-B):
      Ss_adj = Ss * (floor + (1-floor) * Sc)   # soft center gate
      Sm = frame_diff_grid                      # motion detection
      S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm

    Applies EMA temporal smoothing to prevent ROI jitter.

    Args:
        request_type: Initial request type. Default is GENERAL.
        ema_alpha: EMA smoothing factor (0-1). Default is 0.7.
    """

    def __init__(
        self,
        request_type: RequestType = RequestType.GENERAL,
        ema_alpha: float = 0.7,
    ) -> None:
        self._center = CenterCropStrategy()
        self._text = TextROIStrategy()
        self._saliency = SaliencyROIStrategy()
        self._request_type = request_type
        self._ema_alpha = ema_alpha
        self._prev_score: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
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

    def _motion_score_map(
        self, frame: np.ndarray, grid: tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """Compute motion score map from frame-to-frame absdiff.

        Args:
            frame: Input image (BGR or grayscale).
            grid: Grid size (columns, rows).

        Returns:
            Motion score map of shape (rows, cols) with values 0.0-1.0.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        cols, rows = grid

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return np.zeros((rows, cols), dtype=np.float32)

        diff = cv2.absdiff(self._prev_gray, gray).astype(np.float32)
        self._prev_gray = gray.copy()

        # Blur to reduce webcam noise sensitivity
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Threshold to ignore minor pixel noise (< 10/255)
        diff[diff < 10] = 0

        h, w = diff.shape[:2]
        cell_h, cell_w = h / rows, w / cols

        score = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                score[r, c] = diff[y1:y2, x1:x2].mean()

        s_max = score.max()
        if s_max > 0:
            score /= s_max

        return score

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
        wc, wt, ws, wm = _WEIGHTS[self._request_type]

        sc = self._center.score_map(frame, grid)
        st = self._text.score_map(frame, grid)

        if self._battery_mode == BatteryMode.POWER_SAVE:
            ss = np.zeros_like(sc)
            wc += ws
            ws = 0.0
        else:
            ss = self._saliency.score_map(frame, grid)

        # Soft center gate: preserves at least _CENTER_GATE_FLOOR
        # of saliency even at edges (paper: prevents total suppression)
        center_gate = _CENTER_GATE_FLOOR + (1 - _CENTER_GATE_FLOOR) * sc
        ss_adjusted = ss * center_gate

        # Motion score map
        sm = self._motion_score_map(frame, grid)

        fused = wc * sc + wt * st + ws * ss_adjusted + wm * sm

        # Normalize
        f_max = fused.max()
        if f_max > 0:
            fused /= f_max

        # EMA temporal smoothing (paper: prevents ROI jitter)
        if self._prev_score is not None and self._prev_score.shape == fused.shape:
            fused = self._ema_alpha * fused + (1 - self._ema_alpha) * self._prev_score

        self._prev_score = fused.copy()
        return fused

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop the frame based on the fused score map.

        Selects grid cells above a threshold and crops to their
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

        # Keep cells in the top 40% of the score range
        s_min, s_max = score.min(), score.max()
        threshold = s_min + 0.6 * (s_max - s_min)
        mask = score >= threshold

        coords = np.argwhere(mask)
        if len(coords) == 0:
            return frame

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)

        y1 = max(0, int(r_min * cell_h))
        y2 = min(h, int((r_max + 1) * cell_h))
        x1 = max(0, int(c_min * cell_w))
        x2 = min(w, int((c_max + 1) * cell_w))

        logger.debug("hybrid_crop region=(%d,%d,%d,%d)", x1, y1, x2, y2)
        return frame[y1:y2, x1:x2]
