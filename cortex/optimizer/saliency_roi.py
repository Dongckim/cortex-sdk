"""Saliency-based ROI detection using spectral residual method."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SaliencyROIStrategy:
    """Detects salient regions using the spectral residual method.

    Computes a saliency map by analyzing the log spectrum of the image
    and extracting the spectral residual.
    """

    def _saliency_map(self, frame: np.ndarray) -> np.ndarray:
        """Compute the spectral residual saliency map.

        Args:
            frame: Input image (BGR or grayscale).

        Returns:
            Saliency map normalized to 0.0-1.0.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray = cv2.resize(gray, (64, 64)).astype(np.float32)

        # Spectral residual
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
        log_mag = np.log(magnitude + 1e-9)
        smoothed = cv2.blur(log_mag, (3, 3))
        residual = log_mag - smoothed

        # Reconstruct
        exp_residual = np.exp(residual)
        real = exp_residual * np.cos(phase)
        imag = exp_residual * np.sin(phase)
        combined = np.stack([real, imag], axis=-1)
        inv = cv2.idft(combined)
        saliency = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

        # Post-process
        saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
        saliency = saliency**2

        # Normalize
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 0:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency.astype(np.float32)

    def score_map(
        self, frame: np.ndarray, grid: tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """Generate a saliency score map on a grid.

        Args:
            frame: Input image.
            grid: Grid size (columns, rows).

        Returns:
            Score map of shape (rows, cols) with values 0.0-1.0.
        """
        saliency = self._saliency_map(frame)
        cols, rows = grid
        sh, sw = saliency.shape[:2]
        cell_h, cell_w = sh / rows, sw / cols

        score = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                score[r, c] = saliency[y1:y2, x1:x2].mean()

        s_max = score.max()
        if s_max > 0:
            score /= s_max

        return score

    def crop(
        self, frame: np.ndarray, top_percent: float = 0.4
    ) -> np.ndarray:
        """Crop to the region containing the most salient area.

        Args:
            frame: Input image.
            top_percent: Fraction of saliency to keep. Default 0.4.

        Returns:
            Cropped frame around the most salient region.
        """
        h, w = frame.shape[:2]
        saliency = self._saliency_map(frame)
        saliency_full = cv2.resize(saliency, (w, h))

        threshold = np.percentile(saliency_full, (1 - top_percent) * 100)
        mask = (saliency_full >= threshold).astype(np.uint8)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return frame

        x, y, rw, rh = cv2.boundingRect(coords)

        # Add small margin
        margin = int(max(rw, rh) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        rw = min(w - x, rw + 2 * margin)
        rh = min(h - y, rh + 2 * margin)

        logger.debug("saliency_crop region=(%d,%d,%d,%d)", x, y, rw, rh)
        return frame[y : y + rh, x : x + rw]
