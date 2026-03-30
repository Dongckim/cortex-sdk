"""Center-weighted crop strategy."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CenterCropStrategy:
    """Crops the center portion of a frame with Gaussian-weighted scoring.

    Zero computation cost — uses a precomputed Gaussian center weight map.
    """

    def crop(self, frame: np.ndarray, ratio: float = 0.7) -> np.ndarray:
        """Crop the center portion of the frame.

        Args:
            frame: Input image as a numpy array.
            ratio: Fraction of the frame to keep (0.0 to 1.0). Default is 0.7.

        Returns:
            Cropped center region.
        """
        h, w = frame.shape[:2]
        new_h, new_w = int(h * ratio), int(w * ratio)
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        cropped = frame[y_start : y_start + new_h, x_start : x_start + new_w]
        logger.debug("center_crop ratio=%.2f %dx%d -> %dx%d", ratio, w, h, new_w, new_h)
        return cropped

    def score_map(
        self, frame: np.ndarray, grid: tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """Generate a Gaussian center-weighted score map.

        Args:
            frame: Input image (used only for dimensions).
            grid: Grid size (columns, rows). Default is (8, 6).

        Returns:
            Score map as a numpy array of shape (rows, cols) with values 0.0-1.0.
        """
        cols, rows = grid
        cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
        sigma_x, sigma_y = cols / 3.0, rows / 3.0

        y_idx, x_idx = np.mgrid[0:rows, 0:cols]
        score = np.exp(
            -((x_idx - cx) ** 2 / (2 * sigma_x**2))
            - ((y_idx - cy) ** 2 / (2 * sigma_y**2))
        )
        # Normalize to 0-1
        score = score / score.max()
        return score.astype(np.float32)
