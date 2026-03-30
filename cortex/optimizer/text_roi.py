"""Text region detection using OpenCV MSER + morphology.

Uses MSER (Maximally Stable Extremal Regions) to detect text-like regions
without requiring OCR dependencies (pytesseract/easyocr).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TextROIStrategy:
    """Detects text regions in frames and provides ROI cropping.

    Uses MSER for text-like region detection with morphological
    post-processing to merge nearby regions.
    """

    def __init__(self) -> None:
        self._mser = cv2.MSER.create()
        self._mser.setMinArea(60)
        self._mser.setMaxArea(14400)

    def detect_regions(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Detect text-like regions in the frame.

        Args:
            frame: Input image (BGR or grayscale).

        Returns:
            List of bounding boxes as (x, y, w, h).
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        regions, _ = self._mser.detectRegions(gray)
        boxes = [cv2.boundingRect(r.reshape(-1, 1, 2)) for r in regions]

        # Merge overlapping boxes via morphology
        if not boxes:
            logger.debug("no text regions detected")
            return []

        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        for x, y, w, h in boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged = [cv2.boundingRect(c) for c in contours]

        logger.debug("text regions: %d raw -> %d merged", len(boxes), len(merged))
        return merged

    def score_map(
        self, frame: np.ndarray, grid: tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """Generate a score map with high values where text is detected.

        Args:
            frame: Input image.
            grid: Grid size (columns, rows).

        Returns:
            Score map of shape (rows, cols) with values 0.0-1.0.
        """
        h, w = frame.shape[:2]
        cols, rows = grid
        cell_w, cell_h = w / cols, h / rows

        score = np.zeros((rows, cols), dtype=np.float32)
        regions = self.detect_regions(frame)

        for rx, ry, rw, rh in regions:
            # Mark grid cells that overlap with text regions
            c_start = max(0, int(rx / cell_w))
            c_end = min(cols, int((rx + rw) / cell_w) + 1)
            r_start = max(0, int(ry / cell_h))
            r_end = min(rows, int((ry + rh) / cell_h) + 1)
            score[r_start:r_end, c_start:c_end] = 1.0

        return score

    def crop(self, frame: np.ndarray, margin: float = 0.2) -> np.ndarray:
        """Crop to bounding box of all text regions with margin.

        Args:
            frame: Input image.
            margin: Extra margin around text regions (fraction). Default 0.2.

        Returns:
            Cropped frame. Returns original if no text detected.
        """
        regions = self.detect_regions(frame)
        if not regions:
            return frame

        h, w = frame.shape[:2]
        x_min = min(r[0] for r in regions)
        y_min = min(r[1] for r in regions)
        x_max = max(r[0] + r[2] for r in regions)
        y_max = max(r[1] + r[3] for r in regions)

        # Add margin
        margin_x = int((x_max - x_min) * margin)
        margin_y = int((y_max - y_min) * margin)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)

        logger.debug("text_crop region=(%d,%d,%d,%d)", x_min, y_min, x_max, y_max)
        return frame[y_min:y_max, x_min:x_max]
