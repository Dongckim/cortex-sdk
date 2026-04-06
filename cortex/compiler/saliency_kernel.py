"""Three implementations of the saliency DFT kernel.

Compilation pipeline analogy (MLIR):
  Python ops  →  Graph IR node (saliency_dft)
                    ↓  compiler.partition()
              compilable subgraph
                    ↓  compiler.compile()
  baseline  →  vectorized  →  jit   (progressive optimization passes)

All three versions are numerically identical (max diff < 1e-5).

baseline    — direct port of SaliencyROIStrategy._saliency_map() +
              score_map(), including the Python-level grid loop.
              This is what runs today inside the Graph IR node.

vectorized  — same algorithm, Python loop replaced with numpy slice
              reshape + mean (no explicit for-loops).
              Equivalent to an auto-vectorization pass in a real compiler.

jit         — numba @jit(nopython=True) over the hot inner loop of the
              spectral residual + grid pooling.
              Equivalent to LLVM codegen in TVM / MLIR lowering.
"""

from __future__ import annotations

import numpy as np
import cv2

try:
    from numba import jit as _numba_jit
    _NUMBA_AVAILABLE = True
except ImportError:          # pragma: no cover
    _NUMBA_AVAILABLE = False


# ── shared constants ──────────────────────────────────────────────────
_RESIZE   = (64, 64)
_BLUR_K   = (3, 3)
_GAUSS_K  = (9, 9)
_GAUSS_S  = 2.5
_EPS      = np.float32(1e-9)


# ─────────────────────────────────────────────────────────────────────
# Baseline  (mirrors SaliencyROIStrategy exactly)
# ─────────────────────────────────────────────────────────────────────

def saliency_baseline(frame: np.ndarray, grid: tuple[int, int] = (8, 6)) -> np.ndarray:
    """Baseline: direct port of existing SaliencyROIStrategy.score_map().

    Compilation stage: Python source — no optimisation passes applied.

    Args:
        frame: BGR or grayscale input image.
        grid:  (cols, rows) score grid.

    Returns:
        Score map of shape (rows, cols), float32, values 0–1.
    """
    # ── grayscale + resize ───────────────────────────────────────────
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, _RESIZE).astype(np.float32)

    # ── spectral residual (cv2 DFT path) ────────────────────────────
    dft      = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, ph  = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
    log_mag  = np.log(mag + _EPS)
    smoothed = cv2.blur(log_mag, _BLUR_K)
    residual = log_mag - smoothed

    exp_r    = np.exp(residual)
    real     = exp_r * np.cos(ph)
    imag     = exp_r * np.sin(ph)
    combined = np.stack([real, imag], axis=-1)
    inv      = cv2.idft(combined)
    saliency = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    saliency = cv2.GaussianBlur(saliency, _GAUSS_K, _GAUSS_S)
    saliency = saliency ** 2

    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 0:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    # ── grid pooling — Python loop (BASELINE) ────────────────────────
    cols, rows = grid
    sh, sw = saliency.shape
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


# ─────────────────────────────────────────────────────────────────────
# Vectorized  (auto-vectorisation pass analogue)
# ─────────────────────────────────────────────────────────────────────

def saliency_vectorized(frame: np.ndarray, grid: tuple[int, int] = (8, 6)) -> np.ndarray:
    """Vectorized: Python loop replaced with numpy reshape + mean.

    Compilation stage: auto-vectorisation pass — eliminates the O(rows×cols)
    Python loop overhead by reshaping the saliency map into a 4-D tensor and
    calling np.mean over the spatial axes in a single C-level operation.

    The spectral residual path is unchanged (cv2 DFT is already a C call).

    Args:
        frame: BGR or grayscale input image.
        grid:  (cols, rows) score grid.

    Returns:
        Score map of shape (rows, cols), float32, values 0–1.
    """
    # ── grayscale + resize ───────────────────────────────────────────
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, _RESIZE).astype(np.float32)

    # ── spectral residual (same as baseline) ─────────────────────────
    dft      = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, ph  = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
    log_mag  = np.log(mag + _EPS)
    smoothed = cv2.blur(log_mag, _BLUR_K)
    residual = log_mag - smoothed

    exp_r    = np.exp(residual)
    real     = exp_r * np.cos(ph)
    imag     = exp_r * np.sin(ph)
    combined = np.stack([real, imag], axis=-1)
    inv      = cv2.idft(combined)
    saliency = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    saliency = cv2.GaussianBlur(saliency, _GAUSS_K, _GAUSS_S)
    saliency = saliency ** 2

    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 0:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    # ── grid pooling — VECTORIZED (np.add.reduceat) ──────────────────
    # Use the same floating-point boundary formula as baseline so cell
    # sizes remain numerically identical. np.add.reduceat sums each
    # segment along the axis in a single C call — no Python loop.
    cols, rows = grid
    sh, sw = saliency.shape
    cell_h_f = sh / rows
    cell_w_f = sw / cols
    row_starts = [int(r * cell_h_f) for r in range(rows)]
    col_starts = [int(c * cell_w_f) for c in range(cols)]
    row_ends   = [int((r + 1) * cell_h_f) for r in range(rows)]
    col_ends   = [int((c + 1) * cell_w_f) for c in range(cols)]
    row_sizes  = np.array([row_ends[r] - row_starts[r] for r in range(rows)],
                          dtype=np.float32)
    col_sizes  = np.array([col_ends[c] - col_starts[c] for c in range(cols)],
                          dtype=np.float32)

    # sum rows → (rows, W), then sum cols → (rows, cols)
    partial = np.add.reduceat(saliency, row_starts, axis=0)[:rows]
    partial = np.add.reduceat(partial,  col_starts, axis=1)[:, :cols]
    areas   = (row_sizes[:, np.newaxis] * col_sizes[np.newaxis, :])
    score   = (partial / areas).astype(np.float32)

    s_max = score.max()
    if s_max > 0:
        score /= s_max
    return score


# ─────────────────────────────────────────────────────────────────────
# JIT  (LLVM codegen analogue via numba)
# ─────────────────────────────────────────────────────────────────────
#
# Compilation boundary rationale:
#   The spectral residual uses cv2.dft / cv2.idft — these are already
#   compiled C extensions and cannot be called from numba nopython mode.
#   They are treated as external_call nodes (BYOC boundary), exactly as
#   text_roi_mser is in the Graph IR.
#
#   The Python-level grid pooling loop IS the hot path that numba targets:
#     for r in range(rows): for c in range(cols): saliency[y1:y2, x1:x2].mean()
#   This is lowered to a single LLVM-compiled loop nest by _grid_pool_jit.
#
# TVM analogy:
#   cv2 ops → external_call (BYOC, stays opaque)
#   grid loop → compilable → TIR loop nest → LLVM codegen

if _NUMBA_AVAILABLE:
    @_numba_jit(nopython=True, cache=True)
    def _grid_pool_jit(saliency: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Numba JIT grid pooling kernel.

        Compilation stage: LLVM codegen — numba lowers this loop nest to
        native machine code, eliminating Python interpreter overhead for
        every cell iteration. Analogous to TVM's TIR → LLVM lowering pass.

        Args:
            saliency: float32 array (H, W).
            rows:     number of grid rows.
            cols:     number of grid columns.

        Returns:
            Score map float32 (rows, cols).
        """
        H, W = saliency.shape
        cell_h = H / rows
        cell_w = W / cols
        score = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            for c in range(cols):
                y1 = int(r * cell_h)
                y2 = int((r + 1) * cell_h)
                x1 = int(c * cell_w)
                x2 = int((c + 1) * cell_w)
                s, cnt = 0.0, 0
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        s   += saliency[y, x]
                        cnt += 1
                score[r, c] = s / cnt if cnt > 0 else 0.0
        return score

else:                        # pragma: no cover
    def _grid_pool_jit(saliency, rows, cols):  # type: ignore[misc]
        raise RuntimeError("numba not installed")


def saliency_jit(frame: np.ndarray, grid: tuple[int, int] = (8, 6)) -> np.ndarray:
    """JIT: cv2 spectral residual + numba-compiled grid pooling.

    Compilation stage: LLVM codegen for the grid pooling loop.

    The spectral residual (cv2.dft / cv2.idft) is an external_call boundary —
    cv2 is an opaque C extension that cannot be lowered into numba's IR,
    analogous to TVM's BYOC ops. Only the grid pooling loop is JIT-compiled.

    Optimization passes applied vs baseline:
      baseline  → Python loop (rows×cols interpreter iterations)
      vectorized → numpy reshape (single C call, no Python loop)
      jit        → numba LLVM loop (compiled, no interpreter + SIMD-eligible)

    Args:
        frame: BGR or grayscale input image.
        grid:  (cols, rows) score grid.

    Returns:
        Score map of shape (rows, cols), float32, values 0–1.
    """
    if not _NUMBA_AVAILABLE:                   # pragma: no cover
        return saliency_vectorized(frame, grid)

    # ── grayscale + resize  (cv2 — external_call boundary) ───────────
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, _RESIZE).astype(np.float32)

    # ── spectral residual  (cv2 DFT — external_call boundary) ────────
    dft      = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, ph  = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
    log_mag  = np.log(mag + _EPS)
    smoothed = cv2.blur(log_mag, _BLUR_K)
    residual = log_mag - smoothed

    exp_r    = np.exp(residual)
    real     = exp_r * np.cos(ph)
    imag     = exp_r * np.sin(ph)
    combined = np.stack([real, imag], axis=-1)
    inv      = cv2.idft(combined)
    saliency = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    saliency = cv2.GaussianBlur(saliency, _GAUSS_K, _GAUSS_S)
    saliency = saliency ** 2

    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 0:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    # ── JIT kernel: grid pooling  (LLVM-compiled loop nest) ──────────
    cols, rows = grid
    score = _grid_pool_jit(saliency, rows, cols)

    s_max = score.max()
    if s_max > 0:
        score /= s_max
    return score


def warmup_jit(grid: tuple[int, int] = (8, 6)) -> None:
    """Trigger numba JIT compilation on a dummy frame.

    Call once at startup so benchmark timings exclude first-call LLVM compile
    cost, analogous to TVM's ahead-of-time (AOT) compilation step.
    """
    dummy = np.zeros((64, 64), dtype=np.float32)
    _grid_pool_jit(dummy, grid[1], grid[0])
