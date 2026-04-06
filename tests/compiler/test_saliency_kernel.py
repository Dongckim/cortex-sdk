"""Tests for saliency kernel variants — correctness and equivalence."""

import numpy as np
import pytest

from cortex.compiler import (
    saliency_baseline,
    saliency_jit,
    saliency_vectorized,
    warmup_jit,
)

GRID = (8, 6)
ATOL = 1e-4   # tolerance for floating-point equivalence across kernels


@pytest.fixture(scope="module", autouse=True)
def _warmup():
    """Pre-compile JIT kernel once for the entire test module."""
    warmup_jit()


@pytest.fixture
def frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:380, 100:540] = 160
    return img


@pytest.fixture
def gray_frame():
    img = np.zeros((480, 640), dtype=np.uint8)
    img[100:380, 100:540] = 160
    return img


# ── output shape and range ────────────────────────────────────────────

@pytest.mark.parametrize("fn", [saliency_baseline, saliency_vectorized, saliency_jit])
def test_output_shape(fn, frame):
    cols, rows = GRID
    result = fn(frame, GRID)
    assert result.shape == (rows, cols), f"{fn.__name__}: expected ({rows},{cols}), got {result.shape}"


@pytest.mark.parametrize("fn", [saliency_baseline, saliency_vectorized, saliency_jit])
def test_output_dtype_float32(fn, frame):
    result = fn(frame, GRID)
    assert result.dtype == np.float32


@pytest.mark.parametrize("fn", [saliency_baseline, saliency_vectorized, saliency_jit])
def test_output_range(fn, frame):
    result = fn(frame, GRID)
    assert result.min() >= 0.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("fn", [saliency_baseline, saliency_vectorized, saliency_jit])
def test_grayscale_input_accepted(fn, gray_frame):
    result = fn(gray_frame, GRID)
    assert result.shape == (GRID[1], GRID[0])


# ── numerical equivalence ─────────────────────────────────────────────

def test_vectorized_matches_baseline(frame):
    base = saliency_baseline(frame, GRID)
    vec  = saliency_vectorized(frame, GRID)
    assert np.allclose(base, vec, atol=ATOL), (
        f"max diff: {np.abs(base - vec).max():.2e}"
    )


def test_jit_matches_baseline(frame):
    base = saliency_baseline(frame, GRID)
    jit  = saliency_jit(frame, GRID)
    assert np.allclose(base, jit, atol=ATOL), (
        f"max diff: {np.abs(base - jit).max():.2e}"
    )


def test_all_three_consistent(frame):
    base = saliency_baseline(frame, GRID)
    vec  = saliency_vectorized(frame, GRID)
    jit  = saliency_jit(frame, GRID)
    assert np.allclose(base, vec, atol=ATOL)
    assert np.allclose(base, jit, atol=ATOL)
    assert np.allclose(vec,  jit, atol=ATOL)


# ── uniform frame edge case ───────────────────────────────────────────

@pytest.mark.parametrize("fn", [saliency_baseline, saliency_vectorized, saliency_jit])
def test_uniform_frame_no_crash(fn):
    """Uniform frames produce zero-saliency — should not raise or return NaN."""
    uniform = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = fn(uniform, GRID)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
