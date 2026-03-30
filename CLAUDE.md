# CORTEX — Camera Optimized Realtime Transmission Exchange

## What is this?
Battery-aware VLM middleware SDK for wearable devices (smart glasses).
Sits between camera source and VLM API, optimizing images for cost/battery/latency.

## Architecture (4 layers)
- L1 Capture: IMU gate → blur filter → SSIM scene change detection
- L2 Compress: Hybrid ROI crop (center + text + saliency) → adaptive WebP/JPEG
- L3 Bridge: Unified VLM adapter (Claude, GPT, Gemini, Ollama) + circuit breaker
- L4 Memory: Sliding window + hierarchical summary + embedding retrieval

## Tech stack
- Python 3.11+
- opencv-python (capture, image processing)
- scikit-image (SSIM)
- numpy, pillow
- httpx (async VLM API calls)
- pytest (testing)

## Project structure
cortex/
  capture/       # L1
  optimizer/     # L2
  bridge/        # L3
  memory/        # L4
  pipeline.py    # Main CortexPipeline
  config.py      # Configuration

## Code conventions
- Type hints on all functions
- Docstrings on all public classes/methods (Google style)
- Each class in its own file
- Tests mirror source structure in tests/
- No print() — use logging module
- Line length: 88 chars (black formatter)
- Import order: stdlib → third-party → local
