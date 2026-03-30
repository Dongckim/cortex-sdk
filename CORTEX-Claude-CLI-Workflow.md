# CORTEX — Claude CLI Development Workflow

> **Camera Optimized Realtime Transmission Exchange**
> Battery-aware VLM middleware SDK for wearable devices

---

## Overview

This document is your step-by-step guide to building CORTEX using Claude CLI (Claude Code).
Follow each step in order. Do not skip ahead.

**Core principle**: One class at a time → test → commit → next.

---

## Prerequisites

- Python 3.11+
- Git
- Claude CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Anthropic API key configured

---

## Step 0: Project Initialization

Run these commands in your terminal:

```bash
# Create project
mkdir cortex-sdk && cd cortex-sdk

# Git
git init

# Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core dependencies
pip install opencv-python scikit-image numpy pillow httpx

# Dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Create initial structure
mkdir -p cortex/capture cortex/optimizer cortex/bridge cortex/memory
mkdir -p tests/capture tests/optimizer tests/bridge tests/memory
mkdir -p examples docs

# Create package init files
touch cortex/__init__.py
touch cortex/capture/__init__.py
touch cortex/optimizer/__init__.py
touch cortex/bridge/__init__.py
touch cortex/memory/__init__.py
touch tests/__init__.py
touch tests/capture/__init__.py
touch tests/optimizer/__init__.py
touch tests/bridge/__init__.py
touch tests/memory/__init__.py

# Initial commit
git add -A && git commit -m "chore: initial project structure"
```

---

## Step 1: Create CLAUDE.md

**DO THIS MANUALLY — do not ask Claude CLI to write it.**

Create `CLAUDE.md` in the project root:

```markdown
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
```

Then commit:

```bash
git add CLAUDE.md && git commit -m "docs: add CLAUDE.md project context"
```

---

## Step 2: Launch Claude CLI

```bash
claude
```

Claude will automatically read CLAUDE.md and understand the project.

---

## Phase 1: Capture Engine (Layer 1)

### 2-1. pyproject.toml

**Prompt to Claude CLI:**

```
Create pyproject.toml for cortex-sdk.
- Package name: cortex-sdk
- Version: 0.1.0
- License: MIT
- Author: Dongchan Kim <dck.alx@gmail.com>
- Requires Python >=3.11
- Dependencies: opencv-python, scikit-image, numpy, pillow, httpx
- Dev dependencies (optional): pytest, pytest-asyncio, pytest-cov
- Entry point: cortex
```

**After:** `git add -A && git commit -m "build: add pyproject.toml"`

---

### 2-2. BlurDetector

**Prompt to Claude CLI:**

```
Create cortex/capture/blur_detector.py:
- BlurDetector class
- Uses Laplacian variance method to detect motion blur
- Method: detect(frame: np.ndarray) -> bool
  - Returns True if frame is sharp, False if blurry
- Constructor takes threshold: float = 100.0
- Add logging for each detection result
- Also create tests/capture/test_blur_detector.py with:
  - Test with a sharp synthetic image (high variance)
  - Test with a blurry synthetic image (gaussian blur applied)
  - Test threshold customization
```

**After:**
```
/run pytest tests/capture/test_blur_detector.py -v
```
Then: `git add -A && git commit -m "feat(capture): add BlurDetector with Laplacian variance"`

---

### 2-3. SceneChangeDetector

**Prompt to Claude CLI:**

```
Create cortex/capture/scene_change.py:
- SceneChangeDetector class
- Uses SSIM (from skimage.metrics) to compare current frame vs last accepted frame
- Method: detect(frame: np.ndarray) -> bool
  - Returns True if scene has changed (SSIM below threshold)
  - First frame always returns True
  - Internally stores last_accepted_frame
- Method: reset() to clear stored frame
- Constructor takes threshold: float = 0.85
- Frames are converted to grayscale before SSIM comparison
- Also create tests/capture/test_scene_change.py with:
  - Test first frame always accepted
  - Test identical frames rejected
  - Test different frames accepted
  - Test threshold customization
```

**After:**
```
/run pytest tests/capture/test_scene_change.py -v
```
Then: `git add -A && git commit -m "feat(capture): add SceneChangeDetector with SSIM"`

---

### 2-4. IMUGate

**Prompt to Claude CLI:**

```
Create cortex/capture/imu_gate.py:
- BatteryMode enum: AGGRESSIVE, BALANCED, POWER_SAVE
- IMUGate class
- Simulates IMU-based motion detection for wearable devices
- Method: update(accel: tuple[float,float,float], gyro: tuple[float,float,float]) -> bool
  - Computes motion_score = accel_weight * accel_delta + gyro_weight * gyro_delta
  - Returns True if motion_score exceeds threshold
  - Stores previous values internally
- Method: set_battery_mode(mode: BatteryMode)
  - AGGRESSIVE: threshold=0.5 (captures more)
  - BALANCED: threshold=1.0
  - POWER_SAVE: threshold=2.0 (captures less)
- Property: motion_score -> float (last computed score)
- First call always returns True (no previous data)
- Also create tests/capture/test_imu_gate.py with:
  - Test stationary data returns False
  - Test motion data returns True
  - Test battery mode changes threshold
  - Test first call returns True
```

**After:**
```
/run pytest tests/capture/test_imu_gate.py -v
```
Then: `git add -A && git commit -m "feat(capture): add IMUGate with battery-aware thresholds"`

---

### 2-5. CaptureEngine (Integration)

**Prompt to Claude CLI:**

```
Create cortex/capture/engine.py:
- CaptureResult dataclass:
  - accepted: bool
  - reason: str (e.g., "no_motion", "blurry", "no_change", "accepted")
  - stats: dict (motion_score, blur_score, ssim_score, battery_mode)
- CaptureEngine class:
  - Constructor takes optional BlurDetector, SceneChangeDetector, IMUGate instances
  - If not provided, creates defaults
  - Method: process_frame(frame: np.ndarray, imu_data: dict | None = None) -> CaptureResult
    - Pipeline: IMU gate → blur check → SSIM check
    - If imu_data is None, skip IMU gate
    - Returns CaptureResult with reason for accept/reject
  - Method: set_battery_mode(mode: BatteryMode)
    - Updates IMU gate threshold
    - Updates blur detector threshold (POWER_SAVE=50, BALANCED=100, AGGRESSIVE=150)
    - Updates scene change threshold (POWER_SAVE=0.75, BALANCED=0.85, AGGRESSIVE=0.92)
  - Property: stats -> dict (total_frames, accepted_frames, acceptance_rate)
  - Callback support: on_accepted(callback), on_rejected(callback)
- Update cortex/capture/__init__.py to export all classes
- Create tests/capture/test_engine.py with:
  - Test full pipeline with sharp + changed frame → accepted
  - Test blurry frame → rejected
  - Test unchanged frame → rejected
  - Test battery mode switching
  - Test stats tracking
  - Test callbacks
```

**After:**
```
/run pytest tests/capture/ -v
```
Then: `git add -A && git commit -m "feat(capture): add CaptureEngine pipeline integration"`

---

### 2-6. Webcam Demo

**Prompt to Claude CLI:**

```
Create examples/webcam_demo.py:
- Opens default webcam with OpenCV
- Passes each frame through CaptureEngine
- Display window shows:
  - Left: original feed
  - Right: last accepted frame (or black if none)
  - Overlay text at top:
    - "FPS: XX | Accepted: XX% | Mode: BALANCED"
    - "Total: XX | Accepted: XX | Rejected: XX"
    - Current CaptureResult.reason
- Keyboard controls:
  - '1': AGGRESSIVE mode
  - '2': BALANCED mode
  - '3': POWER_SAVE mode
  - 'r': Reset stats
  - 'q': Quit
- Print stats summary on exit
- No IMU data (skip IMU gate for webcam demo)
```

**After:** Run it manually:
```bash
python examples/webcam_demo.py
```
Then: `git add -A && git commit -m "feat(examples): add webcam demo with real-time stats"`

---

## Phase 1 Complete Checkpoint

At this point you should have:

```
cortex-sdk/
├── CLAUDE.md
├── pyproject.toml
├── cortex/
│   ├── __init__.py
│   └── capture/
│       ├── __init__.py
│       ├── blur_detector.py      ✅
│       ├── scene_change.py       ✅
│       ├── imu_gate.py           ✅
│       └── engine.py             ✅
├── tests/
│   └── capture/
│       ├── test_blur_detector.py ✅
│       ├── test_scene_change.py  ✅
│       ├── test_imu_gate.py      ✅
│       └── test_engine.py        ✅
├── examples/
│   └── webcam_demo.py            ✅
└── README.md
```

Run full test suite:
```
/run pytest tests/ -v --cov=cortex
```

Tag the release:
```bash
git tag v0.1.0-alpha
```

---

## Phase 2: Image Optimizer (Layer 2)

### 3-1. CenterCropStrategy

**Prompt:**
```
Create cortex/optimizer/center_crop.py:
- CenterCropStrategy class
- Method: crop(frame: np.ndarray, ratio: float = 0.7) -> np.ndarray
  - Crops center portion of frame (default 70%)
- Method: score_map(frame: np.ndarray, grid: tuple = (8,6)) -> np.ndarray
  - Returns 8x6 grid with Gaussian center-weighted scores
- Zero computation cost
- Tests in tests/optimizer/test_center_crop.py
```

### 3-2. TextROIStrategy

**Prompt:**
```
Create cortex/optimizer/text_roi.py:
- TextROIStrategy class
- Uses pytesseract or easyocr to detect text regions
- Method: detect_regions(frame: np.ndarray) -> list[tuple[int,int,int,int]]
  - Returns bounding boxes of text regions
- Method: score_map(frame: np.ndarray, grid: tuple = (8,6)) -> np.ndarray
  - Returns grid with high scores where text is detected
- Method: crop(frame: np.ndarray, margin: float = 0.2) -> np.ndarray
  - Crops to bounding box of all text regions + margin
- Fallback: if no text detected, returns original frame
- Tests with synthetic image containing text
```

### 3-3. SaliencyROIStrategy

**Prompt:**
```
Create cortex/optimizer/saliency_roi.py:
- SaliencyROIStrategy class
- Uses OpenCV's saliency module (cv2.saliency) or simple spectral residual method
- Method: score_map(frame: np.ndarray, grid: tuple = (8,6)) -> np.ndarray
- Method: crop(frame: np.ndarray, top_percent: float = 0.4) -> np.ndarray
  - Crops to region containing top 40% of saliency
- Tests with synthetic images (uniform vs high-contrast object)
```

### 3-4. HybridROI

**Prompt:**
```
Create cortex/optimizer/hybrid_roi.py:
- RequestType enum: TEXT_RECOGNITION, OBJECT_SCENE, NAVIGATION, GENERAL
- HybridROI class
- Combines three strategies with weighted score map fusion:
  - S = wc*Sc + wt*St + ws*Ss
- Method: set_request_type(type: RequestType)
  - TEXT: wc=0.2, wt=0.6, ws=0.2
  - OBJECT: wc=0.2, wt=0.2, ws=0.6
  - GENERAL: wc=0.4, wt=0.3, ws=0.3
- Method: crop(frame: np.ndarray) -> np.ndarray
  - Fuses score maps → selects top cells → crops
- EMA temporal smoothing on score map (alpha=0.7)
- Time budget: skip saliency if battery_mode == POWER_SAVE
- Tests for each request type
```

### 3-5. AdaptiveEncoder

**Prompt:**
```
Create cortex/optimizer/encoder.py:
- NetworkCondition enum: WIFI, LTE, WEAK, OFFLINE
- AdaptiveEncoder class
- Method: encode(frame: np.ndarray, condition: NetworkCondition) -> bytes
  - WIFI: resolution 1024px, WebP quality 85
  - LTE: resolution 640px, WebP quality 70
  - WEAK: resolution 320px, JPEG quality 60
  - OFFLINE: queue frame, return None
- Method: estimate_tokens(encoded: bytes) -> int
  - Rough estimate based on byte size
- Method: estimate_cost(tokens: int, model: str) -> float
  - Per-model token pricing
- Property: compression_stats -> dict
- Tests for each network condition
```

### 3-6. RequestClassifier

**Prompt:**
```
Create cortex/optimizer/classifier.py:
- RequestClassifier class
- Method: classify_voice(text: str) -> RequestType
  - Keyword matching first (read/translate → TEXT, what is/describe → OBJECT)
  - Fallback: sentence similarity using simple TF-IDF or keyword vectors
- Method: classify_implicit(has_text: bool, is_moving: bool) -> RequestType
  - has_text + stationary → TEXT
  - no_text + moving → NAVIGATION
  - no_text + stationary → OBJECT
  - else → GENERAL
- Property: confidence -> float
- Tests for various input phrases and implicit signals
```

### 3-7. Updated Demo

**Prompt:**
```
Update examples/webcam_demo.py to include Phase 2:
- After CaptureEngine accepts a frame, pass it through:
  1. RequestClassifier (default GENERAL)
  2. HybridROI crop
  3. AdaptiveEncoder (default WIFI)
- Show on screen:
  - Original frame | ROI-cropped frame | Encoded size
  - "Payload reduction: XX%" 
  - "Estimated tokens: XX | Cost: $X.XXX"
- Keyboard: 't' for TEXT mode, 'o' for OBJECT mode, 'g' for GENERAL
```

---

## Phase 2 Complete Checkpoint

```
/run pytest tests/ -v --cov=cortex
```

Expected coverage: >80%

Tag: `git tag v0.1.0-beta`

---

## Phase 3: Context Memory (Layer 4)

Prompts follow the same pattern. Key classes:
- `cortex/memory/context_store.py` — ContextEvent dataclass + sliding window FIFO
- `cortex/memory/summarizer.py` — VLM-generated [SUMMARY] parser + rule-based fallback
- `cortex/memory/retriever.py` — Sentence embedding + cosine similarity search
- `cortex/memory/injector.py` — Assembles final prompt with compressed context

---

## Phase 4: VLM Bridge (Layer 3)

Key classes:
- `cortex/bridge/base_adapter.py` — VLMAdapter abstract base class
- `cortex/bridge/claude_adapter.py` — Anthropic API integration
- `cortex/bridge/openai_adapter.py` — OpenAI API integration
- `cortex/bridge/ollama_adapter.py` — Local Ollama integration
- `cortex/bridge/router.py` — Priority/Condition/Race routing
- `cortex/bridge/circuit_breaker.py` — CLOSED/OPEN/HALF-OPEN states

---

## Phase 5: Full Pipeline + README + Release

**Prompt:**
```
Create cortex/pipeline.py:
- CortexPipeline class that chains all 4 layers
- Simple API: pipeline = CortexPipeline(config); pipeline.start()
- Create a comprehensive README.md with:
  - 30-second description
  - Installation: pip install cortex-sdk
  - Quick start code example
  - Architecture diagram (text-based)
  - Link to arXiv paper
  - Benchmarks table
  - Contributing guide
  - MIT License
```

---

## Claude CLI Tips & Tricks

### Useful commands inside Claude CLI

| Command | Use |
|---------|-----|
| `/run pytest tests/ -v` | Run all tests |
| `/run python examples/webcam_demo.py` | Run demo |
| `/cost` | Check token usage |
| `/clear` | Clear conversation (keep CLAUDE.md) |
| `/compact` | Summarize conversation to save context |

### Best practices

1. **One class per prompt.** Don't ask for multiple files at once.
2. **Always ask for tests.** Say "also create tests" in every prompt.
3. **Run tests after each step.** `/run pytest` before committing.
4. **Commit frequently.** Small, meaningful commits look great on GitHub.
5. **Review before accepting.** Read the generated code. Ask Claude to explain if unclear.
6. **Use /compact when context gets long.** This saves tokens and keeps Claude focused.

### When things break

```
# Show the error to Claude
I got this error when running tests:
[paste error]
Fix it.
```

```
# Ask for refactoring
Review cortex/capture/engine.py for:
- Missing edge cases
- Type hint issues
- Performance improvements
Suggest and apply fixes.
```

### Commit message convention

```
feat(capture): add BlurDetector with Laplacian variance
feat(capture): add SceneChangeDetector with SSIM
feat(capture): add IMUGate with battery-aware thresholds
feat(capture): add CaptureEngine pipeline integration
feat(examples): add webcam demo with real-time stats
feat(optimizer): add HybridROI with score-map fusion
feat(bridge): add VLM adapter pattern with circuit breaker
feat(memory): add context store with hierarchical summary
feat(pipeline): add CortexPipeline full integration
docs: add README with quick start and architecture
build: add pyproject.toml
test: add comprehensive test suite (>80% coverage)
```

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Phase 1 | Capture engine + webcam demo |
| 2-3 | Phase 2 | Image optimizer + classifier |
| 3-4 | Phase 3 | Context memory module |
| 4-5 | Phase 4 | VLM bridge + full pipeline |
| 5 | Release | README + PyPI + GitHub push |

---

## After completion

1. Push to GitHub: `git remote add origin git@github.com:YOUR_USER/cortex-sdk.git && git push -u origin main`
2. Record demo GIF with webcam_demo.py
3. Upload to PyPI: `python -m build && twine upload dist/*`
4. Update arXiv paper with benchmark results
5. Post on LinkedIn + X

---

*CORTEX — The brain between your glasses and AI.*
