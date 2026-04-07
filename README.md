# CORTEX

**Camera Optimized Realtime Transmission Exchange**

Battery-aware VLM middleware SDK for wearable devices (e.g., smart glasses). CORTEX sits between a camera source and Vision-Language Model (VLM) APIs, optimizing capture and image payloads for **battery**, **bandwidth/cost**, and **end-to-end latency**.

> Paper (Apr 2026): [`Dongchan_Kim_2026_cortex_middleware_graph_compiler.pdf`](./Dongchan_Kim_2026_cortex_middleware_graph_compiler.pdf)

---

## Architecture

```
Smart Glasses
     |
     v
┌──────────────────────────────────────────┐
│  L1: CAPTURE                             │
│  IMU Gate → Blur Filter → SSIM Change    │
│  "Should we capture this frame?"         │
└──────────────┬───────────────────────────┘
               │ accepted frames only
               v
┌──────────────────────────────────────────┐
│  L2: COMPRESS                            │
│  Classify → Hybrid ROI Crop → Encode     │
│  "What part matters? How small can it be?"│
└──────────────┬───────────────────────────┘
               │ optimized payload
               v
┌──────────────────────────────────────────┐
│  L3: BRIDGE                              │
│  Router → Circuit Breaker → VLM Adapters │
│  Claude / GPT / Gemini / Ollama          │
└──────────────┬───────────────────────────┘
               │ VLM response
               v
┌──────────────────────────────────────────┐
│  L4: MEMORY                              │
│  Sliding Window → Summary → Retrieval    │
│  "What did we see 2 minutes ago?"        │
└──────────────────────────────────────────┘
```

### Design goals

- **Battery-aware by construction**: maximize camera-off time and avoid wasteful frames.
- **Payload-aware for VLM economics**: reduce resolution/bytes before the API boundary.
- **Compiler-inspired optimization**: expose L2 as a Graph IR to enable analysis, partitioning, and targeted compilation.

## Installation

```bash
git clone https://github.com/Dongckim/cortex-sdk.git
cd cortex-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
from cortex.capture import CaptureEngine, BatteryMode

engine = CaptureEngine()
engine.set_battery_mode(BatteryMode.BALANCED)

# Process a frame (numpy array from camera)
result = engine.process_frame(frame)

if result.accepted:
    print(f"Frame accepted: {result.stats}")
else:
    print(f"Rejected: {result.reason}")
```

### With Image Optimization (L2)

```python
from cortex.optimizer.hybrid_roi import HybridROI, RequestType
from cortex.optimizer.encoder import AdaptiveEncoder, NetworkCondition

roi = HybridROI(request_type=RequestType.GENERAL)
encoder = AdaptiveEncoder()

# Crop to most relevant region
cropped = roi.crop(frame)

# Encode with network-aware compression
payload = encoder.encode(cropped, NetworkCondition.WIFI)
tokens = encoder.estimate_tokens(payload)
cost = encoder.estimate_cost(tokens)
```

## Webcam Demo

```bash
python examples/webcam_demo.py
```

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | AGGRESSIVE / BALANCED / POWER_SAVE |
| `t` / `o` / `g` | TEXT / OBJECT / GENERAL ROI mode |
| `h` | Toggle heatmap overlay |
| `r` | Reset stats |
| `q` | Quit |

## Layer Details

### L1: Adaptive Capture

Three-stage filter pipeline that maximizes camera-off time:

1. **IMU Gate** — Accelerometer/gyroscope motion scoring with battery-aware thresholds
2. **Blur Detector** — Laplacian variance rejects motion-blurred frames (~2ms)
3. **Scene Change** — SSIM comparison against last accepted frame

Battery modes adjust all thresholds simultaneously:

| Mode | Blur Threshold | SSIM Threshold | Captures |
|------|---------------|----------------|----------|
| AGGRESSIVE | 150 | 0.90 | More |
| BALANCED | 100 | 0.80 | Default |
| POWER_SAVE | 50 | 0.65 | Less |

### L2: Image Optimization

**Hybrid ROI Cropping** uses a score-map fusion over an \(8\times6\) grid. In the paper formulation:

```
S = wc*Sc + wt*St + ws*Ss
```

| Map | Source | What it captures |
|-----|--------|-----------------|
| Sc | Gaussian center weight | Gaze-center bias |
| St | Text-aware map | Text regions (external boundary) |
| Ss | DFT-based saliency map | Visually prominent regions |

Battery-aware operating modes can disable expensive components in power-constrained settings (AGGRESSIVE / BALANCED / POWER_SAVE).

**Adaptive Encoding** adjusts per network condition:

| Network | Resolution | Format | Quality |
|---------|-----------|--------|---------|
| WiFi | 1024px | WebP | 85 |
| LTE | 640px | WebP | 70 |
| Weak | 320px | JPEG | 60 |
| Offline | — | queued | — |

### L3: VLM Bridge (planned)

Unified adapter pattern for Claude, GPT-4o, Gemini, and Ollama with circuit-breaker failover.

### L4: Context Memory (planned)

Sliding window event store with hierarchical summarization and embedding-based retrieval (~130 tokens overhead).

## Graph IR + Compiler Pipeline (Phase 3)

CORTEX augments the L2 pipeline with a **Graph Intermediate Representation (Graph IR)** and a **profiling-driven compiler pipeline**, inspired by ML compilation stacks (TVM / MLIR / XLA):

- **Graph IR**: programmatic inspection of the L2 pipeline as nodes/edges, enabling compiler-style passes.
- **Partitioning**: split the graph into **compilable** nodes vs **external_call** boundaries (BYOC-style).
- **Dead-node elimination**: remove nodes that are provably unused under a given battery/request configuration.
- **Targeted kernel compilation**: apply Numba JIT to the largest compilable hotspot (paper target: `saliency_dft`).

### Benchmarks

Repro script:

```bash
python examples/compiler_benchmark.py
```

What it prints and records:

- Graph IR partitioning (compilable vs external boundary)
- Per-node profiling at **640×480**, **100 frames**
- Kernel micro-benchmark (baseline / vectorized / JIT) + numerical equivalence

Latest run output is written to `examples/profiling_report.txt`.

## Project Structure

```
cortex/
  capture/          # L1 — adaptive capture engine
    blur_detector.py
    scene_change.py
    imu_gate.py
    engine.py
  optimizer/        # L2 — image optimization
    center_crop.py
    text_roi.py
    saliency_roi.py
    hybrid_roi.py
    encoder.py
    classifier.py
  bridge/           # L3 — VLM abstraction (planned)
  memory/           # L4 — context memory (planned)
tests/              # mirrors source structure
examples/
  webcam_demo.py    # full L1+L2 demo with HUD
```

## Testing

```bash
pytest tests/ -v --cov=cortex
```

Current: 80+ tests, 99% coverage on L1+L2.

## Target Performance

| Metric | v0.1 Target | Status |
|--------|-------------|--------|
| Image payload reduction | 60%+ | L2 implemented |
| Battery life extension | 2x | L1 implemented |
| Preprocessing latency | <50ms | L1+L2 <30ms |
| Supported VLM backends | 4 | Planned |

## Tech Stack

- Python 3.11+
- OpenCV (capture, image processing)
- scikit-image (SSIM)
- NumPy, Pillow
- httpx (async VLM API calls)

## License

MIT

## Author

Dongchan Kim — [dck.alx@gmail.com](mailto:dck.alx@gmail.com)
