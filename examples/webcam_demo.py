"""CORTEX — L1 Capture + L2 Optimizer webcam demo.

Pipeline: blur filter → scene change → ROI crop → adaptive encode.

Controls:
    1 / 2 / 3  — AGGRESSIVE / BALANCED / POWER_SAVE
    t / o / g  — TEXT / OBJECT / GENERAL ROI mode
    h          — toggle heatmap overlay
    r          — reset stats
    q          — quit
"""

import cv2
import numpy as np

from cortex.capture import BatteryMode, CaptureEngine
from cortex.optimizer.encoder import AdaptiveEncoder, NetworkCondition
from cortex.optimizer.hybrid_roi import HybridROI, RequestType

engine = CaptureEngine()
roi = HybridROI()
encoder = AdaptiveEncoder()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

last_accepted = None
last_cropped = None
last_encoded_size = 0
last_tokens = 0
last_cost = 0.0
last_score_map = None
reason = ""
show_heatmap = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = engine.process_frame(frame)
    reason = result.reason

    if result.accepted:
        last_accepted = frame.copy()
        last_score_map = roi.fused_score_map(frame)
        last_cropped = roi.crop(frame)
        encoded = encoder.encode(last_cropped, NetworkCondition.WIFI)
        if encoded:
            last_encoded_size = len(encoded)
            last_tokens = encoder.estimate_tokens(encoded)
            last_cost = encoder.estimate_cost(last_tokens)

    # --- HUD ---
    display = frame.copy()
    h, w = display.shape[:2]

    # 1. Heatmap overlay (alpha=0.15 for subtlety)
    if show_heatmap and last_score_map is not None:
        heatmap = cv2.resize(last_score_map, (w, h), interpolation=cv2.INTER_NEAREST)
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.addWeighted(heatmap_color, 0.15, display, 0.85, 0, display)

    # Top bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # Status
    colors = {
        "accepted": (0, 255, 0),
        "blurry": (0, 0, 255),
        "no_change": (0, 165, 255),
    }
    color = colors.get(reason, (255, 255, 255))
    cv2.circle(display, (20, 16), 6, color, -1)
    cv2.putText(
        display, reason.upper().replace("_", " "),
        (36, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    # 4. L1 stats — blur score + SSIM score (realtime)
    stats = engine.stats
    rate = stats["acceptance_rate"] * 100
    blur_val = result.stats.get("blur_score", 0)
    ssim_val = result.stats.get("ssim_score")
    ssim_str = f"{ssim_val:.3f}" if ssim_val is not None else "—"
    l1_info = (
        f"blur {blur_val:.0f} (thr:{engine._blur.threshold:.0f})  |  "
        f"ssim {ssim_str} (thr:{engine._scene.threshold:.2f})  |  "
        f"accept {rate:.1f}%"
    )
    cv2.putText(
        display, l1_info,
        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1,
    )

    # 2. L2 stats — precise compression ratio
    if last_accepted is not None:
        orig_kb = frame.nbytes / 1024
        enc_kb = last_encoded_size / 1024
        reduction = (
            (1 - last_encoded_size / frame.nbytes) * 100
            if frame.nbytes > 0 else 0
        )
        l2_info = (
            f"roi {roi.request_type.value}  |  "
            f"{orig_kb:.0f}KB -> {enc_kb:.1f}KB ({reduction:.1f}% reduced)  |  "
            f"~{last_tokens} tok  ${last_cost:.4f}"
        )
        cv2.putText(
            display, l2_info,
            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 160), 1,
        )

    # Controls hint
    cv2.putText(
        display,
        "1/2/3:battery  t/o/g:roi  h:heatmap  r:reset  q:quit",
        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1,
    )

    # 3. ROI crop preview — 1/3 size, right side
    if last_cropped is not None:
        tw, th = w // 3, h // 3
        small = cv2.resize(last_cropped, (tw, th))
        y1 = h - th - 10
        x1 = w - tw - 10
        # Background
        overlay2 = display.copy()
        cv2.rectangle(
            overlay2, (x1 - 1, y1 - 18), (x1 + tw + 1, y1 + th + 1),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay2, 0.5, display, 0.5, 0, display)
        display[y1:y1 + th, x1:x1 + tw] = small
        cv2.putText(
            display, "roi crop",
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
        )

    cv2.imshow("CORTEX L1+L2 Demo", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        engine.set_battery_mode(BatteryMode.AGGRESSIVE)
    elif key == ord("2"):
        engine.set_battery_mode(BatteryMode.BALANCED)
    elif key == ord("3"):
        engine.set_battery_mode(BatteryMode.POWER_SAVE)
    elif key == ord("t"):
        roi.set_request_type(RequestType.TEXT_RECOGNITION)
    elif key == ord("o"):
        roi.set_request_type(RequestType.OBJECT_SCENE)
    elif key == ord("g"):
        roi.set_request_type(RequestType.GENERAL)
    elif key == ord("h"):
        show_heatmap = not show_heatmap
    elif key == ord("r"):
        engine = CaptureEngine()
        roi = HybridROI()
        encoder = AdaptiveEncoder()
        last_accepted = None
        last_cropped = None
        last_score_map = None

cap.release()
cv2.destroyAllWindows()

# Final stats
comp = encoder.compression_stats
print(f"\n=== CORTEX Stats ===")
print(f"Frames: {stats['accepted_frames']}/{stats['total_frames']} ({rate:.1f}% accepted)")
print(f"Encoded: {comp['encode_count']} frames, {comp['compression_ratio']*100:.1f}% compression")
