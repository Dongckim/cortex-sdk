"""CORTEX — L1 Capture + L2 Optimizer + L4 Memory webcam demo.

Pipeline: blur filter → scene change → ROI crop → encode → memory store.

Controls:
    1 / 2 / 3  — AGGRESSIVE / BALANCED / POWER_SAVE
    t / o / g  — TEXT / OBJECT / GENERAL ROI mode
    h          — toggle heatmap overlay
    m          — query memory (type in terminal)
    r          — reset all
    q          — quit
"""

import time

import cv2
import numpy as np

from cortex.capture import BatteryMode, CaptureEngine
from cortex.memory.context_store import ContextEvent, ContextStore
from cortex.memory.frame_describer import describe_frame
from cortex.memory.injector import ContextInjector
from cortex.optimizer.encoder import AdaptiveEncoder, NetworkCondition
from cortex.optimizer.hybrid_roi import HybridROI, RequestType

engine = CaptureEngine()
roi = HybridROI()
encoder = AdaptiveEncoder()
memory = ContextStore(max_events=20, ttl_seconds=300)
injector = ContextInjector(memory)
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
query_result = ""
query_time = 0.0

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

        # L4: generate fake description and store event
        desc, etype = describe_frame(
            frame,
            blur_score=result.stats.get("blur_score", 0),
        )
        memory.add(ContextEvent(
            timestamp=time.time(),
            event_type=etype,
            description=desc,
        ))

    # --- HUD ---
    display = frame.copy()
    h, w = display.shape[:2]

    # Heatmap overlay
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

    # L1 stats
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

    # L2 stats
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

    # Controls
    cv2.putText(
        display,
        "1/2/3:battery  t/o/g:roi  h:heatmap  m:query  r:reset  q:quit",
        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1,
    )

    # --- L4 Memory: event log (left side) ---
    recent_events = memory.events[-5:]  # last 5
    if recent_events:
        mem_y = 100
        overlay3 = display.copy()
        cv2.rectangle(overlay3, (0, mem_y - 5), (320, mem_y + len(recent_events) * 20 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.5, display, 0.5, 0, display)
        cv2.putText(
            display, f"MEMORY ({memory.size} events)",
            (10, mem_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1,
        )
        for i, evt in enumerate(reversed(recent_events)):
            txt = f"{evt.age_label}: {evt.description[:35]}"
            alpha = max(0.3, 1.0 - i * 0.15)
            col = (int(160 * alpha), int(160 * alpha), int(160 * alpha))
            cv2.putText(
                display, txt,
                (10, mem_y + 28 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1,
            )

    # Query result display (fades after 5 seconds)
    if query_result and (time.time() - query_time) < 5:
        qy = h - 50
        overlay4 = display.copy()
        cv2.rectangle(overlay4, (0, qy - 10), (w, h), (0, 80, 0), -1)
        cv2.addWeighted(overlay4, 0.6, display, 0.4, 0, display)
        cv2.putText(
            display, query_result[:80],
            (10, qy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1,
        )
        if len(query_result) > 80:
            cv2.putText(
                display, query_result[80:160],
                (10, qy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1,
            )

    # ROI crop preview
    if last_cropped is not None:
        tw, th = w // 3, h // 3
        small = cv2.resize(last_cropped, (tw, th))
        y1 = h - th - 10
        x1 = w - tw - 10
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

    cv2.imshow("CORTEX L1+L2+L4 Demo", display)

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
    elif key == ord("m"):
        # Query memory from terminal
        print("\n--- Memory Query ---")
        q = input("Ask about past events: ")
        if q.strip():
            context = injector.build_context(query=q)
            query_result = f"Q: {q} | {context[:120]}"
            query_time = time.time()
            print(f"\n{context}\n")
    elif key == ord("r"):
        engine = CaptureEngine()
        roi = HybridROI()
        encoder = AdaptiveEncoder()
        memory.clear()
        last_accepted = None
        last_cropped = None
        last_score_map = None
        query_result = ""

cap.release()
cv2.destroyAllWindows()

# Final stats
comp = encoder.compression_stats
mem_stats = injector.context_stats
print(f"\n=== CORTEX Stats ===")
print(f"Frames: {stats['accepted_frames']}/{stats['total_frames']} ({rate:.1f}% accepted)")
print(f"Encoded: {comp['encode_count']} frames, {comp['compression_ratio']*100:.1f}% compression")
print(f"Memory: {mem_stats['total_events']} events, ~{mem_stats['estimated_tokens']} context tokens")
