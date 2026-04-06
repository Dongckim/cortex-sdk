"""Live L1 + L2 pipeline demo — fullscreen with efficiency metrics.

Run:
    python examples/demo_graph_live.py

Controls:
    B   BALANCED   mode
    P   POWER_SAVE mode
    A   AGGRESSIVE mode
    Q   quit
"""

import time

import cv2
import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.optimizer.hybrid_roi import RequestType
from cortex.pipeline import CortexPipeline

# ── palette ─────────────────────────────────────────────────────────
C_BG     = (20,  20,  20)
C_PANEL  = (28,  28,  28)
C_PANEL2 = (22,  22,  22)
C_BORDER = (55,  55,  55)
C_WHITE  = (225, 225, 225)
C_DIM    = (100, 100, 100)
C_GREEN  = (80,  210, 120)
C_ORANGE = (50,  150, 240)
C_RED    = (70,   70, 220)
C_YELLOW = (50,  210, 210)
C_ELIM   = (60,   60, 180)
C_TEAL   = (180, 200,  80)

WIN      = "CORTEX  L1 + L2  —  live"
PANEL_W  = 400
BOTTOM_H = 160
EMA      = 0.12
CELL     = 26
FPS_BASE = 30.0   # assumed webcam fps for 5s-interval comparison

MODE_META = {
    BatteryMode.BALANCED:   ("BALANCED",   C_GREEN),
    BatteryMode.POWER_SAVE: ("POWER_SAVE", C_ORANGE),
    BatteryMode.AGGRESSIVE: ("AGGRESSIVE", C_RED),
}
L1_COLOR = {
    "accepted":  C_GREEN,
    "blurry":    C_RED,
    "no_change": C_ORANGE,
    "no_motion": C_DIM,
}
ALL_L2 = ["center_crop", "text_roi_mser", "saliency_dft",
          "motion_map", "score_fusion", "ema_smooth"]
EXT_NODES = {"text_roi_mser"}


# ── drawing helpers ──────────────────────────────────────────────────

def _txt(img, text, pos, scale=0.40, color=C_WHITE, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, bg=C_BORDER):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y),
                  (x + max(2, int(w * min(frac, 1.0))), y + h), color, -1)


def _sep(img, y, x0, x1):
    cv2.line(img, (x0, y), (x1, y), C_BORDER, 1)


def _section(img, y, label, x0=14, x1=None):
    if x1 is None:
        x1 = img.shape[1] - 14
    _sep(img, y, x0, x1)
    _txt(img, label, (x0, y + 12), scale=0.33, color=C_DIM)
    return y + 22


# ── right panel ──────────────────────────────────────────────────────

def draw_right_panel(
    panel, mode, last_reason, l1_stats,
    accepted, total, smoothed, active_nodes,
    elim_saved_ms, ref_balanced_ms, score_map,
):
    panel[:] = C_PANEL
    W, PAD = panel.shape[1], 14
    y = 18

    # mode badge
    label, lc = MODE_META[mode]
    cv2.rectangle(panel, (PAD, y - 12), (W - PAD, y + 12), lc, 1)
    _txt(panel, f"  MODE: {label}", (PAD + 4, y + 4), scale=0.50,
         color=lc, thick=1)
    y += 30
    _txt(panel, "[B] balanced   [P] power_save   [A] aggressive",
         (PAD, y), scale=0.28, color=C_DIM)
    y += 16

    # ── L1 ──────────────────────────────────────────────────────────
    y = _section(panel, y, "L1  capture gate", PAD, W - PAD)

    sc = L1_COLOR.get(last_reason, C_DIM)
    cv2.circle(panel, (PAD + 6, y - 3), 5, sc, -1)
    _txt(panel, last_reason.upper(), (PAD + 18, y + 1), scale=0.42,
         color=sc, thick=1)
    rate = accepted / total * 100 if total > 0 else 0
    _txt(panel, f"{accepted}/{total}  ({rate:.0f}%)",
         (PAD + 140, y + 1), scale=0.33, color=C_DIM)
    y += 20

    blur = l1_stats.get("blur_score", 0.0)
    _txt(panel, f"blur  {blur:5.0f}", (PAD, y), scale=0.33, color=C_WHITE)
    _bar(panel, PAD + 80, y - 7, W - PAD - 80 - PAD, 7,
         min(blur / 300, 1.0), C_GREEN if blur > 30 else C_RED)
    y += 16

    ssim = l1_stats.get("ssim_score") or 0.0
    _txt(panel, f"ssim  {ssim:4.2f}", (PAD, y), scale=0.33, color=C_WHITE)
    _bar(panel, PAD + 80, y - 7, W - PAD - 80 - PAD, 7, ssim, C_ORANGE)
    y += 18

    # ── L2 ──────────────────────────────────────────────────────────
    y = _section(panel, y, "L2  Graph IR  active nodes", PAD, W - PAD)

    BAR_X = 185
    BAR_W = W - BAR_X - PAD - 2
    max_ms = max((smoothed.get(n, 0) for n in ALL_L2), default=0.3)
    max_ms = max(max_ms, 0.3)

    for name in ALL_L2:
        active = name in active_nodes
        ms = smoothed.get(name, 0.0)
        ext = name in EXT_NODES
        dc = (C_ORANGE if ext else C_GREEN) if active else C_ELIM
        nc = C_WHITE if active else C_ELIM

        cv2.circle(panel, (PAD + 6, y - 3), 4, dc, -1)
        tag = "  [ext]" if ext else ""
        _txt(panel, name + tag, (PAD + 16, y), scale=0.35, color=nc)

        if active:
            _txt(panel, f"{ms:5.2f}ms", (BAR_X, y), scale=0.35,
                 color=C_YELLOW)
            _bar(panel, BAR_X + 52, y - 7, BAR_W, 7, ms / max_ms, dc)
        else:
            tx = PAD + 16
            cv2.line(panel, (tx, y - 4),
                     (tx + int(len(name) * 7.5), y - 4), C_ELIM, 1)
            cv2.rectangle(panel, (BAR_X, y - 10),
                          (BAR_X + 100, y + 2), C_ELIM, 1)
            _txt(panel, "x ELIMINATED", (BAR_X + 3, y),
                 scale=0.30, color=C_ELIM)
            if elim_saved_ms > 0:
                _txt(panel, f"saved {elim_saved_ms:.2f}ms",
                     (BAR_X + 104, y), scale=0.28, color=C_GREEN)
        y += 20

    # totals
    active_total = sum(smoothed.get(n, 0) for n in active_nodes)
    _sep(panel, y + 2, PAD, W - PAD)
    y += 14
    _txt(panel, f"L2 total:  {active_total:.2f} ms / frame",
         (PAD, y), scale=0.40, color=C_WHITE, thick=1)
    y += 18
    if mode == BatteryMode.POWER_SAVE and ref_balanced_ms > 0:
        saved = ref_balanced_ms - active_total
        pct = saved / ref_balanced_ms * 100
        _txt(panel, f"vs BALANCED:  -{max(saved,0):.2f}ms  ({pct:.0f}% saved)",
             (PAD, y), scale=0.34, color=C_GREEN)
        y += 16

    # ── score map ───────────────────────────────────────────────────
    y = _section(panel, y + 4, "ROI score map  (6×8)", PAD, W - PAD)

    if score_map is not None:
        rows, cols = score_map.shape
        mw = cols * CELL
        x0 = (W - mw) // 2
        for r in range(rows):
            for c in range(cols):
                v = float(score_map[r, c])
                x1, y1 = x0 + c * CELL, y + r * CELL
                cv2.rectangle(panel, (x1 + 1, y1 + 1),
                              (x1 + CELL - 2, y1 + CELL - 2),
                              (int((1-v)*55), int(v*230), 20), -1)
                if v > 0.25:
                    _txt(panel, f"{v:.1f}", (x1 + 3, y1 + CELL - 5),
                         scale=0.24, color=(15, 15, 15))
        cv2.rectangle(panel, (x0, y),
                      (x0 + mw, y + rows * CELL), C_BORDER, 1)


# ── bottom efficiency bar ────────────────────────────────────────────

def draw_bottom(strip, accepted, total, avg_crop_pct, fps_est, elapsed_s: float = 0.0):
    """Bottom strip: naive 5s timer vs Cortex scene-change detection."""
    strip[:] = C_PANEL2
    H, W = strip.shape[:2]
    PAD = 18
    _sep(strip, 0, 0, W)

    # --- compute metrics ---
    # Naive: sends 1 frame every 5 seconds unconditionally
    naive_calls = max(1, int(elapsed_s / 5.0))
    cortex_calls = accepted          # only sent when scene changed
    saved_calls  = max(0, naive_calls - cortex_calls)
    saved_pct    = saved_calls / naive_calls * 100 if naive_calls > 0 else 0.0
    bar_frac     = cortex_calls / naive_calls if naive_calls > 0 else 0.0

    # combined pixel efficiency relative to naive (which sends full frame)
    avg_crop_pct_safe = max(avg_crop_pct, 0.01)
    combined_pct = (cortex_calls / naive_calls) * avg_crop_pct_safe * 100 \
        if naive_calls > 0 else 0.0
    pixel_reduction = max(0.0, 100.0 - combined_pct)

    # --- column layout (3 equal columns) ---
    col_w = (W - 2 * PAD) // 3

    def col_x(i):
        return PAD + i * col_w

    y_title = 18
    y_val   = 38
    y_bar   = 52
    y_sub   = 68

    # ── Col 0: API calls comparison ──────────────────────────────────
    x = col_x(0)
    cv2.line(strip, (x + col_w - 10, 8),
             (x + col_w - 10, H - 8), C_BORDER, 1)

    _txt(strip, "VLM API calls  (5s window)", (x, y_title),
         scale=0.36, color=C_DIM)

    # show Cortex call count vs naive
    _txt(strip,
         f"Cortex: {cortex_calls}   Naive: {naive_calls}",
         (x, y_val), scale=0.46, color=C_WHITE, thick=1)

    # bar: cortex fill vs naive total width
    _bar(strip, x, y_bar, col_w - 20, 10,
         min(bar_frac, 1.0), C_GREEN if saved_pct >= 0 else C_RED)
    # marker at 100% = where naive sits
    bx = x + col_w - 20
    cv2.line(strip, (bx, y_bar - 2), (bx, y_bar + 12), C_ORANGE, 2)
    _txt(strip, "naive", (bx - 28, y_bar + 10), scale=0.26, color=C_ORANGE)

    _txt(strip,
         f"API calls saved:  {saved_calls}  ({saved_pct:.0f}%)",
         (x, y_sub), scale=0.33,
         color=C_GREEN if saved_pct > 0 else C_ORANGE)
    _txt(strip,
         f"elapsed: {elapsed_s:.0f}s  |  5s windows: {naive_calls}",
         (x, y_sub + 18), scale=0.30, color=C_DIM)

    # ── Col 1: L2 ROI crop ──────────────────────────────────────────
    x = col_x(1)
    cv2.line(strip, (x + col_w - 10, 8),
             (x + col_w - 10, H - 8), C_BORDER, 1)

    _txt(strip, "L2  ROI crop", (x, y_title), scale=0.36, color=C_DIM)
    _txt(strip, f"{avg_crop_pct*100:.1f}%  of frame pixels",
         (x, y_val), scale=0.46, color=C_WHITE, thick=1)
    _bar(strip, x, y_bar, col_w - 20, 10, avg_crop_pct, C_TEAL)
    _txt(strip, "avg crop area / full frame",
         (x, y_sub), scale=0.31, color=C_DIM)
    token_saved = (1 - avg_crop_pct) * 100
    _txt(strip, f"token cost reduction:  ~{token_saved:.0f}%",
         (x, y_sub + 18), scale=0.33, color=C_TEAL)

    # ── Col 2: combined pixel data sent to VLM ───────────────────────
    x = col_x(2)
    _txt(strip, "L1 × L2  vs naive 5s", (x, y_title),
         scale=0.36, color=C_DIM)
    _txt(strip, f"{combined_pct:.1f}%  data vs naive",
         (x, y_val), scale=0.46, color=C_YELLOW, thick=1)
    _bar(strip, x, y_bar, col_w - 20, 10,
         min(combined_pct / 100, 1.0), C_YELLOW)
    _txt(strip, "naive sends full frame every 5s = 100%",
         (x, y_sub), scale=0.30, color=C_DIM)
    _txt(strip,
         f"Cortex sends {pixel_reduction:.0f}% less data to VLM",
         (x, y_sub + 18), scale=0.33, color=C_YELLOW)


# ── crop helper ──────────────────────────────────────────────────────

def crop_from_score_map(frame, score_map, margin: float = 0.05):
    """Crop to high-score region with a small safety margin.

    Keeps top 40% of score range (cells in the highest tier),
    then adds a small margin so border text isn't clipped.
    """
    rows, cols = score_map.shape
    h, w = frame.shape[:2]
    cell_h, cell_w = h / rows, w / cols
    s_min, s_max = score_map.min(), score_map.max()
    if s_max - s_min < 1e-6:
        return None
    coords = np.argwhere(score_map >= s_min + 0.6 * (s_max - s_min))
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    y1 = int(r0 * cell_h)
    y2 = int((r1 + 1) * cell_h)
    x1 = int(c0 * cell_w)
    x2 = int((c1 + 1) * cell_w)
    pad_y = int((y2 - y1) * margin)
    pad_x = int((x2 - x1) * margin)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ── main ─────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam — using synthetic frames.")
        cap = None

    mode     = BatteryMode.BALANCED
    pipeline = CortexPipeline(battery_mode=mode, request_type=RequestType.GENERAL)
    pipeline._capture._blur.threshold = 30.0
    # threshold=0.85: permissive — page turns (SSIM ~0.5–0.7) always trigger.
    # cooldown_s=5.0: at most one send per 5-second window, matching the
    # naive baseline so the bottom strip comparison is apples-to-apples.
    pipeline._capture._scene.threshold = 0.85
    pipeline._capture._scene.cooldown_s = 5.0

    smoothed        = {n: 0.0 for n in ALL_L2}
    ref_balanced_ms = 0.0
    last_score_map  = None
    last_crop       = None
    last_reason     = "accepted"
    total_frames    = 0
    accepted_frames = 0
    crop_area_ema   = 0.5   # EMA of crop_area / full_area
    fps_ema         = FPS_BASE
    t_prev          = time.perf_counter()
    t_start         = time.perf_counter()   # for naive 5s window count

    synth = np.zeros((480, 640, 3), dtype=np.uint8)

    # fullscreen window
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # ── grab frame ───────────────────────────────────────────────
        if cap is not None:
            ret, raw = cap.read()
            frame = raw if ret else synth.copy()
        else:
            t = time.time()
            cx = int(320 + 200 * np.sin(t * 0.7))
            cy = int(240 + 100 * np.sin(t * 0.4))
            synth[:] = 25
            cv2.circle(synth, (cx, cy), 70, (170, 190, 210), -1)
            cv2.circle(synth, (cx + 110, cy - 70), 35, (100, 220, 160), -1)
            frame = synth.copy()

        total_frames += 1

        # fps estimate
        now = time.perf_counter()
        fps_ema = 0.05 * (1.0 / max(now - t_prev, 1e-4)) + 0.95 * fps_ema
        t_prev = now

        # ── pipeline ─────────────────────────────────────────────────
        result = pipeline.process(frame, imu_data=None)
        last_reason = result.reason

        if result.accepted:
            accepted_frames += 1
            if result.score_map is not None:
                crop = crop_from_score_map(frame, result.score_map)
                if crop is not None:
                    last_crop = crop.copy()
                    fh, fw = frame.shape[:2]
                    ch, cw = crop.shape[:2]
                    crop_frac = (ch * cw) / (fh * fw)
                    crop_area_ema = 0.1 * crop_frac + 0.9 * crop_area_ema

        for name, ms in result.l2_timings.items():
            smoothed[name] = EMA * ms + (1 - EMA) * smoothed.get(name, ms)
        if result.score_map is not None:
            last_score_map = result.score_map

        active_nodes = result.active_nodes or set(pipeline.active_node_names)
        l2_total = sum(smoothed.get(n, 0) for n in active_nodes)
        if mode == BatteryMode.BALANCED:
            ref_balanced_ms = EMA * l2_total + (1 - EMA) * ref_balanced_ms

        # ── layout ───────────────────────────────────────────────────
        # Camera shown at native resolution — no scaling avoids letterbox.
        # canvas = [camera | panel] on top + [bottom strip] below.
        fh, fw = frame.shape[:2]
        cam_h  = fh                       # camera at full height
        cam_w  = fw
        canvas_w = cam_w + PANEL_W
        canvas_h = cam_h + BOTTOM_H

        canvas = np.full((canvas_h, canvas_w, 3), C_BG, dtype=np.uint8)

        # ── camera feed ──────────────────────────────────────────────
        cam_img = frame.copy()

        # ROI overlay
        if last_score_map is not None:
            rows, cols = last_score_map.shape
            gh, gw = cam_h // rows, cam_w // cols
            for r in range(rows):
                for c in range(cols):
                    v = float(last_score_map[r, c])
                    if v > 0.5:
                        x1, y1 = c * gw, r * gh
                        ov = cam_img.copy()
                        cv2.rectangle(ov, (x1, y1),
                                      (x1+gw, y1+gh),
                                      (0, int(v*220), 0), -1)
                        cv2.addWeighted(ov, 0.20 + v*0.18,
                                        cam_img, 1-(0.20+v*0.18),
                                        0, cam_img)

        # L1 status bar
        sc = L1_COLOR.get(last_reason, C_DIM)
        cv2.rectangle(cam_img, (0, 0), (cam_w, 26), (0, 0, 0), -1)
        _txt(cam_img, f"L1: {last_reason.upper()}", (10, 18),
             scale=0.48, color=sc, thick=1)
        cs = pipeline.capture_stats
        _txt(cam_img,
             f"accept {cs['accepted_frames']}/{cs['total_frames']}  "
             f"({cs['acceptance_rate']*100:.0f}%)",
             (cam_w - 220, 18), scale=0.36, color=C_DIM)

        # dim + label when L1 rejects
        if last_reason != "accepted":
            ov = cam_img.copy()
            cv2.rectangle(ov, (0, 26), (cam_w, cam_h), (0,0,0), -1)
            cv2.addWeighted(ov, 0.40, cam_img, 0.60, 0, cam_img)
            _txt(cam_img, last_reason.upper(),
                 (cam_w//2 - 60, cam_h//2),
                 scale=0.9, color=sc, thick=2)

        # L3 crop preview — bottom-left of camera
        TW = cam_w // 4
        TH = cam_h // 4
        TX, TY = 8, cam_h - TH - 8
        if last_crop is not None:
            thumb = cv2.resize(last_crop, (TW, TH))
            cv2.rectangle(cam_img, (TX-2, TY-20),
                          (TX+TW+2, TY+TH+2), (0,0,0), -1)
            cam_img[TY:TY+TH, TX:TX+TW] = thumb
            cv2.rectangle(cam_img, (TX, TY),
                          (TX+TW, TY+TH), C_GREEN, 1)
            _txt(cam_img, "-> L3  (last accepted crop)",
                 (TX, TY - 6), scale=0.30, color=C_GREEN)
        else:
            cv2.rectangle(cam_img, (TX, TY),
                          (TX+TW, TY+TH), C_BORDER, 1)
            _txt(cam_img, "-> L3  (waiting...)",
                 (TX, TY - 6), scale=0.30, color=C_DIM)

        canvas[:cam_h, :cam_w] = cam_img

        # ── right panel ──────────────────────────────────────────────
        panel = np.full((cam_h, PANEL_W, 3), C_PANEL, dtype=np.uint8)
        draw_right_panel(
            panel, mode, last_reason,
            result.l1_stats,
            accepted_frames, total_frames,
            dict(smoothed), active_nodes,
            pipeline._dne.time_saved_ms if pipeline._dne else 0.0,
            ref_balanced_ms, last_score_map,
        )
        canvas[:cam_h, cam_w:cam_w + PANEL_W] = panel

        # vertical divider
        cv2.line(canvas, (cam_w, 0), (cam_w, cam_h), C_BORDER, 1)

        # ── bottom efficiency strip ───────────────────────────────────
        elapsed_s = time.perf_counter() - t_start
        strip = np.full((BOTTOM_H, canvas_w, 3), C_PANEL2, dtype=np.uint8)
        draw_bottom(strip, accepted_frames, total_frames,
                    crop_area_ema, fps_ema, elapsed_s)
        canvas[cam_h:cam_h + BOTTOM_H, :] = strip

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"): break
        elif key == ord("b"):
            mode = BatteryMode.BALANCED
            pipeline.set_battery_mode(mode)
        elif key == ord("p"):
            mode = BatteryMode.POWER_SAVE
            pipeline.set_battery_mode(mode)
        elif key == ord("a"):
            mode = BatteryMode.AGGRESSIVE
            pipeline.set_battery_mode(mode)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
