"""Live Graph IR demo — dead_node_elimination in real time.

Run:
    python examples/demo_graph_live.py

Controls:
    B   BALANCED   mode  (6 nodes, saliency_dft active)
    P   POWER_SAVE mode  (5 nodes, saliency_dft eliminated)
    A   AGGRESSIVE mode  (6 nodes, saliency_dft active)
    Q   quit

What to watch:
    - Press P: saliency_dft row gets a red ✕, latency bar disappears
    - Press B: saliency_dft reappears with its latency bar
    - Bottom panel: ROI score map updates every frame
"""

import time

import cv2
import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.graph import build_l2_graph, dead_node_elimination
from cortex.optimizer.hybrid_roi import RequestType

# ── palette ─────────────────────────────────────────────────────────
C_BG      = (20,  20,  20)
C_PANEL   = (28,  28,  28)
C_BORDER  = (55,  55,  55)
C_WHITE   = (225, 225, 225)
C_DIM     = (100, 100, 100)
C_GREEN   = (80,  210, 120)
C_ORANGE  = (50,  150, 240)   # BGR
C_RED     = (70,   70, 220)
C_YELLOW  = (50,  210, 210)
C_ELIM    = (60,  60,  200)   # eliminated node colour (muted red)

PANEL_W   = 370
EMA       = 0.12

MODE_META = {
    BatteryMode.BALANCED:   ("BALANCED",   C_GREEN),
    BatteryMode.POWER_SAVE: ("POWER_SAVE", C_ORANGE),
    BatteryMode.AGGRESSIVE: ("AGGRESSIVE", C_RED),
}

ALL_NODES = [
    "center_crop",
    "text_roi_mser",
    "saliency_dft",
    "motion_map",
    "score_fusion",
    "ema_smooth",
]

NODE_TAGS = {"text_roi_mser": "ext"}

# heatmap cell size (px) — larger = more visible
CELL = 32


def _txt(img, text, pos, scale=0.42, color=C_WHITE, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, bg=C_BORDER):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    filled = max(2, int(w * min(frac, 1.0)))
    cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)


def draw_panel(
    panel: np.ndarray,
    mode: BatteryMode,
    smoothed: dict[str, float],
    active_names: set[str],
    ref_balanced_ms: float,
    score_map: np.ndarray | None,
    elim_saved_ms: float,
) -> None:
    panel[:] = C_PANEL
    H, W = panel.shape[:2]
    PAD = 14
    y = 20

    # ── mode badge ──────────────────────────────────────────────────
    label, lcolor = MODE_META[mode]
    cv2.rectangle(panel, (PAD, y - 13), (W - PAD, y + 13), lcolor, 1)
    _txt(panel, f"  MODE: {label}", (PAD + 6, y + 4),
         scale=0.52, color=lcolor, thick=1)
    y += 34

    _txt(panel, "[B] balanced    [P] power_save    [A] aggressive",
         (PAD, y), scale=0.30, color=C_DIM)
    y += 18

    cv2.line(panel, (PAD, y), (W - PAD, y), C_BORDER, 1)
    y += 12

    # ── node list ───────────────────────────────────────────────────
    _txt(panel, "Graph IR  —  active nodes", (PAD, y),
         scale=0.36, color=C_DIM)
    y += 18

    BAR_X = 192
    BAR_W = W - BAR_X - PAD - 4
    BAR_H = 9
    ROW_H = 24

    max_ms = max((smoothed.get(n, 0) for n in ALL_NODES), default=1.0)
    max_ms = max(max_ms, 0.3)

    for name in ALL_NODES:
        active = name in active_names
        ms     = smoothed.get(name, 0.0)
        tag    = NODE_TAGS.get(name, "")

        cx, cy = PAD + 7, y - 4

        if active:
            dot_col  = C_ORANGE if tag else C_GREEN
            name_col = C_WHITE
            ms_col   = C_YELLOW
        else:
            dot_col  = C_ELIM
            name_col = C_ELIM
            ms_col   = C_ELIM

        cv2.circle(panel, (cx, cy), 5, dot_col, -1)

        # node name
        tag_str = f"  [{tag}]" if tag else ""
        _txt(panel, name + tag_str, (PAD + 18, y), scale=0.38, color=name_col)

        if active:
            _txt(panel, f"{ms:5.2f} ms", (BAR_X, y), scale=0.38, color=ms_col)
            _bar(panel, BAR_X + 58, y - BAR_H + 1, BAR_W, BAR_H,
                 ms / max_ms, dot_col)
        else:
            # strikethrough on node name
            tx, ty = PAD + 18, y - 5
            sw = int(len(name) * 8.5)
            cv2.line(panel, (tx, ty), (tx + sw, ty), C_ELIM, 1)
            # ✕ ELIMINATED badge
            cv2.rectangle(panel, (BAR_X, y - 12), (BAR_X + 118, y + 4),
                          C_ELIM, 1)
            _txt(panel, "x ELIMINATED", (BAR_X + 4, y),
                 scale=0.35, color=C_ELIM)
            # saved ms
            if elim_saved_ms > 0:
                _txt(panel, f"saved {elim_saved_ms:.2f}ms",
                     (BAR_X + 124, y), scale=0.32, color=C_GREEN)

        y += ROW_H

    cv2.line(panel, (PAD, y), (W - PAD, y), C_BORDER, 1)
    y += 12

    # ── totals + savings ────────────────────────────────────────────
    active_total = sum(smoothed.get(n, 0) for n in active_names)
    _txt(panel, f"Pipeline total:  {active_total:.2f} ms / frame",
         (PAD, y), scale=0.42, color=C_WHITE, thick=1)
    y += 22

    if mode == BatteryMode.POWER_SAVE and ref_balanced_ms > 0:
        saved_total = ref_balanced_ms - active_total
        pct = saved_total / ref_balanced_ms * 100
        _txt(panel,
             f"vs BALANCED:  -{max(saved_total,0):.2f} ms  ({pct:.0f}% saved)",
             (PAD, y), scale=0.38, color=C_GREEN)
        y += 20

    cv2.line(panel, (PAD, y), (W - PAD, y), C_BORDER, 1)
    y += 12

    # ── ROI score map ───────────────────────────────────────────────
    _txt(panel, "ROI score map  (6 rows × 8 cols)",
         (PAD, y), scale=0.38, color=C_WHITE)
    y += 16

    if score_map is not None:
        rows, cols = score_map.shape   # (6, 8)
        map_w = cols * CELL
        map_h = rows * CELL
        x0 = (W - map_w) // 2         # centre horizontally

        for r in range(rows):
            for c in range(cols):
                val   = float(score_map[r, c])
                green = int(val * 230)
                blue  = int((1 - val) * 60)
                red   = int((1 - val) * 40)
                color = (blue, green, red)
                x1, y1 = x0 + c * CELL, y + r * CELL
                cv2.rectangle(panel,
                              (x1 + 1, y1 + 1),
                              (x1 + CELL - 2, y1 + CELL - 2),
                              color, -1)
                # value label inside cell
                if val > 0.15:
                    _txt(panel, f"{val:.1f}",
                         (x1 + 4, y1 + CELL - 6),
                         scale=0.28, color=(20, 20, 20))

        # border around map
        cv2.rectangle(panel,
                      (x0, y),
                      (x0 + map_w, y + map_h),
                      C_BORDER, 1)

        # colour legend (low → high)
        lx, ly = x0, y + map_h + 8
        for i in range(map_w):
            frac = i / map_w
            g = int(frac * 230)
            b = int((1 - frac) * 60)
            r = int((1 - frac) * 40)
            cv2.line(panel, (lx + i, ly), (lx + i, ly + 6), (b, g, r), 1)
        _txt(panel, "low", (lx, ly + 18), scale=0.30, color=C_DIM)
        _txt(panel, "high", (lx + map_w - 26, ly + 18), scale=0.30, color=C_DIM)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam — using animated synthetic frames.")
        cap = None

    mode = BatteryMode.BALANCED

    graphs: dict = {}

    def _rebuild():
        nonlocal graphs
        g_full = build_l2_graph(RequestType.GENERAL)
        g_src  = build_l2_graph(RequestType.GENERAL)
        dne    = dead_node_elimination(g_src, mode)
        graphs["full"]      = g_full
        graphs["opt"]       = dne.graph
        graphs["elim_ms"]   = dne.time_saved_ms

    _rebuild()

    smoothed: dict[str, float] = {n: 0.0 for n in ALL_NODES}
    ref_balanced_ms = 0.0
    frame_count = 0

    # animated synthetic for no-webcam fallback
    synth = np.zeros((480, 640, 3), dtype=np.uint8)

    while True:
        if cap is not None:
            ret, raw = cap.read()
            frame = raw if ret else synth.copy()
        else:
            t  = time.time()
            cx = int(320 + 200 * np.sin(t * 0.7))
            cy = int(240 + 120 * np.sin(t * 0.4))
            synth[:] = 25
            cv2.circle(synth, (cx, cy), 70, (170, 190, 210), -1)
            cv2.circle(synth, (cx + 100, cy - 60), 30, (100, 220, 180), -1)
            frame = synth.copy()

        frame_count += 1

        active_graph = graphs["opt"] if mode == BatteryMode.POWER_SAVE else graphs["full"]
        active_names = set(active_graph.node_names())

        ctx, timings = active_graph.profile_execute(frame, battery_mode=mode)

        for name, ms in timings.items():
            smoothed[name] = EMA * ms + (1 - EMA) * smoothed.get(name, ms)

        total_ms = sum(smoothed.get(n, 0) for n in active_names)
        if mode == BatteryMode.BALANCED:
            ref_balanced_ms = EMA * total_ms + (1 - EMA) * ref_balanced_ms

        score_map = ctx.get("score_map")

        # ── panel ────────────────────────────────────────────────────
        fh, fw = frame.shape[:2]
        panel = np.full((fh, PANEL_W, 3), C_PANEL, dtype=np.uint8)
        draw_panel(
            panel, mode, dict(smoothed), active_names,
            ref_balanced_ms, score_map,
            graphs.get("elim_ms", 0.0),
        )

        # ── camera feed with ROI crop overlay ────────────────────────
        display = cv2.resize(frame, (fw, fh))

        # draw score map grid on the camera feed (top-left overlay)
        if score_map is not None:
            rows, cols = score_map.shape
            gh, gw = fh // rows, fw // cols
            for r in range(rows):
                for c in range(cols):
                    val = float(score_map[r, c])
                    if val > 0.5:
                        x1, y1 = c * gw, r * gh
                        alpha  = 0.25 + val * 0.3
                        overlay = display.copy()
                        cv2.rectangle(overlay, (x1, y1),
                                      (x1 + gw, y1 + gh),
                                      (0, int(val * 220), 0), -1)
                        cv2.addWeighted(overlay, alpha, display,
                                        1 - alpha, 0, display)

        # mode label on feed
        ml, mc = MODE_META[mode]
        cv2.rectangle(display, (0, 0), (fw, 28), (0, 0, 0), -1)
        _txt(display, f"CORTEX  |  {ml}  |  frame {frame_count}",
             (10, 19), scale=0.45, color=mc)

        # ── composite ────────────────────────────────────────────────
        canvas = np.full((fh, fw + PANEL_W, 3), C_BG, dtype=np.uint8)
        canvas[:, :fw]  = display
        canvas[:, fw:]  = panel
        cv2.line(canvas, (fw, 0), (fw, fh), C_BORDER, 1)

        cv2.imshow("CORTEX  Graph IR  —  dead_node_elimination live", canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"): break
        elif key == ord("b"):
            mode = BatteryMode.BALANCED
            _rebuild()
        elif key == ord("p"):
            mode = BatteryMode.POWER_SAVE
            _rebuild()
        elif key == ord("a"):
            mode = BatteryMode.AGGRESSIVE
            _rebuild()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
