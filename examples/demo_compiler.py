"""Phase 3 compiler demo — feel baseline vs vectorized vs JIT live.

Run:
    python examples/demo_compiler.py

Controls:
    1   switch to BASELINE   (Python loop)
    2   switch to VECTORIZED (numpy reduceat)
    3   switch to JIT        (numba LLVM)
    +   increase stress (more kernel calls/frame)
    -   decrease stress
    Q   quit

How to feel the difference
--------------------------
Press + a few times until stress reaches 200–400.
At that load, baseline takes ~60–120 ms/frame (< 20 fps).
Switch to JIT with key 3 — the display noticeably speeds up.
Switch back to 1 — it slows down again.

The score map heatmap updates on every frame so you can verify
all three kernels produce visually identical output.
"""

import threading
import time

import cv2
import numpy as np

from cortex.compiler import (
    saliency_baseline,
    saliency_jit,
    saliency_vectorized,
    warmup_jit,
)
from cortex.compiler.saliency_kernel import _NUMBA_AVAILABLE

# ── palette ──────────────────────────────────────────────────────────
C_BG     = (18,  18,  18)
C_PANEL  = (26,  26,  26)
C_BORDER = (50,  50,  50)
C_WHITE  = (220, 220, 220)
C_DIM    = (90,  90,  90)
C_GREEN  = (80,  210, 120)
C_ORANGE = (50,  160, 255)
C_RED    = (70,   70, 220)
C_YELLOW = (50,  210, 210)
C_TEAL   = (180, 200,  80)

KERNELS = {
    "1  BASELINE\n(Python loop)":   saliency_baseline,
    "2  VECTORIZED\n(numpy reduceat)": saliency_vectorized,
    "3  JIT\n(numba LLVM)":         saliency_jit,
}
KERNEL_NAMES  = list(KERNELS.keys())
KERNEL_FNS    = list(KERNELS.values())
KERNEL_COLORS = [C_RED, C_ORANGE, C_GREEN]

WIN      = "CORTEX  Phase 3 — kernel comparison"
PANEL_W  = 420
BOTTOM_H = 140
GRID     = (8, 6)
CELL     = 24


def _txt(img, text, pos, scale=0.40, color=C_WHITE, thick=1):
    for i, line in enumerate(text.split("\n")):
        y = pos[1] + i * int(scale * 30)
        cv2.putText(img, line, (pos[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, bg=C_BORDER):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y),
                  (x + max(2, int(w * min(frac, 1.0))), y + h), color, -1)


def _sep(img, y, x0=0, x1=None):
    if x1 is None:
        x1 = img.shape[1]
    cv2.line(img, (x0, y), (x1, y), C_BORDER, 1)


# ── background benchmark thread ───────────────────────────────────────
class KernelTimer:
    """Continuously benchmarks one kernel on a rolling window."""
    def __init__(self, fn, name, color, frame_ref):
        self._fn      = fn
        self.name     = name
        self.color    = color
        self._frame   = frame_ref   # shared list [frame]
        self.ms       = 0.0
        self.calls_ps = 0.0
        self._lock    = threading.Lock()
        self._stop    = False
        self._t       = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        window = 0.3   # seconds per measurement window
        while not self._stop:
            frame = self._frame[0]
            if frame is None:
                time.sleep(0.01)
                continue
            count = 0
            t0 = time.perf_counter()
            deadline = t0 + window
            while time.perf_counter() < deadline:
                self._fn(frame, GRID)
                count += 1
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.ms       = elapsed / count * 1000 if count else 0.0
                self.calls_ps = count / elapsed if elapsed > 0 else 0.0

    def stats(self):
        with self._lock:
            return self.ms, self.calls_ps

    def stop(self):
        self._stop = True


# ── right panel ───────────────────────────────────────────────────────

def draw_panel(panel, active_idx, timers, score_map, stress, active_ms):
    panel[:] = C_PANEL
    W, PAD = panel.shape[1], 14
    y = 16

    _txt(panel, "KERNEL THROUGHPUT", (PAD, y), scale=0.44,
         color=C_WHITE, thick=1)
    y += 24
    _txt(panel, "background benchmark  (all 3 running simultaneously)",
         (PAD, y), scale=0.28, color=C_DIM)
    y += 18
    _sep(panel, y, PAD, W - PAD)
    y += 14

    # per-kernel throughput bars
    max_cps = max((t.stats()[1] for t in timers), default=1.0)
    max_cps = max(max_cps, 1.0)

    for i, timer in enumerate(timers):
        ms_t, cps = timer.stats()
        is_active = (i == active_idx)
        c = timer.color
        nc = C_WHITE if is_active else C_DIM

        # key hint + name
        short = KERNEL_NAMES[i].split("\n")[0]   # e.g. "1  BASELINE"
        sub   = KERNEL_NAMES[i].split("\n")[1]   # e.g. "(Python loop)"
        if is_active:
            cv2.rectangle(panel, (PAD - 2, y - 12),
                          (W - PAD + 2, y + 34), c, 1)
        _txt(panel, short, (PAD + 4, y), scale=0.42, color=nc, thick=1)
        _txt(panel, sub,   (PAD + 4, y + 16), scale=0.30, color=C_DIM)

        _txt(panel, f"{ms_t:.3f} ms",
             (W - PAD - 110, y), scale=0.40, color=c if is_active else C_DIM)
        _txt(panel, f"{cps:.0f}/s",
             (W - PAD - 55, y + 16), scale=0.30, color=C_DIM)

        BAR_Y = y + 28
        _bar(panel, PAD, BAR_Y, W - 2 * PAD, 6, cps / max_cps, c,
             bg=(35, 35, 35))
        y += 56

    _sep(panel, y, PAD, W - PAD)
    y += 14

    # stress indicator
    _txt(panel, f"stress:  {stress}x  kernel calls / frame",
         (PAD, y), scale=0.38, color=C_YELLOW)
    y += 18
    _txt(panel, "[+] more stress    [-] less",
         (PAD, y), scale=0.28, color=C_DIM)
    y += 24

    # active kernel live ms
    _sep(panel, y, PAD, W - PAD)
    y += 14
    _txt(panel, "Active kernel  (main loop, stressed):",
         (PAD, y), scale=0.33, color=C_DIM)
    y += 18
    ms_color = C_GREEN if active_ms < 10 else (C_ORANGE if active_ms < 40 else C_RED)
    _txt(panel, f"{active_ms:.1f} ms / frame",
         (PAD, y), scale=0.58, color=ms_color, thick=2)
    est_fps = 1000 / active_ms if active_ms > 0 else 0
    y += 30
    _txt(panel, f"≈ {est_fps:.0f} fps  (kernel-only estimate)",
         (PAD, y), scale=0.33, color=C_DIM)
    y += 22

    _txt(panel, "Switch kernels with  1 / 2 / 3",
         (PAD, y), scale=0.30, color=C_DIM)
    y += 20
    _txt(panel, "Raise stress to 200+ to feel the gap",
         (PAD, y), scale=0.30, color=C_YELLOW)
    y += 24

    # score map
    _sep(panel, y, PAD, W - PAD)
    y += 14
    _txt(panel, "ROI score map  (active kernel)",
         (PAD, y), scale=0.33, color=C_DIM)
    y += 16

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
                              (int((1 - v) * 50), int(v * 230), 20), -1)
                if v > 0.3:
                    _txt(panel, f"{v:.1f}", (x1 + 3, y1 + CELL - 4),
                         scale=0.22, color=(10, 10, 10))
        cv2.rectangle(panel, (x0, y),
                      (x0 + mw, y + rows * CELL), C_BORDER, 1)


# ── bottom strip ──────────────────────────────────────────────────────

def draw_bottom(strip, timers, active_idx, stress):
    strip[:] = (20, 20, 20)
    H, W = strip.shape[:2]
    PAD = 16
    _sep(strip, 0, 0, W)

    _txt(strip, "ms / kernel call  (lower is better)",
         (PAD, 20), scale=0.38, color=C_DIM)

    bar_y = 36
    bar_h = 22
    max_ms = max((t.stats()[0] for t in timers), default=0.5)
    max_ms = max(max_ms, 0.1)

    bar_w_total = W - 2 * PAD
    col_w = bar_w_total // len(timers)

    for i, timer in enumerate(timers):
        ms_t, _ = timer.stats()
        x = PAD + i * col_w
        c = timer.color if i == active_idx else C_DIM
        short = KERNEL_NAMES[i].split("\n")[0]
        sub   = KERNEL_NAMES[i].split("\n")[1]

        _bar(strip, x, bar_y, col_w - 20, bar_h,
             ms_t / max_ms, timer.color, bg=(35, 35, 35))
        _txt(strip, f"{ms_t:.3f} ms", (x, bar_y + bar_h + 14),
             scale=0.36, color=c)
        _txt(strip, short, (x, bar_y + bar_h + 30), scale=0.30, color=C_DIM)

        if i == active_idx:
            cv2.rectangle(strip, (x - 2, bar_y - 2),
                          (x + col_w - 22, bar_y + bar_h + 2), c, 1)

    # speedup annotations
    ms_vals = [t.stats()[0] for t in timers]
    if ms_vals[0] > 0:
        for i in range(1, len(timers)):
            if ms_vals[i] > 0:
                su = ms_vals[0] / ms_vals[i]
                x = PAD + i * col_w
                _txt(strip, f"{su:.1f}x vs baseline",
                     (x, bar_y - 14), scale=0.28,
                     color=C_GREEN if su > 1.1 else C_DIM)

    _txt(strip, f"stress={stress}x  |  [+/-] adjust  |  [1/2/3] switch kernel  |  [Q] quit",
         (PAD, H - 10), scale=0.28, color=C_DIM)


# ── main ──────────────────────────────────────────────────────────────

def main():
    if _NUMBA_AVAILABLE:
        print("Warming up JIT kernel (one-time LLVM compile)...", end=" ", flush=True)
        warmup_jit()
        print("done")
    else:
        print("numba not available — JIT kernel falls back to vectorized")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = None
        print("No webcam — using synthetic frames.")

    active_idx = 2   # start on JIT
    stress     = 50
    frame_ref  = [None]

    # start background timers (all 3 run continuously)
    timers = [
        KernelTimer(fn, KERNEL_NAMES[i], KERNEL_COLORS[i], frame_ref)
        for i, fn in enumerate(KERNEL_FNS)
    ]

    score_map  = None
    active_ms  = 0.0
    synth      = np.zeros((480, 640, 3), dtype=np.uint8)

    # macOS fullscreen blocks keyboard input — use resizable window instead
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 780)

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
            cv2.circle(synth, (cx, cy), 80, (160, 185, 200), -1)
            cv2.circle(synth, (cx + 120, cy - 60), 40, (90, 210, 150), -1)
            frame = synth.copy()

        frame_ref[0] = frame.copy()

        # ── run active kernel × stress ───────────────────────────────
        fn = KERNEL_FNS[active_idx]
        t0 = time.perf_counter()
        for _ in range(stress):
            sm = fn(frame, GRID)
        active_ms = (time.perf_counter() - t0) * 1000
        score_map = sm

        # ── layout ───────────────────────────────────────────────────
        fh, fw = frame.shape[:2]
        canvas_w = fw + PANEL_W
        canvas_h = fh + BOTTOM_H
        canvas = np.full((canvas_h, canvas_w, 3), C_BG, dtype=np.uint8)

        # camera feed + score map overlay
        cam_img = frame.copy()
        if score_map is not None:
            rows, cols = score_map.shape
            gh, gw = fh // rows, fw // cols
            for r in range(rows):
                for c in range(cols):
                    v = float(score_map[r, c])
                    if v > 0.45:
                        x1, y1 = c * gw, r * gh
                        ov = cam_img.copy()
                        cv2.rectangle(ov, (x1, y1),
                                      (x1 + gw, y1 + gh),
                                      (0, int(v * 220), 0), -1)
                        cv2.addWeighted(ov, 0.18 + v * 0.15,
                                        cam_img, 1 - (0.18 + v * 0.15),
                                        0, cam_img)

        # header bar
        c_active = KERNEL_COLORS[active_idx]
        cv2.rectangle(cam_img, (0, 0), (fw, 28), (0, 0, 0), -1)
        short = KERNEL_NAMES[active_idx].split("\n")[0]
        _txt(cam_img, f"ACTIVE: {short}",
             (10, 19), scale=0.50, color=c_active, thick=1)
        ms_c = C_GREEN if active_ms < 10 else (C_ORANGE if active_ms < 40 else C_RED)
        _txt(cam_img,
             f"stress={stress}x  |  {active_ms:.1f}ms/frame",
             (fw - 300, 19), scale=0.38, color=ms_c)

        # ── stress overlay — big centred text ────────────────────────
        # Draw stress number large so user can see it change immediately
        stress_str = f"x{stress}"
        ms_c2 = C_GREEN if active_ms < 10 else (C_ORANGE if active_ms < 40 else C_RED)
        # shadow
        cv2.putText(cam_img, stress_str,
                    (fw // 2 - 80, fh // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
        # foreground
        cv2.putText(cam_img, stress_str,
                    (fw // 2 - 80, fh // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, ms_c2, 4, cv2.LINE_AA)
        _txt(cam_img, "STRESS  [UP] more  [DOWN] less",
             (fw // 2 - 120, fh // 2 + 65), scale=0.45, color=C_DIM)

        canvas[:fh, :fw] = cam_img
        cv2.line(canvas, (fw, 0), (fw, fh), C_BORDER, 1)

        # right panel
        panel = np.full((fh, PANEL_W, 3), C_PANEL, dtype=np.uint8)
        draw_panel(panel, active_idx, timers, score_map, stress, active_ms)
        canvas[:fh, fw:fw + PANEL_W] = panel

        # bottom strip
        strip = np.full((BOTTOM_H, canvas_w, 3), (20, 20, 20), dtype=np.uint8)
        draw_bottom(strip, timers, active_idx, stress)
        canvas[fh:fh + BOTTOM_H, :] = strip

        cv2.imshow(WIN, canvas)

        raw_key = cv2.waitKey(1)
        key = raw_key & 0xFF
        if   key == ord("q"):  break
        elif key == ord("1"):  active_idx = 0
        elif key == ord("2"):  active_idx = 1
        elif key == ord("3"):  active_idx = 2
        elif key in (ord("+"), ord("="), 82):   # 82 = UP arrow
            stress = min(stress + 25, 1000)
        elif key in (ord("-"), 84):             # 84 = DOWN arrow
            stress = max(stress - 25, 1)

    for t in timers:
        t.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
