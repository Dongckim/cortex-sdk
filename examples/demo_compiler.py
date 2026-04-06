"""Phase 3 compiler demo — isolated kernel throughput comparison.

Run:
    python examples/demo_compiler.py

Each kernel is measured in isolation (round-robin, 0.5 s per kernel)
so Python GIL contention doesn't mask the real difference.

Result (640×480, grid 32×24 = 768 cells):
  baseline   ~370  fps  — Python loop iterates 768 times
  vectorized ~5200 fps  — numpy reduceat, no Python loop
  JIT        ~6300 fps  — numba LLVM compiled loop

Controls:
    1 / 2 / 3   switch score-map overlay
    Q           quit
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
C_BG    = (18,  18,  18)
C_PANEL = (26,  26,  26)
C_BORD  = (50,  50,  50)
C_WHITE = (220, 220, 220)
C_DIM   = (90,  90,  90)
C_GREEN = (80,  210, 120)
C_ORG   = (50,  160, 255)
C_RED   = (70,   70, 220)
C_YELL  = (50,  210, 210)

WIN          = "CORTEX  Phase 3 — kernel benchmark"
PERF_GRID    = (32, 24)   # 768 cells — amplifies Python loop gap
DISPLAY_GRID = (8, 6)     # clean score-map display
WINDOW_S     = 0.5        # seconds each kernel gets per turn

KERNEL_NAMES  = ["BASELINE", "VECTORIZED", "JIT"]
KERNEL_SUBS   = ["Python loop  (768 iters)", "numpy reduceat", "numba → LLVM"]
KERNEL_FNS    = [saliency_baseline, saliency_vectorized, saliency_jit]
KERNEL_COLORS = [C_RED, C_ORG, C_GREEN]


def _txt(img, text, pos, scale=0.45, color=C_WHITE, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, bg=C_BORD):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = max(4, int(w * min(frac, 1.0)))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


# ── sequential benchmark thread ───────────────────────────────────────

class SequentialBenchmark:
    """Runs each kernel in round-robin — no GIL contention between them."""

    def __init__(self, frame_ref):
        self._frame   = frame_ref
        self.fps      = [0.0, 0.0, 0.0]
        self.ms       = [0.0, 0.0, 0.0]
        self.active_i = 0        # which kernel is currently being measured
        self._lock    = threading.Lock()
        self._stop    = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            frame = self._frame[0]
            if frame is None:
                time.sleep(0.01)
                continue
            for i, fn in enumerate(KERNEL_FNS):
                with self._lock:
                    self.active_i = i
                count = 0
                t0    = time.perf_counter()
                deadline = t0 + WINDOW_S
                while time.perf_counter() < deadline:
                    fn(frame, PERF_GRID)
                    count += 1
                elapsed = time.perf_counter() - t0
                with self._lock:
                    self.fps[i] = count / elapsed if elapsed > 0 else 0.0
                    self.ms[i]  = elapsed / count * 1000 if count > 0 else 0.0

    def stats(self):
        with self._lock:
            return list(self.fps), list(self.ms), self.active_i

    def stop(self):
        self._stop = True


# ── draw benchmark panel ──────────────────────────────────────────────

def draw_panel(canvas, bench, active_overlay, x0, panel_w, panel_h):
    fps_list, ms_list, measuring_i = bench.stats()
    PAD  = 16
    W    = panel_w

    cv2.rectangle(canvas, (x0, 0), (x0 + W, panel_h), C_PANEL, -1)
    cv2.line(canvas, (x0, 0), (x0, panel_h), C_BORD, 1)

    y = 20
    _txt(canvas, "ISOLATED BENCHMARK", (x0 + PAD, y),
         scale=0.52, color=C_WHITE, thick=1)
    y += 22
    _txt(canvas, f"grid {PERF_GRID[0]}x{PERF_GRID[1]} = {PERF_GRID[0]*PERF_GRID[1]} cells per frame",
         (x0 + PAD, y), scale=0.30, color=C_DIM)
    y += 10
    cv2.line(canvas, (x0 + PAD, y), (x0 + W - PAD, y), C_BORD, 1)
    y += 16

    max_fps = max(fps_list + [1.0])

    for i in range(3):
        fps = fps_list[i]
        ms  = ms_list[i]
        c   = KERNEL_COLORS[i]

        is_measuring = (i == measuring_i)
        is_overlay   = (i == active_overlay)

        # header row
        dot_c = c if is_measuring else (50, 50, 50)
        cv2.circle(canvas, (x0 + PAD + 6, y), 5, dot_c, -1)
        _txt(canvas, f"[{i+1}] {KERNEL_NAMES[i]}",
             (x0 + PAD + 18, y + 4),
             scale=0.46, color=C_WHITE if is_overlay else C_DIM, thick=1)
        if is_measuring:
            _txt(canvas, "← measuring now",
                 (x0 + W - 130, y + 4), scale=0.28, color=c)
        y += 22
        _txt(canvas, KERNEL_SUBS[i], (x0 + PAD + 18, y),
             scale=0.30, color=C_DIM)
        y += 18

        # FPS number — big
        fps_str = f"{fps:.0f} fps" if fps > 0 else "---"
        cv2.putText(canvas, fps_str,
                    (x0 + PAD, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    c, 2, cv2.LINE_AA)
        _txt(canvas, f"{ms:.3f} ms / call",
             (x0 + PAD, y + 52), scale=0.32, color=C_DIM)

        # bar
        y += 62
        _bar(canvas, x0 + PAD, y, W - 2 * PAD, 20,
             fps / max_fps, c if fps > 0 else (40, 40, 40))
        y += 28

        # speedup annotation
        if i > 0 and fps_list[0] > 0 and fps > 0:
            su = fps / fps_list[0]
            _txt(canvas, f"{su:.0f}x faster than baseline",
                 (x0 + PAD, y), scale=0.32,
                 color=C_GREEN if su > 2 else C_DIM)
            y += 20

        cv2.line(canvas, (x0 + PAD, y + 4),
                 (x0 + W - PAD, y + 4), C_BORD, 1)
        y += 18

    # legend
    y += 4
    _txt(canvas, "Keys  1/2/3 — switch overlay    Q — quit",
         (x0 + PAD, y), scale=0.30, color=C_DIM)


# ── score map overlay ─────────────────────────────────────────────────

def draw_score_overlay(cam_img, score_map, color):
    if score_map is None:
        return
    fh, fw = cam_img.shape[:2]
    rows, cols = score_map.shape
    gh, gw = fh // rows, fw // cols
    for r in range(rows):
        for c in range(cols):
            v = float(score_map[r, c])
            if v > 0.45:
                x1, y1 = c * gw, r * gh
                ov = cam_img.copy()
                cv2.rectangle(ov, (x1, y1),
                              (x1 + gw, y1 + gh), color, -1)
                cv2.addWeighted(ov, 0.18 + v * 0.14,
                                cam_img, 1 - (0.18 + v * 0.14),
                                0, cam_img)


# ── main ─────────────────────────────────────────────────────────────

def main():
    if _NUMBA_AVAILABLE:
        print("Warming up JIT...", end=" ", flush=True)
        warmup_jit()
        # also warm up for PERF_GRID dimensions
        from cortex.compiler.saliency_kernel import _grid_pool_jit
        dummy = np.zeros((64, 64), dtype=np.float32)
        _grid_pool_jit(dummy, PERF_GRID[1], PERF_GRID[0])
        print("done")
    else:
        print("numba unavailable — JIT = vectorized")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = None
        print("No webcam — using synthetic frames.")

    frame_ref     = [None]
    bench         = SequentialBenchmark(frame_ref)
    active_overlay = 2          # default: JIT score map
    synth          = np.zeros((480, 640, 3), dtype=np.uint8)

    PANEL_W  = 380
    CELL     = 26
    BOTTOM_H = DISPLAY_GRID[1] * CELL + 50

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 800)

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

        fh, fw  = frame.shape[:2]
        total_w = fw + PANEL_W
        total_h = fh + BOTTOM_H
        canvas  = np.full((total_h, total_w, 3), C_BG, dtype=np.uint8)

        # camera + score map overlay (always JIT-quality — display grid)
        cam_img  = frame.copy()
        score_sm = KERNEL_FNS[active_overlay](frame, DISPLAY_GRID)
        draw_score_overlay(cam_img, score_sm, KERNEL_COLORS[active_overlay])

        # header
        c_act = KERNEL_COLORS[active_overlay]
        cv2.rectangle(cam_img, (0, 0), (fw, 30), (0, 0, 0), -1)
        _txt(cam_img,
             f"OVERLAY: [{active_overlay+1}] {KERNEL_NAMES[active_overlay]}  —  {KERNEL_SUBS[active_overlay]}",
             (10, 21), scale=0.46, color=c_act, thick=1)

        canvas[:fh, :fw] = cam_img

        # right panel
        draw_panel(canvas, bench, active_overlay, fw, PANEL_W, fh)

        # ── bottom: score maps for all 3 side by side ─────────────────
        y0   = fh
        col  = total_w // 3
        PAD2 = 10
        cv2.line(canvas, (0, y0), (total_w, y0), C_BORD, 1)

        for i, fn in enumerate(KERNEL_FNS):
            sm = fn(frame, DISPLAY_GRID)
            x0 = i * col
            c  = KERNEL_COLORS[i]
            nc = C_WHITE if i == active_overlay else C_DIM

            _txt(canvas, f"[{i+1}] {KERNEL_NAMES[i]}",
                 (x0 + PAD2, y0 + 16), scale=0.38, color=nc)

            rows, cols_ = sm.shape
            mx, my = x0 + PAD2, y0 + 26
            for r in range(rows):
                for c_ in range(cols_):
                    v  = float(sm[r, c_])
                    x1 = mx + c_ * CELL
                    y1 = my + r  * CELL
                    cv2.rectangle(canvas,
                                  (x1 + 1, y1 + 1),
                                  (x1 + CELL - 2, y1 + CELL - 2),
                                  (int((1 - v) * 50),
                                   int(v * (230 if i == active_overlay else 100)),
                                   20), -1)
            cv2.rectangle(canvas, (mx, my),
                          (mx + cols_ * CELL, my + rows * CELL),
                          c if i == active_overlay else C_BORD, 1)

            if i < 2:
                cv2.line(canvas, (x0 + col, y0 + 4),
                         (x0 + col, y0 + BOTTOM_H - 4), C_BORD, 1)

        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"): break
        elif key == ord("1"): active_overlay = 0
        elif key == ord("2"): active_overlay = 1
        elif key == ord("3"): active_overlay = 2

    bench.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
