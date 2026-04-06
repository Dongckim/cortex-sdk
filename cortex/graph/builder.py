"""Frontend: lowers the cortex L2 pipeline into Graph IR nodes.

This is the ingestion / lowering stage — analogous to MLIR's frontend
that converts Python ops into the IR dialect (here: cortex.l2).

Node compilation targets (for Phase 3):
  compilable      → saliency_dft, motion_map, score_fusion, ema_smooth
  external_call   → text_roi_mser  (OpenCV MSER + morphology)

TVM analogy:
  external_call = bring_your_own_codegen (BYOC) boundary.
  Ops the compiler cannot decompose stay as opaque function calls.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from cortex.capture.imu_gate import BatteryMode
from cortex.graph.graph import Graph
from cortex.graph.node import Node
from cortex.optimizer.center_crop import CenterCropStrategy
from cortex.optimizer.hybrid_roi import (
    RequestType,
    _CENTER_GATE_FLOOR,
    _WEIGHTS,
)
from cortex.optimizer.saliency_roi import SaliencyROIStrategy
from cortex.optimizer.text_roi import TextROIStrategy

logger = logging.getLogger(__name__)


def build_l2_graph(
    request_type: RequestType = RequestType.GENERAL,
    ema_alpha: float = 0.7,
    grid: tuple[int, int] = (8, 6),
) -> Graph:
    """Build the full L2 scoring pipeline as a Graph IR.

    Produces a Graph whose nodes mirror the stages in HybridROI.fused_score_map().
    The graph is battery-mode agnostic at build time — pass battery_mode at
    execute() time (or apply dead_node_elimination before executing).

    Pipeline (topological order):
      1. center_crop      [compilable]    Gaussian window → Sc
      2. text_roi_mser    [external_call] OpenCV MSER → St
      3. saliency_dft     [compilable]    Spectral residual FFT → Ss
      4. motion_map       [compilable]    Frame absdiff + grid pool → Sm
      5. score_fusion     [compilable]    S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm
      6. ema_smooth       [compilable]    EMA temporal smoothing → score_map

    Args:
        request_type: Weight configuration (wc, wt, ws, wm).
        ema_alpha: EMA smoothing coefficient.
        grid: Score map grid as (cols, rows). Default (8, 6).

    Returns:
        Graph ready for execute() or pass application.
    """
    graph = Graph()

    _center = CenterCropStrategy()
    _saliency = SaliencyROIStrategy()
    _text = TextROIStrategy()

    cols, rows = grid

    # ------------------------------------------------------------------
    # Node 1: center_crop   dialect=cortex.l2
    # Pure numpy Gaussian — trivially compilable.
    # No inputs from other nodes; reads only the frame.
    # ------------------------------------------------------------------
    def _fn_center_crop(ctx: dict[str, Any]) -> dict[str, Any]:
        sc = _center.score_map(ctx["frame"], grid)
        return {"sc": sc}

    n_center = Node(
        name="center_crop",
        op_type="gaussian_window",
        inputs=[],
        outputs=["sc"],
        metadata={
            "fn": _fn_center_crop,
            "dialect": "cortex.l2",
            "description": "Gaussian center-weighted score map → Sc",
        },
        is_compilable=True,
    )
    graph.add_node(n_center)

    # ------------------------------------------------------------------
    # Node 2: text_roi_mser   dialect=cortex.l2
    # OpenCV MSER + morphological dilation — external_call boundary.
    # Cannot be decomposed into lowerable ops.
    # TVM analogy: bring_your_own_codegen (BYOC) — stays as an opaque call.
    # ------------------------------------------------------------------
    def _fn_text_roi(ctx: dict[str, Any]) -> dict[str, Any]:
        st = _text.score_map(ctx["frame"], grid)
        return {"st": st}

    n_text = Node(
        name="text_roi_mser",
        op_type="external_call",
        inputs=[n_center],
        outputs=["st"],
        metadata={
            "fn": _fn_text_roi,
            "dialect": "cortex.l2",
            "description": "MSER text region detection → St  [OpenCV black-box]",
            "external_lib": "opencv.mser",
        },
        is_compilable=False,
    )
    graph.add_node(n_text)

    # ------------------------------------------------------------------
    # Node 3: saliency_dft   dialect=cortex.l2
    # Spectral residual via numpy FFT — compilable.
    # Primary compilation target in Phase 3 (cortex/compiler/).
    # Dead in POWER_SAVE mode → eliminated by dead_node_elimination.
    # ------------------------------------------------------------------
    def _fn_saliency_dft(ctx: dict[str, Any]) -> dict[str, Any]:
        ss = _saliency.score_map(ctx["frame"], grid)
        return {"ss": ss}

    n_saliency = Node(
        name="saliency_dft",
        op_type="spectral_residual",
        inputs=[n_center],
        outputs=["ss"],
        metadata={
            "fn": _fn_saliency_dft,
            "dialect": "cortex.l2",
            "description": "Spectral residual saliency FFT → Ss  [Phase 3 compile target]",
        },
        is_compilable=True,
    )
    graph.add_node(n_saliency)

    # ------------------------------------------------------------------
    # Node 4: motion_map   dialect=cortex.l2
    # Frame absdiff + Gaussian blur + grid pool — compilable.
    # Reads and updates prev_gray (explicit temporal state).
    # ------------------------------------------------------------------
    def _fn_motion_map(ctx: dict[str, Any]) -> dict[str, Any]:
        frame = ctx["frame"]
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3
            else frame.copy()
        )

        prev_gray = ctx.get("prev_gray")
        if prev_gray is None or prev_gray.shape != gray.shape:
            sm = np.zeros((rows, cols), dtype=np.float32)
        else:
            diff = cv2.absdiff(prev_gray, gray).astype(np.float32)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            diff[diff < 10] = 0

            h, w = diff.shape[:2]
            cell_h, cell_w = h / rows, w / cols
            sm = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
                    x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
                    sm[r, c] = diff[y1:y2, x1:x2].mean()
            s_max = sm.max()
            if s_max > 0:
                sm /= s_max

        return {"sm": sm, "prev_gray": gray}

    n_motion = Node(
        name="motion_map",
        op_type="frame_diff",
        inputs=[n_center],
        outputs=["sm", "prev_gray"],
        metadata={
            "fn": _fn_motion_map,
            "dialect": "cortex.l2",
            "description": "Frame-to-frame motion map → Sm, updates prev_gray",
        },
        is_compilable=True,
    )
    graph.add_node(n_motion)

    # ------------------------------------------------------------------
    # Node 5: score_fusion   dialect=cortex.l2
    # S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm
    # Pure element-wise ops — operator fusion target in Phase 3.
    #
    # Battery mode handling:
    #   POWER_SAVE: "ss" absent from context (eliminated by DNE pass)
    #               → redistributes ws into wc, mirrors HybridROI behavior.
    #   BALANCED / AGGRESSIVE: uses ss normally.
    # ------------------------------------------------------------------
    def _fn_score_fusion(ctx: dict[str, Any]) -> dict[str, Any]:
        wc, wt, ws, wm = _WEIGHTS[request_type]
        sc = ctx["sc"]
        st = ctx["st"]
        sm = ctx["sm"]

        # ss is absent when saliency_dft was eliminated (POWER_SAVE + DNE)
        # or when battery_mode forces it off. Mirrors HybridROI.fused_score_map().
        ss = ctx.get("ss")
        battery_mode = ctx.get("battery_mode", BatteryMode.BALANCED)

        if ss is None or battery_mode == BatteryMode.POWER_SAVE:
            ss = np.zeros_like(sc)
            wc = wc + ws
            ws = 0.0

        center_gate = _CENTER_GATE_FLOOR + (1 - _CENTER_GATE_FLOOR) * sc
        ss_adj = ss * center_gate

        fused = wc * sc + wt * st + ws * ss_adj + wm * sm
        f_max = fused.max()
        if f_max > 0:
            fused /= f_max

        return {"fused": fused}

    n_fusion = Node(
        name="score_fusion",
        op_type="weighted_sum",
        inputs=[n_center, n_text, n_saliency, n_motion],
        outputs=["fused"],
        metadata={
            "fn": _fn_score_fusion,
            "dialect": "cortex.l2",
            "description": "S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm  [operator fusion target]",
        },
        is_compilable=True,
    )
    graph.add_node(n_fusion)

    # ------------------------------------------------------------------
    # Node 6: ema_smooth   dialect=cortex.l2
    # EMA temporal smoothing to prevent ROI jitter.
    # Reads and updates prev_score (explicit temporal state).
    # ------------------------------------------------------------------
    def _fn_ema_smooth(ctx: dict[str, Any]) -> dict[str, Any]:
        fused = ctx["fused"]
        prev_score = ctx.get("prev_score")
        if prev_score is not None and prev_score.shape == fused.shape:
            smoothed = ema_alpha * fused + (1 - ema_alpha) * prev_score
        else:
            smoothed = fused
        return {"score_map": smoothed, "prev_score": smoothed.copy()}

    n_ema = Node(
        name="ema_smooth",
        op_type="ema",
        inputs=[n_fusion],
        outputs=["score_map", "prev_score"],
        metadata={
            "fn": _fn_ema_smooth,
            "dialect": "cortex.l2",
            "description": f"EMA smoothing alpha={ema_alpha}, updates prev_score",
        },
        is_compilable=True,
    )
    graph.add_node(n_ema)

    logger.debug(
        "built l2 graph: %d nodes, request_type=%s",
        len(graph),
        request_type.value,
    )
    return graph
