"""CortexPipeline — wires L1 capture gating into L2 Graph IR.

Flow:
    frame
      └─ L1 CaptureEngine (IMU gate → blur → scene change)
              ├─ rejected → PipelineResult(accepted=False)
              └─ accepted → L2 Graph IR (score map, ROI crop)
                                └─ PipelineResult(accepted=True, score_map, ...)

Battery mode is shared across both layers:
  POWER_SAVE → L1 loosens thresholds + L2 eliminates saliency_dft node
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from cortex.capture.engine import CaptureEngine
from cortex.capture.imu_gate import BatteryMode
from cortex.graph.builder import build_l2_graph
from cortex.graph.passes import EliminationResult, dead_node_elimination
from cortex.optimizer.hybrid_roi import RequestType

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of one frame through the full L1 + L2 pipeline.

    Attributes:
        accepted: True if L1 passed the frame to L2.
        reason: L1 decision string: "accepted", "blurry",
            "no_change", or "no_motion".
        score_map: L2 fused score map (6×8). None if L1 rejected.
        l1_stats: Blur score, SSIM score, motion score from L1.
        l2_timings: Per-node latency dict from L2 (ms). Empty if rejected.
        active_nodes: Set of node names that ran in L2.
    """

    accepted: bool
    reason: str
    score_map: np.ndarray | None = None
    l1_stats: dict = field(default_factory=dict)
    l2_timings: dict[str, float] = field(default_factory=dict)
    active_nodes: set[str] = field(default_factory=set)


class CortexPipeline:
    """Full L1 → L2 pipeline for wearable camera frames.

    L1 gates noisy/redundant frames so L2 only runs on useful ones.
    L2 builds a Graph IR and applies dead_node_elimination based on
    the current battery mode.

    Args:
        battery_mode: Initial battery mode. Applied to both layers.
        request_type: L2 ROI weight configuration.
        ema_alpha: L2 EMA smoothing coefficient.
    """

    def __init__(
        self,
        battery_mode: BatteryMode = BatteryMode.BALANCED,
        request_type: RequestType = RequestType.GENERAL,
        ema_alpha: float = 0.7,
    ) -> None:
        self._request_type = request_type
        self._ema_alpha = ema_alpha
        self._capture = CaptureEngine()
        self._dne: EliminationResult | None = None
        self._l2_graph = None
        self.set_battery_mode(battery_mode)

    @property
    def battery_mode(self) -> BatteryMode:
        return self._battery_mode

    @property
    def active_node_names(self) -> list[str]:
        """Node names currently in the L2 graph (after DNE)."""
        return self._l2_graph.node_names() if self._l2_graph else []

    @property
    def eliminated_nodes(self) -> list[str]:
        """Nodes removed by dead_node_elimination for current mode."""
        return self._dne.eliminated if self._dne else []

    @property
    def capture_stats(self) -> dict:
        """L1 acceptance statistics (total, accepted, rate)."""
        return self._capture.stats

    def set_battery_mode(self, mode: BatteryMode) -> None:
        """Switch battery mode — updates both L1 thresholds and L2 graph.

        In POWER_SAVE:
          L1: loosens blur/scene thresholds (fewer rejections, saves compute)
          L2: dead_node_elimination removes saliency_dft (ws=0 → dead)

        Args:
            mode: New battery mode.
        """
        self._battery_mode = mode
        self._capture.set_battery_mode(mode)

        # Rebuild L2 graph with DNE applied for this mode
        g = build_l2_graph(self._request_type, self._ema_alpha)
        self._dne = dead_node_elimination(g, mode)
        self._l2_graph = self._dne.graph
        logger.debug(
            "battery_mode=%s  l2_nodes=%s  eliminated=%s",
            mode.value,
            self._l2_graph.node_names(),
            self._dne.eliminated,
        )

    def process(
        self,
        frame: np.ndarray,
        imu_data: dict | None = None,
    ) -> PipelineResult:
        """Process one frame through L1 gate and L2 scoring.

        L1 decides whether to forward the frame to a VLM (expensive).
        L2 always runs — score map is a cheap local operation and should
        stay live regardless of L1's gating decision.

        Args:
            frame: BGR camera frame.
            imu_data: Optional dict with "accel" and "gyro" tuples.
                If None, IMU gate is skipped.

        Returns:
            PipelineResult where:
              accepted=True  → frame passed L1, worth sending to VLM
              accepted=False → frame gated by L1, skip VLM call
              score_map      → always present (L2 always runs)
        """
        # ── L1: capture gating (VLM forwarding decision) ─────────────
        l1 = self._capture.process_frame(frame, imu_data)

        # ── L2: Graph IR always runs (score map is local + cheap) ────
        ctx, timings = self._l2_graph.profile_execute(
            frame, battery_mode=self._battery_mode
        )

        return PipelineResult(
            accepted=l1.accepted,
            reason=l1.reason,
            score_map=ctx.get("score_map"),
            l1_stats=l1.stats,
            l2_timings=timings,
            active_nodes=set(self._l2_graph.node_names()),
        )
