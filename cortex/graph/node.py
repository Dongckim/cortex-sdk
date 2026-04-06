"""Graph IR Node — analogous to an MLIR Operation.

Each Node represents one operation in the cortex L2 pipeline.
Nodes are pure data; execution logic lives in metadata["fn"].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A single operation in the cortex Graph IR.

    In MLIR terms this is an Operation: it has a name (op_type),
    operands (inputs), results (outputs), and attributes (metadata).

    Attributes:
        name: Unique identifier within the graph, e.g. "saliency_dft".
        op_type: Operation category string, e.g. "spectral_residual",
            "weighted_sum", "external_call".
            Use "external_call" for OpenCV/scikit-image black-boxes
            (MSER, SSIM, Laplacian) that cannot be decomposed or compiled.
            This is the cortex equivalent of TVM's bring_your_own_codegen
            (BYOC) boundary.
        inputs: Upstream Node references (operands in MLIR terminology).
        outputs: Output slot names produced by this node.
        metadata: Dialect-specific attributes. Required key:
            "fn": Callable[[dict], dict] — the actual computation.
            Optional keys: "dialect", "description", "external_lib".
        is_compilable: True for pure numpy/math operations that can be
            lowered to native code (e.g. saliency DFT, score fusion).
            False for external_call boundaries.
    """

    name: str
    op_type: str
    inputs: list[Node | str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_compilable: bool = False

    def __repr__(self) -> str:
        tag = "compilable" if self.is_compilable else "external_call"
        return f"Node({self.name!r}, op={self.op_type!r}, [{tag}])"
