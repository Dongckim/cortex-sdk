# Compiler Subgraph Boundaries

## What makes a node compilable?

A node is `is_compilable=True` if its implementation consists entirely of
lowerable operations — pure numpy array math, explicit loops, and standard
math functions that a JIT compiler (numba/LLVM) or tensor compiler (TVM) can
introspect and lower into native machine code.

| Criterion | Compilable | External call |
|---|---|---|
| Implementation | Pure numpy / explicit loops | cv2 / scikit-image C extension |
| Compiler can inspect ops | Yes | No (opaque binary) |
| Can be lowered to IR | Yes | No |
| numba nopython mode | Yes | No |
| TVM analogy | Standard op | BYOC boundary |

---

## Node classification in cortex L2 Graph IR

### Compilable nodes (`is_compilable=True`)

| Node | op_type | Why compilable |
|---|---|---|
| `center_crop` | `gaussian_window` | Pure numpy: `np.exp`, element-wise multiply |
| `saliency_dft` | `spectral_residual` | numpy FFT + log/exp + explicit grid loop |
| `motion_map` | `frame_diff` | numpy absdiff + explicit grid loop |
| `score_fusion` | `weighted_sum` | 48-element weighted sum, pure numpy |
| `ema_smooth` | `ema` | Single linear blend, pure numpy |

**Primary compilation target: `saliency_dft`**

- Operates on a 64×64 float32 array (4,096 elements) — large enough for
  kernel launch overhead to be amortised.
- Contains an explicit Python-level `for r / for c` grid loop that numba
  eliminates entirely.
- All ops (FFT, log, exp, cos, sin) are available in numba `nopython=True`.

**Why `score_fusion` is NOT the target:**

`S = wc*Sc + wt*St + ws*Ss_adj + wm*Sm` operates on 48 cells (6×8).
numpy already executes this in ~1 μs. numba/TVM kernel launch overhead
(typically 5–20 μs) exceeds the computation itself. Compiling this node
would make it slower, not faster.

Rule: compile only when `computation_time >> kernel_launch_overhead`.

---

### External call nodes (`is_compilable=False`)

| Node | op_type | Library | Why external |
|---|---|---|---|
| `text_roi_mser` | `external_call` | `opencv.mser` | OpenCV MSER is a closed C++ implementation. Cannot be decomposed into lowerable ops. |

**TVM / MLIR analogy:**

In TVM, operations that cannot be lowered through the compiler's IR are
handled by `bring_your_own_codegen` (BYOC). The compiler emits a call to
an external runtime function at a well-defined boundary instead of
generating code for it.

In MLIR, this corresponds to marking an operation as an `external` function
call — the op stays in the IR as an opaque node, and the runtime resolves it
at link time.

`text_roi_mser` maps exactly to this pattern:
- It is registered as `op_type="external_call"` in the Graph IR.
- `is_compilable=False` signals the partitioning pass to exclude it.
- `metadata["external_lib"] = "opencv.mser"` records the runtime dependency.
- At execution time it is called as a Python function (the cv2 black-box).

---

## Partitioning pass (`compiler.partition()`)

```
Graph IR nodes (topological order)
       ↓
for each node:
    if node.is_compilable:
        → compilable subgraph (JIT target)
    else:
        → external_call boundary (stays as opaque call)
```

The two subgraphs are not disconnected — external nodes can consume outputs
of compilable nodes and vice versa. The boundary only determines which nodes
are handed to the code generator.

---

## Compilation pipeline mapping

```
Python source (SaliencyROIStrategy)
      ↓  cortex.graph.build_l2_graph()        Frontend / ingestion
Graph IR  (Node + is_compilable flags)
      ↓  CortexCompiler.partition()            Partitioning pass
compilable subgraph  |  external_call stubs
      ↓  CortexCompiler.compile() / numba      Codegen  (LLVM via numba)
      ↓                           / TVM        Codegen  (TIR → LLVM / CUDA)
Native kernel  +  external_call stubs
      ↓  CortexCompiler.benchmark()            Runtime profiling
Latency report (baseline / vectorized / jit)
```
