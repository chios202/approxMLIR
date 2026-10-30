"""Fused softmax kernel whose exp is an approximation knob.

`softmax_kernel` computes softmax along each row of a 2D tensor (the standard
fused pattern from Triton's softmax tutorial, reduced to one program per row). For the exp
it calls `softmax_exp`, a @triton.jit function with a @Knob: a decision tree
that picks one of nine exp versions at runtime, from the row length
(by default: exact below 512 columns, minimax2 from 512 columns on). With
ar.enable_triton_approx() on, ApproxMLIR replaces that one call inside the
fused kernel; the rest of the kernel is unchanged. To use other versions, change
the tree's `decisions` / `thresholds` (or let ar.tune_triton search them).

All polynomial variants use the same range reduction:
    exp(x) = 2^i * 2^f,  i = floor(x * log2(e)),  f in [0, 1)
and differ only in how 2^f is approximated. Max relative errors of 2^f come
from examples/exp_minimax.sollya (run it with Sollya to regenerate the coefficients):

    version 0: tl.exp (exact, hardware ex2.approx)
    version 1-3: Taylor series of 2^f = e^(f ln2), degree 1/2/3    15.3%  3.33%  0.556%
    version 4-6: float32 minimax (Sollya fpminimax), degree 1/2/3   2.98%  0.172% 0.0075%
    version 7:   least-squares degree 3 with p(0)=1, p(1)=2         ~0.014%
    version 8:   Schraudolph's bit trick (no polynomial)            ~5.6%
    version 9-10: minimax degree 4/5                          2.6e-4%  7.6e-6%
                 (degree 5 is MORE accurate than the hardware ex2.approx)

Used by: examples/example_triton_softmax.py, tests/test_triton_frontend.py.
"""

import approx_runtime as ar  # before triton: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton
import triton.language as tl

LOG2_E = tl.constexpr(1.4426950408889634)
LN_2 = tl.constexpr(0.6931471805599453)
MIN_INPUT = tl.constexpr(-87.0)  # below this, 2^i is no longer a normal float32


@triton.jit
def _split_exp(x):
    """exp(x) = two_i * 2^f: returns (two_i, f) with f in [0, 1)."""
    t = tl.maximum(x, MIN_INPUT) * LOG2_E
    i = tl.floor(t)
    two_i = ((i.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)
    return two_i, t - i


@triton.jit
def exp_taylor1(x):
    two_i, f = _split_exp(x)
    r = f * LN_2  # 2^f = e^r
    return two_i * (1.0 + r)


@triton.jit
def exp_taylor2(x):
    two_i, f = _split_exp(x)
    r = f * LN_2
    return two_i * (1.0 + r * (1.0 + r * 0.5))


@triton.jit
def exp_taylor3(x):
    two_i, f = _split_exp(x)
    r = f * LN_2
    return two_i * (1.0 + r * (1.0 + r * (0.5 + r * 0.16666666666666666)))


@triton.jit
def exp_minimax1(x):
    two_i, f = _split_exp(x)
    return two_i * (0.970178782939910888671875 + f * 0.970178782939910888671875)


@triton.jit
def exp_minimax2(x):
    two_i, f = _split_exp(x)
    return two_i * (1.001724720001220703125
                    + f * (0.657636463642120361328125 + f * 0.3371892869472503662109375))


@triton.jit
def exp_minimax3(x):
    two_i, f = _split_exp(x)
    return two_i * (0.999925196170806884765625
                    + f * (0.695833742618560791015625
                           + f * (0.22606670856475830078125 + f * 7.8024797141551971435546875e-2)))


@triton.jit
def exp_minimax4(x):
    two_i, f = _split_exp(x)
    return two_i * (1.0000026226043701171875
                    + f * (0.6930034160614013671875
                           + f * (0.2414443492889404296875
                                  + f * (5.20092435181140899658203125e-2
                                         + f * 1.3535187579691410064697265625e-2))))


@triton.jit
def exp_minimax5(x):
    two_i, f = _split_exp(x)
    return two_i * (0.999999940395355224609375
                    + f * (0.693152725696563720703125
                           + f * (0.2401557266712188720703125
                                  + f * (5.58212064206600189208984375e-2
                                         + f * (8.994684554636478424072265625e-3
                                                + f * 1.875563524663448333740234375e-3)))))


@triton.jit
def exp_lsq3(x):
    two_i, f = _split_exp(x)
    return two_i * (1.0 + f * (0.6959 + f * (0.2249 + f * 0.0791)))


@triton.jit
def exp_schraudolph(x):
    x = tl.maximum(x, MIN_INPUT)
    bits = (x * 12102203.0 + 1065292415.0).to(tl.int32)  # 2^23 / ln2, and 127 * 2^23 - 60801
    return bits.to(tl.float32, bitcast=True)


# func_substitute redirects the call to softmax_exp(x, n) to a version without
# changing its arguments, so every version must take the same (x, n). These
# wrappers give the exp_*(x) functions above that signature; n is unused.

@triton.jit
def _taylor1(x, n):
    return exp_taylor1(x)


@triton.jit
def _taylor2(x, n):
    return exp_taylor2(x)


@triton.jit
def _taylor3(x, n):
    return exp_taylor3(x)


@triton.jit
def _minimax1(x, n):
    return exp_minimax1(x)


@triton.jit
def _minimax2(x, n):
    return exp_minimax2(x)


@triton.jit
def _minimax3(x, n):
    return exp_minimax3(x)


@triton.jit
def _lsq3(x, n):
    return exp_lsq3(x)


@triton.jit
def _schraudolph(x, n):
    return exp_schraudolph(x)


@triton.jit
def _minimax4(x, n):
    return exp_minimax4(x)


@triton.jit
def _minimax5(x, n):
    return exp_minimax5(x)


# Knob value (a decision in the tree) -> name of the exp version, for printing results.
VERSION_NAMES = {0: "exact", 1: "taylor1", 2: "taylor2", 3: "taylor3", 4: "minimax1",
                 5: "minimax2", 6: "minimax3", 7: "lsq3", 8: "schraudolph",
                 9: "minimax4", 10: "minimax5"}


@triton.jit
def row_length(n):
    """The decision tree's state function: its result (the row length) is compared to the thresholds."""
    return n


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=row_length,
    state_indices=[1],                  # state = n, the row length
    thresholds=[512],                   # rows < 512 | rows >= 512
    decisions=[0, 5],                   # exact for short rows, minimax2 for long rows
    transform_type="func_substitute",
    approx_kernels={1: _taylor1, 2: _taylor2, 3: _taylor3, 4: _minimax1,
                    5: _minimax2, 6: _minimax3, 7: _lsq3, 8: _schraudolph,
                    9: _minimax4, 10: _minimax5},
    # Search space for ar.tune:
    thresholds_lower=[0],
    thresholds_upper=[8192],
    decision_values=list(VERSION_NAMES),
))
@triton.jit
def softmax_exp(x, n):
    """The exact exp (knob value 0). `n` is the row length, the decision tree's state."""
    return tl.exp(x)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    """Softmax of one row per program: exp(x - max) / sum, with the exp from softmax_exp."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    numerator = softmax_exp(x - tl.max(x, axis=0), n_cols)
    tl.store(out_ptr + row * n_cols + offs, numerator / tl.sum(numerator, axis=0), mask=mask)


def softmax(x):
    """Softmax along the rows of a 2D float32 CUDA tensor, by launching softmax_kernel (one program per row)."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
