"""Fused attention (FlashAttention forward) whose exp2 is an approximation knob.

`attention_kernel` computes softmax(Q K^T * scale) V in blocks, like the Triton
FlashAttention tutorial, but simplified for benchmarking: plain pointers,
non-causal, fp16 inputs with fp32 accumulation, sequence length a multiple of
the block sizes (so no masking).

Attention uses exp2 (2^x), not exp: the scale is multiplied by log2(e) once, so
the inner loop only needs 2^x. The knob is on `attn_exp2`, the 2^x applied to
every attention score. A decision tree picks the version from the sequence
length at runtime. The versions are the same menu as the softmax kernel's, with
the same numbering: 2^x = 2^i * 2^f with i = floor(x), f in [0, 1), and 2^f by
a polynomial (coefficients from examples/exp_minimax.sollya; max relative
error of 2^f):

    version 0: tl.math.exp2 (exact, hardware ex2.approx)
    version 1-3: Taylor series of 2^f = e^(f ln2), degree 1/2/3    15.3%  3.33%  0.556%
    version 4-6: float32 minimax (Sollya fpminimax), degree 1/2/3   2.98%  0.172% 0.0075%
    version 7:   least-squares degree 3 with p(0)=1, p(1)=2         ~0.014%
    version 8:   Schraudolph's bit trick for 2^x (no polynomial)    ~3%
    version 9-10: minimax degree 4/5                          2.6e-4%  7.6e-6%

The running-maximum correction (alpha) stays exact: it is one exp2 per row,
cheap, and errors in it rescale the whole accumulator.

Used by: examples/example_triton_attention.py.
"""

import approx_runtime as ar  # before triton: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton
import triton.language as tl

MIN_EXP2_INPUT = tl.constexpr(-126.0)  # below this, 2^i is no longer a normal float32
LN_2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _split_exp2(x):
    """2^x = two_i * 2^f: returns (two_i, f) with f in [0, 1)."""
    x = tl.maximum(x, MIN_EXP2_INPUT)
    i = tl.floor(x)
    two_i = ((i.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)
    return two_i, x - i


@triton.jit
def exp2_taylor1(x, seq_len):
    two_i, f = _split_exp2(x)
    r = f * LN_2  # 2^f = e^r
    return two_i * (1.0 + r)


@triton.jit
def exp2_taylor2(x, seq_len):
    two_i, f = _split_exp2(x)
    r = f * LN_2
    return two_i * (1.0 + r * (1.0 + r * 0.5))


@triton.jit
def exp2_taylor3(x, seq_len):
    two_i, f = _split_exp2(x)
    r = f * LN_2
    return two_i * (1.0 + r * (1.0 + r * (0.5 + r * 0.16666666666666666)))


@triton.jit
def exp2_minimax1(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (0.970178782939910888671875 + f * 0.970178782939910888671875)


@triton.jit
def exp2_minimax2(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (1.001724720001220703125
                    + f * (0.657636463642120361328125 + f * 0.3371892869472503662109375))


@triton.jit
def exp2_minimax3(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (0.999925196170806884765625
                    + f * (0.695833742618560791015625
                           + f * (0.22606670856475830078125 + f * 7.8024797141551971435546875e-2)))


@triton.jit
def exp2_lsq3(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (1.0 + f * (0.6959 + f * (0.2249 + f * 0.0791)))


@triton.jit
def exp2_schraudolph(x, seq_len):
    # 2^x by writing x straight into the float32 bits: bits = (x + bias) * 2^23.
    # The fraction spills into the mantissa, approximating 2^f linearly;
    # 126.94269504 = 127 - 0.05730496 centers the linear error (~3%).
    x = tl.maximum(x, MIN_EXP2_INPUT)
    bits = ((x + 126.94269504) * 8388608.0).to(tl.int32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def exp2_minimax4(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (1.0000026226043701171875
                    + f * (0.6930034160614013671875
                           + f * (0.2414443492889404296875
                                  + f * (5.20092435181140899658203125e-2
                                         + f * 1.3535187579691410064697265625e-2))))


@triton.jit
def exp2_minimax5(x, seq_len):
    two_i, f = _split_exp2(x)
    return two_i * (0.999999940395355224609375
                    + f * (0.693152725696563720703125
                           + f * (0.2401557266712188720703125
                                  + f * (5.58212064206600189208984375e-2
                                         + f * (8.994684554636478424072265625e-3
                                                + f * 1.875563524663448333740234375e-3)))))


# Same numbering as kernels/triton_kernels.py, so results line up across kernels.
VERSION_NAMES = {0: "exact", 1: "taylor1", 2: "taylor2", 3: "taylor3", 4: "minimax1",
                 5: "minimax2", 6: "minimax3", 7: "lsq3", 8: "schraudolph",
                 9: "minimax4", 10: "minimax5"}


@triton.jit
def attn_seq_len(seq_len):
    """The decision tree's state function: the sequence length."""
    return seq_len


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=attn_seq_len,
    state_indices=[1],                  # state = seq_len
    thresholds=[2048],                  # short sequences exact, long ones minimax2
    decisions=[0, 5],
    transform_type="func_substitute",
    approx_kernels={1: exp2_taylor1, 2: exp2_taylor2, 3: exp2_taylor3,
                    4: exp2_minimax1, 5: exp2_minimax2, 6: exp2_minimax3,
                    7: exp2_lsq3, 8: exp2_schraudolph,
                    9: exp2_minimax4, 10: exp2_minimax5},
    thresholds_lower=[0],
    thresholds_upper=[16384],
    decision_values=list(VERSION_NAMES),
))
@triton.jit
def attn_exp2(x, seq_len):
    """2^x for a block of attention scores (knob value 0: exact)."""
    return tl.math.exp2(x)


@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                     seq_len, qk_scale,
                     HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One (batch, head): rows [pid_m*BLOCK_M, ...) of the output.

    Layout: contiguous (batch*heads, seq_len, HEAD_DIM). qk_scale already
    contains log2(e), so the loop uses exp2.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * seq_len * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        k = tl.load(k_ptr + base + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = attn_exp2(qk - m_ij[:, None], seq_len)          # the knob: one 2^x per score
        alpha = tl.math.exp2(m_i - m_ij)                    # exact: one 2^x per row

        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + base + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], acc.to(tl.float16))


def attention(q, k, v, sm_scale=None, BLOCK_M=64, BLOCK_N=64):
    """softmax(Q K^T * scale) V for contiguous fp16 (batch, heads, seq, head_dim) tensors."""
    batch, heads, seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    assert seq_len % BLOCK_M == 0 and seq_len % BLOCK_N == 0, "no masking in this kernel"
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    attention_kernel[grid](q, k, v, out, seq_len, sm_scale * 1.4426950408889634,
                           HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
