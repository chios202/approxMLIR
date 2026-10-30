"""Flash Attention v2 (Triton's reference implementation), with our exp2 knob.

The kernel is copied from Triton's 06-fused-attention.py (Flash Attention v2,
credits: OpenAI kernel team), forward pass only, trimmed to what
the A10 runs: float16 in/out, float32 accumulation, no FP8, no Hopper/Blackwell
warp specialization, no backward. Causal and non-causal both work. Variable
names follow the tutorial file so a diff against it stays readable.

Exactly ONE line of the compute changed: the tutorial's

    p = tl.math.exp2(qk)

became

    p = attn_exp2(qk, N_CTX)

where attn_exp2 is the SAME knob already used by our hand-written attention in
attention_kernels.py (versions: hardware exp2, minimax degree 1-3). Nothing
else was edited for ApproxMLIR - which is the point: a knob written once drops
into an existing production-style kernel.

The original @triton.autotune list is cut down to 4 block configurations so
that re-tuning stays cheap when ar.tune_triton recompiles per decision-tree
config.

Used by: examples/example_triton_flash_attention.py.
"""

import approx_runtime as ar  # before triton: puts the plugin on TRITON_PLUGIN_PATHS  # noqa: F401

import torch
import triton
import triton.language as tl

from kernels.attention_kernels import VERSION_NAMES, attn_exp2  # noqa: F401


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = attn_exp2(qk, N_CTX)  # tutorial: p = tl.math.exp2(qk) -- the knob
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([offsetv_y, 0])
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


# Triton's original sweeps 24+ configurations; this list keeps the 4 that matter on
# an A10, so ar.tune_triton (which recompiles per decision-tree config) stays cheap.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM, BN, s, w in [(64, 32, 2, 4), (64, 64, 2, 4), (128, 64, 2, 4), (128, 64, 3, 8)]
]


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    # Filter out configs where BLOCK_M > N_CTX, and BLOCK_M < BLOCK_N when causal.
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(tl.float16))


def flash_attention(q, k, v, causal=False, sm_scale=None):
    """Forward fused attention on (batch, heads, seq, head_dim) float16 tensors.

    This is the original launcher without the autograd class: it allocates the
    output, builds the grid from the autotuned BLOCK_M, and launches _attn_fwd
    with plain pointers (the kernel builds tensor descriptors on the device).
    """
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM == k.shape[-1] == v.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    if sm_scale is None:
        sm_scale = HEAD_DIM ** -0.5
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    _attn_fwd[grid](
        sm_scale, M,  #
        q.shape[0], q.shape[1],  #
        q, k, v, o,  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM,  #
        STAGE=stage,  #
    )
    return o
