#!/usr/bin/env python3
"""Benchmark the approximate-exp2 fused attention: error and time per version.

For each sequence length, runs every exp2 version (tree decisions [v, v]) and
measures time (triton.testing.do_bench) and error against
torch.nn.functional.scaled_dot_product_attention computed in float32.
PyTorch's own fused attention (fp16) is timed as context.

Run on a GPU node with the plugin path set before Python starts:
    export TRITON_PASS_PLUGIN_PATH=/path/to/libApproxTritonPlugin.so
    python runtime/examples/example_triton_attention.py
"""

import approx_runtime as ar  # first: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton

from kernels.attention_kernels import VERSION_NAMES, attention, attn_exp2

BATCH, HEADS, HEAD_DIM = 4, 16, 64
SEQ_LENS = (1024, 4096)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    tree = ar.get_config(attn_exp2)["decision_tree"]
    ar.enable_triton_approx()

    for seq_len in SEQ_LENS:
        shape = (BATCH, HEADS, seq_len, HEAD_DIM)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float())
        torch_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

        print(f"\nbatch={BATCH} heads={HEADS} seq_len={seq_len} head_dim={HEAD_DIM}   "
              f"torch fused attention (fp16): {torch_ms:.4f} ms")
        print(f"{'version':>7} {'name':>9} {'max |err|':>11} {'mean |err|':>11} "
              f"{'time (ms)':>10} {'vs exact':>9}")
        exact_ms = None
        for version, name in VERSION_NAMES.items():
            tree.decisions = [version, version]
            out = attention(q, k, v)
            err = (out.float() - reference).abs()
            ms = triton.testing.do_bench(lambda: attention(q, k, v))
            if version == 0:
                exact_ms = ms
            print(f"{version:>7} {name:>9} {err.max().item():>11.3e} "
                  f"{err.mean().item():>11.3e} {ms:>10.4f} {ms / exact_ms:>8.3f}x")

    tree.decisions = [0, 2]
    ar.disable_triton_approx()


if __name__ == "__main__":
    main()
