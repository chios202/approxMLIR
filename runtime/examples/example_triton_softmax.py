#!/usr/bin/env python3
"""Benchmark the approximate-exp softmax: error vs torch.softmax and runtime per knob.

The kernel, its knob, and the exp approximations are in kernels/triton_kernels.py
(see its docstring for the exp versions and their errors). The knob is a
decision tree on the row length; this script sets its `decisions` to compare
every version, then runs the tree's default choice.

Run on a GPU node with the plugin path set before Python starts:

    export TRITON_PASS_PLUGIN_PATH=/path/to/libApproxTritonPlugin.so
    python runtime/examples/example_triton_softmax.py
"""

import approx_runtime as ar  # first: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton

from kernels.triton_kernels import VERSION_NAMES, softmax, softmax_exp


def measure(x, reference):
    """(max |error|, mean |error|, time in ms) of softmax(x) against `reference`."""
    out = softmax(x)
    err = (out - reference).abs()
    ms = triton.testing.do_bench(lambda: softmax(x))
    return err.max().item(), err.mean().item(), ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32) * 4
    reference = torch.softmax(x, dim=1)

    tree = ar.get_config(softmax_exp)["decision_tree"]
    ar.enable_triton_approx(dump_directory="/tmp/approx_triton_softmax_dump")

    # The same exp version for every row length: decisions = [v, v].
    print(f"{'version':>7} {'name':>12} {'max |err|':>11} {'mean |err|':>11} {'time (ms)':>10}")
    for version, name in VERSION_NAMES.items():
        tree.decisions = [version, version]
        max_err, mean_err, ms = measure(x, reference)
        print(f"{version:>7} {name:>12} {max_err:>11.3e} {mean_err:>11.3e} {ms:>10.4f}")

    # The decision tree choosing at runtime: rows of 1024 columns are >= the
    # threshold, so they get decisions[1].
    tree.decisions = [0, 5]
    max_err, mean_err, ms = measure(x, reference)
    print(f"tree thresholds={tree.thresholds} decisions={tree.decisions}: "
          f"max |err| {max_err:.3e}, mean |err| {mean_err:.3e}, {ms:.4f} ms")
    ar.disable_triton_approx()

    torch.testing.assert_close(softmax(x), reference, rtol=1e-5, atol=1e-6)
    print("exact path after disable_triton_approx(): matches torch.softmax")


if __name__ == "__main__":
    main()
