#!/usr/bin/env python3
"""Benchmark + auto-tune the exp2 knob inside Flash Attention v2.

The kernel is kernels/flash_attention_kernels.py: Triton's reference Flash
Attention v2 (forward), where the one exp2 line is our attn_exp2 knob
(shared with the hand-written attention in attention_kernels.py).

Part 1 measures every exp2 version at two sequence lengths against
torch.nn.functional.scaled_dot_product_attention in float32.
Part 2 lets ar.tune_triton search the knob's decision tree (threshold on the
sequence length + one version per band) and prints the Pareto frontier.

Run on a GPU node with OpenTuner installed and the plugin path set before
Python starts:

    export TRITON_PASS_PLUGIN_PATH=/path/to/libApproxTritonPlugin.so
    python runtime/examples/example_triton_flash_attention.py
"""

import approx_runtime as ar  # first: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton

from kernels.flash_attention_kernels import (VERSION_NAMES, attn_exp2,
                                             flash_attention)

BATCH, HEADS, HEAD_DIM = 4, 16, 64
SEQUENCE_LENGTHS = (1024, 4096)
ACCURACY_THRESHOLD = 0.999
TUNING_TIME_BUDGET_SECONDS = 300


def make_inputs(sequence_length):
    q, k, v = (torch.randn(BATCH, HEADS, sequence_length, HEAD_DIM,
                           device="cuda", dtype=torch.float16) * 0.5
               for _ in range(3))
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=HEAD_DIM ** -0.5)
    return q, k, v, reference


def measure(inputs):
    """(max |error|, time in ms) of flash_attention against the fp32 reference."""
    q, k, v, reference = inputs
    out = flash_attention(q, k, v)
    max_error = (out.float() - reference).abs().max().item()
    time_ms = triton.testing.do_bench(lambda: flash_attention(q, k, v))
    return max_error, time_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    all_inputs = {n: make_inputs(n) for n in SEQUENCE_LENGTHS}

    tree = ar.get_config(attn_exp2)["decision_tree"]
    ar.enable_triton_approx()

    # Part 1: the same exp2 version for every sequence length: decisions = [v, v].
    for sequence_length, inputs in all_inputs.items():
        print(f"\nseq_len = {sequence_length}")
        print(f"{'version':>8} {'max |err|':>11} {'time (ms)':>10} {'vs exact':>9}")
        exact_time = None
        for version, name in VERSION_NAMES.items():
            tree.decisions = [version, version]
            max_error, time_ms = measure(inputs)
            exact_time = exact_time or time_ms
            print(f"{name:>8} {max_error:>11.3e} {time_ms:>10.4f} "
                  f"{time_ms / exact_time:>8.3f}x")

    # Part 2: OpenTuner searches the decision tree over both sequence lengths.
    def evaluate(configuration):
        total_time_milliseconds = 0.0
        worst_relative_error = 0.0
        for q, k, v, reference in all_inputs.values():
            out = flash_attention(q, k, v)
            relative_error = ((out.float() - reference).abs().max()
                              / reference.abs().max()).item()
            worst_relative_error = max(worst_relative_error, relative_error)
            total_time_milliseconds += triton.testing.do_bench(
                lambda: flash_attention(q, k, v))
        return total_time_milliseconds, 1.0 - worst_relative_error

    all_results = []
    tuning_result = ar.tune_triton(
        attn_exp2, evaluate,
        accuracy_threshold=ACCURACY_THRESHOLD,
        time_budget=TUNING_TIME_BUDGET_SECONDS,
        result_callback=lambda configuration, time_milliseconds, accuracy:
            all_results.append((time_milliseconds, accuracy, dict(configuration))),
    )

    ar.apply_config_to_tree(tree, "attn_exp2", tuning_result["best_config"])
    decision_names = [VERSION_NAMES[decision] for decision in tree.decisions]
    print(f"\nbest: thresholds={tree.thresholds} decisions={tree.decisions} {decision_names}")
    print(f"      time {tuning_result['best_time']:.4f} ms, "
          f"accuracy {tuning_result['best_accuracy']:.6f}")

    # Pareto frontier: the configurations that no other configuration beats in
    # both time and accuracy.
    frontier = []
    for time_milliseconds, accuracy, configuration in sorted(
            all_results, key=lambda result: (result[0], -result[1])):
        if not frontier or accuracy > frontier[-1][1]:
            frontier.append((time_milliseconds, accuracy, configuration))
    print(f"\nPareto frontier ({len(frontier)} of {len(all_results)} configurations):")
    for time_milliseconds, accuracy, configuration in frontier:
        print(f"  {time_milliseconds:.4f} ms  accuracy {accuracy:.6f}  {configuration}")
    ar.disable_triton_approx()


if __name__ == "__main__":
    main()
