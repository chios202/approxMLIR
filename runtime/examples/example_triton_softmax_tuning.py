#!/usr/bin/env python3
"""Auto-tune the decision tree of the approximate-exp softmax with OpenTuner.
"""

import approx_runtime  # first: puts the plugin on TRITON_PLUGIN_PATHS

import torch
import triton

from kernels.triton_kernels import VERSION_NAMES, softmax, softmax_exp

ACCURACY_THRESHOLD = 0.999
TUNING_TIME_BUDGET_SECONDS = 120


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    input_matrices = [torch.randn(4096, column_count, device="cuda", dtype=torch.float32) * 4
                      for column_count in (256, 2048)]
    reference_outputs = [torch.softmax(matrix, dim=1) for matrix in input_matrices]

    def evaluate(configuration):
        """Runs the kernel on both matrices. Returns (total time in ms, accuracy)."""
        total_time_milliseconds = 0.0
        worst_relative_error = 0.0
        for input_matrix, reference_output in zip(input_matrices, reference_outputs):
            output = softmax(input_matrix)
            relative_error = ((output - reference_output).abs().max()
                              / reference_output.abs().max()).item()
            worst_relative_error = max(worst_relative_error, relative_error)
            total_time_milliseconds += triton.testing.do_bench(lambda: softmax(input_matrix))
        return total_time_milliseconds, 1.0 - worst_relative_error

    all_results = []
    approx_runtime.enable_triton_approx()
    tuning_result = approx_runtime.tune_triton(
        softmax_exp, evaluate,
        accuracy_threshold=ACCURACY_THRESHOLD,
        time_budget=TUNING_TIME_BUDGET_SECONDS,
        result_callback=lambda configuration, time_milliseconds, accuracy:
            all_results.append((time_milliseconds, accuracy, dict(configuration))),
    )

    decision_tree = approx_runtime.get_config(softmax_exp)["decision_tree"]
    approx_runtime.apply_config_to_tree(decision_tree, "softmax_exp",
                                        tuning_result["best_config"])
    decision_names = [VERSION_NAMES[decision] for decision in decision_tree.decisions]
    print(f"best: thresholds={decision_tree.thresholds} "
          f"decisions={decision_tree.decisions} {decision_names}")
    print(f"      time {tuning_result['best_time']:.4f} ms, "
          f"accuracy {tuning_result['best_accuracy']:.6f}")

    # Pareto frontier: the configurations that no other configuration beats in
    # both time and accuracy. Walk the results from fastest to slowest; keep a
    # configuration only if it is more accurate than every faster one.
    frontier = []
    for time_milliseconds, accuracy, configuration in sorted(
            all_results, key=lambda result: (result[0], -result[1])):
        if not frontier or accuracy > frontier[-1][1]:
            frontier.append((time_milliseconds, accuracy, configuration))
    print(f"\nPareto frontier ({len(frontier)} of {len(all_results)} configurations):")
    for time_milliseconds, accuracy, configuration in frontier:
        print(f"  {time_milliseconds:.4f} ms  accuracy {accuracy:.6f}  {configuration}")
    approx_runtime.disable_triton_approx()


if __name__ == "__main__":
    main()
