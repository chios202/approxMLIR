"""Tests for core/annotations.py: MLIR annotation generation."""

from approx_runtime import (
    DecisionTree,
    SafetyContract,
    StaticTransform,
    generate_annotations,
    inject_annotations,
    collect_helper_functions,
    needs_pre_emit_transform,
)

from .mlir_check import as_module, assert_parses, filecheck


def get_state(x):
    return x


def checker(result, x):
    return True


def recover(x):
    return x


def approx_v1(x):
    return x


def approx_v2(x):
    return x * 2


def knob_config(decision_tree=None, safety_contract=None, static_transform=None):
    """A knob config dict, as @Knob stores it."""
    return {"decision_tree": decision_tree, "safety_contract": safety_contract,
            "static_transform": static_transform}


def tree(transform_type="loop_perforate", thresholds=(100,), state_indices=(0,), **kwargs):
    """A DecisionTree on get_state with decisions 0, 1, ... (one per region)."""
    if transform_type == "func_substitute":
        kwargs.setdefault("approx_kernels", {1: approx_v1})
    return DecisionTree(state_function=get_state, state_indices=list(state_indices),
                        thresholds=list(thresholds), decisions=list(range(len(thresholds) + 1)),
                        transform_type=transform_type, **kwargs)


SAFETY = SafetyContract(checker=checker, recover=recover)


class TestGenerateAnnotations:
    def test_decision_tree_annotation(self):
        config = knob_config(tree(thresholds=(100, 500), state_indices=(0, 1)))

        result = generate_annotations("my_func", config)

        filecheck(result, '''
            CHECK:      "approx.util.annotation.decision_tree"() <{
            CHECK-NEXT:   func_name = "my_func",
            CHECK-NEXT:   transform_type = "loop_perforate",
            CHECK-NEXT:   state_indices = array<i64: 0, 1>,
            CHECK-NEXT:   state_function = "get_state",
            CHECK-NEXT:   num_thresholds = 2 : i32,
            CHECK-NEXT:   thresholds_uppers = array<i32: 100>,
            CHECK-NEXT:   thresholds_lowers = array<i32: 0>,
            CHECK-NEXT:   decision_values = array<i32: 0, 1, 2>,
            CHECK-NEXT:   thresholds = array<i32: 100, 500>,
            CHECK-NEXT:   decisions = array<i32: 0, 1, 2>
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_safety_contract_annotation(self):
        config = knob_config(safety_contract=SAFETY)

        result = generate_annotations("my_func", config)

        filecheck(result, '''
            CHECK:      "approx.util.annotation.knob"() <{
            CHECK-NEXT:   func_name = "my_func"
            CHECK-NEXT: }> : () -> ()
            CHECK-NEXT: "approx.util.annotation.try"() <{
            CHECK-NEXT:   func_name = "my_func",
            CHECK-NEXT:   recover = "recover",
            CHECK-NEXT:   checker = "checker"
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_static_transform_annotation(self):
        config = knob_config(static_transform=StaticTransform("loop_perforate", 3))

        result = generate_annotations("kernel", config)

        filecheck(result, '''
            CHECK:      "approx.util.annotation.knob"() <{
            CHECK-NEXT:   func_name = "kernel"
            CHECK-NEXT: }> : () -> ()
            CHECK-NEXT: "approx.util.annotation.transform"() <{
            CHECK-NEXT:   func_name = "kernel",
            CHECK-NEXT:   transform_type = "loop_perforate",
            CHECK-NEXT:   knob_val = 3 : i32
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_combined_annotations(self):
        config = knob_config(tree(), safety_contract=SAFETY)

        result = generate_annotations("my_func", config)

        # A decision tree creates the knob itself, so no separate knob annotation.
        filecheck(result, '''
            CHECK:      "approx.util.annotation.decision_tree"() <{
            CHECK-NEXT:   func_name = "my_func",
            CHECK-NEXT:   transform_type = "loop_perforate",
            CHECK-NEXT:   state_indices = array<i64: 0>,
            CHECK-NEXT:   state_function = "get_state",
            CHECK-NEXT:   num_thresholds = 1 : i32,
            CHECK-NEXT:   thresholds_uppers = array<i32: 100>,
            CHECK-NEXT:   thresholds_lowers = array<i32: 0>,
            CHECK-NEXT:   decision_values = array<i32: 0, 1>,
            CHECK-NEXT:   thresholds = array<i32: 100>,
            CHECK-NEXT:   decisions = array<i32: 0, 1>
            CHECK-NEXT: }> : () -> ()
            CHECK-NEXT: "approx.util.annotation.try"() <{
            CHECK-NEXT:   func_name = "my_func",
            CHECK-NEXT:   recover = "recover",
            CHECK-NEXT:   checker = "checker"
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_empty_config(self):
        config = knob_config()

        result = generate_annotations("my_func", config)
        assert result == ""

    def test_func_substitute_emits_convert_to_call(self):
        config = knob_config(tree("func_substitute"))

        result = generate_annotations("my_func", config)

        filecheck(result, '''
            CHECK:      "approx.util.annotation.convert_to_call"() <{
            CHECK-NEXT:   func_name = "my_func"
            CHECK-NEXT: }> : () -> ()
            CHECK-NEXT: "approx.util.annotation.decision_tree"() <{
            CHECK-NEXT:   func_name = "my_func",
            CHECK-NEXT:   transform_type = "func_substitute",
            CHECK-NEXT:   state_indices = array<i64: 0>,
            CHECK-NEXT:   state_function = "get_state",
            CHECK-NEXT:   num_thresholds = 1 : i32,
            CHECK-NEXT:   thresholds_uppers = array<i32: 100>,
            CHECK-NEXT:   thresholds_lowers = array<i32: 0>,
            CHECK-NEXT:   decision_values = array<i32: 0, 1>,
            CHECK-NEXT:   thresholds = array<i32: 100>,
            CHECK-NEXT:   decisions = array<i32: 0, 1>
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_state_function_name_override(self):
        config = knob_config(tree("func_substitute", thresholds=(512,), state_indices=(1,)))

        result = generate_annotations("__main__.my_exp__fp32S128S_i32", config,
                                      state_function_name="__main__.row_len__i32")

        filecheck(result, '''
            CHECK:      "approx.util.annotation.decision_tree"() <{
            CHECK-NEXT:   func_name = "__main__.my_exp__fp32S128S_i32",
            CHECK-NEXT:   transform_type = "func_substitute",
            CHECK-NEXT:   state_indices = array<i64: 1>,
            CHECK-NEXT:   state_function = "__main__.row_len__i32",
        ''', match_full_lines=True)
        assert_parses(as_module(result))


    def test_checker_and_recover_name_override(self):
        config = knob_config(safety_contract=SAFETY)

        result = generate_annotations("pkg.f__fp32", config,
                                      checker_name="pkg.checker__fp32_fp32",
                                      recover_name="pkg.recover__fp32")

        filecheck(result, '''
            CHECK:      "approx.util.annotation.try"() <{
            CHECK-NEXT:   func_name = "pkg.f__fp32",
            CHECK-NEXT:   recover = "pkg.recover__fp32",
            CHECK-NEXT:   checker = "pkg.checker__fp32_fp32"
            CHECK-NEXT: }> : () -> ()
        ''', match_full_lines=True)
        assert_parses(as_module(result))


class TestCollectHelperFunctions:
    def test_collect_state_function(self):
        config = knob_config(tree())

        helpers = collect_helper_functions("my_func", config)

        assert helpers == [("get_state", get_state)]

    def test_collect_safety_contract_functions(self):
        config = knob_config(safety_contract=SAFETY)

        helpers = collect_helper_functions("my_func", config)

        assert helpers == [("checker", checker), ("recover", recover)]

    def test_collect_approx_kernels(self):
        config = knob_config(tree("func_substitute", thresholds=(100, 200),
                                  approx_kernels={1: approx_v1, 2: approx_v2}))

        helpers = collect_helper_functions("my_kernel", config)

        assert helpers == [
            ("get_state", get_state),
            ("approx_my_kernel_1", approx_v1),
            ("approx_my_kernel_2", approx_v2),
        ]

    def test_collect_static_approx_kernel(self):
        config = knob_config(static_transform=StaticTransform("func_substitute", 1, approx_kernel=approx_v1))

        helpers = collect_helper_functions("my_func", config)

        assert helpers == [("approx_my_func_1", approx_v1)]


class TestNeedsPreEmitTransform:
    def test_loop_perforate_no_pre_emit(self):
        config = knob_config(tree())

        assert needs_pre_emit_transform(config) is False

    def test_func_substitute_needs_pre_emit(self):
        config = knob_config(tree("func_substitute"))

        assert needs_pre_emit_transform(config) is True

    def test_static_func_substitute_needs_pre_emit(self):
        config = knob_config(static_transform=StaticTransform("func_substitute", 1, approx_kernel=approx_v1))

        assert needs_pre_emit_transform(config) is True


class TestInjectAnnotations:
    def test_inject_into_module(self):
        mlir = '''module {
  func.func @my_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}'''
        config = knob_config(static_transform=StaticTransform("loop_perforate", 2))

        result = inject_annotations(mlir, "my_func", config)

        filecheck(result, '''
            CHECK:      module {
            CHECK-NEXT:   "approx.util.annotation.knob"() <{
            CHECK-NEXT:     func_name = "my_func"
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   "approx.util.annotation.transform"() <{
            CHECK-NEXT:     func_name = "my_func",
            CHECK-NEXT:     transform_type = "loop_perforate",
            CHECK-NEXT:     knob_val = 2 : i32
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   func.func @my_func(%arg0: i32) -> i32 {
            CHECK-NEXT:     return %arg0 : i32
            CHECK-NEXT:   }
            CHECK-NEXT: }
        ''', match_full_lines=True)
        assert_parses(result)

    def test_inject_into_named_module(self):
        mlir = '''module @mymodule {
  func.func @kernel() {
    return
  }
}'''
        config = knob_config(static_transform=StaticTransform("task_skipping", 1))

        result = inject_annotations(mlir, "kernel", config)

        filecheck(result, '''
            CHECK:      module @mymodule {
            CHECK-NEXT:   "approx.util.annotation.knob"() <{
            CHECK-NEXT:     func_name = "kernel"
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   "approx.util.annotation.transform"() <{
            CHECK-NEXT:     func_name = "kernel",
            CHECK-NEXT:     transform_type = "task_skipping",
            CHECK-NEXT:     knob_val = 1 : i32
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   func.func @kernel() {
            CHECK-NEXT:     return
            CHECK-NEXT:   }
            CHECK-NEXT: }
        ''', match_full_lines=True)
        assert_parses(result)

    def test_inject_empty_config(self):
        mlir = '''module {
  func.func @f() { return }
}'''

        result = inject_annotations(mlir, "f", None)
        assert result == mlir

        result = inject_annotations(mlir, "f", {})
        assert result == mlir

    def test_full_decision_tree_injection(self):
        mlir = '''module {
  func.func @process(%n: i32, %data: tensor<1000xf32>) -> tensor<1000xf32> {
    return %data : tensor<1000xf32>
  }
}'''
        config = knob_config(tree(thresholds=(100, 500),
                                  thresholds_lower=[0], thresholds_upper=[1000]))

        result = inject_annotations(mlir, "process", config)

        filecheck(result, '''
            CHECK:      module {
            CHECK-NEXT:   "approx.util.annotation.decision_tree"() <{
            CHECK-NEXT:     func_name = "process",
            CHECK-NEXT:     transform_type = "loop_perforate",
            CHECK-NEXT:     state_indices = array<i64: 0>,
            CHECK-NEXT:     state_function = "get_state",
            CHECK-NEXT:     num_thresholds = 2 : i32,
            CHECK-NEXT:     thresholds_uppers = array<i32: 1000>,
            CHECK-NEXT:     thresholds_lowers = array<i32: 0>,
            CHECK-NEXT:     decision_values = array<i32: 0, 1, 2>,
            CHECK-NEXT:     thresholds = array<i32: 100, 500>,
            CHECK-NEXT:     decisions = array<i32: 0, 1, 2>
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   func.func @process(%n: i32, %data: tensor<1000xf32>) -> tensor<1000xf32> {
        ''', match_full_lines=True)
        assert_parses(result)

    def test_inject_into_jax_module_with_attributes(self):
        mlir = '''#loc1 = loc("data")
module @jit_my_kernel attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32} {
  func.func public @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    return %arg0 : tensor<1024xf32>
  }
}'''
        config = knob_config(static_transform=StaticTransform("loop_perforate", 2))

        result = inject_annotations(mlir, "my_func", config)

        # Annotations go inside the module, after its attribute dictionary.
        filecheck(result, '''
            CHECK:      #loc1 = loc("data")
            CHECK-NEXT: module @jit_my_kernel attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32} {
            CHECK-NEXT:   "approx.util.annotation.knob"() <{
            CHECK-NEXT:     func_name = "my_func"
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   "approx.util.annotation.transform"() <{
            CHECK-NEXT:     func_name = "my_func",
            CHECK-NEXT:     transform_type = "loop_perforate",
            CHECK-NEXT:     knob_val = 2 : i32
            CHECK-NEXT:   }> : () -> ()
            CHECK-NEXT:   func.func public @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
        ''', match_full_lines=True)
        assert_parses(result)
