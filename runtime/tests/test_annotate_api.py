"""Tests for jax/api.py non-invasive API."""

import sys
from pathlib import Path

import pytest

from approx_runtime.jax.api import with_config, export_with_config
from approx_runtime.core.knob import DecisionTree
from approx_runtime.jax.export import JAX_AVAILABLE

from .mlir_check import assert_parses, filecheck, run_passes


def _mock_state(x):
    return x


def test_with_config_attaches_config():
    def f(x):
        return x

    tree = DecisionTree(
        state_function=_mock_state,
        state_indices=[0],
        thresholds=[1],
        decisions=[0, 1],
        transform_type="loop_perforate",
    )
    wrapped = with_config(f, decision_tree=tree)
    assert wrapped._approx_config == {
        "decision_tree": tree,
        "safety_contract": None,
        "static_transform": None,
    }
    assert wrapped(3) == 3


def test_export_with_config_rejects_safety_contract():
    def f(x):
        return x

    config = {"safety_contract": object()}
    with pytest.raises(ValueError, match="does not infer shapes for safety_contract"):
        export_with_config(f, (1,), config)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_export_with_config_basic():
    import jax.numpy as jnp

    def f(x):
        return x + 1

    config = {
        "decision_tree": DecisionTree(
            state_function=_mock_state,
            state_indices=[0],
            thresholds=[1],
            decisions=[0, 1],
            transform_type="loop_perforate",
        )
    }

    mlir = export_with_config(f, (jnp.array(1),), config)

    # Annotation first, then the kernel, then its state function.
    filecheck(mlir, '''
        CHECK:      module {
        CHECK-NEXT:   "approx.util.annotation.decision_tree"() <{
        CHECK-NEXT:     func_name = "f",
        CHECK-NEXT:     transform_type = "loop_perforate",
        CHECK-NEXT:     state_indices = array<i64: 0>,
        CHECK-NEXT:     state_function = "_mock_state",
        CHECK-NEXT:     num_thresholds = 1 : i32,
        CHECK-NEXT:     thresholds_uppers = array<i32: 100>,
        CHECK-NEXT:     thresholds_lowers = array<i32: 0>,
        CHECK-NEXT:     decision_values = array<i32: 0, 1>,
        CHECK-NEXT:     thresholds = array<i32: 1>,
        CHECK-NEXT:     decisions = array<i32: 0, 1>
        CHECK-NEXT:   }> : () -> ()
        CHECK-NEXT:   func.func @f(%[[X:arg[0-9]+]]: tensor<i32>{{.*}}) -> (tensor<i32>
        CHECK-NEXT:     %[[ONE:c]] = stablehlo.constant dense<1> : tensor<i32>
        CHECK-NEXT:     %[[SUM:[0-9]+]] = stablehlo.add %[[X]], %[[ONE]] : tensor<i32>
        CHECK-NEXT:     return %[[SUM]] : tensor<i32>
        CHECK-NEXT:   }
        CHECK-NEXT:   func.func @_mock_state(%[[N:arg[0-9]+]]: tensor<i32>{{.*}}) -> (tensor<i32>
        CHECK-NEXT:     return %[[N]] : tensor<i32>
        CHECK-NEXT:   }
        CHECK-NEXT: }
    ''')
    assert_parses(mlir)


def _jax_kernels():
    """examples/kernels/jax_kernels.py, the kernel the JAX examples run."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from kernels import jax_kernels
    return jax_kernels


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_kernel_exported_with_annotations():
    import approx_runtime as ar

    jk = _jax_kernels()
    mlir = export_with_config(jk.my_kernel, jk.ARG_SHAPES, ar.get_config(jk.my_kernel))

    # Annotations, then the exact kernel, its state function, and both
    # approximations, all with the (n, data) signature func_substitute needs.
    filecheck(mlir, '''
        CHECK:      module {
        CHECK-NEXT:   "approx.util.annotation.convert_to_call"() <{
        CHECK-NEXT:     func_name = "my_kernel"
        CHECK-NEXT:   }> : () -> ()
        CHECK-NEXT:   "approx.util.annotation.decision_tree"() <{
        CHECK-NEXT:     func_name = "my_kernel",
        CHECK-NEXT:     transform_type = "func_substitute",
        CHECK-NEXT:     state_indices = array<i64: 0>,
        CHECK-NEXT:     state_function = "get_state",
        CHECK-NEXT:     num_thresholds = 2 : i32,
        CHECK-NEXT:     thresholds_uppers = array<i32: 500, 1000>,
        CHECK-NEXT:     thresholds_lowers = array<i32: 0, 100>,
        CHECK-NEXT:     decision_values = array<i32: 0, 1, 2>,
        CHECK-NEXT:     thresholds = array<i32: 100, 500>,
        CHECK-NEXT:     decisions = array<i32: 0, 1, 2>
        CHECK-NEXT:   }> : () -> ()

        CHECK-NEXT:   func.func @my_kernel(%[[N:arg0]]: tensor<i32>{{.*}}, %[[DATA:arg1]]: tensor<1024xf32>{{.*}}) -> (tensor<1024xf32>
        CHECK:          %[[SIN:[0-9]+]] = stablehlo.sine %[[DATA]] : tensor<1024xf32>
        CHECK:          return %{{[0-9]+}} : tensor<1024xf32>
        CHECK-NEXT:   }

        CHECK-NEXT:   func.func @get_state(%[[STATE_N:arg0]]: tensor<i32>{{.*}}) -> (tensor<i32>
        CHECK-NEXT:     return %[[STATE_N]] : tensor<i32>
        CHECK-NEXT:   }

        CHECK-NEXT:   func.func @approx_my_kernel_1(%{{arg0}}: tensor<i32>{{.*}}, %{{arg1}}: tensor<1024xf32>{{.*}}) -> (tensor<1024xf32>
        CHECK-NEXT:     %{{cst}} = stablehlo.constant dense<2.000000e+00> : tensor<f32>
        CHECK:          return
        CHECK-NEXT:   }

        CHECK-NEXT:   func.func @approx_my_kernel_2(%{{arg0}}: tensor<i32>{{.*}}, %{{arg1}}: tensor<1024xf32>{{.*}}) -> (tensor<1024xf32>
        CHECK-NEXT:     %{{cst}} = stablehlo.constant dense<1.900000e+00> : tensor<f32>
        CHECK:          return
        CHECK-NEXT:   }
        CHECK-NEXT: }
    ''')
    assert "stablehlo.sine" not in mlir.split("@approx_my_kernel_1")[1]
    assert_parses(mlir)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_kernel_through_pipeline():
    """The glued module goes through the full ML pass pipeline."""
    import approx_runtime as ar

    jk = _jax_kernels()
    config = ar.get_config(jk.my_kernel)
    mlir = export_with_config(jk.my_kernel, jk.ARG_SHAPES, config)

    out = run_passes(mlir, ar.get_pipeline_for_config(config))

    # state < 100 -> exact, 100 <= state < 500 -> approx 1, state >= 500 -> approx 2.
    filecheck(out, '''
        CHECK:      func.func @__internal_my_kernel(
        CHECK:        stablehlo.sine

        CHECK:      func.func @my_kernel(%[[N:arg0]]: tensor<i32>, %[[DATA:arg1]]: tensor<1024xf32>)
        CHECK-DAG:    %[[C500:[a-z_0-9]+]] = stablehlo.constant dense<500> : tensor<i32>
        CHECK-DAG:    %[[C100:[a-z_0-9]+]] = stablehlo.constant dense<100> : tensor<i32>
        CHECK:        %[[STATE:[0-9]+]] = call @get_state(%[[N]]) : (tensor<i32>) -> tensor<i32>
        CHECK-NEXT:   %{{[0-9]+}} = stablehlo.compare GE, %[[STATE]], %[[C100]]
        CHECK:        %{{[0-9]+}} = stablehlo.compare GE, %[[STATE]], %[[C500]]

        CHECK:        "stablehlo.case"(%{{[0-9]+}}) ({
        CHECK-NEXT:     func.call @__internal_my_kernel(%[[N]], %[[DATA]])
        CHECK:        }, {
        CHECK-NEXT:     func.call @approx_my_kernel_1(%[[N]], %[[DATA]])
        CHECK:        }, {
        CHECK-NEXT:     func.call @approx_my_kernel_2(%[[N]], %[[DATA]])
        CHECK:        }, {
        CHECK-NEXT:     func.call @__internal_my_kernel(%[[N]], %[[DATA]])
        CHECK:        }) : (tensor<i32>) -> tensor<1024xf32>
    ''')
    assert "approx." not in out
    assert_parses(out)
