"""Tests for jax/export.py - JAX to MLIR export."""

import pytest
from approx_runtime import get_function_name
from approx_runtime.jax.export import JAX_AVAILABLE, _extract_and_rename_func

from .mlir_check import as_module, assert_parses, filecheck

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


class TestGetFunctionName:
    def test_simple_function(self):
        def my_func(x):
            return x

        assert get_function_name(my_func) == "my_func"

    def test_lambda(self):
        f = lambda x: x
        assert get_function_name(f) == "<lambda>"


class TestExtractAndRenameFunc:

    def test_rename_simple_func(self):
        mlir = '''module @test {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}'''
        result = _extract_and_rename_func(mlir, "my_kernel")

        filecheck(result, '''
            CHECK:      func.func @my_kernel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            CHECK-NEXT:   return %arg0 : tensor<4xf32>
            CHECK-NEXT: }
        ''', match_full_lines=True)
        assert_parses(as_module(result))

    def test_rename_with_operations(self):
        mlir = '''module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f32>
    return %0 : tensor<f32>
  }
}'''
        result = _extract_and_rename_func(mlir, "scale_fn")

        filecheck(result, '''
            CHECK:      func.func @scale_fn(%arg0: tensor<f32>) -> tensor<f32> {
            CHECK-NEXT:   %cst = stablehlo.constant dense<2.0> : tensor<f32>
            CHECK-NEXT:   %0 = stablehlo.multiply %arg0, %cst : tensor<f32>
            CHECK-NEXT:   return %0 : tensor<f32>
            CHECK-NEXT: }
        ''', match_full_lines=True)
        assert_parses(as_module(result))


class TestExportToMlir:
    def test_export_simple_function(self):
        import jax.numpy as jnp
        from approx_runtime import export_to_mlir

        def add(x, y):
            return x + y

        mlir = export_to_mlir(add, jnp.array(1.0), jnp.array(2.0))

        filecheck(mlir, '''
            CHECK:      module @jit_add
            CHECK-NEXT:   func.func public @main(%[[X:arg[0-9]+]]: tensor<f32>{{.*}}, %[[Y:arg[0-9]+]]: tensor<f32>{{.*}}) -> (tensor<f32>
            CHECK-NEXT:     %[[SUM:[0-9]+]] = stablehlo.add %[[X]], %[[Y]] : tensor<f32>
            CHECK-NEXT:     return %[[SUM]] : tensor<f32>
            CHECK-NEXT:   }
        ''')
        assert_parses(mlir)

    def test_export_with_knob_decorator(self):
        import jax.numpy as jnp
        from approx_runtime import export_to_mlir, Knob, StaticTransform

        @Knob(static_transform=StaticTransform("loop_perforate", 2))
        def my_kernel(x):
            return jnp.sum(x)

        mlir = export_to_mlir(my_kernel, jnp.zeros(100))

        # The decorator must not change the exported code (no annotations yet).
        filecheck(mlir, '''
            CHECK:      module @jit_my_kernel
            CHECK-NEXT:   func.func public @main(%[[X:arg[0-9]+]]: tensor<100xf32>{{.*}}) -> (tensor<f32>
            CHECK:          stablehlo.reduce(%[[X]] init: %{{.*}}) applies stablehlo.add across dimensions = [0] : (tensor<100xf32>, tensor<f32>) -> tensor<f32>
        ''')
        assert "approx." not in mlir
        assert_parses(mlir)


class TestExportModule:
    def test_export_single_function(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module

        def my_func(x):
            return x * 2.0

        functions = {
            "my_func": (my_func, (jax.ShapeDtypeStruct((4,), jnp.float32),))
        }
        mlir = export_module(functions)

        filecheck(mlir, '''
            CHECK:      module {
            CHECK-NEXT:   func.func @my_func(%[[X:arg[0-9]+]]: tensor<4xf32>{{.*}}) -> (tensor<4xf32>
            CHECK-NEXT:     %[[TWO:cst]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
            CHECK-NEXT:     %[[TWOS:[0-9]+]] = stablehlo.broadcast_in_dim %[[TWO]], dims = [] : (tensor<f32>) -> tensor<4xf32>
            CHECK-NEXT:     %[[PRODUCT:[0-9]+]] = stablehlo.multiply %[[X]], %[[TWOS]] : tensor<4xf32>
            CHECK-NEXT:     return %[[PRODUCT]] : tensor<4xf32>
            CHECK-NEXT:   }
            CHECK-NEXT: }
        ''')
        assert "@main" not in mlir
        assert_parses(mlir)

    def test_export_multiple_functions(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module

        def kernel(data):
            return data * 2.0

        def get_state(n):
            return n

        functions = {
            "my_kernel": (kernel, (jax.ShapeDtypeStruct((1024,), jnp.float32),)),
            "get_state": (get_state, (jax.ShapeDtypeStruct((), jnp.int32),)),
        }
        mlir = export_module(functions)

        # Both functions, renamed from @main, in one module, in the given order.
        filecheck(mlir, '''
            CHECK:      module {
            CHECK-NEXT:   func.func @my_kernel(%{{arg[0-9]+}}: tensor<1024xf32>{{.*}}) -> (tensor<1024xf32>
            CHECK:          return %{{[0-9]+}} : tensor<1024xf32>
            CHECK-NEXT:   }
            CHECK-NEXT:   func.func @get_state(%[[N:arg[0-9]+]]: tensor<i32>{{.*}}) -> (tensor<i32>
            CHECK-NEXT:     return %[[N]] : tensor<i32>
            CHECK-NEXT:   }
            CHECK-NEXT: }
        ''')
        assert "@main" not in mlir
        assert_parses(mlir)

    def test_export_empty_returns_empty_module(self):
        from approx_runtime import export_module

        mlir = export_module({})
        assert mlir == "module {}\n"
        assert_parses(mlir)

    def test_export_with_module_name(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module

        def f(x):
            return x

        functions = {"f": (f, (jax.ShapeDtypeStruct((), jnp.float32),))}
        mlir = export_module(functions, module_name="my_module")

        filecheck(mlir, '''
            CHECK:      module @my_module {
            CHECK-NEXT:   func.func @f(%[[X:arg[0-9]+]]: tensor<f32>{{.*}}) -> (tensor<f32>
            CHECK-NEXT:     return %[[X]] : tensor<f32>
            CHECK-NEXT:   }
            CHECK-NEXT: }
        ''')
        assert_parses(mlir)
