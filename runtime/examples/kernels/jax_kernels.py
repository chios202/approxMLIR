"""Annotated JAX kernel shared by the JAX examples and tests.

`my_kernel` computes `data * 2 + sin(data)`. Its decision tree picks, from the
runtime state `n` (via `get_state`), one of:

    decision 0: my_kernel itself (exact)
    decision 1: approx_my_kernel_1, drops sin()          (n in [100, 500))
    decision 2: approx_my_kernel_2, drops sin(), * 1.9   (n >= 500)

The `n * 0` terms keep `n` as a real input of every function, so all of them
share the signature (n, data) required by func_substitute.

Used by: examples/example_wo_tuning.py, examples/example_w_tuning.py,
tests/test_annotate_api.py.
"""

import jax
import jax.numpy as jnp

import approx_runtime as ar

N_SHAPE = jax.ShapeDtypeStruct((), jnp.int32)
DATA_SHAPE = jax.ShapeDtypeStruct((1024,), jnp.float32)
ARG_SHAPES = (N_SHAPE, DATA_SHAPE)


def get_state(n: jax.Array) -> jax.Array:
    """The decision tree's state function: its result is compared to the thresholds."""
    return n


def approx_my_kernel_1(n: jax.Array, data: jax.Array) -> jax.Array:
    """Approximation 1: skip sin()."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


def approx_my_kernel_2(n: jax.Array, data: jax.Array) -> jax.Array:
    """Approximation 2: skip sin() and scale by 1.9 (cheaper, less accurate)."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.9 + dummy


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100, 500],
    decisions=[0, 1, 2],
    transform_type="func_substitute",
    approx_kernels={1: approx_my_kernel_1, 2: approx_my_kernel_2},
    # Search bounds for the auto-tuner (example_w_tuning.py).
    thresholds_lower=[0, 100],
    thresholds_upper=[500, 1000],
))
def my_kernel(n: jax.Array, data: jax.Array) -> jax.Array:
    """The exact computation (knob value 0): data * 2 + sin(data)."""
    dummy = (n * 0).astype(data.dtype)
    return (data * 2.0 + jnp.sin(data)) + dummy


# Every function the exported MLIR module must contain (the kernel, its state
# function and both approximations), with example argument shapes, in the form
# ar.export_module() takes.
EXPORT_FUNCTIONS = {
    "my_kernel": (my_kernel, ARG_SHAPES),
    "get_state": (get_state, (N_SHAPE,)),
    "approx_my_kernel_1": (approx_my_kernel_1, ARG_SHAPES),
    "approx_my_kernel_2": (approx_my_kernel_2, ARG_SHAPES),
}
