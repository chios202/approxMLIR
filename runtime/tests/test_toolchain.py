"""Tests for toolchain configuration."""

from approx_runtime.core.toolchain import (
    ToolchainConfig, WorkloadType, with_pre_emit_transform)
from approx_runtime.core.knob import DecisionTree


def _mock_state(x):
    return x


def test_get_opt_path():
    tc = ToolchainConfig(ml_opt_path="ml-opt", cpp_opt_path="cpp-opt")
    assert tc.get_opt_path(WorkloadType.ML) == "ml-opt"
    assert tc.get_opt_path(WorkloadType.CPP) == "cpp-opt"


def test_get_pipeline_pre_emit_transform():
    tc = ToolchainConfig(
        ml_pipeline=["emit-approx"],
        cpp_pipeline=["emit-approx"],
    )
    config = {
        "decision_tree": DecisionTree(
            state_function=_mock_state,
            state_indices=[0],
            thresholds=[1],
            decisions=[0, 1],
            transform_type="func_substitute",
            approx_kernels={1: _mock_state},
        )
    }
    pipeline = tc.get_pipeline(WorkloadType.ML, config)
    assert pipeline[0] == "pre-emit-transform"
    assert "emit-approx" in pipeline


def test_get_pipeline_safety_contract_removes_legalize():
    tc = ToolchainConfig(
        ml_pipeline=["emit-approx", "legalize-to-stablehlo"],
        cpp_pipeline=["emit-approx"],
    )
    config = {"safety_contract": object()}
    pipeline = tc.get_pipeline(WorkloadType.ML, config)
    assert "legalize-to-stablehlo" not in pipeline


def test_get_pipeline_triton_uses_triton_pipeline():
    tc = ToolchainConfig(
        ml_pipeline=["emit-approx", "legalize-to-stablehlo"],
        cpp_pipeline=["emit-approx"],
        triton_pipeline=["emit-approx", "finalize-approx"],
    )
    pipeline = tc.get_pipeline(WorkloadType.TRITON, {})
    assert pipeline == ["emit-approx", "finalize-approx"]


def test_default_pipelines():
    tc = ToolchainConfig()
    assert tc.get_pipeline(WorkloadType.ML) == ["approx-pipeline=workload=ml"]
    assert tc.get_pipeline(WorkloadType.CPP)[0] == "emit-approx"
    assert tc.get_pipeline(WorkloadType.TRITON) == ["approx-triton-pipeline"]


def test_get_pipeline_safety_contract_disables_legalize_in_named_pipeline():
    tc = ToolchainConfig()
    pipeline = tc.get_pipeline(WorkloadType.ML, {"safety_contract": object()})
    assert pipeline == ["approx-pipeline=workload=ml legalize=false"]


def test_with_pre_emit_transform():
    assert with_pre_emit_transform(["emit-approx"], True) == ["pre-emit-transform", "emit-approx"]
    assert with_pre_emit_transform(["emit-approx"], False) == ["emit-approx"]
    assert with_pre_emit_transform(["pre-emit-transform", "emit-approx"], True) == [
        "pre-emit-transform", "emit-approx"]
    assert with_pre_emit_transform(["approx-triton-pipeline"], True) == ["approx-triton-pipeline"]
