"""Which tool runs the passes, and which pipeline runs, per workload type.

The pass order itself is defined once, in C++ (lib/approx/Passes/Pipelines.cpp):

    pre-emit-transform -> emit-approx -> emit-management -> config-approx
      -> transform-approx -> finalize-approx -> <host adapter>

and exposed as named pipelines: ``--approx-pipeline="workload=ml|cpp|triton"``
in approx-opt, ``approx-triton-pipeline`` in the Triton plugin. This module only
picks which one to run.

- ``WorkloadType``: ML (JAX/StableHLO via approx-opt -> IREE), CPP (Polygeist
  via approx-opt-cpp), TRITON (TTIR via the Triton plugin, no approx-opt).
- ``ToolchainConfig``: paths to the tools (env vars APPROX_OPT_ML, APPROX_OPT_CPP,
  TRITON_PASS_PLUGIN_PATH) and the default pipeline per workload. A pipeline is
  a list of pass names; the ML and TRITON defaults are one-element lists naming
  a C++ pipeline. CPP defaults to the explicit pass list, since its binary
  (polygeist-opt) is built outside this repo.
  ``get_pipeline(workload, config)`` adapts the pipeline to the config.
- ``with_pre_emit_transform(pipeline, needed)``: prepend pre-emit-transform to an
  explicit pass list when func_substitute needs it (named pipelines include it).
- ``get_toolchain()`` / ``set_toolchain()``: the process-wide default config.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import List, Optional

from .annotations import needs_pre_emit_transform

__all__ = [
    "WorkloadType",
    "ToolchainConfig",
    "get_toolchain",
    "set_toolchain",
    "with_pre_emit_transform",
    "is_named_pipeline",
    "approx_pipeline",
]

# Names registered in C++ (lib/approx/Passes/Pipelines.cpp and the Triton plugin).
APPROX_PIPELINE = "approx-pipeline"
APPROX_TRITON_PIPELINE = "approx-triton-pipeline"


class WorkloadType(Enum):
    """Type of workload for compiler selection."""

    ML = "ml"    # JAX/StableHLO -> IREE
    CPP = "cpp"  # C++/Polygeist -> native
    TRITON = "triton"  # Triton TTIR -> plugin pass manager


def approx_pipeline(workload: "WorkloadType") -> str:
    """The C++ named pipeline for `workload`, e.g. ``approx-pipeline=workload=ml``."""
    return f"{APPROX_PIPELINE}=workload={workload.value}"


def is_named_pipeline(pass_name: str) -> bool:
    """True for a C++ pipeline name (with or without options), not a single pass."""
    name = pass_name.split("=", 1)[0]
    return name in (APPROX_PIPELINE, APPROX_TRITON_PIPELINE)


def with_pre_emit_transform(pipeline: List[str], needed: bool) -> List[str]:
    """Prepend pre-emit-transform when `needed`, unless the pipeline already runs it.

    Named pipelines always run pre-emit-transform first, so they are returned
    unchanged; explicit pass lists get it prepended if it is missing.
    """
    pipeline = list(pipeline)
    if (not needed or "pre-emit-transform" in pipeline
            or any(is_named_pipeline(p) for p in pipeline)):
        return pipeline
    return ["pre-emit-transform"] + pipeline


@dataclass
class ToolchainConfig:
    """Configuration for ApproxMLIR toolchains."""

    ml_opt_path: str = field(
        default_factory=lambda: os.environ.get("APPROX_OPT_ML", "approx-opt")
    )
    cpp_opt_path: str = field(
        default_factory=lambda: os.environ.get("APPROX_OPT_CPP", "approx-opt-cpp")
    )
    triton_plugin_path: str = field(
        default_factory=lambda: os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    )

    ml_pipeline: List[str] = field(
        default_factory=lambda: [approx_pipeline(WorkloadType.ML)])
    # Explicit list: approx-opt-cpp is usually polygeist-opt, built outside this
    # repo, which may not register approx-pipeline. With approx-opt, use
    # ["approx-pipeline=workload=cpp"].
    cpp_pipeline: List[str] = field(
        default_factory=lambda: [
            "emit-approx", "emit-management", "config-approx",
            "transform-approx", "finalize-approx",
        ])
    triton_pipeline: List[str] = field(
        default_factory=lambda: [APPROX_TRITON_PIPELINE])

    def get_opt_path(self, workload: WorkloadType) -> str:
        """Get approx-opt path for workload type."""
        if workload == WorkloadType.ML:
            return self.ml_opt_path
        if workload == WorkloadType.CPP:
            return self.cpp_opt_path
        raise ValueError(
            "WorkloadType.TRITON does not use approx-opt; "
            "use plugin pass manager integration instead."
        )

    def get_pipeline(self, workload: WorkloadType, config: Optional[dict] = None) -> List[str]:
        """The pipeline for `workload`, adjusted for `config`.

        - func_substitute needs pre-emit-transform (named pipelines already run it).
        - A safety contract skips legalize-to-stablehlo: approx.try lowers to
          scf.if, which that pass does not handle.
        """
        pipelines = {WorkloadType.ML: self.ml_pipeline, WorkloadType.CPP: self.cpp_pipeline,
                     WorkloadType.TRITON: self.triton_pipeline}
        if workload not in pipelines:
            raise ValueError(f"Unsupported workload type: {workload}")
        config = config or {}
        pipeline = with_pre_emit_transform(pipelines[workload],
                                           bool(config) and needs_pre_emit_transform(config))
        if config.get("safety_contract"):
            pipeline = _without_stablehlo_legalization(pipeline)
        return pipeline


def _without_stablehlo_legalization(pipeline: List[str]) -> List[str]:
    """Drop legalize-to-stablehlo from an explicit list, or turn it off in the named ML pipeline."""
    named_ml = approx_pipeline(WorkloadType.ML)
    return [f"{p} legalize=false" if p.startswith(named_ml) else p
            for p in pipeline if p != "legalize-to-stablehlo"]


_DEFAULT_TOOLCHAIN = ToolchainConfig()


def get_toolchain() -> ToolchainConfig:
    """Get the current toolchain configuration."""
    return _DEFAULT_TOOLCHAIN


def set_toolchain(config: ToolchainConfig) -> None:
    """Set the global toolchain configuration."""
    global _DEFAULT_TOOLCHAIN
    _DEFAULT_TOOLCHAIN = config
