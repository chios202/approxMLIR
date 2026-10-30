"""Compile annotated C++ with ApproxMLIR.

``compile_cpp_source``: C++ source -> cgeist (Polygeist) -> MLIR, then the
``// @approx`` comments in the source become approx annotations
(cpp/annotations.py), then approx-opt-cpp runs the CPP pipeline
(the explicit CPP pass list by default, see core/toolchain.py).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import subprocess
import tempfile
from pathlib import Path

from .annotations import parse_cpp_annotations, generate_cpp_annotation_mlir
from ..core.annotations import inject_annotations_text
from ..core.toolchain import (
    WorkloadType, ToolchainConfig, get_toolchain, with_pre_emit_transform)

__all__ = [
    "CgeistConfig",
    "compile_cpp_source",
]


@dataclass
class CgeistConfig:
    """Configuration for running cgeist."""

    cgeist_path: str = field(
        default_factory=lambda: os.environ.get("CGEIST_PATH", "cgeist")
    )
    resource_dir: Optional[str] = None
    include_dirs: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)


def compile_cpp_source(
    cpp_source: str,
    *,
    cgeist_config: Optional[CgeistConfig] = None,
    toolchain: Optional[ToolchainConfig] = None,
    emit: str = "transformed",
) -> str:
    """Run cgeist, inject annotations, and optionally run CPP approx-opt pipeline.

    Args:
        cpp_source: C/C++ source text with @approx comments
        cgeist_config: Optional cgeist configuration
        toolchain: Toolchain configuration (default: global config)

    Returns:
        MLIR text. If emit == "annotated", returns annotated MLIR before approx-opt.
        If emit == "transformed", returns MLIR after CPP approx-opt pipeline.
    """
    cgeist_config = cgeist_config or CgeistConfig()
    toolchain = toolchain or get_toolchain()

    if emit not in ("annotated", "transformed"):
        raise ValueError("emit must be 'annotated' or 'transformed'")

    with tempfile.NamedTemporaryFile(
        suffix=".c", mode="w", delete=False
    ) as f:
        f.write(cpp_source)
        f.flush()
        cpp_path = f.name

    try:
        cgeist_cmd = [cgeist_config.cgeist_path, "-O0", "-S", cpp_path]
        if cgeist_config.resource_dir:
            cgeist_cmd.append(f"-resource-dir={cgeist_config.resource_dir}")
        for inc in cgeist_config.include_dirs:
            cgeist_cmd.extend(["-I", inc])
        cgeist_cmd.extend(cgeist_config.extra_args)

        cgeist_result = subprocess.run(
            cgeist_cmd, capture_output=True, text=True
        )
        if cgeist_result.returncode != 0:
            raise RuntimeError(
                f"cgeist failed with code {cgeist_result.returncode}:\n"
                f"{cgeist_result.stderr}"
            )

        base_mlir = cgeist_result.stdout
        parsed_annotations = parse_cpp_annotations(cpp_source)
        annotations_text = generate_cpp_annotation_mlir(parsed_annotations)
        annotated = inject_annotations_text(base_mlir, annotations_text)
        if emit == "annotated":
            return annotated

        cpp_passes = with_pre_emit_transform(
            toolchain.get_pipeline(WorkloadType.CPP),
            any(a.transform_type == "func_substitute" for a in parsed_annotations))
        approx_opt_path = toolchain.get_opt_path(WorkloadType.CPP)

        with tempfile.NamedTemporaryFile(
            suffix=".mlir", mode="w", delete=False
        ) as mf:
            mf.write(annotated)
            mf.flush()
            mlir_path = mf.name

        try:
            cmd = [approx_opt_path, mlir_path] + [f"--{p}" for p in cpp_passes]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"approx-opt-cpp failed with code {result.returncode}:\n"
                    f"{result.stderr}"
                )
            return result.stdout
        finally:
            Path(mlir_path).unlink(missing_ok=True)
    finally:
        Path(cpp_path).unlink(missing_ok=True)
