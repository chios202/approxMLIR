"""FileCheck-style checks for MLIR text produced by the Python runtime.

- ``filecheck(text, checks)``: run LLVM's FileCheck on `text`. `checks` holds
  ``CHECK:`` / ``CHECK-NEXT:`` / ``CHECK-SAME:`` lines exactly as in a lit test.
  With ``match_full_lines=True`` every checked line must match completely, so a
  missing comma, type, or bracket fails.
- ``assert_parses(text)``: run ``approx-opt`` on `text`, which parses and
  verifies it; any syntax error fails the test. Returns the printed module.

Tools are found via the FILECHECK / APPROX_OPT_ML environment variables, then
PATH, then the usual build locations. Tests skip when a tool is missing.
"""

import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[2]  # approxMLIR/


def _find_tool(env_var: str, name: str, candidates: List[Path]) -> Optional[str]:
    """Path to a tool: from `env_var`, else on PATH, else the first existing `candidates`."""
    path = os.environ.get(env_var)
    if path and Path(path).is_file():
        return path
    found = shutil.which(name)
    if found:
        return found
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


FILECHECK = _find_tool("FILECHECK", "FileCheck", [
    _REPO / "llvm-project" / "build" / "bin" / "FileCheck",
    _REPO.parent / "llvm-project" / "build" / "bin" / "FileCheck",
])
APPROX_OPT = _find_tool("APPROX_OPT_ML", "approx-opt", [
    _REPO / "build" / "bin" / "approx-opt",
])


def filecheck(text: str, checks: str, match_full_lines: bool = False) -> None:
    """Fail the test unless `text` matches the FileCheck `checks`."""
    if FILECHECK is None:
        pytest.skip("FileCheck not found (set FILECHECK=/path/to/FileCheck)")
    with tempfile.NamedTemporaryFile("w", suffix=".check", delete=False) as f:
        f.write(textwrap.dedent(checks))
        check_path = f.name
    try:
        cmd = [FILECHECK, check_path, "--dump-input=fail"]
        if match_full_lines:
            cmd.append("--match-full-lines")
        result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    finally:
        Path(check_path).unlink(missing_ok=True)
    if result.returncode != 0:
        pytest.fail(f"FileCheck failed:\n{result.stderr}", pytrace=False)


def assert_parses(text: str) -> str:
    """Fail the test unless approx-opt parses and verifies `text`; return the printed module."""
    return _approx_opt(text, [], "could not parse the MLIR", show_input=True)


def run_passes(text: str, passes: List[str]) -> str:
    """Run approx-opt with `passes` on `text` and return the result."""
    return _approx_opt(text, [f"--{p}" for p in passes], f"{' '.join(passes)} failed")


def _approx_opt(text: str, args: List[str], failure: str, show_input: bool = False) -> str:
    """Run approx-opt with `args` on `text`; fail the test with `failure` if it exits non-zero."""
    if APPROX_OPT is None:
        pytest.skip("approx-opt not found (set APPROX_OPT_ML=/path/to/approx-opt)")
    result = subprocess.run([APPROX_OPT, "-"] + args, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        details = f"\n--- input ---\n{text}" if show_input else ""
        pytest.fail(f"approx-opt {failure}:\n{result.stderr}{details}", pytrace=False)
    return result.stdout


def as_module(ops: str) -> str:
    """Wrap bare annotation ops in a module so they can be parsed."""
    return "module {\n" + ops + "\n}\n"
