"""Tests for triton/frontend.py: @Knob on Triton device functions.

The softmax kernel and its exp approximations come from
examples/kernels/triton_kernels.py, the same code the GPU example runs.

Kernels are compiled with triton.compile for an explicit GPU target, which runs
Triton's full pipeline (including ptxas) without needing a GPU. Requires Triton
and the ApproxMLIR plugin (TRITON_PASS_PLUGIN_PATH or TRITON_PLUGIN_PATHS set
before the test session starts).
"""

import re
import sys
from pathlib import Path

import pytest

import approx_runtime as ar

from .mlir_check import filecheck

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

try:
    from triton._C.libtriton import passes as _triton_passes
    PLUGIN_LOADED = hasattr(_triton_passes.plugin, "emit-approx")
except Exception:  # pragma: no cover - Triton without the C extension
    PLUGIN_LOADED = False

pytestmark = pytest.mark.skipif(
    not PLUGIN_LOADED, reason="ApproxMLIR Triton plugin not loaded (set TRITON_PASS_PLUGIN_PATH)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from kernels import triton_kernels  # noqa: E402

TARGET = GPUTarget("cuda", 80, 32)
SOFTMAX_SIG = {"out_ptr": "*fp32", "in_ptr": "*fp32", "n_cols": "i32", "BLOCK": "constexpr"}
TREE = ar.get_config(triton_kernels.softmax_exp)["decision_tree"]
DEFAULT_DECISIONS = list(TREE.decisions)  # [0, 5]: exact below 512 columns, minimax2 from 512


@pytest.fixture(autouse=True)
def frontend_off_after_test(monkeypatch):
    """Compile from scratch in every test, then turn the frontend off and restore the tree."""
    # A kernel found in Triton's cache would skip the stages, so the hooks (and dumps) would not run.
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
    yield
    ar.disable_triton_approx()
    TREE.decisions = list(DEFAULT_DECISIONS)


def functions_in(ttir):
    """Names of all tt.func in the TTIR, in order."""
    return re.findall(r"tt\.func (?:public |private )?@([\w.$]+)", ttir)


def compile_kernel(kernel, signature=SOFTMAX_SIG, block=128):
    """Compile `kernel` for an A100-class target (sm_80) without launching it; returns the CompiledKernel."""
    return triton.compile(ASTSource(kernel, signature=signature, constexprs={"BLOCK": block}),
                          target=TARGET)


def compile_softmax(decisions, dump_directory=None):
    """Final TTIR of triton_kernels.softmax_kernel with the given tree decisions."""
    TREE.decisions = list(decisions)
    ar.enable_triton_approx(dump_directory=str(dump_directory) if dump_directory else None)
    return compile_kernel(triton_kernels.softmax_kernel).asm["ttir"]


# Softmax with its decision tree (exact below 512 columns, minimax2 from 512)

def compile_softmax_stages(tmp_path):
    """(annotated TTIR, TTIR after the approx passes, final TTIR) for the default tree."""
    final = compile_softmax(DEFAULT_DECISIONS, dump_directory=tmp_path)
    annotated = next(tmp_path.glob("*.annotated.ttir")).read_text()
    after_passes = next(tmp_path.glob("*.approx.ttir")).read_text()
    return annotated, after_passes, final


def test_softmax_annotations_injected(tmp_path):
    annotated, _, _ = compile_softmax_stages(tmp_path)

    filecheck(annotated, """
        CHECK:      module {
        CHECK-NEXT:   "approx.util.annotation.convert_to_call"() <{
        CHECK-NEXT:     func_name = "[[EXP:kernels.triton_kernels.softmax_exp__fp32S128S_i32]]"
        CHECK-NEXT:   }> : () -> ()
        CHECK-NEXT:   "approx.util.annotation.decision_tree"() <{
        CHECK-NEXT:     func_name = "[[EXP]]",
        CHECK-NEXT:     transform_type = "func_substitute",
        CHECK-NEXT:     state_indices = array<i64: 1>,
        CHECK-NEXT:     state_function = "[[STATE:kernels.triton_kernels.row_length__i32]]",
        CHECK-NEXT:     num_thresholds = 1 : i32,
        CHECK-NEXT:     thresholds_uppers = array<i32: 8192>,
        CHECK-NEXT:     thresholds_lowers = array<i32: 0>,
        CHECK-NEXT:     decision_values = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>,
        CHECK-NEXT:     thresholds = array<i32: 512>,
        CHECK-NEXT:     decisions = array<i32: 0, 5>
        CHECK-NEXT:   }> : () -> ()
        CHECK-NEXT:   tt.func public @softmax_kernel(%{{out_ptr}}: !tt.ptr<f32>{{.*}}, %{{in_ptr}}: !tt.ptr<f32>{{.*}}, %[[N_COLS:n_cols]]: i32

        CHECK:        %[[MAX:[a-z_0-9]+]] = tt.call @triton.language.standard.max__fp32S128S{{.*}}(%[[ROW:[a-z_0-9]+]])
        CHECK-NEXT:   %[[MAX_SPLAT:[a-z_0-9]+]] = tt.splat %[[MAX]] : f32 -> tensor<128xf32>
        CHECK-NEXT:   %[[SHIFTED:[a-z_0-9]+]] = arith.subf %[[ROW]], %[[MAX_SPLAT]] : tensor<128xf32>
        CHECK-NEXT:   %{{[a-z_0-9]+}} = tt.call @[[EXP]](%[[SHIFTED]], %[[N_COLS]]) : (tensor<128xf32>, i32) -> tensor<128xf32>

        CHECK:      tt.func private @[[EXP]](%[[EXACT_ARG:x]]: tensor<128xf32>{{.*}}, %{{n}}: i32
        CHECK-NEXT:   %[[EXACT:[0-9]+]] = math.exp %[[EXACT_ARG]] : tensor<128xf32>
        CHECK-NEXT:   tt.return %[[EXACT]] : tensor<128xf32>

        CHECK:      tt.func private @approx_[[EXP]]_5(%[[APPROX_X:x]]: tensor<128xf32>{{.*}}, %{{n}}: i32
        CHECK-NEXT:   %[[RESULT:[0-9]+]] = tt.call @kernels.triton_kernels.exp_minimax2__fp32S128S(%[[APPROX_X]])
        CHECK-NEXT:   tt.return %[[RESULT]] : tensor<128xf32>

        CHECK:      tt.func private @[[STATE]](%[[STATE_N:n]]: i32
        CHECK-NEXT:   tt.return %[[STATE_N]] : i32
    """)
    # One approximate version per decision value the tuner may pick.
    approx_functions = {f for f in functions_in(annotated) if f.startswith("approx_")}
    assert approx_functions == {f"approx_kernels.triton_kernels.softmax_exp__fp32S128S_i32_{v}"
                                for v in range(1, 11)}


def test_softmax_after_approx_passes(tmp_path):
    _, after_passes, _ = compile_softmax_stages(tmp_path)

    # The knob function now computes the region from the row length and calls
    # the exact body (region 0, and the default) or approximation 5 (region 1).
    filecheck(after_passes, """
        CHECK:      tt.func private @[[EXP:kernels.triton_kernels.softmax_exp__fp32S128S_i32]](%[[X:arg0]]: tensor<128xf32>{{.*}}, %[[N:arg1]]: i32
        CHECK-DAG:    %[[IDX0:c0]] = arith.constant 0 : index
        CHECK-DAG:    %[[IDX1:c1]] = arith.constant 1 : index
        CHECK-DAG:    %[[C512:c512_i32]] = arith.constant 512 : i32
        CHECK:        %[[STATE:[0-9]+]] = tt.call @kernels.triton_kernels.row_length__i32(%[[N]]) : (i32) -> i32
        CHECK-NEXT:   %[[LONG_ROW:[0-9]+]] = arith.cmpi sge, %[[STATE]], %[[C512]] : i32
        CHECK-NEXT:   %[[REGION:[0-9]+]] = arith.select %[[LONG_ROW]], %{{c1_i32}}, %{{c0_i32}} : i32
        CHECK-NEXT:   %[[REGION_INDEX:[0-9]+]] = arith.index_cast %[[REGION]] : i32 to index
        CHECK-NEXT:   %[[IS_REGION_0:[0-9]+]] = arith.cmpi eq, %[[REGION_INDEX]], %[[IDX0]] : index
        CHECK-NEXT:   %[[RESULT:[0-9]+]] = scf.if %[[IS_REGION_0]] -> (tensor<128xf32>) {
        CHECK-NEXT:     %[[EXACT:[0-9]+]] = tt.call @__internal_[[EXP]](%[[X]], %[[N]])
        CHECK-NEXT:     scf.yield %[[EXACT]] : tensor<128xf32>
        CHECK-NEXT:   } else {
        CHECK-NEXT:     %[[IS_REGION_1:[0-9]+]] = arith.cmpi eq, %[[REGION_INDEX]], %[[IDX1]] : index
        CHECK-NEXT:     %[[INNER:[0-9]+]] = scf.if %[[IS_REGION_1]] -> (tensor<128xf32>) {
        CHECK-NEXT:       %[[APPROX:[0-9]+]] = tt.call @approx_[[EXP]]_5(%[[X]], %[[N]])
        CHECK-NEXT:       scf.yield %[[APPROX]] : tensor<128xf32>
        CHECK-NEXT:     } else {
        CHECK-NEXT:       %[[DEFAULT:[0-9]+]] = tt.call @__internal_[[EXP]](%[[X]], %[[N]])
        CHECK-NEXT:       scf.yield %[[DEFAULT]] : tensor<128xf32>
        CHECK-NEXT:     }
        CHECK-NEXT:     scf.yield %[[INNER]] : tensor<128xf32>
        CHECK-NEXT:   }
        CHECK-NEXT:   tt.return %[[RESULT]] : tensor<128xf32>
        CHECK-NEXT: }
    """)
    assert "approx." not in after_passes.replace("@approx_", "")


def test_softmax_final_ttir_has_approx_inlined(tmp_path):
    _, _, final = compile_softmax_stages(tmp_path)

    # One function left; exp(x - max) became a runtime choice on n_cols >= 512
    # between the hardware exp and the inlined minimax2 polynomial.
    filecheck(final, """
        CHECK:      module {
        CHECK-NEXT:   tt.func public @softmax_kernel(%{{out_ptr}}: !tt.ptr<f32>{{.*}}, %{{in_ptr}}: !tt.ptr<f32>{{.*}}, %[[N_COLS:n_cols]]: i32
        CHECK-DAG:      %[[C512:[a-z_0-9]+]] = arith.constant 512 : i32
        CHECK-DAG:      %[[C2:[a-z_0-9]+]] = arith.constant dense<0.337189287> : tensor<128xf32>
        CHECK-DAG:      %[[C1:[a-z_0-9]+]] = arith.constant dense<0.657636463> : tensor<128xf32>
        CHECK-DAG:      %[[C0:[a-z_0-9]+]] = arith.constant dense<1.00172472> : tensor<128xf32>

        CHECK:          %[[ROW:[a-z_0-9]+]] = tt.load
        CHECK:          %[[MAX:[a-z_0-9]+]] = "tt.reduce"(%[[ROW]])
        CHECK:          %[[MAX_SPLAT:[a-z_0-9]+]] = tt.splat %[[MAX]] : f32 -> tensor<128xf32>
        CHECK-NEXT:     %[[SHIFTED:[a-z_0-9]+]] = arith.subf %[[ROW]], %[[MAX_SPLAT]] : tensor<128xf32>
        CHECK-NEXT:     %[[LONG_ROW:[a-z_0-9]+]] = arith.cmpi sge, %[[N_COLS]], %[[C512]] : i32

        CHECK:          %[[NUMERATOR:[a-z_0-9]+]] = scf.if %{{[a-z_0-9]+}} -> (tensor<128xf32>) {
        CHECK-NEXT:       %[[EXACT:[a-z_0-9]+]] = math.exp %[[SHIFTED]] : tensor<128xf32>
        CHECK-NEXT:       scf.yield %[[EXACT]] : tensor<128xf32>
        CHECK-NEXT:     } else {
        CHECK:            scf.if %{{[a-z_0-9]+}} -> (tensor<128xf32>) {
        CHECK-NEXT:         %[[CLAMPED:[a-z_0-9]+]] = arith.maxnumf %[[SHIFTED]], %{{[a-z_0-9]+}} : tensor<128xf32>
        CHECK-NEXT:         %[[T:[a-z_0-9]+]] = arith.mulf %[[CLAMPED]], %{{[a-z_0-9]+}} : tensor<128xf32>
        CHECK-NEXT:         %[[I:[a-z_0-9]+]] = math.floor %[[T]] : tensor<128xf32>
        CHECK-NEXT:         %[[I_INT:[a-z_0-9]+]] = arith.fptosi %[[I]] : tensor<128xf32> to tensor<128xi32>
        CHECK-NEXT:         %[[EXPONENT:[a-z_0-9]+]] = arith.addi %[[I_INT]], %{{[a-z_0-9]+}} : tensor<128xi32>
        CHECK-NEXT:         %[[BITS:[a-z_0-9]+]] = arith.shli %[[EXPONENT]], %{{[a-z_0-9]+}} : tensor<128xi32>
        CHECK-NEXT:         %[[TWO_I:[a-z_0-9]+]] = tt.bitcast %[[BITS]] : tensor<128xi32> -> tensor<128xf32>
        CHECK-NEXT:         %[[F:[a-z_0-9]+]] = arith.subf %[[T]], %[[I]] : tensor<128xf32>
        CHECK-NEXT:         %[[P1:[a-z_0-9]+]] = arith.mulf %[[F]], %[[C2]] : tensor<128xf32>
        CHECK-NEXT:         %[[P2:[a-z_0-9]+]] = arith.addf %[[P1]], %[[C1]] : tensor<128xf32>
        CHECK-NEXT:         %[[P3:[a-z_0-9]+]] = arith.mulf %[[F]], %[[P2]] : tensor<128xf32>
        CHECK-NEXT:         %[[POLY:[a-z_0-9]+]] = arith.addf %[[P3]], %[[C0]] : tensor<128xf32>
        CHECK-NEXT:         %[[APPROX:[a-z_0-9]+]] = arith.mulf %[[TWO_I]], %[[POLY]] : tensor<128xf32>
        CHECK-NEXT:         scf.yield %[[APPROX]] : tensor<128xf32>
        CHECK-NEXT:       } else {
        CHECK-NEXT:         %[[DEFAULT:[a-z_0-9]+]] = math.exp %[[SHIFTED]] : tensor<128xf32>

        CHECK:          %[[SUM:[0-9]+]] = "tt.reduce"(%[[NUMERATOR]])
        CHECK:          %[[SUM_SPLAT:[0-9]+]] = tt.splat %[[SUM]] : f32 -> tensor<128xf32>
        CHECK-NEXT:     %[[RESULT:[0-9]+]] = arith.divf %[[NUMERATOR]], %[[SUM_SPLAT]] : tensor<128xf32>
        CHECK-NEXT:     tt.store %{{[0-9]+}}, %[[RESULT]], %{{[a-z_0-9]+}} : tensor<128x!tt.ptr<f32>>
        CHECK-NEXT:     tt.return
        CHECK-NEXT:   }
        CHECK-NEXT: }
    """)


# Every exp version compiles to the right computation (decisions = [v, v])

# The first ops of the first branch of the decision, per version.
EXACT_EXP = """
    CHECK:      %[[SHIFTED:[a-z_0-9]+]] = arith.subf
    CHECK:      scf.if %{{[a-z_0-9]+}} -> (tensor<128xf32>) {
    CHECK-NEXT:   %{{[a-z_0-9]+}} = math.exp %[[SHIFTED]] : tensor<128xf32>
"""
RANGE_REDUCED_POLYNOMIAL = """
    CHECK:      %[[SHIFTED:[a-z_0-9]+]] = arith.subf %{{[a-z_0-9]+}}, %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK:      scf.if %{{[a-z_0-9]+}} -> (tensor<128xf32>) {
    CHECK-NEXT:   %[[CLAMPED:[a-z_0-9]+]] = arith.maxnumf %[[SHIFTED]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   %[[T:[a-z_0-9]+]] = arith.mulf %[[CLAMPED]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   %[[I:[a-z_0-9]+]] = math.floor %[[T]] : tensor<128xf32>
    CHECK:        %[[TWO_I:[a-z_0-9]+]] = tt.bitcast %{{[a-z_0-9]+}} : tensor<128xi32> -> tensor<128xf32>
    CHECK-NEXT:   %{{[a-z_0-9]+}} = arith.subf %[[T]], %[[I]] : tensor<128xf32>
    CHECK:        %[[APPROX:[a-z_0-9]+]] = arith.mulf %[[TWO_I]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   scf.yield %[[APPROX]] : tensor<128xf32>
"""
SCHRAUDOLPH = """
    CHECK:      %[[SHIFTED:[a-z_0-9]+]] = arith.subf %{{[a-z_0-9]+}}, %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK:      scf.if %{{[a-z_0-9]+}} -> (tensor<128xf32>) {
    CHECK-NEXT:   %[[CLAMPED:[a-z_0-9]+]] = arith.maxnumf %[[SHIFTED]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   %[[SCALED:[a-z_0-9]+]] = arith.mulf %[[CLAMPED]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   %[[BIASED:[a-z_0-9]+]] = arith.addf %[[SCALED]], %{{[a-z_0-9]+}} : tensor<128xf32>
    CHECK-NEXT:   %[[BITS:[a-z_0-9]+]] = arith.fptosi %[[BIASED]] : tensor<128xf32> to tensor<128xi32>
    CHECK-NEXT:   %[[APPROX:[a-z_0-9]+]] = tt.bitcast %[[BITS]] : tensor<128xi32> -> tensor<128xf32>
    CHECK-NEXT:   scf.yield %[[APPROX]] : tensor<128xf32>
"""
EXPECTED_EXP = {0: EXACT_EXP, 8: SCHRAUDOLPH}


@pytest.mark.parametrize("version", sorted(triton_kernels.VERSION_NAMES))
def test_softmax_every_version_compiles_to_its_exp(version):
    final = compile_softmax([version, version])

    assert functions_in(final) == ["softmax_kernel"]
    filecheck(final, EXPECTED_EXP.get(version, RANGE_REDUCED_POLYNOMIAL))
    assert ("math.exp" in final) == (version == 0)


# Frontend behavior

def test_knob_registers_triton_function():
    assert TREE.approx_kernels[5] is triton_kernels._minimax2
    assert triton_kernels.softmax_exp in ar.knobbed_functions()


def test_disable_restores_exact_compilation():
    ar.enable_triton_approx()
    ar.disable_triton_approx()
    final = compile_kernel(triton_kernels.softmax_kernel).asm["ttir"]
    # No knob at all: exp(x - max) is the plain inlined math.exp, no branch.
    filecheck(final, """
        CHECK:      %[[SHIFTED:[a-z_0-9]+]] = arith.subf
        CHECK-NEXT: %{{[a-z_0-9]+}} = math.exp %[[SHIFTED]] : tensor<128xf32>
    """)
    assert "scf.if" not in final


@triton.jit
def plain_exp_kernel(out_ptr, in_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.exp(tl.load(in_ptr + offs)))


def test_kernel_without_knobs_is_unchanged():
    ar.enable_triton_approx()
    final = compile_kernel(plain_exp_kernel, {"out_ptr": "*fp32", "in_ptr": "*fp32", "BLOCK": "constexpr"},
                           block=64).asm["ttir"]
    filecheck(final, '''
        CHECK:      %[[X:[a-z_0-9]+]] = tt.load
        CHECK-NEXT: %[[EXP:[a-z_0-9]+]] = math.exp %[[X]] : tensor<64xf32>
        CHECK-NEXT: tt.store %{{[a-z_0-9]+}}, %[[EXP]]
    ''')


def test_cache_key_tracks_config_changes():
    ar.enable_triton_approx()
    default_key = ar.triton.frontend._cache_key()
    TREE.decisions = [0, 6]
    assert ar.triton.frontend._cache_key() != default_key
    TREE.decisions = list(DEFAULT_DECISIONS)
    assert ar.triton.frontend._cache_key() == default_key


# tune_triton: OpenTuner replaced by a stand-in that calls evaluate once

def without_locations(ttir):
    """TTIR without debug locations (they name a different temporary file per compile)."""
    lines = [line for line in ttir.splitlines() if not line.startswith("#loc")]
    return re.sub(r" loc\(#loc\d*\)", "", "\n".join(lines))


def test_tune_triton_recompiles_with_each_config_and_restores_tree(monkeypatch):
    calls = {}

    def one_round_tune(mlir_source, evaluate, accuracy_threshold, **kwargs):
        calls["params"] = sorted(ar.MLIRConfigManager().parse_annotations(mlir_source))
        evaluate({"softmax_exp_threshold_0": 512,
                  "softmax_exp_decision_0": 3, "softmax_exp_decision_1": 3})
        return {"best_config": {}}

    def evaluate_fn(config):
        calls["tree"] = (list(TREE.thresholds), list(TREE.decisions))
        calls["ttir"] = compile_kernel(triton_kernels.softmax_kernel).asm["ttir"]
        return 1.0, 1.0

    monkeypatch.setattr(ar.tuning.tuner, "tune", one_round_tune)
    ar.enable_triton_approx()
    ar.tune_triton(triton_kernels.softmax_exp, evaluate_fn)

    assert calls["params"] == ["softmax_exp_decision_0", "softmax_exp_decision_1",
                               "softmax_exp_threshold_0"]
    assert calls["tree"] == ([512], [3, 3])
    tuned = without_locations(calls["ttir"])
    assert tuned == without_locations(compile_softmax([3, 3]))
    assert tuned != without_locations(compile_softmax(DEFAULT_DECISIONS))


def test_tune_triton_restores_tree_when_evaluate_fails(monkeypatch):
    def one_round_tune(mlir_source, evaluate, accuracy_threshold, **kwargs):
        evaluate({"softmax_exp_decision_1": 8})

    def failing_evaluate(config):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(ar.tuning.tuner, "tune", one_round_tune)
    ar.enable_triton_approx()
    with pytest.raises(RuntimeError, match="launch failed"):
        ar.tune_triton(triton_kernels.softmax_exp, failing_evaluate)
    assert TREE.decisions == DEFAULT_DECISIONS


def test_tune_triton_requires_enabled_frontend():
    ar.disable_triton_approx()
    with pytest.raises(RuntimeError, match="enable_triton_approx"):
        ar.tune_triton(triton_kernels.softmax_exp, lambda config: (1.0, 1.0))


def test_tune_triton_requires_decision_tree():
    ar.enable_triton_approx()
    with pytest.raises(ValueError, match="no @Knob DecisionTree"):
        ar.tune_triton(triton_kernels.row_length, lambda config: (1.0, 1.0))


# Decision tree: two call sites with different types in one kernel

@triton.jit
def row_length(n):
    return n


@triton.jit
def cheap_exp(x, n):
    return x + 1.0


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=row_length, state_indices=[1], thresholds=[512], decisions=[0, 1],
    transform_type="func_substitute", approx_kernels={1: cheap_exp}))
@triton.jit
def tree_exp(x, n):
    return tl.exp(x)


@triton.jit
def two_call_sites_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs, mask=offs < n_cols, other=0.0)
    block_exp = tree_exp(x, n_cols)
    scalar_exp = tree_exp(tl.sum(x, axis=0), n_cols)
    tl.store(out_ptr + offs, block_exp + scalar_exp, mask=offs < n_cols)


def test_decision_tree_two_call_sites(tmp_path):
    ar.enable_triton_approx(dump_directory=str(tmp_path))
    asm = compile_kernel(two_call_sites_kernel).asm

    assert functions_in(asm["ttir"]) == ["two_call_sites_kernel"]
    assert "scf.index_switch" not in asm["ttir"]
    assert re.search(r"arith\.cmpi sge, .*%c512_i32", asm["ttir"])
    assert len(asm["cubin"]) > 0

    annotated = next(tmp_path.glob("*.annotated.ttir")).read_text()
    sites = re.findall(r'decision_tree"\(\) <\{\s*func_name = "([^"]+)"', annotated)
    assert sorted(sites) == sorted([
        f"{tree_exp.__module__}.tree_exp__fp32S128S_i32",
        f"{tree_exp.__module__}.tree_exp__fp32_i32",
    ])
    assert re.findall(r'state_function = "([^"]+)"', annotated) == \
        [f"{row_length.__module__}.row_length__i32"] * 2
    approx_funcs = set(functions_in(annotated)) & {f"approx_{s}_1" for s in sites}
    assert len(approx_funcs) == 2


# Safety contract: try-check-recover

@triton.jit
def all_positive(result, x, n):
    bad_lanes = tl.sum((result <= 0.0).to(tl.int32), axis=0)
    return bad_lanes == 0


@triton.jit
def exact_exp(x, n):
    return tl.exp(x)


@ar.Knob(safety_contract=ar.SafetyContract(checker=all_positive, recover=exact_exp))
@triton.jit
def guarded_exp(x, n):
    return 1.0 + x  # cheap body; can go non-positive, which the checker catches


@triton.jit
def contract_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs, mask=offs < n_cols, other=0.0)
    tl.store(out_ptr + offs, guarded_exp(x, n_cols), mask=offs < n_cols)


def test_safety_contract_compiles_to_check_and_recover(tmp_path):
    ar.enable_triton_approx(dump_directory=str(tmp_path))
    final = compile_kernel(contract_kernel).asm["ttir"]

    module = guarded_exp.__module__
    annotated = next(tmp_path.glob("*.annotated.ttir")).read_text()
    filecheck(annotated, f"""
        CHECK:      "approx.util.annotation.knob"() <{{
        CHECK-NEXT:   func_name = "[[EXP:{module}.guarded_exp__fp32S128S_i32]]"
        CHECK-NEXT: }}> : () -> ()
        CHECK-NEXT: "approx.util.annotation.try"() <{{
        CHECK-NEXT:   func_name = "[[EXP]]",
        CHECK-NEXT:   recover = "{module}.exact_exp__fp32S128S_i32",
        CHECK-NEXT:   checker = "{module}.all_positive__fp32S128S_fp32S128S_i32"
        CHECK-NEXT: }}> : () -> ()
    """)

    # After the passes: run the body, check the result, recover on failure.
    after_passes = next(tmp_path.glob("*.approx.ttir")).read_text()
    filecheck(after_passes, f"""
        CHECK:      tt.func private @[[EXP:{module}.guarded_exp__fp32S128S_i32]](%[[X:arg0]]: tensor<128xf32>{{{{.*}}}}, %[[N:arg1]]: i32
        CHECK:        %[[RESULT:[0-9]+]] = arith.addf
        CHECK:        %[[OK:[0-9]+]] = tt.call @{module}.all_positive__fp32S128S_fp32S128S_i32(%[[RESULT]], %[[X]], %[[N]])
        CHECK-NEXT:   %[[FINAL:[0-9]+]] = scf.if %[[OK]] -> (tensor<128xf32>) {{
        CHECK-NEXT:     scf.yield %[[RESULT]] : tensor<128xf32>
        CHECK-NEXT:   }} else {{
        CHECK-NEXT:     %[[EXACT:[0-9]+]] = tt.call @{module}.exact_exp__fp32S128S_i32(%[[X]], %[[N]])
        CHECK-NEXT:     scf.yield %[[EXACT]] : tensor<128xf32>
        CHECK-NEXT:   }}
        CHECK-NEXT:   tt.return %[[FINAL]] : tensor<128xf32>
    """)
    assert "approx." not in after_passes.replace("@approx_", "")
    assert "scf.if" in final  # survives make_ttir


@triton.jit
def checker_wrong_type(result, x):
    return result  # a tensor, not one bool


@triton.jit
def recover_fine(x):
    return tl.exp(x)


@ar.Knob(safety_contract=ar.SafetyContract(checker=checker_wrong_type, recover=recover_fine))
@triton.jit
def badly_guarded_exp(x):
    return tl.exp(x)


def test_error_checker_must_return_one_bool():
    assert "must return a scalar bool" in _compile_error(badly_guarded_exp)


# Errors for approximate versions that cannot replace the exact one

def _compile_error(knob_fn):
    @triton.jit
    def kernel(out_ptr, in_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, knob_fn(tl.load(in_ptr + offs)))

    ar.enable_triton_approx()
    with pytest.raises(Exception) as info:
        compile_kernel(kernel, {"out_ptr": "*fp32", "in_ptr": "*fp32", "BLOCK": "constexpr"}, block=64)
    return str(info.value)


@triton.jit
def takes_two(x, y):
    return x


@ar.Knob(static_transform=ar.StaticTransform("func_substitute", 1, approx_kernel=takes_two))
@triton.jit
def wrong_arity_exp(x):
    return tl.exp(x)


@triton.jit
def returns_int(x):
    return x.to(tl.int32)


@ar.Knob(static_transform=ar.StaticTransform("func_substitute", 1, approx_kernel=returns_int))
@triton.jit
def wrong_return_exp(x):
    return tl.exp(x)


def not_jitted(x):
    return x


@ar.Knob(static_transform=ar.StaticTransform("func_substitute", 1, approx_kernel=not_jitted))
@triton.jit
def not_jitted_exp(x):
    return tl.exp(x)


def test_error_approx_kernel_wrong_arity():
    assert "takes 2 arguments but 'wrong_arity_exp' takes 1" in _compile_error(wrong_arity_exp)


def test_error_approx_kernel_wrong_return_type():
    assert "approx kernel 'returns_int' returns" in _compile_error(wrong_return_exp)


def test_error_approx_kernel_not_jitted():
    assert "must be a @triton.jit function" in _compile_error(not_jitted_exp)
