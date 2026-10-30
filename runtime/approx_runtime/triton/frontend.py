"""Triton frontend: makes Triton use the approximations from @Knob.

@Knob only saves a config on a function. Nothing changes until you call
enable_triton_approx(). It changes two things in Triton:

1. It replaces CodeGenerator.call_JitFunction with _call_with_versions.
   Triton calls call_JitFunction every time one @triton.jit function calls
   another. _call_with_versions first runs Triton's original. Then, if the
   called function has a @Knob, it also generates the approximate versions
   (named approx_<name>_<k>) and the state function.

2. It adds a hook before Triton's make_ttir step.
   The hook adds the annotation ops to the Triton IR (TTIR) and runs the ApproxMLIR
   passes (approx-triton-pipeline). The passes build the decision tree.
   It has to run before make_ttir, because make_ttir inlines all functions
   into the kernel.

disable_triton_approx() puts Triton's originals back.

Only calls to a @Knob function are changed. A @Knob on the kernel itself
does not work, because nothing calls the kernel from Triton code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..core.annotations import inject_annotations, needs_pre_emit_transform
from ..core.knob import get_config, knobbed_functions
from ..core.toolchain import get_toolchain, with_pre_emit_transform
from .passes import TritonCompilationError

__all__ = [
    "enable_triton_approx",
    "disable_triton_approx",
    "is_triton_approx_enabled",
    "approx_ttir",
]

PLUGIN_NOT_LOADED = ("Set TRITON_PLUGIN_PATHS (or TRITON_PASS_PLUGIN_PATH) to "
                     "libApproxTritonPlugin.so before importing triton.")

# Matches the name of every function ("tt.func @name(") in Triton IR text.
_TRITON_FUNCTION_NAME = re.compile(r"^\s*tt\.func\b[^@]*@([\w.$]+)\(", re.MULTILINE)


@dataclass
class _State:
    original_call: Optional[Callable] = None  # Triton's CodeGenerator.call_JitFunction
    previous_hook: Optional[Callable] = None  # the stages hook that was installed before ours
    passes: Optional[List[str]] = None  # plugin pass names; None means the toolchain default
    dump_directory: Optional[str] = None  # where approx_ttir writes the Triton IR, before/after
    plugin: str = ""  # TRITON_PLUGIN_PATHS when the frontend was enabled (part of the cache key)
    # Generated name of a knob function -> generated name of its helper.
    # Generated names are always the same for the same function and types,
    # so the entries stay valid across compilations.
    state_functions: Dict[str, str] = field(default_factory=dict)
    checker_functions: Dict[str, str] = field(default_factory=dict)
    recover_functions: Dict[str, str] = field(default_factory=dict)


_state = _State()


def _full_name(function) -> str:
    """The Python path of a function, e.g. kernels.triton_kernels.softmax_exp.

    Triton names the functions it generates with this path, then "__", then the
    argument types: kernels.triton_kernels.softmax_exp__fp32S128S_i32.
    """
    return f"{function.__module__}.{function.__qualname__}"


def _config_of(function) -> Optional[dict]:
    """The @Knob config of a function, or None.

    @Knob can be written above or below @triton.jit, so we look on the
    JITFunction and on the Python function inside it (.fn).
    """
    return get_config(function) or get_config(getattr(function, "fn", None))


def _approx_versions(config: dict) -> Dict[int, object]:
    """The approximate versions listed in a config, as {knob value: @triton.jit function}.

    Only func_substitute has separate versions: a DecisionTree's approx_kernels,
    or a StaticTransform's approx_kernel. The other transforms change the
    function's own body, so for them this is empty.
    """
    versions = {}
    tree = config.get("decision_tree")
    if tree is not None and tree.transform_type == "func_substitute":
        versions.update(tree.approx_kernels)
    static = config.get("static_transform")
    if (static is not None and static.transform_type == "func_substitute"
            and static.knob_val and static.approx_kernel is not None):
        versions[static.knob_val] = static.approx_kernel
    return versions


def _knob_configs() -> Dict[str, dict]:
    """{Python path: config} for every function that has a @Knob."""
    return {_full_name(function): config
            for function in knobbed_functions()
            if (config := _config_of(function))}


# Hook 1: code generation

def _call_with_versions(self, fn, args, kwargs, caller_context=None):
    """Runs instead of Triton's CodeGenerator.call_JitFunction.

    Triton calls it for every call from one @triton.jit function to another.
    It first runs Triton's original, which generates the called function and the call.
    Then, if the called function (fn) has a @Knob, it also generates its
    approximate versions and its state function.

    The parameters must keep Triton's names (fn, args, kwargs): Triton's own
    code calls this by keyword, e.g. call_JitFunction(combine_fn, args, kwargs={}).
    """
    result = _state.original_call(self, fn, args, kwargs, caller_context)
    config = _config_of(fn)
    if config:
        _generate_versions(self, fn, config, args, kwargs,
                           caller_context or self.caller_context)
    return result


def _generate_versions(code_generator, knob_function, config, positional_arguments,
                       keyword_arguments, caller_context) -> None:
    """Generate the helper functions of knob_function into the module.

    knob_function is the @Knob function being called, e.g. softmax_exp.
    The helpers are: the approximate versions, the decision tree's state
    function, and the safety contract's checker and recover functions.
    Each gets the argument types of this call. Each version is named
    approx_<name of knob_function>_<knob value>; the func_substitute pass
    looks for that name.
    It raises an error if a helper does not fit knob_function: not a
    @triton.jit function, wrong number of arguments, or wrong types.
    A checker receives (result, arguments...) for a whole block and must
    return ONE bool for the block (reduce with tl.sum or tl.min first).
    """
    import triton.language
    from triton.compiler.code_generator import mangle_fn

    argument_types = _argument_types(knob_function, positional_arguments, keyword_arguments)
    name = mangle_fn(_full_name(knob_function), argument_types, caller_context)
    exact_type = code_generator.function_ret_types[name]

    for knob_value, version in sorted(_approx_versions(config).items()):
        _require_triton_jit_function(version, f"approx kernel {knob_value}", knob_function)
        if len(version.arg_names) != len(knob_function.arg_names):
            raise TypeError(
                f"approx kernel '{version.__name__}' takes {len(version.arg_names)} "
                f"arguments but '{knob_function.__name__}' takes {len(knob_function.arg_names)}")
        version_type = _generate(code_generator, version, argument_types,
                                 f"approx_{name}_{knob_value}", caller_context)
        if version_type != exact_type:
            raise TypeError(
                f"approx kernel '{version.__name__}' returns {version_type} but "
                f"'{knob_function.__name__}' returns {exact_type} (arguments {argument_types})")

    tree = config.get("decision_tree")
    if tree is not None:
        _require_triton_jit_function(tree.state_function, "state_function", knob_function)
        state_types = [argument_types[index] for index in tree.state_indices]
        state_name = mangle_fn(_full_name(tree.state_function), state_types, caller_context)
        state_type = _generate(code_generator, tree.state_function, state_types, state_name,
                               caller_context)
        if state_type != triton.language.int32:
            raise TypeError(f"state_function '{tree.state_function.__name__}' must return "
                            f"a scalar int32, got {state_type}")
        _state.state_functions[name] = state_name

    contract = config.get("safety_contract")
    if contract is not None:
        _require_triton_jit_function(contract.recover, "recover", knob_function)
        if len(contract.recover.arg_names) != len(knob_function.arg_names):
            raise TypeError(
                f"recover '{contract.recover.__name__}' takes {len(contract.recover.arg_names)} "
                f"arguments but '{knob_function.__name__}' takes {len(knob_function.arg_names)}")
        recover_name = mangle_fn(_full_name(contract.recover), argument_types, caller_context)
        recover_type = _generate(code_generator, contract.recover, argument_types,
                                 recover_name, caller_context)
        if recover_type != exact_type:
            raise TypeError(
                f"recover '{contract.recover.__name__}' returns {recover_type} but "
                f"'{knob_function.__name__}' returns {exact_type}")

        _require_triton_jit_function(contract.checker, "checker", knob_function)
        checker_types = [exact_type] + list(argument_types)
        if len(contract.checker.arg_names) != len(checker_types):
            raise TypeError(
                f"checker '{contract.checker.__name__}' takes {len(contract.checker.arg_names)} "
                f"arguments but must take {len(checker_types)}: the result, then "
                f"'{knob_function.__name__}'s arguments")
        checker_name = mangle_fn(_full_name(contract.checker), checker_types, caller_context)
        checker_type = _generate(code_generator, contract.checker, checker_types,
                                 checker_name, caller_context)
        if checker_type != triton.language.int1:
            raise TypeError(
                f"checker '{contract.checker.__name__}' must return a scalar bool (one "
                f"tl.int1 for the whole block), got {checker_type}")
        _state.recover_functions[name] = recover_name
        _state.checker_functions[name] = checker_name


def _argument_types(function, positional_arguments, keyword_arguments) -> list:
    """The types of the arguments in this call, e.g. [tensor<128xf32>, i32].

    Triton builds a function's generated name from these types:
    softmax_exp called with (tensor<128xf32>, i32) becomes
    kernels.triton_kernels.softmax_exp__fp32S128S_i32.
    This function copies the type computation from Triton's call_JitFunction.
    So the types are the same as Triton's, and the version names we build
    from them (approx_<that name>_<k>) point at the right function.
    """
    from triton.language.core import base_value, constexpr
    from triton.runtime.jit import JITCallable

    bound = function.signature.bind(*positional_arguments, **keyword_arguments)
    bound.apply_defaults()
    values = [bound.arguments[argument_name] for argument_name in function.arg_names]
    return [(value if isinstance(value, base_value) and not isinstance(value, JITCallable)
             else constexpr(value)).type
            for value in values]


def _generate(code_generator, function, argument_types, name: str, caller_context):
    """Generate function into the module, under the given name. Return its return type.

    This is the part of Triton's call_JitFunction (compiler/code_generator.py)
    that generates the called function. It differs in two ways: we choose the
    name, and it does not create a call. If the module already has a function
    with that name, it does nothing.
    """
    from triton.compiler.code_generator import ASTFunction, CodeGenerator

    if code_generator.module.has_function(name):
        return code_generator.function_ret_types[name]
    # The keyword names below (jit_fn, codegen_fns, ...) are Triton's own.
    new_generator = CodeGenerator(
        code_generator.context, ASTFunction([], argument_types, dict()),
        function.get_capture_scope(),
        module=code_generator.module, jit_fn=function, function_name=name,
        function_types=code_generator.function_ret_types, noinline=function.noinline,
        file_name=function.file_name, begin_line=function.def_file_line_number,
        begin_col=function.def_file_col_number, options=code_generator.builder.options,
        codegen_fns=code_generator.builder.codegen_fns,
        module_map=code_generator.builder.module_map,
        caller_context=caller_context, is_gluon=code_generator.is_gluon)
    new_generator.visit(function.parse())
    code_generator.function_ret_types[name] = new_generator.ret_type
    return new_generator.ret_type


def _require_triton_jit_function(candidate, role: str, knob_function) -> None:
    """Raise an error unless candidate is a @triton.jit function (a JITFunction)."""
    from triton.runtime.jit import JITFunction

    if not isinstance(candidate, JITFunction):
        raise TypeError(f"{role} of @Knob function '{knob_function.__name__}' must be a "
                        f"@triton.jit function, got {candidate!r}")


# Hook 2: the Triton IR stage

def approx_ttir(module, passes: Optional[List[str]] = None,
                dump_directory: Optional[str] = None):
    """Run the ApproxMLIR passes on a Triton IR module. Return the new module.

    module is the Triton IR that Triton generated from the kernel's Python code.
    All functions are still separate in it.
    For each function that came from a @Knob function, it adds the annotation ops.
    Then it runs the plugin passes. The result has the decision tree built in.
    If the kernel calls no @Knob function, it returns module unchanged.
    """
    text = str(module)
    sites = _knob_sites(text)
    if not sites:
        return module

    for name, config in sites:
        text = inject_annotations(text, name, config,
                                  state_function_name=_state.state_functions.get(name),
                                  checker_name=_state.checker_functions.get(name),
                                  recover_name=_state.recover_functions.get(name))
    pipeline = with_pre_emit_transform(
        passes or get_toolchain().triton_pipeline,
        any(needs_pre_emit_transform(config) for _, config in sites))
    new_module = _run_plugin(text, pipeline, module.context)

    result = str(new_module)
    if dump_directory:
        _dump(dump_directory, text, result)
    if "approx." in result:
        # The passes can report an error (e.g. a missing approx_<name>_<k>) without failing.
        raise TritonCompilationError(
            "ApproxMLIR passes left approx ops in the module; see the MLIR diagnostics "
            f"above. Knob sites: {[name for name, _ in sites]}")
    return new_module


def _knob_sites(triton_ir_text: str) -> List[Tuple[str, dict]]:
    """(generated name, config) for every function in the module that came from a @Knob function.

    A function matches when its name starts with the @Knob function's Python
    path followed by "__". The "__" starts Triton's argument-type part.
    """
    configs = _knob_configs()
    return [(function_name, config)
            for function_name in _TRITON_FUNCTION_NAME.findall(triton_ir_text)
            for full_name, config in configs.items()
            if function_name.startswith(full_name + "__")]


def _run_plugin(text: str, pipeline: List[str], context):
    """Parse the MLIR text in Triton's context and run the plugin passes on it."""
    from triton._C.libtriton import ir
    from triton._C.libtriton import passes as triton_passes

    missing = [pass_name for pass_name in pipeline if not hasattr(triton_passes.plugin, pass_name)]
    if missing:
        raise TritonCompilationError(
            f"ApproxMLIR plugin passes not loaded: {missing}. " + PLUGIN_NOT_LOADED)
    with tempfile.TemporaryDirectory() as temporary_directory:  # Triton only parses MLIR from files
        path = Path(temporary_directory) / "annotated.ttir"
        path.write_text(text, encoding="utf-8")
        new_module = ir.parse_mlir_module(str(path), context)
    new_module.context = context  # a module parsed from a file does not keep the context

    pass_manager = ir.pass_manager(context)
    for pass_name in pipeline:
        getattr(triton_passes.plugin, pass_name)(pass_manager)
    pass_manager.run(new_module, "approx")
    return new_module


def _dump(dump_directory: str, annotated: str, transformed: str) -> None:
    """Write <hash>.annotated.ttir (before the passes) and <hash>.approx.ttir (after)."""
    output_directory = Path(dump_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(annotated.encode()).hexdigest()[:12]
    (output_directory / f"{tag}.annotated.ttir").write_text(annotated, encoding="utf-8")
    (output_directory / f"{tag}.approx.ttir").write_text(transformed, encoding="utf-8")


def _cache_key() -> str:
    """An extra key for Triton's caches of compiled kernels.

    Triton's own key only covers the kernel's source code. Without this key,
    a changed decision tree or an edited approximate version would reuse an
    old compiled kernel. This key covers every @Knob config, the source code of
    every approximate version and state function, the pass list and the plugin path.
    """
    entries = []
    for full_name, config in sorted(_knob_configs().items()):
        functions = [version for _, version in sorted(_approx_versions(config).items())]
        tree = config.get("decision_tree")
        if tree is not None:
            functions.append(tree.state_function)
        contract = config.get("safety_contract")
        if contract is not None:
            functions.extend([contract.checker, contract.recover])
        sources = [getattr(function, "cache_key", repr(function)) for function in functions]
        entries.append([full_name, repr(config), sources])
    return "approx_triton::" + json.dumps([entries, _state.passes, _state.plugin], default=repr)


def _stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    """The stages hook. Triton calls it while compiling (knobs.runtime.add_stages_inspection_hook).

    Triton calls it twice per compilation.
    The first call has no arguments. Triton only wants the extra cache key.
    The second call has stages. We wrap the "ttir" stage, so approx_ttir runs
    before Triton's make_ttir.
    """
    key = _cache_key()
    if stages is not None and "ttir" in stages:
        make_ttir = stages["ttir"]
        stages["ttir"] = lambda source, metadata: make_ttir(
            approx_ttir(source, _state.passes, _state.dump_directory), metadata)
    return key, hashlib.sha256(key.encode("utf-8")).hexdigest()


# Public switch

def enable_triton_approx(passes: Optional[List[str]] = None,
                         dump_directory: Optional[str] = None) -> None:
    """Make Triton compile kernels with the approximations from @Knob.

    @Knob only saves a config on a function. This installs the two hooks
    that make Triton's compiler use it.

    Args:
        passes: Plugin pass names to run. Default: the TRITON toolchain
            pipeline, which is the C++ approx-triton-pipeline.
        dump_directory: If set, write each kernel's Triton IR there, before and
            after the passes (<hash>.annotated.ttir, <hash>.approx.ttir).

    Raises TritonCompilationError if the ApproxMLIR plugin is not loaded.
    """
    from triton import knobs
    from triton._C.libtriton import passes as triton_passes
    from triton.compiler.code_generator import CodeGenerator

    if not hasattr(triton_passes.plugin, "emit-approx"):
        raise TritonCompilationError("ApproxMLIR plugin is not loaded. " + PLUGIN_NOT_LOADED)
    _state.passes = list(passes) if passes else None
    _state.dump_directory = dump_directory
    _state.plugin = os.environ.get("TRITON_PLUGIN_PATHS", "")
    if _state.original_call is None:
        _state.original_call = CodeGenerator.call_JitFunction
        CodeGenerator.call_JitFunction = _call_with_versions
    if knobs.runtime.add_stages_inspection_hook is not _stages_hook:
        _state.previous_hook = knobs.runtime.add_stages_inspection_hook
        knobs.runtime.add_stages_inspection_hook = _stages_hook


def is_triton_approx_enabled() -> bool:
    """True between enable_triton_approx() and disable_triton_approx()."""
    return _state.original_call is not None


def disable_triton_approx() -> None:
    """Put Triton's originals back. Kernels compile as without ApproxMLIR.

    The @Knob configs stay on the functions. They have no effect until the
    next enable_triton_approx().
    """
    from triton import knobs
    from triton.compiler.code_generator import CodeGenerator

    if _state.original_call is not None:
        CodeGenerator.call_JitFunction = _state.original_call
        _state.original_call = None
    if knobs.runtime.add_stages_inspection_hook is _stages_hook:
        knobs.runtime.add_stages_inspection_hook = _state.previous_hook
    _state.previous_hook = None
