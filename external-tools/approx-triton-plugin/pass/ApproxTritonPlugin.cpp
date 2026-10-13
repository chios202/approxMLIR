/// ApproxTritonPlugin.cpp — Triton pass plugin entry point
///
/// Registers approxMLIR's dialect and passes so they can be loaded by Triton
/// via TRITON_PASS_PLUGIN_PATH.
///
/// Entries:
///   approx-triton-pipeline  the whole pipeline, built by buildApproxPipeline
///                           (lib/approx/Passes/Pipelines.cpp) for "triton"
///   pre-emit-transform, emit-approx, emit-management, config-approx,
///   transform-approx, finalize-approx, index-switch-to-if
///                           each pass on its own
///
/// Dialect: "approx" (approxDialect) — registered via the dialect plugin API
/// so Triton's MLIRContext can parse approx.* ops.

#include "mlir/Pass/PassRegistry.h"
#include "approx/Dialect.h"
#include "approx/Ops.h"
#include "approx/Passes/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Config/llvm-config.h"

namespace {

/// Pass name constants — these are the names users reference when adding
/// passes to Triton's pipeline.
constexpr const char *kPassNames[] = {
    "emit-approx",
    "emit-management",
    "config-approx",
    "pre-emit-transform",
    "transform-approx",
    "finalize-approx",
    "index-switch-to-if",
    "approx-triton-pipeline",
};
constexpr uint32_t kNumPasses = sizeof(kPassNames) / sizeof(kPassNames[0]);

std::unique_ptr<mlir::Pass> createPassByName(const char *name) {
  llvm::StringRef n(name);
  if (n == "emit-approx")
    return mlir::approx::createEmitApproxPass();
  if (n == "emit-management")
    return mlir::approx::createEmitManagementPass();
  if (n == "config-approx")
    return mlir::approx::createConfigApproxPass();
  if (n == "pre-emit-transform")
    return mlir::approx::createPreEmitTransformationPass();
  if (n == "transform-approx")
    return mlir::approx::createTransformApproxPass();
  if (n == "finalize-approx")
    return mlir::approx::createFinalizeApproxPass();
  if (n == "index-switch-to-if")
    return mlir::approx::createIndexSwitchToIfPass();
  return nullptr;
}

void addPassByName(mlir::PassManager *pm, const std::vector<std::string> &args,
                   const char *name) {
  (void)args;
  if (auto pass = createPassByName(name))
    pm->addPass(std::move(pass));
}

void registerPassByName(const char *name) {
  mlir::registerPass([name]() { return createPassByName(name); });
}

void registerApproxDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::approx::approxDialect>();
  mlir::func::registerInlinerExtension(*registry);
}

void addEmitApproxPass(mlir::PassManager *pm,
                       const std::vector<std::string> &args) {
  addPassByName(pm, args, "emit-approx");
}

void addEmitManagementPass(mlir::PassManager *pm,
                           const std::vector<std::string> &args) {
  addPassByName(pm, args, "emit-management");
}

void addConfigApproxPass(mlir::PassManager *pm,
                         const std::vector<std::string> &args) {
  addPassByName(pm, args, "config-approx");
}

void addPreEmitTransformPass(mlir::PassManager *pm,
                             const std::vector<std::string> &args) {
  addPassByName(pm, args, "pre-emit-transform");
}

void addTransformApproxPass(mlir::PassManager *pm,
                            const std::vector<std::string> &args) {
  addPassByName(pm, args, "transform-approx");
}

void addFinalizeApproxPass(mlir::PassManager *pm,
                           const std::vector<std::string> &args) {
  addPassByName(pm, args, "finalize-approx");
}

void addIndexSwitchToIfPass(mlir::PassManager *pm,
                            const std::vector<std::string> &args) {
  addPassByName(pm, args, "index-switch-to-if");
}

void registerEmitApproxPass() { registerPassByName("emit-approx"); }
void registerEmitManagementPass() { registerPassByName("emit-management"); }
void registerConfigApproxPass() { registerPassByName("config-approx"); }
void registerPreEmitTransformPass() {
  registerPassByName("pre-emit-transform");
}
void registerTransformApproxPass() { registerPassByName("transform-approx"); }
void registerFinalizeApproxPass() { registerPassByName("finalize-approx"); }
void registerIndexSwitchToIfPass() {
  registerPassByName("index-switch-to-if");
}

/// The whole ApproxMLIR pipeline for Triton in one entry (same passes, same
/// order as `approx-opt --approx-pipeline="workload=triton"`).
void addApproxTritonPipeline(mlir::PassManager *pm,
                             const std::vector<std::string> &args) {
  (void)args;
  mlir::approx::buildApproxPipeline(*pm, "triton");
}

void registerApproxTritonPipeline() {
  mlir::PassPipelineRegistration<>(
      "approx-triton-pipeline", "Full ApproxMLIR lowering for Triton TTIR",
      [](mlir::OpPassManager &pm) {
        mlir::approx::buildApproxPipeline(pm, "triton");
      });
}

} // namespace

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo passes[] = {
      {"emit-approx", LLVM_VERSION_STRING, addEmitApproxPass,
       registerEmitApproxPass},
      {"emit-management", LLVM_VERSION_STRING, addEmitManagementPass,
       registerEmitManagementPass},
      {"config-approx", LLVM_VERSION_STRING, addConfigApproxPass,
       registerConfigApproxPass},
      {"pre-emit-transform", LLVM_VERSION_STRING, addPreEmitTransformPass,
       registerPreEmitTransformPass},
      {"transform-approx", LLVM_VERSION_STRING, addTransformApproxPass,
       registerTransformApproxPass},
      {"finalize-approx", LLVM_VERSION_STRING, addFinalizeApproxPass,
       registerFinalizeApproxPass},
      {"index-switch-to-if", LLVM_VERSION_STRING, addIndexSwitchToIfPass,
       registerIndexSwitchToIfPass},
      {"approx-triton-pipeline", LLVM_VERSION_STRING, addApproxTritonPipeline,
       registerApproxTritonPipeline},
  };
  static mlir::triton::plugin::DialectInfo dialects[] = {
      {"approx", LLVM_VERSION_STRING, registerApproxDialect},
  };
  static mlir::triton::plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "ApproxPlugin",
      LLVM_VERSION_STRING,
      passes,
      kNumPasses,
      dialects,
      1,
      nullptr,
      0,
#ifdef TRITON_VERSION
      // Newer Triton adds a tritonVersion field to PluginInfo; older Triton
      // (e.g. 3f0a0085a) has no such field and no TRITON_VERSION macro.
      TRITON_VERSION,
#endif
  };
  return &info;
}
