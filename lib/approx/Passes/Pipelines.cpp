/**
 * Pipelines.cpp - Named ApproxMLIR pass pipelines
 *
 * One place that defines the order of the ApproxMLIR passes:
 *
 *   pre-emit-transform -> emit-approx -> emit-management -> config-approx
 *     -> transform-approx -> finalize-approx -> <host adapter>
 *
 * The workload (same values as WorkloadType in the Python runtime) picks the
 * host adapter, which turns the decision tree's scf.index_switch into
 * something the workload's host compiler accepts:
 *   ml     (StableHLO -> IREE): legalize-to-stablehlo  (switch -> stablehlo.case)
 *   triton (TTIR -> TTGIR):     index-switch-to-if     (switch -> scf.if chain)
 *   cpp    (Polygeist):         none
 *
 * pre-emit-transform only acts on convert_to_call annotations, so it is always
 * included and is a no-op for transforms other than func_substitute.
 *
 * Usage: approx-opt input.mlir --approx-pipeline="workload=ml"
 *        approx-opt input.mlir --approx-pipeline="workload=ml legalize=false"
 */
#include "approx/Passes/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

enum class Workload { Ml, Triton, Cpp };

llvm::StringRef workloadName(Workload workload) {
  switch (workload) {
  case Workload::Ml:
    return "ml";
  case Workload::Triton:
    return "triton";
  case Workload::Cpp:
    return "cpp";
  }
  llvm_unreachable("unknown workload");
}

struct ApproxPipelineOptions
    : public PassPipelineOptions<ApproxPipelineOptions> {
  Option<Workload> workload{
      *this, "workload", llvm::cl::desc("Workload type (picks the host adapter pass)"),
      llvm::cl::init(Workload::Ml),
      llvm::cl::values(
          clEnumValN(Workload::Ml, "ml", "JAX / StableHLO -> IREE (adds legalize-to-stablehlo)"),
          clEnumValN(Workload::Triton, "triton", "Triton TTIR (adds index-switch-to-if)"),
          clEnumValN(Workload::Cpp, "cpp", "C++ / Polygeist (no adapter pass)"))};
  Option<bool> legalize{
      *this, "legalize",
      llvm::cl::desc("Run the host adapter pass (ml: legalize-to-stablehlo)"),
      llvm::cl::init(true)};
};

} // namespace

namespace mlir {
namespace approx {

void buildApproxPipeline(OpPassManager &pm, StringRef workload, bool legalize) {
  if (workload != "ml" && workload != "triton" && workload != "cpp")
    llvm::report_fatal_error(llvm::Twine("approx-pipeline: unknown workload '") +
                             workload + "' (expected ml, triton, or cpp)");

  pm.addPass(createPreEmitTransformationPass());
  pm.addPass(createEmitApproxPass());
  pm.addPass(createEmitManagementPass());
  pm.addPass(createConfigApproxPass());
  pm.addPass(createTransformApproxPass());
  pm.addPass(createFinalizeApproxPass());

  if (!legalize)
    return;
  if (workload == "ml")
    pm.addPass(createLegalizeToStablehloPass());
  else if (workload == "triton")
    pm.addPass(createIndexSwitchToIfPass());
}

void registerApproxPipelines() {
  PassPipelineRegistration<ApproxPipelineOptions>(
      "approx-pipeline",
      "Full ApproxMLIR lowering: annotations -> approx ops -> transformed "
      "host IR (options: workload=ml|triton|cpp, legalize=true|false)",
      [](OpPassManager &pm, const ApproxPipelineOptions &options) {
        buildApproxPipeline(pm, workloadName(options.workload), options.legalize);
      });
}

} // namespace approx
} // namespace mlir
