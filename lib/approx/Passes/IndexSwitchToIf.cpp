/**
 * IndexSwitchToIf.cpp - Rewrite scf.index_switch as a chain of scf.if
 *
 *   %r = scf.index_switch %idx -> T
 *        case 0 {A}  case 1 {B}  default {D}
 * becomes
 *   %r = scf.if (%idx == 0) -> T {A}
 *        else { scf.if (%idx == 1) -> T {B} else {D} }
 *
 * config-approx lowers a decision tree to scf.index_switch (one case per
 * region of the tree). Triton's TTIR -> TTGIR conversion has no rule for
 * scf.index_switch with tensor results, but handles scf.if, so the Triton
 * pipeline (approx-pipeline="workload=triton") ends with this pass.
 */
#include "PassDetails.h"
#include "approx/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace approx;

namespace {

struct IndexSwitchToIfPattern : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  /// Builds `if (idx == case i) {case i} else {...}` at the insertion point;
  /// the else branch is the chain for cases i+1.. or, after the last case,
  /// the default region. Moves the case regions out of `switchOp`.
  static ValueRange buildChain(PatternRewriter &rewriter,
                               scf::IndexSwitchOp switchOp, unsigned i) {
    Location loc = switchOp.getLoc();
    Value caseValue =
        arith::ConstantIndexOp::create(rewriter, loc, switchOp.getCases()[i]);
    Value cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                       switchOp.getArg(), caseValue);
    auto ifOp = scf::IfOp::create(rewriter, loc, switchOp.getResultTypes(),
                                  cond, /*addThenBlock=*/false,
                                  /*addElseBlock=*/false);
    rewriter.inlineRegionBefore(switchOp.getCaseRegions()[i],
                                ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());

    if (i + 1 == switchOp.getNumCases()) {
      rewriter.inlineRegionBefore(switchOp.getDefaultRegion(),
                                  ifOp.getElseRegion(),
                                  ifOp.getElseRegion().end());
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ifOp.getElseRegion());
      ValueRange inner = buildChain(rewriter, switchOp, i + 1);
      scf::YieldOp::create(rewriter, loc, inner);
    }
    return ifOp.getResults();
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    // A switch with only a default region is left as is (nothing to compare).
    if (switchOp.getNumCases() == 0)
      return rewriter.notifyMatchFailure(switchOp, "switch has no cases");
    rewriter.setInsertionPoint(switchOp);
    ValueRange results = buildChain(rewriter, switchOp, 0);
    rewriter.replaceOp(switchOp, results);
    return success();
  }
};

struct IndexSwitchToIfPass
    : public mlir::approx::impl::IndexSwitchToIfPassBase<IndexSwitchToIfPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<IndexSwitchToIfPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace approx {
std::unique_ptr<Pass> createIndexSwitchToIfPass() {
  return std::make_unique<IndexSwitchToIfPass>();
}
} // namespace approx
} // namespace mlir
