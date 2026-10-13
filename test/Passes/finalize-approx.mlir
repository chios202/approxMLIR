// RUN: approx-opt %s --finalize-approx | FileCheck %s

func.func @foo(%x: f32) -> f32 {
  %r = "approx.knob"(%x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
    %0 = arith.addf %x, %x : f32
    approx.yield %0 : f32
  }) : (f32) -> f32
  %s = arith.mulf %r, %r : f32
  return %s : f32
}

// CHECK:       module {
// CHECK-NEXT:    func.func @foo
// CHECK-SAME:    (%[[ARG_X:.*]]: f32)
// CHECK-NEXT:    %[[SUM:.*]] = arith.addf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:    %[[PRODUCT:.*]] = arith.mulf %[[SUM]], %[[SUM]] : f32
// CHECK-NEXT:    return %[[PRODUCT]] : f32
