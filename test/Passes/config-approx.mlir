// RUN: approx-opt %s --split-input-file --config-approx | FileCheck %s

module {
  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @foo(%n: i32, %x: f32) -> f32 {
    %0 = "approx.knob"(%n, %x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
      "approx.decide"(%n) <{decision_values = array<i32: 1, 2>, decisions = array<i32: 1, 2>, num_thresholds = 1 : i32, thresholds = array<i32: 50>, thresholds_lowers = array<i32: 0>, thresholds_uppers = array<i32: 100>, transform_type = "loop_perforate"}> ({
      ^bb0(%s: i32):
        %st = func.call @get_state(%s) : (i32) -> i32
        approx.yield %st : i32
      }) : (i32) -> ()
      %1 = arith.addf %x, %x : f32
      approx.yield %1 : f32
    }) : (i32, f32) -> f32
    return %0 : f32
  }
}

// CHECK-LABEL: func.func @foo
// CHECK-SAME:    (%[[ARG_N:.*]]: i32, %[[ARG_X:.*]]: f32)
// CHECK-NEXT:    "approx.knob"

// CHECK-NEXT:      %[[STATE:.*]] = func.call @get_state(%[[ARG_N]])
// CHECK-NEXT:      %[[THRESHOLD:.*]] = arith.constant 50 : i32
// CHECK-NEXT:      %[[STATE_GE_THRESHOLD:.*]] = arith.cmpi sge, %[[STATE]], %[[THRESHOLD]] : i32
// CHECK:           %[[REGION:.*]] = arith.select %[[STATE_GE_THRESHOLD]]
// CHECK:           %[[REGION_INDEX:.*]] = arith.index_cast %[[REGION]] : i32 to index

// CHECK:           %[[SWITCH_RESULT:.*]] = scf.index_switch %[[REGION_INDEX]] -> f32
//
// CHECK-NEXT:      case 0 {
// CHECK-NEXT:        "approx.transform"() <{knob_val = 1 : i32, transform_type = "loop_perforate"}>
// CHECK-NEXT:        %[[CASE0_SUM:.*]] = arith.addf %[[ARG_X]], %[[ARG_X]]
// CHECK-NEXT:        scf.yield %[[CASE0_SUM]] : f32
//
// CHECK:           case 1 {
// CHECK-NEXT:        "approx.transform"() <{knob_val = 2 : i32, transform_type = "loop_perforate"}>
//
// CHECK:           default {
// CHECK-NEXT:        "approx.transform"() <{knob_val = 1 : i32, transform_type = "loop_perforate"}>

// CHECK:           approx.yield %[[SWITCH_RESULT]] : f32

// -----

module {
  func.func private @recover_impl(%a: f32, %b: f32) -> f32
  func.func private @checker_impl(%r: f32, %a: f32, %b: f32) -> i1

  func.func @safe_add(%a: f32, %b: f32) -> f32 {
    %0 = "approx.knob"(%a, %b) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
      %1 = arith.addf %a, %b : f32
      "approx.try"(%a, %b) <{recover = "recover_impl"}> ({
      ^bb0(%ca: f32, %cb: f32):
        %ok = func.call @checker_impl(%1, %ca, %cb) : (f32, f32, f32) -> i1
        approx.yield %ok : i1
      }) : (f32, f32) -> ()
      approx.yield %1 : f32
    }) : (f32, f32) -> f32
    return %0 : f32
  }
}

// CHECK-LABEL: func.func @safe_add
// CHECK-SAME:    (%[[ARG_A:.*]]: f32, %[[ARG_B:.*]]: f32)
// CHECK-NEXT:    "approx.knob"

// CHECK-NEXT:      %[[SUM:.*]] = arith.addf %[[ARG_A]], %[[ARG_B]] : f32
// CHECK-NEXT:      %[[CHECK_PASSED:.*]] = func.call @checker_impl(%[[SUM]], %[[ARG_A]], %[[ARG_B]])

// CHECK-NEXT:      %[[IF_RESULT:.*]] = scf.if %[[CHECK_PASSED]] -> (f32) {
// CHECK-NEXT:        scf.yield %[[SUM]] : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[RECOVERED:.*]] = func.call @recover_impl(%[[ARG_A]], %[[ARG_B]])
// CHECK-NEXT:        scf.yield %[[RECOVERED]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      approx.yield %[[IF_RESULT]] : f32
