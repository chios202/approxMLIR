// RUN: approx-opt %s --split-input-file --emit-approx | FileCheck %s

module {
  "approx.util.annotation.knob"() <{func_name = "foo"}> : () -> ()
  "approx.util.annotation.transform"() <{func_name = "foo", transform_type = "loop_perforate", knob_val = 2 : i32}> : () -> ()

  func.func @foo(%a: i32, %x: f32) -> f32 {
    %0 = arith.addf %x, %x : f32
    return %0 : f32
  }

  func.func @untouched(%x: f32) -> f32 {
    return %x : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    "approx.util.annotation.transform"()
// CHECK-SAME:    func_name = "foo"
// CHECK-SAME:    knob_val = 2 : i32
// CHECK-SAME:    transform_type = "loop_perforate"

// CHECK-NEXT:    func.func @foo
// CHECK-SAME:    (%[[ARG_A:.*]]: i32, %[[ARG_X:.*]]: f32) -> f32
// CHECK-NEXT:    %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_A]], %[[ARG_X]])
// CHECK-SAME:      <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}>
// CHECK-NEXT:      %[[SUM:.*]] = arith.addf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:      approx.yield %[[SUM]] : f32
// CHECK-NEXT:    }) : (i32, f32) -> f32

// CHECK-NEXT:    return %[[KNOB_RESULT]] : f32
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @untouched
// CHECK-SAME:    (%[[UNTOUCHED_ARG:.*]]: f32) -> f32
// CHECK-NEXT:    return %[[UNTOUCHED_ARG]] : f32
// CHECK-NEXT:  }

// -----

module {
  "approx.util.annotation.knob"() <{func_name = "sum"}> : () -> ()
  "approx.util.annotation.transform"() <{func_name = "sum", transform_type = "loop_perforate", knob_val = 2 : i32}> : () -> ()

  func.func @sum(%n: index, %m: memref<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) -> f32 {
      %v = memref.load %m[%i] : memref<?xf32>
      %a = arith.addf %acc, %v : f32
      scf.yield %a : f32
    }
    return %s : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    "approx.util.annotation.transform"()
// CHECK-SAME:    func_name = "sum"
// CHECK-SAME:    knob_val = 2 : i32
// CHECK-SAME:    transform_type = "loop_perforate"

// CHECK-NEXT:    func.func @sum
// CHECK-SAME:    (%[[ARG_N:.*]]: index, %[[ARG_MEMREF:.*]]: memref<?xf32>) -> f32
// CHECK-NEXT:    %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_N]], %[[ARG_MEMREF]])
// CHECK-NEXT:      %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ONE_INDEX:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[ZERO_F32:.*]] = arith.constant 0.000000e+00 : f32

// CHECK-NEXT:      %[[LOOP_RESULT:.*]] = scf.for %[[LOOP_IV:.*]] = %[[ZERO_INDEX]] to %[[ARG_N]] step %[[ONE_INDEX]] iter_args(%[[ACCUMULATOR:.*]] = %[[ZERO_F32]]) -> (f32) {
// CHECK-NEXT:        %[[LOADED:.*]] = memref.load %[[ARG_MEMREF]][%[[LOOP_IV]]] : memref<?xf32>
// CHECK-NEXT:        %[[NEW_ACCUMULATOR:.*]] = arith.addf %[[ACCUMULATOR]], %[[LOADED]] : f32
// CHECK-NEXT:        scf.yield %[[NEW_ACCUMULATOR]] : f32
// CHECK-NEXT:      }

// CHECK-NEXT:      approx.yield %[[LOOP_RESULT]] : f32
// CHECK-NEXT:    }) : (index, memref<?xf32>) -> f32
// CHECK-NEXT:    return %[[KNOB_RESULT]] : f32

// -----

module {
  "approx.util.annotation.decision_tree"() <{func_name = "bar", transform_type = "loop_perforate", state_indices = array<i64: 0>, state_function = "get_state", num_thresholds = 1 : i32, thresholds_uppers = array<i32: 100>, thresholds_lowers = array<i32: 0>, decision_values = array<i32: 1, 2>, thresholds = array<i32: 50>, decisions = array<i32: 1, 2>}> : () -> ()

  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @bar(%n: i32, %x: f32) -> f32 {
    %0 = arith.addf %x, %x : f32
    return %0 : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    "approx.util.annotation.decision_tree"()
// CHECK-SAME:    decision_values = array<i32: 1, 2>
// CHECK-SAME:    decisions = array<i32: 1, 2>
// CHECK-SAME:    func_name = "bar"
// CHECK-SAME:    num_thresholds = 1 : i32
// CHECK-SAME:    state_function = "get_state"
// CHECK-SAME:    state_indices = array<i64: 0>
// CHECK-SAME:    thresholds = array<i32: 50>
// CHECK-SAME:    thresholds_lowers = array<i32: 0>
// CHECK-SAME:    thresholds_uppers = array<i32: 100>
// CHECK-SAME:    transform_type = "loop_perforate"

// CHECK-NEXT:    func.func @get_state
// CHECK-SAME:    (%[[STATE_ARG:.*]]: i32) -> i32
// CHECK-NEXT:      return %[[STATE_ARG]] : i32
// CHECK-NEXT:    }

// CHECK-LABEL: func.func @bar
// CHECK-SAME:    (%[[ARG_N:.*]]: i32, %[[ARG_X:.*]]: f32) -> f32
// CHECK-NEXT:    %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_N]], %[[ARG_X]])
// CHECK-NEXT:      %[[SUM:.*]] = arith.addf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:      approx.yield %[[SUM]] : f32
// CHECK-NEXT:    }) : (i32, f32) -> f32
// CHECK-NEXT:    return %[[KNOB_RESULT]] : f32
