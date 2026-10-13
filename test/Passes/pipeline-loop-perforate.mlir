// RUN: approx-opt %s --approx-pipeline="workload=cpp" | FileCheck %s

module {
  "approx.util.annotation.decision_tree"() <{func_name = "sum", transform_type = "loop_perforate", state_indices = array<i64: 0>, state_function = "get_state", num_thresholds = 1 : i32, thresholds_uppers = array<i32: 100>, thresholds_lowers = array<i32: 0>, decision_values = array<i32: 1, 2>, thresholds = array<i32: 50>, decisions = array<i32: 1, 2>}> : () -> ()

  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @sum(%n: i32, %m: memref<?xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %ub = arith.index_cast %n : i32 to index
    %s = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %z) -> f32 {
      %v = memref.load %m[%i] : memref<?xf32>
      %a = arith.addf %acc, %v : f32
      scf.yield %a : f32
    }
    return %s : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @get_state

// CHECK-LABEL: func.func @sum
// CHECK-DAG:     %[[STEP_1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[STEP_2:.*]] = arith.constant 2 : index

// CHECK:         %[[STATE:.*]] = call @get_state
// CHECK:         arith.cmpi sge, %[[STATE]]

// CHECK:         %[[SWITCH_RESULT:.*]] = scf.index_switch
// CHECK-NEXT:    case 0 {
// CHECK:           scf.for {{.*}} step %[[STEP_1]]
//
// CHECK:         case 1 {
// CHECK:           scf.for {{.*}} step %[[STEP_2]]
//
// CHECK:         default {
// CHECK:           scf.for {{.*}} step %[[STEP_1]]

// CHECK:         return %[[SWITCH_RESULT]] : f32
