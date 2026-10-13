// RUN: approx-opt %s --approx-pipeline="workload=cpp" | FileCheck %s

module {
  "approx.util.annotation.convert_to_call"() <{func_name = "my_exp"}> : () -> ()
  "approx.util.annotation.decision_tree"() <{func_name = "my_exp", transform_type = "func_substitute", state_indices = array<i64: 0>, state_function = "get_state", num_thresholds = 1 : i32, thresholds_uppers = array<i32: 100>, thresholds_lowers = array<i32: 0>, decision_values = array<i32: 0, 1>, thresholds = array<i32: 50>, decisions = array<i32: 0, 1>}> : () -> ()

  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @my_exp(%n: i32, %x: f32) -> f32 {
    %0 = math.exp %x : f32
    return %0 : f32
  }

  func.func @approx_my_exp_1(%n: i32, %x: f32) -> f32 {
    %c = arith.constant 1.0 : f32
    %0 = arith.addf %x, %c : f32
    return %0 : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @get_state

// CHECK-LABEL: func.func @__internal_my_exp
// CHECK:         math.exp

// CHECK-LABEL: func.func @my_exp
// CHECK-SAME:    (%[[ARG_N:.*]]: i32, %[[ARG_X:.*]]: f32)
// CHECK:         call @get_state(%[[ARG_N]])
// CHECK:         %[[SWITCH_RESULT:.*]] = scf.index_switch

// CHECK-NEXT:    case 0 {
// CHECK-NEXT:      func.call @__internal_my_exp(%[[ARG_N]], %[[ARG_X]])
//
// CHECK:         case 1 {
// CHECK-NEXT:      func.call @approx_my_exp_1(%[[ARG_N]], %[[ARG_X]])
//
// CHECK:         default {
// CHECK-NEXT:      func.call @__internal_my_exp(%[[ARG_N]], %[[ARG_X]])

// CHECK:         return %[[SWITCH_RESULT]] : f32

// CHECK-LABEL: func.func @approx_my_exp_1
