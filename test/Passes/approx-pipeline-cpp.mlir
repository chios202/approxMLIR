// RUN: approx-opt %s --approx-pipeline="workload=cpp" | FileCheck %s

module {
  "approx.util.annotation.convert_to_call"() <{func_name = "my_exp"}> : () -> ()
  "approx.util.annotation.decision_tree"() <{
    func_name = "my_exp",
    transform_type = "func_substitute",
    state_indices = array<i64: 0>,
    state_function = "get_state",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 1000>,
    thresholds_lowers = array<i32: 0>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 100>,
    decisions = array<i32: 0, 1>
  }> : () -> ()

  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @my_exp(%n: i32, %x: f32) -> f32 {
    %e = math.exp %x : f32
    return %e : f32
  }

  func.func @approx_my_exp_1(%n: i32, %x: f32) -> f32 {
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %sixth = arith.constant 0.166666672 : f32
    %x_squared = arith.mulf %x, %x : f32
    %x_cubed = arith.mulf %x_squared, %x : f32
    %second_order_term = arith.mulf %x_squared, %half : f32
    %third_order_term = arith.mulf %x_cubed, %sixth : f32
    %first_order_sum = arith.addf %one, %x : f32
    %second_order_sum = arith.addf %first_order_sum, %second_order_term : f32
    %taylor_order_3 = arith.addf %second_order_sum, %third_order_term : f32
    return %taylor_order_3 : f32
  }
}

// CHECK-LABEL: func.func @my_exp(
// CHECK-SAME:      %[[N:.*]]: i32, %[[X:.*]]: f32) -> f32
// CHECK:         %[[DECISION:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:    %[[RESULT:.*]] = scf.index_switch %[[DECISION]] -> f32
// CHECK-NEXT:    case 0 {
// CHECK-NEXT:      %[[EXACT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:      scf.yield %[[EXACT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %[[APPROX:.*]] = func.call @approx_my_exp_1(%[[N]], %[[X]])
// CHECK-NEXT:      scf.yield %[[APPROX]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      %[[DEFAULT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:      scf.yield %[[DEFAULT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : f32
