// RUN: approx-opt %s --approx-pipeline="workload=ml" | FileCheck %s

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

  func.func @get_state(%n: tensor<i32>) -> tensor<i32> {
    return %n : tensor<i32>
  }

  func.func @my_exp(%n: tensor<i32>, %x: tensor<f32>) -> tensor<f32> {
    %e = stablehlo.exponential %x : tensor<f32>
    return %e : tensor<f32>
  }

  func.func @approx_my_exp_1(%n: tensor<i32>, %x: tensor<f32>) -> tensor<f32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %r = stablehlo.add %one, %x : tensor<f32>
    return %r : tensor<f32>
  }
}

// CHECK-LABEL: func.func @my_exp(
// CHECK-SAME:      %[[N:.*]]: tensor<i32>, %[[X:.*]]: tensor<f32>) -> tensor<f32>
// CHECK:         %[[IN_RANGE:.*]] = stablehlo.compare LT, %[[RAW:.*]], %[[MAX:.*]] :
// CHECK-NEXT:    %[[DECISION:.*]] = stablehlo.select %[[IN_RANGE]], %[[RAW]], %[[MAX]] : tensor<i1>, tensor<i32>
// CHECK-NEXT:    %[[RESULT:.*]] = "stablehlo.case"(%[[DECISION]]) ({
// CHECK-NEXT:      %[[EXACT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:      stablehlo.return %[[EXACT]] : tensor<f32>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %[[APPROX:.*]] = func.call @approx_my_exp_1(%[[N]], %[[X]])
// CHECK-NEXT:      stablehlo.return %[[APPROX]] : tensor<f32>
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %[[DEFAULT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:      stablehlo.return %[[DEFAULT]] : tensor<f32>
// CHECK-NEXT:    }) : (tensor<i32>) -> tensor<f32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<f32>
