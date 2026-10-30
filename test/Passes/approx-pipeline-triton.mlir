// RUN: approx-opt %s --approx-pipeline="workload=triton" | FileCheck %s

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

  func.func @my_exp(%n: i32, %x: tensor<128xf32>) -> tensor<128xf32> {
    %e = math.exp %x : tensor<128xf32>
    return %e : tensor<128xf32>
  }

  func.func @approx_my_exp_1(%n: i32, %x: tensor<128xf32>) -> tensor<128xf32> {
    %one = arith.constant dense<1.0> : tensor<128xf32>
    %half = arith.constant dense<0.5> : tensor<128xf32>
    %sixth = arith.constant dense<0.166666672> : tensor<128xf32>
    %x_squared = arith.mulf %x, %x : tensor<128xf32>
    %x_cubed = arith.mulf %x_squared, %x : tensor<128xf32>
    %second_order_term = arith.mulf %x_squared, %half : tensor<128xf32>
    %third_order_term = arith.mulf %x_cubed, %sixth : tensor<128xf32>
    %first_order_sum = arith.addf %one, %x : tensor<128xf32>
    %second_order_sum = arith.addf %first_order_sum, %second_order_term : tensor<128xf32>
    %taylor_order_3 = arith.addf %second_order_sum, %third_order_term : tensor<128xf32>
    return %taylor_order_3 : tensor<128xf32>
  }
}

// CHECK-LABEL: func.func @my_exp(
// CHECK-SAME:      %[[N:.*]]: i32, %[[X:.*]]: tensor<128xf32>) -> tensor<128xf32>
// CHECK:         %[[DECISION:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:    %[[IS_EXACT:.*]] = arith.cmpi eq, %[[DECISION]], %c0 : index
// CHECK-NEXT:    %[[RESULT:.*]] = scf.if %[[IS_EXACT]] -> (tensor<128xf32>) {
// CHECK-NEXT:      %[[EXACT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:      scf.yield %[[EXACT]] : tensor<128xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[IS_APPROX:.*]] = arith.cmpi eq, %[[DECISION]], %c1 : index
// CHECK-NEXT:      %[[INNER:.*]] = scf.if %[[IS_APPROX]] -> (tensor<128xf32>) {
// CHECK-NEXT:        %[[APPROX:.*]] = func.call @approx_my_exp_1(%[[N]], %[[X]])
// CHECK-NEXT:        scf.yield %[[APPROX]] : tensor<128xf32>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[DEFAULT:.*]] = func.call @__internal_my_exp(%[[N]], %[[X]])
// CHECK-NEXT:        scf.yield %[[DEFAULT]] : tensor<128xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[INNER]] : tensor<128xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RESULT]] : tensor<128xf32>
