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
    %r = arith.addf %one, %x : tensor<128xf32>
    return %r : tensor<128xf32>
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
