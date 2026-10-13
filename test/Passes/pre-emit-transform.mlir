// RUN: approx-opt %s --pre-emit-transform | FileCheck %s

module {
  "approx.util.annotation.convert_to_call"() <{func_name = "my_exp"}> : () -> ()

  func.func @my_exp(%a: i32, %x: f32) -> f32 {
    %0 = math.exp %x : f32
    return %0 : f32
  }

  func.func @untouched(%x: f32) -> f32 {
    return %x : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @__internal_my_exp
// CHECK-SAME:    (%[[ARG_A:.*]]: i32, %[[ARG_X:.*]]: f32) -> f32
// CHECK-NEXT:    %[[EXP:.*]] = math.exp %[[ARG_X]] : f32
// CHECK-NEXT:    return %[[EXP]] : f32

// CHECK-LABEL: func.func @my_exp
// CHECK-SAME:    (%[[ARG_A:.*]]: i32, %[[ARG_X:.*]]: f32) -> f32
// CHECK-NEXT:    %[[CALL_RESULT:.*]] = call @__internal_my_exp(%[[ARG_A]], %[[ARG_X]]) : (i32, f32) -> f32
// CHECK-NEXT:    return %[[CALL_RESULT]] : f32

// CHECK-LABEL: func.func @untouched
// CHECK-SAME:    (%[[UNTOUCHED_ARG:.*]]: f32) -> f32
// CHECK-NEXT:    return %[[UNTOUCHED_ARG]] : f32
