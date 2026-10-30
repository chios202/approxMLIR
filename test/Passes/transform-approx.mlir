// RUN: approx-opt %s --split-input-file --transform-approx | FileCheck %s

func.func @loop_perforate(%n: index, %m: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0.0 : f32
  %r = "approx.knob"(%n, %m) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
    "approx.transform"() <{knob_val = 3 : i32, transform_type = "loop_perforate"}> : () -> ()
    %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) -> f32 {
      %v = memref.load %m[%i] : memref<?xf32>
      %a = arith.addf %acc, %v : f32
      scf.yield %a : f32
    }
    approx.yield %s : f32
  }) : (index, memref<?xf32>) -> f32
  return %r : f32
}

// CHECK:       module {
// CHECK-NEXT:    func.func @loop_perforate(%[[ARG_N:.*]]: index, %[[ARG_M:.*]]: memref<?xf32>) -> f32 {
// CHECK-NEXT:      %[[ZERO_INDEX:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ZERO_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_N]], %[[ARG_M]]) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
// CHECK-NEXT:        %[[STEP_3:.*]] = arith.constant 3 : index
// CHECK-NEXT:        %[[LOOP_RESULT:.*]] = scf.for %[[LOOP_IV:.*]] = %[[ZERO_INDEX]] to %[[ARG_N]] step %[[STEP_3]] iter_args(%[[ACCUMULATOR:.*]] = %[[ZERO_F32]]) -> (f32) {
// CHECK-NEXT:          %[[LOADED:.*]] = memref.load %[[ARG_M]][%[[LOOP_IV]]] : memref<?xf32>
// CHECK-NEXT:          %[[NEW_ACCUMULATOR:.*]] = arith.addf %[[ACCUMULATOR]], %[[LOADED]] : f32
// CHECK-NEXT:          scf.yield %[[NEW_ACCUMULATOR]] : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        approx.yield %[[LOOP_RESULT]] : f32
// CHECK-NEXT:      }) : (index, memref<?xf32>) -> f32
// CHECK-NEXT:      return %[[KNOB_RESULT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @__internal_my_exp(%x: f32) -> f32 {
  %0 = math.exp %x : f32
  return %0 : f32
}
func.func @approx_my_exp_2(%x: f32) -> f32 {
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
func.func @my_exp(%x: f32) -> f32 {
  %r = "approx.knob"(%x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
    "approx.transform"() <{knob_val = 2 : i32, transform_type = "func_substitute"}> : () -> ()
    %0 = func.call @__internal_my_exp(%x) : (f32) -> f32
    approx.yield %0 : f32
  }) : (f32) -> f32
  return %r : f32
}

// CHECK:       module {
// CHECK-NEXT:    func.func @__internal_my_exp(%[[INTERNAL_ARG:.*]]: f32) -> f32 {
// CHECK-NEXT:      %[[EXP:.*]] = math.exp %[[INTERNAL_ARG]] : f32
// CHECK-NEXT:      return %[[EXP]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @approx_my_exp_2(%[[APPROX_ARG:.*]]: f32) -> f32 {
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:      %[[SIXTH:.*]] = arith.constant 0.166666672 : f32
// CHECK-NEXT:      %[[X_SQUARED:.*]] = arith.mulf %[[APPROX_ARG]], %[[APPROX_ARG]] : f32
// CHECK-NEXT:      %[[X_CUBED:.*]] = arith.mulf %[[X_SQUARED]], %[[APPROX_ARG]] : f32
// CHECK-NEXT:      %[[SECOND_ORDER_TERM:.*]] = arith.mulf %[[X_SQUARED]], %[[HALF]] : f32
// CHECK-NEXT:      %[[THIRD_ORDER_TERM:.*]] = arith.mulf %[[X_CUBED]], %[[SIXTH]] : f32
// CHECK-NEXT:      %[[FIRST_ORDER_SUM:.*]] = arith.addf %[[APPROX_ARG]], %[[ONE]] : f32
// CHECK-NEXT:      %[[SECOND_ORDER_SUM:.*]] = arith.addf %[[FIRST_ORDER_SUM]], %[[SECOND_ORDER_TERM]] : f32
// CHECK-NEXT:      %[[TAYLOR_ORDER_3:.*]] = arith.addf %[[SECOND_ORDER_SUM]], %[[THIRD_ORDER_TERM]] : f32
// CHECK-NEXT:      return %[[TAYLOR_ORDER_3]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @my_exp(%[[ARG_X:.*]]: f32) -> f32 {
// CHECK-NEXT:      %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_X]]) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
// CHECK-NEXT:        %[[APPROX_RESULT:.*]] = func.call @approx_my_exp_2(%[[ARG_X]]) : (f32) -> f32
// CHECK-NEXT:        approx.yield %[[APPROX_RESULT]] : f32
// CHECK-NEXT:      }) : (f32) -> f32
// CHECK-NEXT:      return %[[KNOB_RESULT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func @task_skip_else(%c: i1, %x: f32) -> f32 {
  %r = "approx.knob"(%c, %x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
    "approx.transform"() <{knob_val = 2 : i32, transform_type = "task_skipping"}> : () -> ()
    %y = scf.if %c -> f32 {
      %a = arith.addf %x, %x : f32
      scf.yield %a : f32
    } else {
      %b = arith.mulf %x, %x : f32
      scf.yield %b : f32
    }
    approx.yield %y : f32
  }) : (i1, f32) -> f32
  return %r : f32
}

// CHECK:       module {
// CHECK-NEXT:    func.func @task_skip_else(%[[ARG_C:.*]]: i1, %[[ARG_X:.*]]: f32) -> f32 {
// CHECK-NEXT:      %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_C]], %[[ARG_X]]) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
// CHECK-NEXT:        %[[PRODUCT:.*]] = arith.mulf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:        approx.yield %[[PRODUCT]] : f32
// CHECK-NEXT:      }) : (i1, f32) -> f32
// CHECK-NEXT:      return %[[KNOB_RESULT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
