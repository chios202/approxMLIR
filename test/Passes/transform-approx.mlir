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
  return %x : f32
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
// CHECK-NEXT:      return %[[APPROX_ARG]] : f32
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
