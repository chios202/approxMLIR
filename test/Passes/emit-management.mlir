// RUN: approx-opt %s --split-input-file --emit-management | FileCheck %s

module {
  "approx.util.annotation.decision_tree"() <{func_name = "relu_sum", transform_type = "loop_perforate", state_indices = array<i64: 0>, state_function = "get_state", num_thresholds = 2 : i32, thresholds_uppers = array<i32: 1000, 1000>, thresholds_lowers = array<i32: 0, 0>, decision_values = array<i32: 1, 2, 4>, thresholds = array<i32: 100, 500>, decisions = array<i32: 1, 2, 4>}> : () -> ()

  func.func @get_state(%n: i32) -> i32 {
    return %n : i32
  }

  func.func @relu_sum(%n: i32, %m: memref<?xf32>, %scale: f32) -> f32 {
    %0 = "approx.knob"(%n, %m, %scale) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0.0 : f32
      %ub = arith.index_cast %n : i32 to index
      %sum = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %zero) -> f32 {
        %v = memref.load %m[%i] : memref<?xf32>
        %pos = arith.cmpf ogt, %v, %zero : f32
        %r = scf.if %pos -> f32 {
          %s = arith.mulf %v, %scale : f32
          scf.yield %s : f32
        } else {
          scf.yield %zero : f32
        }
        %a = arith.addf %acc, %r : f32
        scf.yield %a : f32
      }
      %nf = arith.sitofp %n : i32 to f32
      %mean = arith.divf %sum, %nf : f32
      approx.yield %mean : f32
    }) : (i32, memref<?xf32>, f32) -> f32
    return %0 : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @get_state

// CHECK-LABEL: func.func @relu_sum
// CHECK-SAME:    (%[[ARG_N:.*]]: i32, %[[ARG_MEMREF:.*]]: memref<?xf32>, %[[ARG_SCALE:.*]]: f32) -> f32
// CHECK-DAG:     %[[ZERO_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[ONE_INDEX:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[ZERO_INDEX:.*]] = arith.constant 0 : index

// CHECK:         %[[KNOB_RESULT:.*]] = "approx.knob"(%[[ARG_N]], %[[ARG_MEMREF]], %[[ARG_SCALE]])
// CHECK-NEXT:      "approx.decide"(%[[ARG_N]])
// CHECK-SAME:        decision_values = array<i32: 1, 2, 4>
// CHECK-SAME:        decisions = array<i32: 1, 2, 4>
// CHECK-SAME:        num_thresholds = 2 : i32
// CHECK-SAME:        thresholds = array<i32: 100, 500>
// CHECK-SAME:        thresholds_lowers = array<i32: 0, 0>
// CHECK-SAME:        thresholds_uppers = array<i32: 1000, 1000>
// CHECK-SAME:        transform_type = "loop_perforate"

// CHECK-NEXT:      ^bb0(%[[STATE_ARG:.*]]: i32):
// CHECK-NEXT:        %[[STATE:.*]] = func.call @get_state(%[[STATE_ARG]]) : (i32) -> i32
// CHECK-NEXT:        approx.yield %[[STATE]] : i32
// CHECK-NEXT:      }) : (i32) -> ()

// CHECK-NEXT:      %[[UPPER_BOUND:.*]] = arith.index_cast %[[ARG_N]] : i32 to index
// CHECK-NEXT:      %[[SUM:.*]] = scf.for %[[LOOP_IV:.*]] = %[[ZERO_INDEX]] to %[[UPPER_BOUND]] step %[[ONE_INDEX]] iter_args(%[[ACCUMULATOR:.*]] = %[[ZERO_F32]]) -> (f32) {
// CHECK-NEXT:        %[[LOADED:.*]] = memref.load %[[ARG_MEMREF]][%[[LOOP_IV]]] : memref<?xf32>
// CHECK-NEXT:        %[[IS_POSITIVE:.*]] = arith.cmpf ogt, %[[LOADED]], %[[ZERO_F32]] : f32
//
// CHECK-NEXT:        %[[RELU:.*]] = scf.if %[[IS_POSITIVE]] -> (f32) {
// CHECK-NEXT:          %[[SCALED:.*]] = arith.mulf %[[LOADED]], %[[ARG_SCALE]] : f32
// CHECK-NEXT:          scf.yield %[[SCALED]] : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[ZERO_F32]] : f32
// CHECK-NEXT:        }
//
// CHECK-NEXT:        %[[NEW_ACCUMULATOR:.*]] = arith.addf %[[ACCUMULATOR]], %[[RELU]] : f32
// CHECK-NEXT:        scf.yield %[[NEW_ACCUMULATOR]] : f32
// CHECK-NEXT:      }

// CHECK-NEXT:      %[[N_AS_FLOAT:.*]] = arith.sitofp %[[ARG_N]] : i32 to f32
// CHECK-NEXT:      %[[MEAN:.*]] = arith.divf %[[SUM]], %[[N_AS_FLOAT]] : f32
// CHECK-NEXT:      approx.yield %[[MEAN]] : f32
// CHECK-NEXT:    }) : (i32, memref<?xf32>, f32) -> f32
// CHECK-NEXT:    return %[[KNOB_RESULT]] : f32

// -----

module {
  "approx.util.annotation.transform"() <{func_name = "foo", transform_type = "loop_perforate", knob_val = 3 : i32}> : () -> ()

  func.func @foo(%n: i32, %x: f32) -> f32 {
    %0 = "approx.knob"(%n, %x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
      %1 = arith.addf %x, %x : f32
      approx.yield %1 : f32
    }) : (i32, f32) -> f32
    return %0 : f32
  }
}

// CHECK:       module {
// CHECK-NEXT:    func.func @foo
// CHECK-NEXT:      "approx.knob"
// CHECK-NEXT:      "approx.transform"() <{knob_val = 3 : i32, transform_type = "loop_perforate"}>
// CHECK-NEXT:      arith.addf

// -----

module {
  func.func private @recover_impl(%a: f32, %b: f32) -> f32
  func.func private @checker_impl(%r: f32, %a: f32, %b: f32) -> i1

  "approx.util.annotation.try"() <{func_name = "gauss", recover = "@recover_impl", checker = "@checker_impl"}> : () -> ()

  func.func @gauss(%a: f32, %b: f32) -> f32 {
    %0 = "approx.knob"(%a, %b) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
      %d = arith.subf %a, %b : f32
      %sq = arith.mulf %d, %d : f32
      %e = math.exp %sq : f32
      %r = arith.addf %e, %a : f32
      approx.yield %r : f32
    }) : (f32, f32) -> f32
    return %0 : f32
  }
}

// CHECK-LABEL: func.func @gauss
// CHECK-SAME:    (%[[ARG_A:.*]]: f32, %[[ARG_B:.*]]: f32)
// CHECK:         "approx.knob"(%[[ARG_A]], %[[ARG_B]])
// CHECK-NEXT:      %[[DIFF:.*]] = arith.subf %[[ARG_A]], %[[ARG_B]] : f32
// CHECK-NEXT:      %[[DIFF_SQUARED:.*]] = arith.mulf %[[DIFF]], %[[DIFF]] : f32
// CHECK-NEXT:      %[[EXP:.*]] = math.exp %[[DIFF_SQUARED]] : f32
// CHECK-NEXT:      %[[RESULT:.*]] = arith.addf %[[EXP]], %[[ARG_A]] : f32

// CHECK-NEXT:      "approx.try"(%[[ARG_A]], %[[ARG_B]]) <{recover = "recover_impl"}>
// CHECK-NEXT:      ^bb0(%[[CHECK_ARG_A:.*]]: f32, %[[CHECK_ARG_B:.*]]: f32):
// CHECK-NEXT:        %[[CHECK_PASSED:.*]] = func.call @checker_impl(%[[RESULT]], %[[CHECK_ARG_A]], %[[CHECK_ARG_B]])
// CHECK-NEXT:        approx.yield %[[CHECK_PASSED]] : i1
// CHECK-NEXT:      }) : (f32, f32) -> ()

// CHECK-NEXT:      approx.yield %[[RESULT]] : f32
