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
    %min_input = arith.constant -8.700000e+01 : f32
    %log2_e = arith.constant 1.44269502 : f32
    %exponent_bias = arith.constant 127 : i32
    %mantissa_bits = arith.constant 23 : i32
    %ln_2 = arith.constant 0.693147182 : f32
    %sixth = arith.constant 0.166666672 : f32
    %half = arith.constant 5.000000e-01 : f32
    %one = arith.constant 1.000000e+00 : f32
    %clamped = arith.maxnumf %x, %min_input : f32
    %t = arith.mulf %clamped, %log2_e : f32
    %i = math.floor %t : f32
    %i_int = arith.fptosi %i : f32 to i32
    %biased_exponent = arith.addi %i_int, %exponent_bias : i32
    %exponent_field = arith.shli %biased_exponent, %mantissa_bits : i32
    %two_i = arith.bitcast %exponent_field : i32 to f32
    %f = arith.subf %t, %i : f32
    %r = arith.mulf %f, %ln_2 : f32
    %r_sixth = arith.mulf %r, %sixth : f32
    %inner = arith.addf %r_sixth, %half : f32
    %r_inner = arith.mulf %r, %inner : f32
    %middle = arith.addf %r_inner, %one : f32
    %r_middle = arith.mulf %r, %middle : f32
    %taylor_order_3 = arith.addf %r_middle, %one : f32
    %result = arith.mulf %two_i, %taylor_order_3 : f32
    return %result : f32
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
// CHECK-SAME:    (%{{.*}}: i32, %[[X:.*]]: f32) -> f32
// CHECK-NEXT:    %[[MIN_INPUT:.*]] = arith.constant -8.700000e+01 : f32
// CHECK-NEXT:    %[[LOG2_E:.*]] = arith.constant 1.44269502 : f32
// CHECK-NEXT:    %[[EXPONENT_BIAS:.*]] = arith.constant 127 : i32
// CHECK-NEXT:    %[[MANTISSA_BITS:.*]] = arith.constant 23 : i32
// CHECK-NEXT:    %[[LN_2:.*]] = arith.constant 0.693147182 : f32
// CHECK-NEXT:    %[[SIXTH:.*]] = arith.constant 0.166666672 : f32
// CHECK-NEXT:    %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1.000000e+00 : f32

// CHECK-NEXT:    %[[CLAMPED:.*]] = arith.maxnumf %[[X]], %[[MIN_INPUT]] : f32
// CHECK-NEXT:    %[[T:.*]] = arith.mulf %[[CLAMPED]], %[[LOG2_E]] : f32
// CHECK-NEXT:    %[[I:.*]] = math.floor %[[T]] : f32
// CHECK-NEXT:    %[[I_INT:.*]] = arith.fptosi %[[I]] : f32 to i32
// CHECK-NEXT:    %[[BIASED_EXPONENT:.*]] = arith.addi %[[I_INT]], %[[EXPONENT_BIAS]] : i32
// CHECK-NEXT:    %[[EXPONENT_FIELD:.*]] = arith.shli %[[BIASED_EXPONENT]], %[[MANTISSA_BITS]] : i32
// CHECK-NEXT:    %[[TWO_I:.*]] = arith.bitcast %[[EXPONENT_FIELD]] : i32 to f32
// CHECK-NEXT:    %[[F:.*]] = arith.subf %[[T]], %[[I]] : f32
// CHECK-NEXT:    %[[R:.*]] = arith.mulf %[[F]], %[[LN_2]] : f32

// CHECK-NEXT:    %[[R_SIXTH:.*]] = arith.mulf %[[R]], %[[SIXTH]] : f32
// CHECK-NEXT:    %[[INNER:.*]] = arith.addf %[[R_SIXTH]], %[[HALF]] : f32
// CHECK-NEXT:    %[[R_INNER:.*]] = arith.mulf %[[R]], %[[INNER]] : f32
// CHECK-NEXT:    %[[MIDDLE:.*]] = arith.addf %[[R_INNER]], %[[ONE]] : f32
// CHECK-NEXT:    %[[R_MIDDLE:.*]] = arith.mulf %[[R]], %[[MIDDLE]] : f32
// CHECK-NEXT:    %[[TAYLOR_ORDER_3:.*]] = arith.addf %[[R_MIDDLE]], %[[ONE]] : f32
// CHECK-NEXT:    %[[RESULT:.*]] = arith.mulf %[[TWO_I]], %[[TAYLOR_ORDER_3]] : f32
// CHECK-NEXT:    return %[[RESULT]] : f32
