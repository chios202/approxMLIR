// RUN: approx-opt %s --legalize-to-stablehlo | FileCheck %s

func.func @get_state(%n: tensor<i32>) -> tensor<i32> {
  return %n : tensor<i32>
}

func.func @foo(%n: tensor<i32>, %x: tensor<4xf32>) -> tensor<4xf32> {
  %st = call @get_state(%n) : (tensor<i32>) -> tensor<i32>
  %s = tensor.extract %st[] : tensor<i32>
  %c50 = arith.constant 50 : i32
  %ge = arith.cmpi sge, %s, %c50 : i32
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %sel = arith.select %ge, %c1, %c0 : i32
  %idx = arith.index_cast %sel : i32 to index
  %r = scf.index_switch %idx -> tensor<4xf32>
  case 0 {
    %a = stablehlo.add %x, %x : tensor<4xf32>
    scf.yield %a : tensor<4xf32>
  }
  case 1 {
    %m = stablehlo.multiply %x, %x : tensor<4xf32>
    scf.yield %m : tensor<4xf32>
  }
  default {
    %a = stablehlo.add %x, %x : tensor<4xf32>
    scf.yield %a : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @foo
// CHECK-SAME:    (%[[ARG_N:.*]]: tensor<i32>, %[[ARG_X:.*]]: tensor<4xf32>)

// CHECK:         %[[STATE:.*]] = call @get_state(%[[ARG_N]])
// CHECK:         stablehlo.compare GE, %[[STATE]]

// CHECK:         %[[CASE_RESULT:.*]] = "stablehlo.case"
// CHECK-NEXT:      stablehlo.add %[[ARG_X]], %[[ARG_X]]
// CHECK:           stablehlo.return
//
// CHECK:         }, {
// CHECK-NEXT:      stablehlo.multiply %[[ARG_X]], %[[ARG_X]]
// CHECK:           stablehlo.return
//
// CHECK:         }, {
// CHECK-NEXT:      stablehlo.add %[[ARG_X]], %[[ARG_X]]
// CHECK:           stablehlo.return
// CHECK:         }) : (tensor<i32>) -> tensor<4xf32>

// CHECK:         return %[[CASE_RESULT]] : tensor<4xf32>
