// RUN: approx-opt %s --index-switch-to-if | FileCheck %s

func.func @pick(%idx: index, %x: f32) -> f32 {
  %r = scf.index_switch %idx -> f32
  case 0 {
    %a = arith.addf %x, %x : f32
    scf.yield %a : f32
  }
  case 1 {
    %m = arith.mulf %x, %x : f32
    scf.yield %m : f32
  }
  default {
    scf.yield %x : f32
  }
  return %r : f32
}

// CHECK:       module {
// CHECK-NEXT:    func.func @pick(%[[ARG_IDX:.*]]: index, %[[ARG_X:.*]]: f32) -> f32 {
// CHECK-NEXT:      %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[IS_CASE_0:.*]] = arith.cmpi eq, %[[ARG_IDX]], %[[ZERO]] : index
// CHECK-NEXT:      %[[RESULT:.*]] = scf.if %[[IS_CASE_0]] -> (f32) {
// CHECK-NEXT:        %[[SUM:.*]] = arith.addf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:        scf.yield %[[SUM]] : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[IS_CASE_1:.*]] = arith.cmpi eq, %[[ARG_IDX]], %[[ONE]] : index
// CHECK-NEXT:        %[[INNER_RESULT:.*]] = scf.if %[[IS_CASE_1]] -> (f32) {
// CHECK-NEXT:          %[[PRODUCT:.*]] = arith.mulf %[[ARG_X]], %[[ARG_X]] : f32
// CHECK-NEXT:          scf.yield %[[PRODUCT]] : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.yield %[[ARG_X]] : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[INNER_RESULT]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[RESULT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
