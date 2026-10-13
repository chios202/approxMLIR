// RUN: approx-opt %s --transform-approx -verify-diagnostics

func.func @__internal_my_exp(%x: f32) -> f32 {
  %0 = math.exp %x : f32
  return %0 : f32
}
func.func @my_exp(%x: f32) -> f32 {
  %r = "approx.knob"(%x) <{QoS_in = array<i32>, QoS_out = array<i32>, id = 0 : i32, rf = 0 : i32}> ({
    // expected-error @+1 {{func_substitute target function not found in module}}
    "approx.transform"() <{knob_val = 5 : i32, transform_type = "func_substitute"}> : () -> ()
    %0 = func.call @__internal_my_exp(%x) : (f32) -> f32
    approx.yield %0 : f32
  }) : (f32) -> f32
  return %r : f32
}
