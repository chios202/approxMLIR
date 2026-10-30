import approx_runtime as ar 

import torch
import triton
import triton.language as tl
LOG2_E = tl.constexpr(1.4426950408889634)
LN_2 = tl.constexpr(0.6931471805599453)
MIN_INPUT = tl.constexpr(-87.0) 


@triton.jit
def _split_exp(x):
    t = tl.maximum(x, MIN_INPUT) * LOG2_E
    i = tl.floor(t)
    two_i = ((i.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)
    return two_i, t - i


@triton.jit
def taylor(x, n):
    two_i, f = _split_exp(x)
    r = f * LN_2
    return two_i * (1.0 + r * (1.0 + r * 0.5))


@triton.jit
def minimax(x, n):
    two_i, f = _split_exp(x)
    return two_i * (1.001724720001220703125
                    + f * (0.657636463642120361328125 + f * 0.3371892869472503662109375))


@triton.jit
def row_length(n):
    return n


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=row_length,
    state_indices=[1],                  
    thresholds=[512],                   
    decisions=[1, 2],       
    transform_type="func_substitute",
    approx_kernels={1: taylor, 2: minimax},
))
@triton.jit
def softmax_exp(x, n):
    return tl.exp(x)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x = tl.load(in_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    numerator = softmax_exp(x - tl.max(x, axis=0), n_cols)
    tl.store(out_ptr + row * n_cols + offsets,
             numerator / tl.sum(numerator, axis=0), mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.manual_seed(0)
    short_rows = torch.randn(1024, 256, device="cuda") * 4 
    long_rows = torch.randn(1024, 2048, device="cuda") * 4    
    ar.enable_triton_approx()
    for name, x in [("256 columns (taylor)", short_rows),
                    ("2048 columns (minimax)", long_rows)]:
        error = (softmax(x) - torch.softmax(x, dim=1)).abs().max().item()
        print(f"{name:>24}: max |error| vs torch.softmax = {error:.2e}")
    ar.disable_triton_approx()

    torch.testing.assert_close(softmax(long_rows), torch.softmax(long_rows, dim=1),
                               rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    main()
