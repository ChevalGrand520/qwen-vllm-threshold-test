# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 T2026103389912286

from vllm.triton_utils import triton
import triton.language as tl
import triton.testing

import torch

import copy
import typing as tp

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


_t_ranges = {
    (1, 15): (0, [1, 2, 4, 8, 16]),
    (16, 31): (1, [16, 32]),
    (32, 63): (2, [32, 64]),
    (64, 127): (3, [32, 64, 128]),
    (128, 255): (4, [32, 64, 128]),
    (256, 511): (5, [32, 64, 128]),
    (512, 767): (6, [32, 64, 128]),
    (768, 1023): (7, [32, 64, 128, 256]),
    (1024, 2047): (8, [32, 64, 128, 256]),
    None: (9, [32, 64, 128, 256]),
}


def _get_t_range_idx(T) -> int:
    for rng, (range_idx, t_range) in _t_ranges.items():
        if rng is None:
            continue
        low, high = rng
        if low <= T <= high:
            return range_idx
    return _t_ranges[None][0]


def _get_t_autotune_range(t_range_idx) -> list[int]:
    for rng, (range_idx, t_range) in _t_ranges.items():
        if range_idx == t_range_idx:
            return t_range
    return _t_ranges[None][1]


def _early_config_prune(
    configs: tp.Sequence[triton.Config], named_args: dict[str, tp.Any], **kwargs
) -> list[triton.Config]:
    T_RANGE_IDX = named_args.get("T_RANGE_IDX", kwargs.get("T_RANGE_IDX"))
    ts = _get_t_autotune_range(T_RANGE_IDX)
    prune_configs = []
    for t in ts:
        for config in configs:
            cfg = copy.deepcopy(config)
            cfg.kwargs["T_BLOCK_SIZE"] = t
            prune_configs.append(cfg)
    return prune_configs


def _do_bench_ignore_exception(
    fn, warmup=25, rep=100, grad_to_none=None, quantiles=None
):
    try:
        return triton.testing.do_bench(
            fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, quantiles=quantiles
        )
    except Exception as e:
        print(f"Skipping failed config due to error: {e}")
        if quantiles:
            return [float("inf")] * len(quantiles)
        return float("inf")


@triton.autotune(
    configs=[
        triton.Config(
            {
                "I_BLOCK_SIZE": i_block_size,
                "H_BLOCK_SIZE": h_block_size,
                "GROUP_SIZE": group_size,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for h_block_size in [32, 64]
        for i_block_size in [64, 128]
        for group_size in [1, 2, 4]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["T_RANGE_IDX", "H", "I"],
    prune_configs_by={"early_config_prune": _early_config_prune},
    do_bench=_do_bench_ignore_exception,
)
@triton.heuristics(
    {
        "X_BOUNDARY_CHECK": lambda args: args["T"] % args["T_BLOCK_SIZE"] != 0,
    }
)
@triton.jit
def _gemm_silu_and_mul_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    T,
    T_RANGE_IDX: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    T_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
    I_BLOCK_SIZE: tl.constexpr,
    X_BOUNDARY_CHECK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    WEIGHT_BOUNDARY_CHECK = I % I_BLOCK_SIZE != 0
    OUT_BOUNDARY_CHECK = T % T_BLOCK_SIZE != 0 or I % I_BLOCK_SIZE != 0

    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)

    if GROUP_SIZE > 1 and tl.cdiv(T, T_BLOCK_SIZE) >= GROUP_SIZE:
        pid_t, pid_i = tl.swizzle2d(
            pid_t, pid_i, tl.cdiv(T, T_BLOCK_SIZE), tl.cdiv(I, I_BLOCK_SIZE), GROUP_SIZE
        )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(T, H),
        strides=(H, 1),
        offsets=(pid_t * T_BLOCK_SIZE, 0),
        block_shape=(T_BLOCK_SIZE, H_BLOCK_SIZE),
        order=(1, 0),
    )
    weight1_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(H, I),
        strides=(1, H),
        offsets=(0, pid_i * I_BLOCK_SIZE),
        block_shape=(H_BLOCK_SIZE, I_BLOCK_SIZE),
        order=(0, 1),
    )
    weight2_block_ptr = tl.make_block_ptr(
        base=weight_ptr + I * H,
        shape=(H, I),
        strides=(1, H),
        offsets=(0, pid_i * I_BLOCK_SIZE),
        block_shape=(H_BLOCK_SIZE, I_BLOCK_SIZE),
        order=(0, 1),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(T, I),
        strides=(I, 1),
        offsets=(pid_t * T_BLOCK_SIZE, pid_i * I_BLOCK_SIZE),
        block_shape=(T_BLOCK_SIZE, I_BLOCK_SIZE),
        order=(1, 0),
    )

    acc1 = tl.zeros((T_BLOCK_SIZE, I_BLOCK_SIZE), dtype=tl.float32)
    acc2 = tl.zeros((T_BLOCK_SIZE, I_BLOCK_SIZE), dtype=tl.float32)

    for h in range(0, H, H_BLOCK_SIZE):
        x = (
            tl.load(x_block_ptr, boundary_check=(0, 1))
            if X_BOUNDARY_CHECK
            else tl.load(x_block_ptr)
        )
        weight1 = (
            tl.load(weight1_block_ptr, boundary_check=(0, 1))
            if WEIGHT_BOUNDARY_CHECK
            else tl.load(weight1_block_ptr)
        )
        weight2 = (
            tl.load(weight2_block_ptr, boundary_check=(0, 1))
            if WEIGHT_BOUNDARY_CHECK
            else tl.load(weight2_block_ptr)
        )
        acc1 = tl.dot(x, weight1, acc1)
        acc2 = tl.dot(x, weight2, acc2)

        x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
        weight1_block_ptr = tl.advance(weight1_block_ptr, [H_BLOCK_SIZE, 0])
        weight2_block_ptr = tl.advance(weight2_block_ptr, [H_BLOCK_SIZE, 0])

    gate = acc1.to(out_ptr.dtype.element_ty).to(tl.float32)
    up = acc2.to(out_ptr.dtype.element_ty)
    res = silu(gate) * up
    if OUT_BOUNDARY_CHECK:
        tl.store(out_block_ptr, res.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(out_block_ptr, res.to(out_ptr.dtype.element_ty))


def gemm_silu_and_mul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    T, H = x.shape
    assert H == weight.shape[1]
    assert weight.shape[0] % 2 == 0
    I = weight.shape[0] // 2

    out = torch.empty((T, I), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(T, meta["T_BLOCK_SIZE"]),
        triton.cdiv(I, meta["I_BLOCK_SIZE"]),
    )

    _gemm_silu_and_mul_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        T=T,
        H=H,
        I=I,
        T_RANGE_IDX=_get_t_range_idx(T),
    )
    return out


def _get_gemm_silu_and_mul_warmup_lens() -> list[int]:
    lens = []
    for rng in _t_ranges.keys():
        if rng is None:
            lens.append(4096)
        else:
            low, high = rng
            lens.append((low + high) // 2)
    return lens


_gemm_silu_and_mul_warmuped = False


def warmup_gemm_silu_and_mul(
    intermediate_size: int, hidden_size: int, x_dtype, weight_dtype
) -> None:
    global _gemm_silu_and_mul_warmuped
    if _gemm_silu_and_mul_warmuped:
        return
    _gemm_silu_and_mul_warmuped = True

    weight = torch.empty(
        (intermediate_size * 2, hidden_size), device="cuda", dtype=weight_dtype
    )
    warmup_lens = _get_gemm_silu_and_mul_warmup_lens()
    xs = torch.empty((max(warmup_lens) * hidden_size,), device="cuda", dtype=x_dtype)
    for T in warmup_lens:
        x = xs[: T * hidden_size].reshape(T, hidden_size)
        gemm_silu_and_mul(x, weight)
    del xs


def _gemm_silu_and_mul_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    T = x.shape[0]
    I = weight.shape[0] // 2
    return torch.empty((T, I), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="gemm_silu_and_mul",
    op_func=gemm_silu_and_mul,
    mutates_args=[],
    fake_impl=_gemm_silu_and_mul_fake,
)
