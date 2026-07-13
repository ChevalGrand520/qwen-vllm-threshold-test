# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Contest specialization developed with OpenAI Codex assistance and team review.


def select_prefix_prefill_tile(
    physical_block_size: int,
    head_size: int,
    *,
    is_rocm: bool,
) -> tuple[int, int]:
    is_power_of_two = physical_block_size > 0 and (
        physical_block_size & (physical_block_size - 1) == 0
    )
    if is_power_of_two:
        return 128, 64
    # BW1000 measurements showed that widening only the query tile preserves
    # bit-identical output for Qwen3.5's hybrid-cache geometry.
    if is_rocm and physical_block_size == 784 and head_size == 256:
        return 64, 32
    return 32, 32


def select_prefix_prefill_num_unroll_cache(
    physical_block_size: int,
    head_size: int,
    *,
    is_rocm: bool,
) -> int:
    if is_rocm and physical_block_size == 784 and head_size == 256:
        return 2
    return 4
