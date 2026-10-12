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
        return 1
    return 4


def select_prefix_prefill_page_address_fastpath(
    physical_block_size: int,
    head_size: int,
    *,
    is_rocm: bool,
    tile_size: int = 32,
) -> bool:
    """Whether a tile can be mapped with at most two page-table entries.

    The kernel walks fixed-size 32-token address tiles.  The mapping is valid
    for any page at least as large as that tile; a page smaller than a tile
    would require three or more page-table entries and stays on the reference
    vector div/mod path.

    ``head_size`` is retained in the signature for selector call-site
    compatibility.  It does not affect the page arithmetic.
    """
    del head_size
    return is_rocm and tile_size > 0 and physical_block_size >= tile_size
