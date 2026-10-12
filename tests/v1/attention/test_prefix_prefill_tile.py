# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Contest-specific tests were developed with OpenAI Codex assistance and team review.

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm"
    / "v1"
    / "attention"
    / "ops"
    / "prefix_prefill_tile.py"
)


def _load_tile_module():
    assert MODULE_PATH.exists(), "prefix prefill tile selector module must exist"
    spec = importlib.util.spec_from_file_location("prefix_prefill_tile", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("physical_block_size", "head_size", "is_rocm", "expected"),
    [
        (1024, 256, True, (128, 64)),
        (784, 256, True, (64, 32)),
        (784, 256, False, (32, 32)),
        (784, 128, True, (32, 32)),
        (544, 256, True, (32, 32)),
    ],
)
def test_select_prefix_prefill_tile(physical_block_size, head_size, is_rocm, expected):
    select_prefix_prefill_tile = _load_tile_module().select_prefix_prefill_tile
    assert (
        select_prefix_prefill_tile(
            physical_block_size,
            head_size,
            is_rocm=is_rocm,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("physical_block_size", "head_size", "is_rocm", "expected"),
    [
        (784, 256, True, 1),
        (784, 256, False, 4),
        (784, 128, True, 4),
        (544, 256, True, 4),
        (1024, 256, True, 4),
    ],
)
def test_select_prefix_prefill_cache_unroll(
    physical_block_size, head_size, is_rocm, expected
):
    select_cache_unroll = _load_tile_module().select_prefix_prefill_num_unroll_cache
    assert (
        select_cache_unroll(
            physical_block_size,
            head_size,
            is_rocm=is_rocm,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("physical_block_size", "head_size", "is_rocm", "expected"),
    [
        (784, 256, True, True),
        (784, 256, False, False),
        (784, 128, True, True),
        (544, 256, True, True),
        (1024, 256, True, True),
        (31, 256, True, False),
    ],
)
def test_select_prefix_prefill_page_address_fastpath(
    physical_block_size, head_size, is_rocm, expected
):
    select_fastpath = _load_tile_module().select_prefix_prefill_page_address_fastpath
    assert (
        select_fastpath(
            physical_block_size,
            head_size,
            is_rocm=is_rocm,
        )
        == expected
    )


def test_page784_fast_address_mapping_matches_divmod():
    page_size = 784
    tile_size = 32

    for tile_index in range(page_size // 16):
        start_n = tile_index * tile_size
        logical_block = start_n // page_size
        internal_start = start_n - logical_block * page_size

        for token_offset in range(tile_size):
            next_page = internal_start + token_offset >= page_size
            fast_block = logical_block + int(next_page)
            fast_offset = internal_start + token_offset - int(next_page) * page_size

            expected_block, expected_offset = divmod(
                start_n + token_offset,
                page_size,
            )
            assert (fast_block, fast_offset) == (expected_block, expected_offset)


@pytest.mark.parametrize("page_size", [32, 33, 544, 784, 1024, 1568])
def test_generic_fast_address_mapping_matches_divmod(page_size):
    tile_size = 32
    for start_n in range(0, page_size * 3, tile_size):
        for token_offset in range(tile_size):
            logical_block = start_n // page_size
            internal_start = start_n - logical_block * page_size
            crosses_page = internal_start + token_offset >= page_size
            fast_block = logical_block + int(crosses_page)
            fast_offset = internal_start + token_offset - int(crosses_page) * page_size
            assert (fast_block, fast_offset) == divmod(
                start_n + token_offset, page_size
            )
