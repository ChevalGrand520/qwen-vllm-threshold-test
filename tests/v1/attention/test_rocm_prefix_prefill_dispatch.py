# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.triton_attn import (
    _should_use_rocm_prefix_prefill,
    _view_rocm_prefix_kv_cache,
)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"is_rocm": False}, False),
        ({"num_prefill_tokens": 0}, False),
        ({"has_key_value": False}, False),
        ({"logits_soft_cap": 1.0}, False),
        ({"use_alibi_sqrt": True}, False),
        ({"has_mm_prefix": True}, False),
    ],
)
def test_should_use_rocm_prefix_prefill(overrides, expected):
    args = {
        "is_rocm": True,
        "num_prefill_tokens": 4096,
        "has_key_value": True,
        "logits_soft_cap": 0.0,
        "use_alibi_sqrt": False,
        "has_mm_prefix": False,
    }
    args.update(overrides)

    assert _should_use_rocm_prefix_prefill(**args) is expected


def test_view_rocm_prefix_kv_cache_preserves_token_major_mapping():
    key = torch.arange(3 * 16 * 2 * 64, dtype=torch.int16).reshape(3, 16, 2, 64)
    value = key + 10_000

    key_view, value_view = _view_rocm_prefix_kv_cache(
        key, value, num_kv_heads=2, head_size=64
    )

    assert key_view.shape == (3, 2, 8, 16, 8)
    assert value_view.shape == (3, 2, 64, 16)
    assert key_view.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    assert value_view.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert key_view[1, 1, 5, 7, 3] == key[1, 7, 1, 43]
    assert value_view[1, 1, 43, 7] == value[1, 7, 1, 43]


def test_view_rocm_prefix_kv_cache_rejects_incompatible_head_size():
    key = torch.empty((1, 16, 2, 66), dtype=torch.bfloat16)
    value = torch.empty_like(key)

    with pytest.raises(ValueError, match="must be divisible"):
        _view_rocm_prefix_kv_cache(
            key, value, num_kv_heads=2, head_size=66
        )
