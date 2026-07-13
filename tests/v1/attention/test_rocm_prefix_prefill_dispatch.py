# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends import triton_attn
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    _select_seq_threshold_3d,
    _shift_prefill_query_start_loc,
    _should_use_rocm_prefix_prefill,
    _view_rocm_prefix_kv_cache,
)


def test_single_decode_capture_preserves_natural_3d_threshold():
    assert _select_seq_threshold_3d(64, [1], preserve_natural=True) == 64
    assert _select_seq_threshold_3d(64, [1], preserve_natural=False) == 1


def test_qwen35_rocm_prefix_prefill_uses_piecewise_prefill_graphs():
    config = MagicMock()
    config.model_config.hf_config.architectures = [
        "Qwen3_5ForConditionalGeneration"
    ]
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.cache_config = None
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    RocmPlatform.check_and_update_config(config)

    assert (
        config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert config.compilation_config.max_cudagraph_capture_size == 1
    assert config.compilation_config.cudagraph_capture_sizes == [1]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"is_rocm": False}, False),
        ({"is_decoder": False}, False),
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
        "is_decoder": True,
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


def test_shift_prefill_query_start_loc_rebases_trailing_requests():
    query_start_loc = torch.tensor([0, 1, 2, 6, 10], dtype=torch.int32)

    prefill_query_start_loc = _shift_prefill_query_start_loc(
        query_start_loc,
        num_decodes=2,
        num_decode_tokens=2,
    )

    assert torch.equal(
        prefill_query_start_loc,
        torch.tensor([0, 4, 8], dtype=torch.int32),
    )


def test_mixed_batch_routes_decode_and_prefill_to_disjoint_kernels(monkeypatch):
    calls = {}

    def record_unified_attention(**kwargs):
        calls["decode"] = kwargs
        kwargs["out"].fill_(1)

    def record_prefix_prefill(**kwargs):
        calls["prefill"] = kwargs
        kwargs["o"].fill_(2)

    monkeypatch.setattr(
        triton_attn,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True),
    )
    monkeypatch.setattr(triton_attn, "unified_attention", record_unified_attention)
    monkeypatch.setattr(
        triton_attn,
        "prefix_prefill_attention_fwd",
        record_prefix_prefill,
    )

    impl = object.__new__(TritonAttentionImpl)
    impl.attn_type = "decoder"
    impl.kv_cache_dtype = "auto"
    impl.fp8_dtype = torch.float8_e4m3fn
    impl.num_kv_heads = 2
    impl.head_size = 64
    impl.logits_soft_cap = 0.0
    impl.use_alibi_sqrt = False
    impl.scale = 0.125
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.sinks = None

    query = torch.zeros((6, 4, 64), dtype=torch.bfloat16)
    key = torch.zeros((6, 2, 64), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 2, 16, 2, 64), dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, 1, 2, 6], dtype=torch.int32)
    seq_lens = torch.tensor([10, 20, 6], dtype=torch.int32)
    block_table = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    metadata = SimpleNamespace(
        num_actual_tokens=6,
        max_query_len=4,
        query_start_loc=query_start_loc,
        max_seq_len=20,
        seq_lens=seq_lens,
        block_table=block_table,
        use_cascade=False,
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=2,
        num_prefill_tokens=4,
        prefill_query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_threshold_3D=16,
        num_par_softmax_segments=4,
        softmax_segm_output=torch.empty(0),
        softmax_segm_max=torch.empty(0),
        softmax_segm_expsum=torch.empty(0),
        mm_prefix_range_tensor=None,
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
        _q_scale_float=1.0,
    )

    result = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output=output,
    )

    assert result is output
    assert calls["decode"]["q"].shape[0] == 2
    assert calls["decode"]["max_seqlen_q"] == 1
    assert torch.equal(calls["decode"]["seqused_k"], seq_lens[:2])
    assert calls["prefill"]["q"].shape[0] == 4
    assert torch.equal(
        calls["prefill"]["b_start_loc"], metadata.prefill_query_start_loc
    )
    assert torch.equal(calls["prefill"]["b_seq_len"], seq_lens[2:])
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2))
