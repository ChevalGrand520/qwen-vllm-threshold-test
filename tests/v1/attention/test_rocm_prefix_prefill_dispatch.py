# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Contest-specific tests were developed with OpenAI Codex assistance and team review.

from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends import triton_attn
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
    _is_pure_prefill_batch,
    _select_seq_threshold_3d,
    _should_use_rocm_prefix_prefill,
    _view_rocm_prefix_kv_cache,
)


def test_single_decode_capture_preserves_natural_3d_threshold():
    assert _select_seq_threshold_3d(64, [1], preserve_natural=True) == 64
    assert _select_seq_threshold_3d(64, [1], preserve_natural=False) == 1


def test_qwen35_rocm_prefix_prefill_uses_piecewise_prefill_graphs():
    config = MagicMock()
    config.model_config.hf_config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.cache_config = None
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

    RocmPlatform.check_and_update_config(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert config.compilation_config.max_cudagraph_capture_size == 1
    assert config.compilation_config.cudagraph_capture_sizes == [1]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"is_rocm": False}, False),
        ({"is_decoder": False}, False),
        ({"is_pure_prefill": False}, False),
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
        "is_pure_prefill": True,
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
        _view_rocm_prefix_kv_cache(key, value, num_kv_heads=2, head_size=66)


@pytest.mark.parametrize(
    ("query_start_loc", "expected"),
    [
        ([0, 4, 8], True),
        ([0, 1, 5], False),
        ([0, 1, 2], False),
        ([0], False),
    ],
)
def test_is_pure_prefill_batch(query_start_loc, expected):
    starts = torch.tensor(query_start_loc, dtype=torch.int32)

    assert _is_pure_prefill_batch(starts) is expected


def test_metadata_builder_does_not_enable_batch_reordering():
    config = MagicMock()
    config.model_config.get_num_attention_heads.return_value = 4
    config.model_config.get_num_kv_heads.return_value = 2
    config.model_config.get_head_size.return_value = 64
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    builder = TritonAttentionMetadataBuilder(
        SimpleNamespace(block_size=16),
        ["model.layers.0.self_attn"],
        config,
        torch.device("cpu"),
    )

    assert builder.reorder_batch_threshold is None


def _run_dispatch_case(monkeypatch, query_lens, *, is_pure_prefill):
    calls = {}

    def record_unified_attention(**kwargs):
        calls["unified"] = kwargs
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

    num_tokens = sum(query_lens)
    num_reqs = len(query_lens)
    query = torch.zeros((num_tokens, 4, 64), dtype=torch.bfloat16)
    key = torch.zeros((num_tokens, 2, 64), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 2, 16, 2, 64), dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, *accumulate(query_lens)], dtype=torch.int32)
    seq_lens = torch.tensor(
        [length if length > 1 else 16 for length in query_lens],
        dtype=torch.int32,
    )
    block_table = torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        query_start_loc=query_start_loc,
        max_seq_len=int(seq_lens.max()),
        seq_lens=seq_lens,
        block_table=block_table,
        use_cascade=False,
        is_pure_prefill=is_pure_prefill,
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

    return calls, result, output, metadata


def test_pure_prefill_routes_to_rocm_prefix_kernel(monkeypatch):
    calls, result, output, metadata = _run_dispatch_case(
        monkeypatch, [4, 4], is_pure_prefill=True
    )

    assert result is output
    assert "unified" not in calls
    assert calls["prefill"]["q"].shape[0] == 8
    assert torch.equal(calls["prefill"]["b_start_loc"], metadata.query_start_loc)
    assert torch.equal(calls["prefill"]["b_seq_len"], metadata.seq_lens)
    assert torch.equal(output, torch.full_like(output, 2))


@pytest.mark.parametrize("query_lens", [[1, 1], [1, 4]])
def test_decode_and_mixed_batches_use_original_triton_path(monkeypatch, query_lens):
    calls, result, output, _ = _run_dispatch_case(
        monkeypatch, query_lens, is_pure_prefill=False
    )

    assert result is output
    assert "prefill" not in calls
    assert calls["unified"]["q"].shape[0] == sum(query_lens)
    assert torch.equal(output, torch.ones_like(output))
