#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xdzs2026_c337
REPO="$ROOT/vllm_cscc"
SRC="$ROOT/vllm_page784_unroll1_address_97b2998"
SITE=/usr/local/lib/python3.10/dist-packages/vllm
COMMIT=97b2998b3928561f016bbc19bf27cf12aff9f51e
BRANCH=rocm-page784-unroll1-address-fastpath-0714
REMOTE=https://gitlab.eduxiji.net/T2026103389912286/qwen-vllm-xiandao-2026.git
LOG="$ROOT/page784-launch-matrix-0714.log"

if pgrep -af 'vllm serve|vllm.entrypoints.openai.api_server|VLLM::EngineCore' \
    > /tmp/vllm-active.txt; then
    echo "ABORT: stop the full vLLM service before running this matrix:"
    cat /tmp/vllm-active.txt
    exit 2
fi

cd "$REPO"
if ! git remote get-url xiandao >/dev/null 2>&1; then
    git remote add xiandao "$REMOTE"
fi
for attempt in 1 2 3; do
    if git fetch xiandao "$BRANCH"; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "Unable to fetch $BRANCH after three attempts."
        exit 3
    fi
    sleep 10
done
test "$(git rev-parse FETCH_HEAD)" = "$COMMIT"
git cat-file -e "$COMMIT^{commit}"

if [ -e "$SRC" ]; then
    test -e "$SRC/.git"
    test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT"
else
    git worktree add --detach "$SRC" "$COMMIT"
fi

FILES=(
    v1/attention/ops/prefix_prefill.py
    v1/attention/ops/prefix_prefill_tile.py
)
for file in "${FILES[@]}"; do
    install -m 0644 "$SRC/vllm/$file" "$SITE/$file"
done

cd /tmp
python -m py_compile "${FILES[@]/#/$SITE/}"
python -m pytest -q \
    --confcutdir="$SRC/tests/v1/attention" \
    "$SRC/tests/v1/attention/test_prefix_prefill_tile.py" \
    "$SRC/tests/v1/attention/test_rocm_prefix_prefill_dispatch.py"

PYTHONUNBUFFERED=1 python - <<'PY' 2>&1 | tee "$LOG"
import json
import math
import statistics
from pathlib import Path

import torch
import vllm.v1.attention.ops.prefix_prefill as prefix
from vllm.v1.attention.ops.prefix_prefill_tile import (
    select_prefix_prefill_num_unroll_cache,
    select_prefix_prefill_page_address_fastpath,
    select_prefix_prefill_tile,
)

MODEL = Path("/public/home/xdzs2026_c337/Qwen3.5-27B/config.json")
PAGE_SIZE = 784
SHAPES = (
    (2048, 4096),
    (2048, 12288),
    (2048, 28672),
    (4096, 4096),
    (4096, 12288),
    (4096, 28672),
)
# label, num_warps, num_unroll_cache, address_fastpath
CONFIGS = (
    ("best77", 4, 2, False),
    ("unroll1", 4, 1, False),
    ("address", 4, 2, True),
    ("combo", 4, 1, True),
    ("address_w2", 2, 2, True),
    ("combo_w2", 2, 1, True),
    ("address_w8", 8, 2, True),
    ("combo_w8", 8, 1, True),
)
REPEATS = 5

config = json.loads(MODEL.read_text())
text_config = config.get("text_config", config)
num_heads = int(text_config["num_attention_heads"])
num_kv_heads = int(text_config["num_key_value_heads"])
head_dim = int(text_config.get("head_dim", text_config["hidden_size"] // num_heads))
assert head_dim == 256, head_dim
assert num_heads % num_kv_heads == 0
assert select_prefix_prefill_tile(784, 256, is_rocm=True) == (64, 32)
assert select_prefix_prefill_num_unroll_cache(784, 256, is_rocm=True) == 1
assert select_prefix_prefill_page_address_fastpath(784, 256, is_rocm=True)

device = torch.device("cuda:0")
dtype = torch.bfloat16
torch.manual_seed(0)
torch.accelerator.set_device_index(0)
x = 16 // torch.empty((), dtype=dtype).element_size()
num_queries_per_kv = num_heads // num_kv_heads

print(
    f"model heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} "
    f"physical_page={PAGE_SIZE} tile=64x32 configs={len(CONFIGS)}"
)


def make_case(query_len, context_len):
    q = torch.randn(query_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(query_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    context_k = torch.randn(
        context_len, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    context_v = torch.randn_like(context_k)
    num_blocks = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    padded_tokens = num_blocks * PAGE_SIZE
    key_base = torch.zeros(
        num_blocks,
        PAGE_SIZE,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value_base = torch.zeros_like(key_base)
    key_base.view(padded_tokens, num_kv_heads, head_dim)[:context_len].copy_(context_k)
    value_base.view(padded_tokens, num_kv_heads, head_dim)[:context_len].copy_(
        context_v
    )
    k_cache = key_base.view(
        num_blocks, PAGE_SIZE, num_kv_heads, head_dim // x, x
    ).permute(0, 2, 3, 1, 4)
    v_cache = value_base.permute(0, 2, 3, 1)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).view(1, -1)
    return {
        "q": q,
        "k": k,
        "v": v,
        "context_k": context_k,
        "context_v": context_v,
        "key_base": key_base,
        "value_base": value_base,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "query_start": torch.tensor(
            [0, query_len], device=device, dtype=torch.int32
        ),
        "seq_lens": torch.tensor(
            [context_len + query_len], device=device, dtype=torch.int32
        ),
        "scale": torch.tensor(1.0, device=device, dtype=torch.float32),
    }


def launch(case, out, num_warps, num_unroll_cache, use_fastpath):
    q = case["q"]
    k = case["k"]
    v = case["v"]
    k_cache = case["k_cache"]
    v_cache = case["v_cache"]
    block_table = case["block_table"]
    grid = (1, num_heads, prefix.triton.cdiv(q.shape[0], 64))
    prefix._fwd_kernel[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        None,
        block_table,
        head_dim**-0.5,
        case["scale"],
        case["scale"],
        1.0,
        case["query_start"],
        case["seq_lens"],
        k_cache.shape[4],
        out,
        block_table.stride(0),
        block_table.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=32,
        PHYSICAL_BLOCK_SIZE=PAGE_SIZE,
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=None,
        BLOCK_DMODEL=head_dim,
        BLOCK_DMODEL_PADDED=prefix.triton.next_power_of_2(head_dim),
        SLIDING_WINDOW=0,
        SKIP_DECODE=False,
        USE_FP8=False,
        BLOCK_M=64,
        BLOCK_N=32,
        num_unroll_cache=num_unroll_cache,
        num_unroll_request=1,
        USE_PAGE_ADDRESS_FASTPATH=use_fastpath,
        num_warps=num_warps,
        num_stages=1,
        USE_SINKS=False,
    )


speedups = {label: [] for label, *_ in CONFIGS}
max_diffs = {label: [] for label, *_ in CONFIGS}

for query_len, context_len in SHAPES:
    case = make_case(query_len, context_len)
    outputs = {label: torch.empty_like(case["q"]) for label, *_ in CONFIGS}
    active = []
    for label, warps, unroll, fastpath in CONFIGS:
        try:
            launch(case, outputs[label], warps, unroll, fastpath)
            torch.accelerator.synchronize()
            active.append((label, warps, unroll, fastpath))
        except Exception as exc:
            print(
                f"query={query_len:4d} ctx={context_len:5d} {label} FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

    if not active or active[0][0] != "best77":
        raise SystemExit("Best77 baseline failed to compile")

    samples = {label: [] for label, *_ in active}
    for repeat in range(REPEATS):
        offset = repeat % len(active)
        order = active[offset:] + active[:offset]
        for label, warps, unroll, fastpath in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(case, outputs[label], warps, unroll, fastpath)
            end.record()
            end.synchronize()
            samples[label].append(start.elapsed_time(end))

    baseline_ms = statistics.median(samples["best77"])
    baseline_out = outputs["best77"].float()
    for label, _, _, _ in active:
        elapsed = statistics.median(samples[label])
        max_abs = (baseline_out - outputs[label].float()).abs().max().item()
        speedup = baseline_ms / elapsed
        speedups[label].append(speedup)
        max_diffs[label].append(max_abs)
        print(
            f"query={query_len:4d} ctx={context_len:5d} {label:10s} "
            f"ms={elapsed:9.3f} speedup={speedup:7.4f}x "
            f"max_abs_diff={max_abs:.3e}"
        )

    del outputs, baseline_out
    case.clear()
    torch.cuda.empty_cache()

print("summary:")
ranking = []
for label, warps, unroll, fastpath in CONFIGS:
    values = speedups[label]
    if len(values) != len(SHAPES):
        continue
    geomean = math.prod(values) ** (1.0 / len(values))
    worst = min(values)
    max_abs = max(max_diffs[label])
    ranking.append((geomean, worst, max_abs, label, warps, unroll, fastpath))

for geomean, worst, max_abs, label, warps, unroll, fastpath in sorted(
    ranking, reverse=True
):
    print(
        f"{label:10s} warps={warps} unroll={unroll} fastpath={fastpath} "
        f"geomean={geomean:.4f}x worst={worst:.4f}x "
        f"max_abs_diff={max_abs:.3e}"
    )

eligible = [
    row
    for row in ranking
    if row[2] == 0.0 and row[1] > 0.99 and row[0] > 1.01
]
if eligible:
    best = max(eligible)
    print(f"MATRIX_WINNER={best[3]}")
else:
    print("NO_MATRIX_CANDIDATE_PASSED")
PY

echo "Results saved to $LOG"
