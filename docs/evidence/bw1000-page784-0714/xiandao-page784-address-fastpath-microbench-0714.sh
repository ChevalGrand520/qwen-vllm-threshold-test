#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/xdzs2026_c337
REPO="$ROOT/vllm_cscc"
SRC="$ROOT/vllm_page784_address_fastpath_f5ef16e"
SITE=/usr/local/lib/python3.10/dist-packages/vllm
COMMIT=f5ef16e37eab2ded4a6c7feb4136cd49185e6f76
BRANCH=rocm-page784-address-fastpath-0714
REMOTE=https://gitlab.eduxiji.net/T2026103389912286/qwen-vllm-xiandao-2026.git
LOG="$ROOT/page784-address-fastpath-microbench-0714.log"

if pgrep -af 'vllm serve|vllm.entrypoints.openai.api_server|VLLM::EngineCore' \
    > /tmp/vllm-active.txt; then
    echo "ABORT: stop the full vLLM service before running this microbenchmark:"
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

MODEL = Path("/public/home/xdzs2026_c337/Qwen3.5-27B/config.json")
PAGE_SIZE = 784
TILE_SIZE = 32
PERF_SHAPES = (
    (2048, 4096),
    (2048, 12288),
    (2048, 28672),
    (4096, 4096),
    (4096, 12288),
    (4096, 28672),
)
BOUNDARY_CONTEXTS = (769, 783, 784, 785, 799, 800, 1567, 1568, 1569)
REPEATS = 7

config = json.loads(MODEL.read_text())
text_config = config.get("text_config", config)
num_heads = int(text_config["num_attention_heads"])
num_kv_heads = int(text_config["num_key_value_heads"])
head_dim = int(text_config.get("head_dim", text_config["hidden_size"] // num_heads))
assert head_dim == 256, head_dim
assert num_heads % num_kv_heads == 0

device = torch.device("cuda:0")
dtype = torch.bfloat16
torch.manual_seed(0)
torch.accelerator.set_device_index(0)
x = 16 // torch.empty((), dtype=dtype).element_size()
num_queries_per_kv = num_heads // num_kv_heads

print(
    f"model heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} "
    f"physical_page={PAGE_SIZE} tile=64x32 unroll=2 warps=4"
)


def make_case(query_len, context_len):
    q = torch.randn(query_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(query_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    context_k = torch.randn(
        context_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
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
        num_blocks,
        PAGE_SIZE,
        num_kv_heads,
        head_dim // x,
        x,
    ).permute(0, 2, 3, 1, 4)
    v_cache = value_base.permute(0, 2, 3, 1)
    block_table = torch.arange(
        num_blocks,
        device=device,
        dtype=torch.int32,
    ).view(1, -1)
    query_start = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor(
        [context_len + query_len],
        device=device,
        dtype=torch.int32,
    )
    scale = torch.tensor(1.0, device=device, dtype=torch.float32)
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
        "query_start": query_start,
        "seq_lens": seq_lens,
        "scale": scale,
    }


def launch(case, out, use_fastpath):
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
        BLOCK_SIZE=TILE_SIZE,
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
        num_unroll_cache=2,
        num_unroll_request=1,
        USE_PAGE_ADDRESS_FASTPATH=use_fastpath,
        num_warps=4,
        num_stages=1,
        USE_SINKS=False,
    )


def clear_case(case):
    case.clear()
    torch.cuda.empty_cache()


print("boundary correctness:")
for context_len in BOUNDARY_CONTEXTS:
    case = make_case(64, context_len)
    baseline_out = torch.empty_like(case["q"])
    fast_out = torch.empty_like(case["q"])
    launch(case, baseline_out, False)
    launch(case, fast_out, True)
    torch.accelerator.synchronize()
    max_abs = (baseline_out.float() - fast_out.float()).abs().max().item()
    print(f"ctx={context_len:4d} max_abs_diff={max_abs:.3e}")
    assert max_abs == 0.0, (context_len, max_abs)
    del baseline_out, fast_out
    clear_case(case)


speedups = []
max_diffs = []
print("performance:")
for query_len, context_len in PERF_SHAPES:
    case = make_case(query_len, context_len)
    outputs = {
        False: torch.empty_like(case["q"]),
        True: torch.empty_like(case["q"]),
    }

    launch(case, outputs[False], False)
    launch(case, outputs[True], True)
    torch.accelerator.synchronize()

    samples = {False: [], True: []}
    for repeat in range(REPEATS):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for use_fastpath in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(case, outputs[use_fastpath], use_fastpath)
            end.record()
            end.synchronize()
            samples[use_fastpath].append(start.elapsed_time(end))

    baseline_ms = statistics.median(samples[False])
    fast_ms = statistics.median(samples[True])
    speedup = baseline_ms / fast_ms
    max_abs = (outputs[False].float() - outputs[True].float()).abs().max().item()
    speedups.append(speedup)
    max_diffs.append(max_abs)
    print(
        f"query={query_len:4d} ctx={context_len:5d} "
        f"baseline_ms={baseline_ms:9.3f} fast_ms={fast_ms:9.3f} "
        f"speedup={speedup:7.4f}x max_abs_diff={max_abs:.3e}"
    )
    del outputs
    clear_case(case)

geomean = math.prod(speedups) ** (1.0 / len(speedups))
worst = min(speedups)
max_abs = max(max_diffs)
print(
    f"summary geomean_speedup={geomean:.4f}x "
    f"worst_speedup={worst:.4f}x max_abs_diff={max_abs:.3e}"
)
if max_abs != 0.0:
    raise SystemExit("REJECT: output differs from the Best77 address path")
if geomean > 1.01 and worst > 0.99:
    print("CANDIDATE_READY_FOR_OFFICIAL_REVIEW")
else:
    print("REJECT_OR_RETEST: insufficient stable speedup")
PY

echo "Results saved to $LOG"
