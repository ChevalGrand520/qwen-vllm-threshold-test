# Qwen3.5-27B 推理优化说明

## 1. 评测结果

本分支对应的官方评测结果如下：

| 版本 | 4K-8K | 8K-16K | 16K-32K | 精度扣分 | 最终得分 |
|---|---:|---:|---:|---:|---:|
| 融合算子初版 | 15.39 | 10.81 | 6.74 | 0.4949 | 65.4892 |
| 双支路 BF16 舍入 | 15.26 | 11.76 | 6.74 | 0.0 | 67.9970 |
| up 支路保留 BF16 | 15.32 | 11.80 | 6.73 | 0.0 | 68.0961 |
| ROCm Prefix + 单请求 Decode Graph 初版 | 15.94 | 12.69 | 7.40 | 0.5367 | 71.0277 |
| 保留自然 3D/2D Attention 阈值 | 15.97 | 12.66 | 7.45 | 0.0 | 71.6201 |
| page784 Prefix query tile `64x32` | 16.23 | 13.74 | 9.13 | 0.0 | 76.2214 |
| page784 cache-loop unroll `2` | 16.09 | 14.10 | 10.16 | 0.0 | 77.8698 |

当前已验证安全版本为 `d42b814`，对应最后一行结果，没有 SLA 或精度扣分。
官方 `main` 上另有 cache-loop unroll `1` 单变量实验 `ab6f410`，尚待官方结果。
本分支从安全版本独立创建，只实验 page784 KV 地址生成快路径，不与未验证的
unroll `1` 叠加。

## 2. 改动概览

### 2.1 GDN 输入投影融合

Qwen3.5 的线性注意力层原先分别执行 `qkvz` 和 `ba` 两次输入投影。
发布版将六个输出分片合并为一次 `in_proj_qkvzba` 投影，再按原尺寸拆分
为 q、k、v、z、b、a。权重加载映射同步调整，张量顺序和后续计算不变。

涉及文件：

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`

### 2.2 MLP 的 GEMM-SiLU 融合

Qwen3.5 文本模型的 MLP 原路径为 `gate_up_proj` 生成中间张量，再调用
`SiluAndMul`。发布版使用 Triton 内核在一次算子中完成两路矩阵乘、SiLU
和逐元素乘法，避免完整 gate/up 中间张量往返显存。

内核按 token 数区间预热和自动选择 tile。模型加载后的 warmup 覆盖比赛
模型的固定形状：hidden size 5120、intermediate size 17408、BF16。

涉及文件：

- `vllm/model_executor/layers/gemm_silu_and_mul.py`
- `vllm/model_executor/models/qwen2_moe.py`
- `vllm/model_executor/warmup/kernel_warmup.py`

### 2.3 BF16 中间舍入位置

两路 GEMM 均保持 FP32 累加。循环结束后，gate 支路先舍入到输出 BF16，
再转回 FP32 执行 sigmoid；up 支路舍入到 BF16 后直接参与乘法。这样保留
未融合路径的 BF16 物化边界，同时避免 up 支路一次额外的显式类型回转。

曾测试过直接把循环累加器改为 BF16，该写法在目标 Triton 编译器中失败，
未进入发布分支。

### 2.4 LayerNorm 调用与编译配置

FLA LayerNorm/RMSNorm 入口去除重复的 Python 输入装饰器检查，底层实现及
数学公式未修改。`serve_args.txt` 只启用框架已有的 `rms_norm_gated` 自定义
算子编译路径，不修改模型、采样、上下文长度或调度参数。

涉及文件：

- `vllm/model_executor/layers/fla/ops/layernorm_guard.py`
- `serve_args.txt`

### 2.5 ROCm Prefix Prefill Attention

纯 Prefill 批次使用 ROCm Prefix Prefill 内核，并通过张量 view 复用原 KV Cache
存储，不复制或持久化新的缓存。Decode 和混合 Prefill/Decode 批次继续使用原始
Triton Unified Attention。分流只读取 Attention Metadata，没有修改或重排
batch scheduler。

涉及文件：

- `vllm/v1/attention/backends/triton_attn.py`
- `tests/v1/attention/test_rocm_prefix_prefill_dispatch.py`

### 2.6 Qwen3.5 CUDA Graph 范围

Qwen3.5 在 ROCm 上把 Full Decode Graph 捕获范围限制为单请求 `[1]`，与初赛
固定并发 1 一致。早期安全版本 `2f8800e` 使用 `FULL_DECODE_ONLY`；后续官方已验证
`FULL_AND_PIECEWISE` 路径，并继续把 Prefix 内核限制在纯 Prefill 批次。Full Graph
最大捕获批次仍固定为 1。

涉及文件：

- `vllm/platforms/rocm.py`

### 2.7 page784 Prefix tile 与 KV 地址快路径

Qwen3.5 的混合缓存物理 Attention page 为 784 token。已验证版本把 Prefix Prefill
query tile 从 `32x32` 扩为 `64x32`，并把 cache-loop unroll 从 4 降为 2，减少长
上下文下的程序数量、代码体积和寄存器压力。

本实验分支进一步保持 `64x32`、unroll 2 和全部 Attention 数学不变，只优化
block table 地址生成。32-token cache tile 的起点相对 784-token page 每 49 次循环
才跨页一次。快路径使用当前页和按需加载的下一页两个标量 block id 重建原向量
映射，替代每个 token 的向量整数除法、取模和重复 block-table 读取。跨页 token、
尾部 mask、K/V 加载顺序与 Softmax 归约顺序均不变。

涉及文件：

- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/prefix_prefill_tile.py`
- `tests/v1/attention/test_prefix_prefill_tile.py`

BW1000 设备端 A/B 结果（Best77 `64x32`、unroll 2、warps 4 为基线）：

| Query | Context | 基线 ms | 快路径 ms | 加速比 | 最大绝对误差 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4096 | 19.720 | 18.594 | 1.0606x | 0 |
| 2048 | 12288 | 56.226 | 53.027 | 1.0603x | 0 |
| 2048 | 28672 | 129.469 | 122.026 | 1.0610x | 0 |
| 4096 | 4096 | 41.297 | 38.827 | 1.0636x | 0 |
| 4096 | 12288 | 114.495 | 107.834 | 1.0618x | 0 |
| 4096 | 28672 | 260.937 | 245.793 | 1.0616x | 0 |

六形状几何平均加速 `1.0615x`，最差形状 `1.0603x`。context 769、783、784、
785、799、800、1567、1568、1569 的页边界输出均 bit-exact；远端聚焦测试
`36 passed`。候选满足进入官方单变量评测的设备门禁。

## 3. 合规边界

- 不修改模型权重、模型层数、注意力头数或 tokenizer。
- 不跳过层、通道、token 或长样本，不截断输入和输出。
- 不使用 MTP、ngram、draft model 或其他投机解码方式。
- 不修改 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 及调度器。
- 不缓存测试集、答案、中间结果或生成可复用的量化权重。
- 不修改评测脚本、服务接口、请求格式、采样参数或统计口径。
- 所有优化均位于模型等价算子、投影融合、Attention 内核、图执行和启动预热范围内。

## 4. 复现与验证

提交前执行：

```bash
git diff --check
python -m py_compile \
  vllm/model_executor/layers/fla/ops/layernorm_guard.py \
  vllm/model_executor/layers/gemm_silu_and_mul.py \
  vllm/model_executor/models/qwen2_moe.py \
  vllm/model_executor/models/qwen3_5.py \
  vllm/model_executor/models/qwen3_next.py \
  vllm/model_executor/warmup/kernel_warmup.py \
  vllm/platforms/rocm.py \
  vllm/v1/attention/backends/triton_attn.py \
  vllm/v1/attention/ops/prefix_prefill.py \
  vllm/v1/attention/ops/prefix_prefill_tile.py \
  tests/v1/attention/test_prefix_prefill_tile.py \
  tests/v1/attention/test_rocm_prefix_prefill_dispatch.py
```

设备端编译、吞吐和精度以比赛统一 DCU 环境的官方结果为准。当前已验证安全版
官方得分为 77.8698。本地址快路径已通过 BW1000 page 边界 bit-exact 检查和六组
中长上下文 A/B 门禁；尚未取得官方端到端分数，不能替代安全版本。

## 5. 来源说明

项目主体沿用比赛指定的 vLLM 源码及其原始 Apache-2.0 许可和文件头说明。
本说明列出的增量实现由参赛队伍在 OpenAI Codex 辅助下完成并由队伍审阅；
未引入新的第三方源码。AI 辅助范围还需在答辩中主动披露。
