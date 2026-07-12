# Qwen3.5-27B 推理优化说明

## 1. 评测结果

本分支对应的官方评测结果如下：

| 版本 | 4K-8K | 8K-16K | 16K-32K | 精度扣分 | 最终得分 |
|---|---:|---:|---:|---:|---:|
| 融合算子初版 | 15.39 | 10.81 | 6.74 | 0.4949 | 65.4892 |
| 双支路 BF16 舍入 | 15.26 | 11.76 | 6.74 | 0.0 | 67.9970 |
| up 支路保留 BF16 | 15.32 | 11.80 | 6.73 | 0.0 | 68.0961 |

最终保留第三行版本。该版本没有 SLA 扣分和精度扣分。

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

## 3. 合规边界

- 不修改模型权重、模型层数、注意力头数或 tokenizer。
- 不跳过层、通道、token 或长样本，不截断输入和输出。
- 不使用 MTP、ngram、draft model 或其他投机解码方式。
- 不修改 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 及调度器。
- 不缓存测试集、答案、中间结果或生成可复用的量化权重。
- 不修改评测脚本、服务接口、请求格式、采样参数或统计口径。
- 所有优化均位于模型等价算子实现、投影融合和启动预热范围内。

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
  vllm/model_executor/warmup/kernel_warmup.py
```

设备端编译、吞吐和精度以比赛统一 DCU 环境的官方结果为准。当前发布版
官方得分为 68.0961。

## 5. 来源说明

项目主体沿用比赛指定的 vLLM 源码及其原始 Apache-2.0 许可和文件头说明。
本说明列出的增量实现由参赛队伍在比赛分支中完成；未引入新的第三方源码。
