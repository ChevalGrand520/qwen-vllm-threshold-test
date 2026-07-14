# 先导杯提交物与复现说明

## 1. 提交范围

本仓库提交 Qwen3.5-27B 的完整推理优化源码、测试和说明文档。方案不包含模型
训练、微调、蒸馏、剪枝或持久化量化，因此没有队伍生成的轻量模型权重需要
提交。正式模型权重由比赛平台提供，保持原始格式和数值，不在 Git 仓库分发。

交付文件：

- `vllm/`：完整 vLLM 推理框架与竞赛增量源码；
- `tests/v1/attention/`：ROCm Prefix Prefill 分流、tile 和页地址测试；
- `serve_args.txt`：唯一新增的服务编译参数；
- `OPTIMIZATION.md`：优化设计、实现、实验数据和模块说明；
- `COMPLIANCE.md`：禁止行为、权重、调度、精度和学术诚信自查；
- `ENVIRONMENT_VARIABLES.md`：环境变量说明；
- `README.md` 文件头：第三方来源、许可证和 AI 辅助披露。

最终性能实现基线为
`1f70f91080e0dbd976bab551f7544d4f19ca128c`，安全分支为
`best-compliant-80.2281-0714`。后续仅补充提交说明，不改变该实现的运行代码。

## 2. 开源来源与第三方依赖

项目基于 [vLLM](https://github.com/vllm-project/vllm)，许可证为 Apache-2.0。
仓库继承使用以下上游项目或库：

- [PyTorch](https://github.com/pytorch/pytorch)：张量与模型执行运行时；
- [Triton](https://github.com/triton-lang/triton)：GPU/DCU kernel 编程与编译；
- [Transformers](https://github.com/huggingface/transformers)：模型配置兼容；
- Qwen 模型适配代码：保留原文件版权和 Apache-2.0 声明；
- Flash Linear Attention 相关代码：由 vLLM 上游引入，来源和许可证保留在
  `vllm/model_executor/layers/fla/ops/` 的源码文件头部。

本次竞赛增量未复制新的第三方源码。新增 kernel、分流逻辑、测试和文档由队伍
在 OpenAI Codex 辅助下完成并由队伍成员审阅。队伍将在答辩中继续主动说明该
辅助范围，并对最终提交负责。

## 3. 运行环境

已验证环境：

```text
比赛镜像: qwen3.5-dtk26.04:0509
加速卡: HG BW1000
Python: 3.10
模型: Qwen3.5-27B（由平台挂载）
最大上下文: 32768（由官方评测锁定）
并发: 1（由官方评测锁定）
```

构建前必须使用组委会镜像预装的 ROCm/DCU 驱动、PyTorch 和编译工具链，不要
自行替换驱动或模型权重。

## 4. 编译安装

```bash
cd /path/to/qwen-vllm-xiandao-2026

# 语法与差异检查
git diff --check
python3 -m py_compile \
  vllm/model_executor/layers/gemm_silu_and_mul.py \
  vllm/model_executor/models/qwen3_5.py \
  vllm/model_executor/models/qwen3_next.py \
  vllm/platforms/rocm.py \
  vllm/v1/attention/backends/triton_attn.py \
  vllm/v1/attention/ops/prefix_prefill.py \
  vllm/v1/attention/ops/prefix_prefill_tile.py

# 在比赛基础镜像内构建并安装 wheel
python3 setup.py bdist_wheel
python3 -m pip install --force-reinstall dist/vllm-*.whl --no-deps
hash -r
which vllm
```

`--no-deps` 用于保留比赛镜像中已验证的 PyTorch/ROCm 依赖，避免安装过程替换
官方运行时。若平台提供自己的构建脚本，应以平台脚本为准，但源码 commit 必须
保持一致。

## 5. 启动服务

正式评测由平台注入并锁定模型、tokenizer、上下文、batch、采样参数和服务接口。
不得修改这些参数。手工复现时可使用同一接口：

```bash
MODEL_DIR=/public/home/xdzs2026_c337/Qwen3.5-27B

vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --compilation-config '{"custom_ops":["none","+rms_norm_gated"]}'
```

上述命令只用于同口径复现。正式提交时以平台锁定命令为准，`serve_args.txt`
仅表达相同的 compilation custom-op 配置，不覆盖 scheduler 或采样参数。

健康检查：

```bash
curl -f http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/models
```

首次启动会进行权重加载、torch.compile 和 kernel warmup；在 HG BW1000 上可能
持续较长时间。应以 `/health` 成功和日志出现应用启动完成为准，不要仅凭一段
时间没有新日志判断服务崩溃。

## 6. 代码模块

| 模块 | 主要文件 | 作用 |
|---|---|---|
| GDN 投影融合 | `qwen3_5.py`, `qwen3_next.py` | 合并等价 q/k/v/z/b/a 投影，保持权重映射与语义 |
| GEMM-SiLU 融合 | `gemm_silu_and_mul.py`, `qwen2_moe.py` | 融合两路 GEMM、SiLU 和乘法 |
| ROCm Prefill 分流 | `triton_attn.py` | 仅纯 Prefill 使用 Prefix kernel |
| page784 kernel | `prefix_prefill.py`, `prefix_prefill_tile.py` | 优化 query tile、循环展开与 KV 地址计算 |
| 图执行 | `vllm/platforms/rocm.py` | 单请求 Decode graph 与 piecewise Prefill |
| 验证 | `test_rocm_prefix_prefill_dispatch.py`, `test_prefix_prefill_tile.py` | 分流、边界和选择逻辑测试 |

## 7. 实验结果

最终官方结果：

```text
状态: AC
得分: 80.2281
4K-8K: 16.42 tok/s
8K-16K: 14.84 tok/s
16K-32K: 11.40 tok/s
SLA 扣分: 0.0
精度扣分: 0.0
```

设备端还完成 page784 九组页边界 bit-exact 检查、六组 A/B、八配置矩阵和
聚焦测试 `36 passed`。详细数据见 `OPTIMIZATION.md`。

## 8. 合规边界

- 不修改或替换模型权重、tokenizer、chat template 和模型输出口径；
- 不训练、蒸馏、微调、剪枝或生成可复用量化权重；
- 不跳层、跳 token、截断输入、缓存答案或针对测试集写死规则；
- 不使用 MTP、draft model、ngram 等投机解码；
- 不修改 batch scheduler、锁定参数、评测脚本、服务接口或统计口径；
- 低精度只保留原 BF16 语义边界和 kernel 内计算，不持久化新权重。

完整逐项证据见 `COMPLIANCE.md`。
