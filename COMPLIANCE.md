# 提交合规自查

检查范围：相对队伍基线 `b386836` 的全部增量文件。

## 自查结论

| 检查项 | 结果 | 说明 |
|---|---|---|
| 模型权重和 tokenizer | 通过 | 未修改权重、权重加载格式、tokenizer 或 chat template |
| 模型有效计算结构 | 通过 | 仅合并等价投影与 MLP 算子，并优化 Attention 内核；未减少层、头、通道或 token |
| 投机解码 | 通过 | 未启用 MTP、ngram、draft model 或其他预测器 |
| 调度与锁定参数 | 通过 | 未修改 scheduler、上下文长度、批处理上限或采样参数 |
| 数据与评测流程 | 通过 | 未修改数据集、评测脚本、接口、统计代码或结果解析 |
| 缓存和预计算 | 通过 | 未缓存答案、测试样本、中间结果或量化权重 |
| 外部依赖与网络 | 通过 | 未增加依赖，推理过程不访问外部网络 |
| 自定义环境变量 | 通过 | 未新增环境变量，详见 `ENVIRONMENT_VARIABLES.md` |
| 优化说明 | 通过 | 全部增量文件和官方结果已记录在 `OPTIMIZATION.md` |
| 许可证 | 通过 | 沿用 vLLM Apache-2.0；新增内核包含 SPDX 标识 |
| 第三方来源披露 | 通过 | 根目录 README 头部和提交说明明确列出 vLLM、继承依赖与许可 |
| 生成式 AI 披露 | 通过 | README 头部、代码注释和本文档均明确披露 OpenAI Codex 辅助；答辩时仍需主动说明 |

## 规则边界说明

GDN 六路投影和 GEMM-SiLU 的变化属于等价算子融合。融合前后的权重、输出
尺寸、张量拆分顺序和后续数学计算保持一致。该改动减少算子启动和中间张量
访存，不属于剪枝、跳层、early-exit 或模型语义变更。

ROCm Prefix Prefill 路径只在纯 Prefill 批次中替换 Attention 内核；Decode 和
混合批次仍走原始 Triton Unified Attention。该路径没有修改 batch scheduler，
也没有调整 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 等锁定参数。

page784 地址快路径只替换 Prefix Prefill 内核内的 KV block-table 地址计算。它按
原公式重建每个 token 的物理 block id 和页内偏移，不改变 K/V 数据、Attention
矩阵、Softmax 归约顺序、输出类型或请求调度；非 ROCm、非 page784、非 head256
配置继续使用原路径。

`qwen3_5.py` 和 `qwen3_next.py` 中仍可看到 vLLM 上游对 speculative config、
MTP 权重和 scheduler config 的通用兼容代码。这些代码在队伍增量之前已经
存在；本提交没有启用相应能力，`serve_args.txt` 也未包含相关参数。

## 已执行检查

- `git diff --check`
- 增量 Python 文件的 `py_compile` 语法检查
- 增量 Attention 文件的 `ruff check` 与 `ruff format --check`
- 相对基线的禁用行为关键词扫描
- Attention 分流和调度边界的 AST 定向检查
- 发布历史、路径和增量文件中的工具命名扫描
- BW1000 page784 九个页边界 bit-exact 检查
- BW1000 六组中长上下文 A/B：geomean `1.0615x`、worst `1.0603x`
- BW1000 八配置矩阵：组合 geomean `1.9716x`、worst `1.8397x`
- SCNet 聚焦测试：`36 passed`

设备端编译与数值正确性以统一 DCU 环境的官方评测结果为准。最终性能实现
`1f70f91080e0dbd976bab551f7544d4f19ca128c` 的官方得分为 80.2281，三档吞吐
为 `16.42 / 14.84 / 11.40`，精度和 SLA 扣分均为 0。安全分支
`best-compliant-80.2281-0714` 保留同一实现；失败或降分实验未合入最终版本。

## 生成式 AI 辅助披露

本项目的方案分析、部分竞赛增量代码、测试、合规检查和文档由参赛队伍在
OpenAI Codex 辅助下完成。队伍成员负责需求判断、代码审阅、设备部署、测试、
官方提交和最终结果确认，并对提交内容承担责任。答辩材料中必须继续主动披露
上述辅助范围，不得将 AI 辅助代码表述为完全独立手写。
