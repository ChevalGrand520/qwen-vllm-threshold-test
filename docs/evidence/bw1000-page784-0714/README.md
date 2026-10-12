# BW1000 page784 原始实验证据

本目录归档 2026-07-14 在 HG BW1000 上运行的 page784 地址快路径 A/B 与八配置
启动矩阵。GitHub 原先只有结果摘要，本目录补充可复核的测试脚本、设备输出和来源
哈希。

## 地址快路径 A/B

- 脚本：[`xiandao-page784-address-fastpath-microbench-0714.sh`](xiandao-page784-address-fastpath-microbench-0714.sh)
- 原始结果：[`page784-address-fastpath-A-B-0714.log`](page784-address-fastpath-A-B-0714.log)
- SCNet 原输出路径：
  `/public/home/xdzs2026_c337/page784-address-fastpath-microbench-0714.log`
- 被测分支：`rocm-page784-address-fastpath-0714`
- 设备测试提交：`f5ef16e37eab2ded4a6c7feb4136cd49185e6f76`
- 实现提交：`3cc47997d7174a331be30526fee64a289901c973`

日志包含 9 个页边界 bit-exact 检查和 6 组性能 A/B：

```text
geomean_speedup=1.0615x
worst_speedup=1.0603x
max_abs_diff=0
```

## 八配置矩阵

- 脚本：[`xiandao-page784-launch-matrix-0714.sh`](xiandao-page784-launch-matrix-0714.sh)
- 未裁剪终端转录：[`page784-launch-matrix-0714-terminal-transcript.txt`](page784-launch-matrix-0714-terminal-transcript.txt)
- 提取后的结果块：[`page784-launch-matrix-0714.log`](page784-launch-matrix-0714.log)
- SCNet 原输出路径：
  `/public/home/xdzs2026_c337/page784-launch-matrix-0714.log`
- 被测分支：`rocm-page784-unroll1-address-fastpath-0714`
- 设备测试提交：`97b2998b3928561f016bbc19bf27cf12aff9f51e`
- 最终记录提交：`1f70f91080e0dbd976bab551f7544d4f19ca128c`

八个配置为 `best77`、`unroll1`、`address`、`combo`、`address_w2`、
`combo_w2`、`address_w8`、`combo_w8`。日志包含 6 个形状下全部 48 行测量：

```text
address  geomean=1.0624x worst=1.0606x max_abs_diff=0
unroll1  geomean=1.7055x worst=1.6628x max_abs_diff=0
combo    geomean=1.9716x worst=1.8397x max_abs_diff=0
MATRIX_WINNER=combo
```

## 具体实现贡献

原通用路径对 tile 内每个 token 计算：

```python
logical_page = token_index // physical_page_size
offset = token_index % physical_page_size
block_id = block_table[logical_page]
```

在 `physical_page_size=784`、cache tile 为 32 token 时，`gcd(784, 32)=16`，
tile 起点相对页边界每 49 次迭代循环一次，且只有一个 tile 会跨页。设备专用快路径
因此改为：

1. 用 tile 起点标量计算当前逻辑页和页内起点；
2. 标量加载当前页 block id；
3. 仅在 tile 确实跨页时 masked 加载下一页 block id；
4. 用逐 lane 比较、选择和减法重建原 token 地址。

它消除了热循环内逐 token 的向量整数除法、取模和重复 block-table 读取，保持 K/V
加载、Attention 数学、Softmax 归约顺序和输出语义不变。当前 selector 只对
`ROCm + page784 + head_size256` 开启，并不是任意非标准 page 的通用实现。

官方端到端对照中，纯 unroll1 版本得分为 `80.0198`，叠加地址快路径后为
`80.2281`；三档吞吐从 `16.35 / 14.78 / 11.29` 提升至
`16.42 / 14.84 / 11.40 tok/s`，约为 `+0.4% / +0.4% / +1.0%`，精度与 SLA
仍为零扣分。这是地址快路径最保守、最可辩护的整服务贡献；不能把 kernel 的
`1.0615x` 或组合矩阵的 `1.9716x` 直接表述为整模型吞吐提升。

## 文件哈希

```text
F5B9655898FD161EE3E71301F822CE4F743CAABDDB0EF351D61BFC95860B97EF  xiandao-page784-address-fastpath-microbench-0714.sh
0F36865FC0DE57F5AE9A1CFF6027B63FC380FAA8013E6EDF1B5C66749D1F9F14  xiandao-page784-launch-matrix-0714.sh
E6E62F8EEFD51EBDC3CDC090129F205243F153BB9659C33A00B6915A79A85D1F  page784-address-fastpath-A-B-0714.log
A8FC2673B37190B7BFF841E999E3DA43FBB0501ECFDC2EB59FB7DDED06226D40  page784-launch-matrix-0714-terminal-transcript.txt
8D90270B3F3274D8F3C0B2F67F17F6BDB43BCA3437397CAA24C6F735FFF8639D  page784-launch-matrix-0714.log
```

## 来源边界

- 原实验 Codex 任务：`指导千问推理加速参赛`
- Thread ID：`019f26a6-ce24-74f3-bfaf-1c0927ecce17`
- A/B 日志由该任务的原始用户终端消息机械提取。
- 八配置完整终端转录来自当时上传的 `pasted-text.txt` 附件；提取日志只去除了
  SSH 登录、Git 拉取和 pytest 前导输出，没有改动测量值。
- 方案分析、部分实现、测试与文档由参赛队伍在 OpenAI Codex 辅助下完成；设备
  部署、运行与结果核对由队伍成员执行。
