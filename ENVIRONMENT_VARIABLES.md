# 环境变量说明

本提交不依赖新增的自定义环境变量。

运行时仅使用比赛基础镜像和 vLLM 原有环境配置。`serve_args.txt` 中的
`rms_norm_gated` 是 vLLM 编译配置项，不是环境变量；本提交未增加或覆盖
`VLLM_*`、`HIP_*`、`HSA_*`、`ROCM_*` 等环境变量。
