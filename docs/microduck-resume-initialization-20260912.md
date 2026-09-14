# Microduck 续训初始化验证

本轮修复了从 checkpoint 续训时首个 episode 的课程配置偏差。修复通过固定版本上游的事件接口实现，训练循环、PPO、BAM、奖励配方及模型加载仍使用上游实现。

## 原因与改动

mjlab 1.3.0 的训练入口先构造 `RslRlVecEnvWrapper`，其构造函数立即调用 `env.reset()`；随后才调用 `runner.load()` 恢复 checkpoint 的 `common_step_counter`。课程计算在 reset 开始时发生，因此中期模型的首个 episode 会按计数 0 初始化。

续训 worker 现在读取已复制的 checkpoint 快照，构造上游 `TrainConfig`，并添加两个局部事件：

1. `startup`：在环境构造完成阶段恢复课程计数，先于 wrapper 的首次 reset。
2. `reset`：仅在第一次执行时检查计数并记录课程计算后的实际配置。后续 reset 不再恢复计数或覆盖记录。

随后仍调用上游 `launch_training`。完整训练状态恢复、自适应学习率初始化、随机 episode 长度、TensorBoard 与 checkpoint 保存方式保持原有语义。新的记录被主入口校验，并写入运行 manifest；记录不一致时不能报告成功。

## GPU 实测

从已有的 `model_1000.pt` 追加 5 次 PPO 更新，使用 64 环境，符合小规模冒烟验证的预算。来源 checkpoint 的课程计数为 24,048，学习率为 `0.00017085937500000006`。

首次 reset 实际记录：

| 配置 | 实测值 |
| --- | --- |
| 课程计数 | 24,048 |
| 动作变化惩罚权重 | -0.6 |
| 站立指令采样概率 | 0.15 |
| neck pitch / head pitch 范围 | 各 ±0.39 rad |
| head yaw 范围 | ±0.49 rad |
| head roll 范围 | ±0.11 rad |

最终生成 `model_1004.pt`，课程计数为 24,168，等于 24,048 + 5 × 24。全部 TensorBoard 标量有限，NaN 状态为零。官方 ONNX 导出通过，61 维输入、14 维输出的推理检查通过。`progress` 正确显示本次完成 5 次追加更新、最后编号 1004。

- [首次 reset 记录](../runs/microduck-resume-initialization-20260912/resume.initialization.json)
- [完整运行记录](../runs/microduck-resume-initialization-20260912/run.json)
- [训练指标验证](../runs/microduck-resume-initialization-20260912/metrics.validation.json)
- [导出模型](../runs/microduck-resume-initialization-20260912/policy.onnx) 与 [推理验证](../runs/microduck-resume-initialization-20260912/policy.validation.json)

## 验证范围

专项测试 **141 passed**；`ef-viewer` 全量 **248 passed、31 skipped**（可选依赖）。新增回归覆盖启动事件接入、学习率初始化、首次 reset 的实际配置、后续 reset 不回退、错误计数拒绝，以及主入口对不一致记录的失败处理。Ruff 和 `git diff --check` 通过，上游源码保持干净。

这次修复保证后续续训实验从正确的课程阶段启动，不证明短训后的步态质量提高，也尚未证明此前后期停步由这一问题导致。此前的 2,000 次追加训练从课程计数 480 开始；其表现仍以已有同条件评估为准。此次 64 环境、5 次更新的模型仅用于流程验证，不能替换经过完整评估的模型作为部署候选。

实际续训命令不变，继续使用 `train --resume ... --num-envs ... --iterations ... --output ...`；用法见 [Microduck RL](microduck.md)。本页 `runs/` 链接指向本机实验产物。
