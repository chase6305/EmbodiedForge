# Microduck 保存阶段对比与逐环境诊断

同条件评估确认：`model_1000.pt` 比最终的 `model_2018.pt` 更能执行前进指令，但转向控制更差。两者都没有通过既定的步态阈值。不能默认最后保存的 checkpoint 最好，也不能仅凭未摔倒判断已经学会行走。

## 三个种子的 5 秒对比

使用种子 0/1/2，各 8 个环境 × 250 控制步，机体坐标系指令 `(0.2, 0, 0)`、无推扰、头部与身体中性指令，课程起点 0、完整 FP32。每个 checkpoint 均通过官方路径导出 ONNX，再在真实观测上检查动作一致性。

| 三个种子的均值 | 最终模型 2018 | 中间模型 1000 |
| --- | ---: | ---: |
| 实际前向速度（m/s） | 0.041020 | 0.134032 |
| 平面速度 RMSE（m/s） | 0.189155 | 0.160992 |
| 转向速度 RMSE（rad/s） | 0.209201 | 0.380478 |
| 首次 episode 完整存活 5 秒 | 24/24 | 24/24 |

中间模型的三个种子也均未满足平面 RMSE ≤ 0.1 m/s、转向 RMSE ≤ 0.1 rad/s 的阈值，评估状态正确记为 `rejected`；全部 750 次 ONNX 动作对照通过。阈值沿用上一轮，没有为使结果通过而调整。

产物：[中间模型汇总](../runs/microduck-stage1000-evaluation-20260912/summary.json)、[验收结果](../runs/microduck-stage1000-evaluation-20260912/acceptance.json)、[同条件配对对比](../runs/microduck-stage-comparison-20260912/comparison.json)、[中间模型 ONNX](../runs/microduck-stage1000-policy-20260912.onnx)。对比文件的 `before` 为最终模型、`after` 为中间模型，差值统一为 `after - before`。

## 延长到 15 秒后，差异来自哪些环境

补充种子 2、各 8 环境 × 750 控制步的同条件评估。两者均有 8/8 个首次 episode 完整存活 15 秒，无摔倒、超时或 NaN 终止。

| 15 秒指标 | 最终模型 2018 | 中间模型 1000 |
| --- | ---: | ---: |
| 实际前向速度均值（m/s） | 0.014699 | 0.130436 |
| 平面速度 RMSE（m/s） | 0.195796 | 0.161100 |
| 转向速度 RMSE（rad/s） | 0.102159 | 0.350997 |

最终模型的 7 个环境平均前向速度只有 0.00032–0.00043 m/s，几乎没有前进；仅环境 2 约为 0.11494 m/s。中间模型的全部 8 个环境都在前进，速度约为 0.11773–0.14158 m/s，仍低于 0.2 m/s 指令。

![各环境平均前向速度](../runs/microduck-stage-long-comparison-20260912/per-environment-velocity.png)

此图展示各环境在观察期内的时间平均值，不是独立训练种子或置信区间。机器人自身摆动可能使瞬时 RMSE 较大；均值也可能让正负误差抵消，因此两类指标都保留。

产物：[最终模型原始报告](../runs/microduck-final-long-evaluation-20260912/evaluation.json)、[中间模型原始报告](../runs/microduck-stage1000-long-evaluation-20260912/evaluation.json)、[15 秒配对对比](../runs/microduck-stage-long-comparison-20260912/comparison.json)、[SVG 图表](../runs/microduck-stage-long-comparison-20260912/per-environment-velocity.svg)、[中间模型录像](../runs/microduck-stage1000-long-evaluation-20260912/policy.mp4)。录像经 FFprobe 确认包含 750 帧、时长 15 秒，并已检查画面。

两次延长评估合计 1,500 次 ONNX/PyTorch 动作对照通过，最大绝对误差约 `6.56e-7`。这仍是由 PyTorch 驱动仿真、抽样对照 ONNX 的验证，不是实机验收。

## 实现改进

`EvaluationMetrics` 现在按环境分别累计指令、实际速度、绝对误差与平方误差，并输出 `velocity_tracking.per_environment`。批量指标由这些累计值合并得到，仍在自动 reset 和指令重采样之前采样，包含终止步；存储随环境数增长，不随步数增长。

每行代表一个环境槽在全部控制步上的统计，包含后续自动重启的 episode；首次 episode 的生存统计仍在独立字段中。离线 `assess`/`compare` 校验逐环境形状、样本数、有限性、有符号偏差，以及均值和合并 RMSE 与原报告的对应关系。没有逐环境字段的旧报告继续兼容。

回归测试覆盖批量均值抵消环境间误差、终止步统计、样本不完整、形状错误、NaN 和数据不一致。Microduck 专项 **137 passed**；`ef-viewer` 全量 **244 passed、31 skipped**（可选依赖），Ruff 和 `git diff --check` 通过。

## 后续实验依据

上游配方在训练后期同时增加动作变化惩罚、站立环境比例、头部指令范围及其他课程强度。本轮确认了前进能力与转向稳定性的取舍，尚未通过消融实验确定是哪一项引起变化。下一步应以中间 checkpoint 为候选起点，逐项控制课程变化并进行同条件评估；目前没有理由直接把最终 checkpoint 当作最佳续训起点。

本轮没有调整奖励、动作过滤、BAM 或上游源码，也未追加长时间训练。结果只覆盖该速度指令、这些种子和观察时长；系统 EGL 注册文件问题沿用上一轮的未修复状态。`runs/` 链接为本机实验产物。
