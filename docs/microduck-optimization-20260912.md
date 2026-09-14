# Microduck 训练与评估结果（2026-09-12）

本轮完成了 4096 环境的 2,000 次追加 PPO 更新，以及官方 ONNX 导出、三个种子的固定指令评估和录像。训练链路正常，但最终模型**未通过本次步态阈值验收**，不能视为已完成稳定行走训练。

用户要求的一小时从北京时间 00:24:50 开始；训练于 00:29:07–01:00:48 完成。随后 EGL 主机诊断的授权等待使墙钟时间超出原定 01:24:50 截止点。等待后未追加训练，仅完成最终导出、配对评估与结果整理。

## 本轮实现

- `progress`：读取已落盘 TensorBoard 事件，区分追加更新数和 checkpoint 编号，报告日志时效、有限性、NaN 状态与剩余时间估计。
- `assess`：离线重新验收原始评估报告，保存输入字节快照与散列，无需训练 SDK 或 GPU。
- `compare`：按种子配对两个模型，校验源码、锁文件、课程起点、指令、精度与采样条件。保留误差、平均实际速度及偏差，不自动宣称总体胜负。
- `evaluate --curriculum-step`：在首次 reset 前统一评估课程起点，并处理模型加载再次恢复课程计数的行为。
- `evaluate --video`：使用上游离屏渲染器，按控制步流式写入 MP4，记录实际 OpenGL 设备；退出时关闭编码器和渲染器。修复了 FFmpeg 收到 SIGINT 后退出码 255 覆盖原中断异常的问题。

用法见 [Microduck RL](microduck.md)。上游源码保持干净，未修改其安装文件、奖励配方或 BAM 执行器。该流程仍使用 mjlab/MuJoCo-Warp，并非 Newton 原生机器人训练。

## 训练

RTX 5090 D v2、驱动 595.84；上游提交 `53b8971b61baf5b7f3c16d135dd7cac37623de4b`，严格沿用其锁定环境。

先分别运行 1024 和 4096 环境、20 次更新的吞吐试验。各自最后一次记录约 33,513 和 98,497 环境步/秒；两次都无非有限指标和 NaN 状态。它们只是短时吞吐测量，环境批量不同，不能用相同更新数比较策略质量。

从 4096 环境的 `model_19.pt` 追加 2,000 次更新，共 196,608,000 个环境步。最终保存 `model_2018.pt`，课程计数由 480 增至 48,480。续训编号沿用上游语义，不能仅按文件名计算更新数。最终训练指标校验通过，全部标量有限、NaN 状态为零。

- [训练运行记录](../runs/microduck-hour-training-20260912/run.json)
- [训练指标校验](../runs/microduck-hour-training-20260912/metrics.validation.json)
- [最终 checkpoint](../runs/microduck-hour-training-20260912/logs/rsl_rl/microduck/2026-09-12_00-29-14_embodiedforge/model_2018.pt)
- [最终 ONNX](../runs/microduck-hour-policy-20260912.onnx) 与 [接口验证](../runs/microduck-hour-policy-20260912.validation.json)

最终 checkpoint SHA-256：`096b5fc09326da3d5cdcd25a6afebaf2cdcdf9960333dad3a2b0c0c996a572fb`。

## 同条件对比

训练前后均使用种子 0/1/2，每个种子 8 个环境、250 控制步（5 秒）；固定机体坐标系指令 `(0.2, 0, 0)`，关闭推扰，头部与身体使用中性指令，课程起点统一为 0。使用完整 FP32，每步抽样一份真实观测执行 ONNX/PyTorch 动作对照。

下表是三个种子的平均值，RMSE 为逐种子 RMSE 的均值：

| 指标 | 追加训练前 | 追加训练后 |
| --- | ---: | ---: |
| 平均实际前向速度（m/s） | 0.001670 | 0.041020 |
| 平面速度 RMSE（m/s） | 0.198684 | 0.189155 |
| 转向速度 RMSE（rad/s） | 0.028273 | 0.209201 |
| 首次 episode 完整存活 5 秒 | 24/24 | 24/24 |

最终模型的前进速度仍远低于指令，转向误差反而增加。三个种子均未满足平面 RMSE ≤ 0.1 m/s、转向 RMSE ≤ 0.1 rad/s 的本次示例阈值；存活比例 ≥ 0.8 通过。批次状态为 `rejected`，退出码 3，全部原始报告与完整汇总均保留。

750 次实际观测上的 ONNX/PyTorch 动作对照全部通过。它验证导出的一致性，不代表 ONNX 独立闭环或实机验收。此处只覆盖一个速度指令和短时观察，未覆盖长时间、多指令或实机表现。

- [训练前评估](../runs/microduck-hour-before-20260912/summary.json)
- [训练后评估](../runs/microduck-hour-after-20260912/summary.json)
- [逐种子验收](../runs/microduck-hour-after-20260912/acceptance.json)
- [配对对比与差值](../runs/microduck-hour-comparison-20260912/comparison.json)
- 最终录像：[种子 0](../runs/microduck-hour-after-20260912/seed-0/policy.mp4)、[种子 1](../runs/microduck-hour-after-20260912/seed-1/policy.mp4)、[种子 2](../runs/microduck-hour-after-20260912/seed-2/policy.mp4)

本轮结束时，中间的 `model_1000.pt` 仅做过 TF32 单种子试验，曾达到约 0.136 m/s。后续已经补齐完整 FP32、多种子以及 15 秒逐环境对比，见 [保存阶段分析](microduck-checkpoint-analysis-20260912.md)：中间模型前进能力更强，但转向误差更大，仍不能视为合格策略。

## 渲染与退出验证

640 × 480、50 fps 录像已实测；最终种子 0 的文件经 FFprobe 确认包含 250 帧、时长 5 秒，并已检查实际画面。真实 Ctrl+C 复测退出码为 130，运行状态为 `interrupted`，保留的 270 帧、5.4 秒部分视频可正常读取。

OpenGL 诊断显示默认离屏渲染使用 AMD RAPHAEL_MENDOCINO 集显，而 CUDA 训练使用 NVIDIA。检查发现 `libnvidia-gl-595:amd64` 已安装，但 `dpkg --verify` 报告 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 等包文件缺失；NVIDIA EGL 库本身存在。这支持 NVIDIA EGL 注册不完整的判断，尚未完成系统包修复或 NVIDIA EGL 渲染复验。未屏蔽 Mesa 的设备探测警告。

此前用户重新安装的是 `libnvidia-compute-595`，缺失的上述注册文件属于 `libnvidia-gl-595`。后续应先修复这个确切版本的图形包，再检查 EGL 实际厂商；本轮未自行修改系统注册文件，也未宣称 RTX 查看器的历史警告已全部解决。

## 测试

Microduck 专项测试 **129 passed**，包含真实 TensorBoard/PyTorch 测试以及录像初始化失败、编码失败、超时、中断和清理顺序。`ef-viewer` 环境全量测试 **237 passed、30 skipped**；跳过项需要该环境未安装的可选依赖，相关训练测试已在专项环境执行。Ruff 和 `git diff --check` 通过。

上述 `runs/` 链接指向本机实验产物，按项目约定不加入 Git。
