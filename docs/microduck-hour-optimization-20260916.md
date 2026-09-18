# Microduck headless 优化记录（2026-09-16～17）

本轮在 `ef` 主入口和独立 Python 3.12 SDK 环境中优化本地 Microduck 工作流。任务于北京时间 9 月 16 日 23:31 开始，原定一小时；后续一次 GPU 工具确认等待跨过该窗口，SIGTERM 实测于 9 月 17 日 11:00 完成。等待不计作持续训练，也未把训练扩展为整夜运行。发现超时后停止新增训练实验并收尾。

## 工程改进

- 新训练、续训、smoke 与评估保存分阶段日志；终端仍可查看输出，`--quiet` 可关闭工作进程回显。
- `phase`、`active_log`、`failed_phase` 指明执行进度和故障位置；运行前检查失败保存 `runtime.failure.json`。
- 本地运行保存约 25 MB 的实现快照，按启动时的文件散列逐项核对；后续任务与导出从快照加载，多种子评估共享同一快照。
- 支持 SIGTERM/SIGHUP 清理，保留 nohup 的挂断忽略行为；中断状态和信号写入运行记录。
- 进度查询核对 Linux 启动器 PID、启动时间和 boot ID，不把复用的 PID 误报为同一作业；跨主机和旧记录保留未知状态。
- `runtime.json` 记录实际显示环境、GL 后端、CUDA 设备选择和检查时显存；显存读数不是训练峰值预测。
- 训练/续训/smoke 支持 `--seed`，默认仍为 0。
- 本地 TensorBoard checkpoint 使用临时文件加原子替换，失败时保留旧完整模型；外部日志上传模式保留 SDK 原行为。

## 验证

| 项目 | 结果 |
| --- | --- |
| Microduck 回归测试 | 203 passed，10 skipped |
| Ruff / diff whitespace | 通过 |
| 实现快照 smoke | 64 环境、5 次更新、完整 ONNX 导出通过；运行记录与实际加载代码散列一致 |
| 有界续训 | 512 环境、1000 次更新，约 8 分 18 秒；指标全部有限、NaN 状态为 0 |
| 实际 SIGTERM | seed=42，首个 checkpoint 发布后发送 SIGTERM；退出码 143、状态 interrupted、信号 15 |
| 中断后模型 | `model_0.pt` 可完整读取；课程计数 24，启动器已退出，无剩余临时 checkpoint |
| 实际 headless 环境 | DISPLAY/Wayland 变量均无，MUJOCO_GL 与 PYOPENGL_PLATFORM 均为 egl |
| wheel | 构建成功；从解包后的 wheel 加载任务、资产和进度查询通过，无需源码目录 |

主要本机产物：

- `runs/microduck-hour-opt-snapshot-smoke-20260916`
- `runs/microduck-hour-opt-training-20260916`
- `runs/microduck-hour-opt-before-20260916`
- `runs/microduck-hour-opt-after-20260916`
- `runs/microduck-hour-opt-comparison-20260916`
- `runs/microduck-hour-opt-sigterm-20260916`

这些目录保留 `run.json`、模型、日志、配置和适用的验证报告；产物不纳入源码分发。

## 续训没有改善本次速度跟踪

以历史 `model_2018.pt` 为起点追加 1000 次更新，保存 `model_3017.pt`。比较使用相同的三个种子（0、1、2）、16 环境、每环境 1000 步（20 秒）、0.2 m/s 前进指令、关闭推扰、课程起点 48480 和 TF32 精度。

| 指标（三个种子的均值） | 续训前 | 续训后 |
| --- | --- | --- |
| 平面速度 RMSE（m/s，越小越好） | 0.18757145 | 0.19222329 |
| 每步平均奖励 | 0.16131981 | 0.15929317 |
| 摔倒次数 | 0 | 0 |

本次续训未显示速度跟踪改善，不将新模型标为更优，也没有覆盖原模型。未调整奖励、观测、动作或课程配方；上述工程改进不能等同于步态质量提升。此对照仅覆盖所列工况，不代替更广泛的行走或实机验收。

使用说明见 [Microduck](microduck.md)。
