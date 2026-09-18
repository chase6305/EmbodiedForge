# Microduck 平地行走移植验证（2026-09-16）

本次将平地行走任务迁入 `src/embodiedforge/locomotion/microduck/`。默认运行不再依赖原 `microduck_rl` 检出；底层 mjlab、MuJoCo-Warp、BAM 与 RSL-RL 继续作为固定版本依赖。主入口仍在 `ef`，训练环境位于 `.cache/microduck-native-venv`（Python 3.12）。

## 范围与来源

基线提交：`53b8971b61baf5b7f3c16d135dd7cac37623de4b`。迁入内容包括任务配置、共享 MDP 辅助代码、对称增强、执行器扩展、平地行走机器人常量、`robot_walk.xml` 及其引用的全部 38 个 STL 网格。保留 Apache-2.0 许可证；源文件路径和散列见包内 `UPSTREAM.json`，修改范围见 `NOTICE`。

只注册 `Mjlab-Velocity-Flat-MicroDuck`。其他上游任务、HF Jobs、发布和实机运行程序未迁入。共享 MDP 中对 RewardManager、PPO 和导出元数据的扩展保留原行为，只在隔离工作进程中加载。原仓库未修改。

## 已完成的验证

| 检查 | 结果 |
| --- | --- |
| Microduck 回归测试 | 172 passed，10 skipped |
| 本地与上游的 train/play/RL 配置 | 统一模块命名空间后完全一致 |
| MuJoCo 编译后的机器人模型 | 468 组数组完全一致；`nq=21`、`nv=20`、`nu=14` |
| 旧 checkpoint 的确定性 actor 推理 | 1,024 组输入，动作逐元素一致，最大差异 0 |
| 带观测归一化的 ONNX 推理 | 1,024 组输入，Torch/ONNX 最大绝对误差 `4.76837158203125e-6` |
| 本地环境的源码隔离 | 未安装 `mjlab-microduck` 分发包；验证进程未导入 `mjlab_microduck` |
| wheel 数据完整性 | XML、38 个网格、许可证和依赖清单完整，内容与源码文件一致 |
| wheel 中任务的模型和推理 | 从解包后的 wheel 加载，重复上述配置、模型与动作对照，通过 |
| 固定依赖清单 | `uv pip sync --dry-run` 解析 141 个包，无需修改；随后默认 `setup` 成功 |
| CUDA 基础检查 | 本地 `check` 曾完成 GPU 矩阵运算，并报告本地组件路径 |

旧模型为 `runs/microduck-hour-training-20260912` 中的 `model_2018.pt`，SHA-256：

```text
096b5fc09326da3d5cdcd25a6afebaf2cdcdf9960333dad3a2b0c0c996a572fb
```

推理对照在 CPU 上执行，输入包括一组零向量和固定种子生成的 1,023 组随机 61 维向量，输出 14 维动作。ONNX 使用同一 mjlab runner 的导出方法及 checkpoint 中的观测归一化。它证明模型加载与推理兼容，**不是实际轨迹、GPU 训练或实机行走验收**。

本机详细产物保存在 `runs/microduck-port-validation-20260916/`：`comparison.json`、三个来源的配置/模型数组/动作/ONNX/摘要，以及执行对照的 `snapshot.py`。运行产物不纳入源码分发。

隔离环境的初始依赖复用了已有安装文件，再通过正式 `setup` 同步固定清单；本次没有声称从空缓存重新下载所有依赖。

## GPU 与 headless 实测已通过

先前因其他进程占用显存而失败的运行仍保留为 `failed`。显存空闲后，以下验证已于同日完成，使用 NVIDIA GeForce RTX 5090 D v2；没有中止其他进程。

| 本机运行目录 | 结果 |
| --- | --- |
| `runs/microduck-headless-smoke-20260916` | 64 环境、5 次 PPO 更新，完整 CLI ONNX 导出和 CPU 推理检查通过 |
| `runs/microduck-native-resume-20260916` | 从历史 `model_2018.pt` 追加 2 次更新，保存 `model_2019.pt`；首次 reset 恢复课程计数 48480 |
| `runs/microduck-native-eval-20260916` | 16 环境 × 250 步，固定前进 0.2 m/s、关闭推扰、seed=0；观测、动作和奖励均有限 |
| `runs/microduck-upstream-eval-20260916` | 原实现采用相同 checkpoint、指令、精度、ONNX 和种子完成对照 |

训练和续训记录均为 `complete`，`run.json.execution` 保存 `headless=true`、`viewer=null`、`video=false`、`mujoco_gl=egl`。启动器在工作进程导入 SDK 前清除了显示服务环境变量。两次训练的指标均全部有限，NaN 状态数均为 0。

实际观测下，移植版本完成 250 次 ONNX/Torch 动作抽样对照，最大绝对误差为 `2.980232238769531e-7`（容差 `1e-4 + 1e-4 * abs(torch)`）。此评估使用前述旧 checkpoint 的 CPU 导出模型；smoke 中新模型的完整 CLI 导出另行通过。

| 5 秒评估指标 | 本地移植 | 上游 |
| --- | --- | --- |
| 平均每步奖励 | 0.16151076 | 0.16148100 |
| 平面速度 RMSE（m/s） | 0.18460187 | 0.18460580 |
| 平均前进速度（m/s） | 0.05629920 | 0.05605575 |
| 完整观察期内未终止的首次 episode | 16/16 | 16/16 |
| 摔倒 / NaN 终止数 | 0 / 0 | 0 / 0 |

GPU 报告并非逐位相等；这里只比较统计报告，没有保存逐步完整轨迹。平面 RMSE 差约 `3.94e-6 m/s`，平均奖励差约 `2.98e-5`。对照产物见 `runs/microduck-port-validation-20260916/gpu-comparison.json`。

这些结果验证了迁移后的训练、续训与导出链路。旧策略的平均前进速度仍显著低于 0.2 m/s 指令；5 秒没有摔倒不能视为步态达标，也不能替代多个种子、长时评估或实机验收。

## 离线诊断启动优化

`progress`、`metrics`、`checkpoint` 和 `onnx` 工作操作现在只加载对应的文件处理依赖，不再注册机器人任务或加载 mjlab、MuJoCo、Warp、BAM。训练、续训、导出和评估仍先注册本地任务。

回归测试覆盖离线操作不注册仿真任务、训练仍按顺序注册。本机另外安装了禁止导入仿真 SDK 的 import hook，四类操作分别在真实训练产物上通过；`progress` 的实际 CLI 输出可直接解析为 JSON。该优化不移除固定依赖安装要求，也不改变 checkpoint 或策略行为。

复现 headless 验证时使用新的输出目录：

```bash
conda activate ef
python -m embodiedforge.microduck train --headless \
  --resume-run runs/microduck-hour-training-20260912 \
  --num-envs 64 --iterations 2 --output runs/microduck-port-resume
```

完整使用说明见 [Microduck](microduck.md)；默认命令不再需要 `--repo`。
