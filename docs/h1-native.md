# H1 原生训练：不依赖 IsaacLab

[English](h1-native.en.md) · [训练与部署](training-deployment.md)

`h1-native` 的模型构建、关节接口、任务、观测、奖励、重置和 PPO 均由本仓库执行。
运行时只需要 MuJoCo 3.11、mjbatch 0.1.0、NumPy 和 PyTorch；不导入 IsaacLab、Isaac Sim、RSL-RL 或上游训练脚本。
当前是 **CPU 批量仿真和 CPU PPO**，并非原 Newton/MuJoCo-Warp GPU 配方的等价替换。
原来的 `h1` 命令继续运行 IsaacLab 配方；两个入口的 checkpoint 不兼容。

## 安装与训练

在仓库根目录执行。H1 MJCF 需要带齐其引用的网格文件；模型资产本身保留原许可。

```bash
conda create -n ef-h1-native python=3.12 pip
conda activate ef-h1-native
python -m pip install -e '.[h1-native]'
python -m embodiedforge h1-native train \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml \
  --num-envs 128 --threads 4 --updates 1000 \
  --output runs/h1-native-first
```

本机已有 `.cache/external/mjbatch/.venv/bin/python`，也可使用这个解释器运行同一命令，无需安装 IsaacLab。
该路径只选择 Python 环境；训练不会导入 mjbatch 的上游 example。

```bash
python -m embodiedforge h1-native train \
  --resume runs/h1-native-first --num-envs 128 --threads 4 --updates 1000 \
  --output runs/h1-native-resumed
```

输出目录必须不存在。`--updates` 表示本次新增更新数，默认 horizon 为 24。
目录包括 `model.mjb`、`checkpoint.pt`、`metrics.jsonl`、`run.json`。MJB 包含编译后的模型与网格，续训和回放无需再次读取原 XML/网格路径。
每 50 轮及最终轮原子保存 checkpoint；续训接受 `complete` 目录，核对模型和 checkpoint 的 SHA256、任务版本、关节顺序和有限数值。
恢复策略与 Adam 优化器，重新初始化环境和随机数；这不是逐步等价恢复。
`--learning-rate` 默认 0.001，续训时使用这次指定的学习率。

## 评估与网页回放

```bash
python -m embodiedforge h1-native evaluate \
  --run runs/h1-native-resumed --steps 500 --velocity 0.5 0 0 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --record-motion --output runs/h1-native-eval
```

每控制步 0.02 秒，500 步为 10 秒。报告包含存活率、平面与转向速度 RMSE、平均速度、回报和最后子步实际关节力矩峰值。
只统计每行的首次试验，包括终止帧；阈值不通过时保存报告并退出 2。
未指定任何阈值时 `accepted=null`，执行成功不等于行为验收通过。
当前评估从确定性的标称姿态开始，默认只有一个环境；增加环境数会重复同一初始状态，不能当作多样本鲁棒性测试。训练时的随机化和观测噪声在评估中关闭。
运动记录仅保存环境 0 的首次试验，最多 10000 步。

在独立查看器环境执行：

```bash
conda activate ef-viewer
python -m embodiedforge replay \
  --model runs/h1-native-resumed/model.mjb \
  --motion runs/h1-native-eval/motion.npz \
  --render-backend rtx --port 8081
```

回放也可选 `mujoco` 或 `gl`。现有 `live` 命令仍只支持 Go1，尚不能在线控制这个 H1 策略。

## 项目内接口

| 能力 | 实现及约定 |
| --- | --- |
| 模型 | `build_model(path)` 编译 H1 MJCF；`ArticulationBatch` 管理同一拓扑的多行仿真 |
| 关节分组 | `robot.group(*patterns)` 完整正则匹配，返回稳定的执行器顺序；空匹配报错，不假设 qpos 顺序 |
| 批量位置指令 | `robot.set_joint_position_targets(values, env_ids=..., joint_ids=...)`；要求单位传动比的位置执行器 |
| 批量速度命令 | `env.set_commands(values, env_ids=...)`，形状 `(选中环境数, 3)`，对应前进/横移/转向；固定指令保留到 reset 后 |
| 参数随机化 | `robot.randomize(...)`，按环境修改滑动摩擦及指定刚体质量/惯量，并重算派生常量；从标称值采样，不累乘 |
| 状态重置 | `robot.reset(ids, qpos=..., qvel=...)` 和任务级 `env.reset(ids)`，仅重置指定行；MuJoCo 四元数为 wxyz |
| 力矩与状态 | `robot.snapshot()` 返回独立数组，含关节位置/速度、目标、刚体姿态、执行器力与 `qfrc_actuator` 对应的关节力矩 |

力矩是上一次控制步**最后一个积分子步**的实际执行器结果，随后刷新的 FK 不会覆盖它；不是整步平均力矩，也不包含全部被动关节力或外力。
H1 分组为 legs、feet、arms，另有用于奖励的 hip、torso 分组，共 19 个关节，观测 69 维。
MuJoCo 的 `qpos0` 保持资产的运动学参考值；站立姿态写入各环境状态，不通过修改 `qpos0` 改变模型运动学。

## 与 IsaacLab 配方的差异

参考本机 IsaacLab revision `2e44ddb2e19536579140496023b5ccb060bc4152` 的 H1 Flat/Rough 配置、Unitree 执行器配置与奖励定义，保留 BSD-3-Clause 归属和许可。
当前已迁入位置 PD、动作缩放、69 维观测组合、主要奖励及 PPO 更新，但不是 IsaacLab 全套功能移植：

- 使用 MJCF 与 MuJoCo CPU 接触求解；脚和躯干使用 touch 传感器，另有高度/倾斜终止保护，未复现上游接触力历史。
- 线速度观测来自躯干 IMU；跟踪奖励使用浮动根节点世界线速度在躯干 yaw 坐标系的分量，与上游刚体状态接口可能不同。
- 随机化在 reset 时进行：滑动摩擦 0.6–0.9、躯干质量/惯量比例 0.8–1.25、根位置/yaw/速度扰动。未迁入独立静/动摩擦、恢复系数、推力事件或完整事件管理器。
- 每 10 秒直接重采样速度命令，没有 heading 控制器、命令课程或地形课程。
- PPO 为三层 128 单元 ELU、固定学习率；没有上游 adaptive-KL 学习率调度、分布式训练或 GPU 批量状态。

## 本机验证：2026-09-15

`runs/h1-native-validation-20260915` 保存训练、续训、评估和 RTX 回放证据。
128 环境 × 24 步 × 500 轮完成 1,536,000 条环境转换，4 个 CPU 线程下训练循环约 75 秒。
500 轮策略在标称起点的 10 秒前进评估中未倒，但平均前进速度约 0.0065 m/s，平面 RMSE 约 0.4996 m/s，**没有通过 0.5 m/s 行走跟踪验收**。
这证明独立训练链路能够执行，不代表已有合格的行走策略。

测试在禁止导入 `isaaclab*`、`isaacsim*`、`omni*`、`rsl_rl*` 的进程中执行训练、续训和评估；另验证关节顺序、选行隔离、随机化常量、实际力矩与单环境 MuJoCo 的一致性，以及记录的独立 FK 回放。
