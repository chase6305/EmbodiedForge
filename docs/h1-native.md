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
python -m embodiedforge h1-native train --headless \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml \
  --num-envs 128 --threads 2 --updates 5000 --seed 0 --learning-rate 0.0003 \
  --output runs/h1-native-first
```

本机已有 `.cache/external/mjbatch/.venv/bin/python`，也可使用这个解释器运行同一命令，无需安装 IsaacLab。
该路径只选择 Python 环境；训练不会导入 mjbatch 的上游 example。

```bash
python -m embodiedforge h1-native train --headless \
  --resume runs/h1-native-first --num-envs 128 --threads 2 --updates 1000 \
  --output runs/h1-native-resumed
```

示例采用 [三种子实验](rl-training-study-20260925.md) 中的 128 环境、5000 轮预算，替换早期的 500/1000 轮示例。这是有实测依据的起始预算，不保证得到合格行走策略；每次训练或续训后仍须按阈值重新评估。

输出目录必须不存在。`--updates` 表示本次新增更新数，默认 horizon 为 24。
训练和评估始终无窗口，不初始化渲染器；可显式传入 `--headless`，省略时行为相同。`--seed` 范围为 `0..2**64-1`。
目录包括 `model.mjb`、带更新轮数的 `checkpoint-000000050.pt` 等权重文件、`metrics.jsonl` 和 `run.json`。MJB 包含编译后的模型与网格，续训和回放无需再次读取原 XML/网格路径。权重与 MJB 均从通过哈希校验的同一份字节加载；续训目录保存实际加载的模型，避免输入文件被替换后记录与运行内容不一致。

训练失败或 Ctrl+C / SIGTERM 中断时会尝试更新 `run.json`。若此时写盘也失败，日志记录文件路径及写盘异常，并保留原始训练异常或中断；文件中的状态可能仍为 `running`。正常训练结束后的最终记录写入失败则直接报错，不返回成功。
每 50 轮及最终轮原子保存 checkpoint；续训和评估接受 `complete`、`interrupted` 或 `failed` 目录中已记录的保存点，核对模型和 checkpoint 的 SHA256、任务版本、关节顺序和有限数值。中断后从最后保存的轮数继续，未落盘的进度不会恢复；首次保存前退出或仍标记为 `running` 的目录不能作为输入。
新权重落盘后才更新 `run.json` 中的 `checkpoint` 文件名与哈希，提交成功后通常保留当前和上一个保存点。若提交中途失败，目录可能留下未引用的权重；加载只认运行记录，不按文件名选择最大的轮数。这样，写入新权重后发生可处理的异常或中断，不会覆盖旧记录引用的权重。

旧目录未记录 `checkpoint` 字段时仍读取 `checkpoint.pt`；新目录不生成该别名，`--resume` / `--run` 继续传目录即可。兼容方向是新程序读取旧目录；旧程序不能直接读取新的带轮数目录布局。该机制不承诺断电持久性，也不自动恢复 SIGKILL 后仍为 `running` 的目录。保存发布故障、完整训练和续训对照见 [实测报告](h1-checkpoint-publication-20260929.md)。

续训记录还保存实际加载的来源权重 SHA256（`resume_checkpoint_sha256`），便于追溯输入；从零训练时为 `null`。加载后续运行不要求来源目录仍存在。恢复策略与 Adam 优化器，重新初始化环境和随机数；这不是逐步等价恢复。恢复优化器前及保存 checkpoint 前，使用与原生 Go1 共用的 Adam 校验，检查参数组、完整状态、动量形状/类型、非负二阶矩和整数步数；损坏的优化器状态会在新运行目录和仿真环境创建前被拒绝。
新训练默认学习率为 `0.0003`。续训省略 `--learning-rate` 时继承 checkpoint 中的学习率；显式指定则覆盖保存值。该默认值来自 [三种子学习率对照](rl-training-study-20260925.md)，验证范围是当前平地任务及标称起点评估。

PPO 的探索参数 `log_std` 使用既有范围 `[-5, 2]`，并在每次 Adam 更新后投回该范围。仅在前向计算时截断会让略微越界的参数梯度变成零，旧权重可能因此无法继续调整对应关节的探索噪声。新实现恢复这种更新能力，同时拒绝非有限参数；没有降低标准差上限。旧 checkpoint 仍可加载，确定性推理不变，但触及边界时的后续训练轨迹会与旧实现不同。核心 `train` PPO 同步修复；Go1 的独立 PPO 不属于本次修改。

完整训练预算下的三种子续训、指令/奖励消融、随机初始状态与核心 PPO 对照见 [转向与探索参数研究](h1-turning-study-20260929.md)。边界修复解决梯度冻结，不保证每个训练种子的转向指标都提高；报告保留退化结果。

后续 [指令采样研究](h1-command-mixture-20260930.md) 包含三种子从零训练与续训配对、四组合消融、连续 120 秒指令切换，以及用 `--record-env` 定位真实失败的示例。候选未通过随机存活筛选，原采样默认值保持不变。

[终止惩罚研究](h1-termination-cost-20260930.md) 进一步配对比较 −200 / −400 与两种指令采样，记录首次终止判据、连续指令存活和事后动作/梯度诊断。报告区分本轮新增训练与复用对照，并保留所有训练种子的筛选结果。两项加倍惩罚候选均未通过筛选，保留原采样和 −200 系数。

[探索熵研究](h1-entropy-study-20260930.md) 补充 0.01 / 0.001 熵系数的完整预算配对、历史权重续训、均值/随机动作诊断及权重大小。降熵改善部分跟踪指标，却降低随机起点稳定性；两项候选均未通过筛选，默认仍为原采样、0.01 熵系数和 −200 终止系数。

## 评估与网页回放

```bash
python -m embodiedforge h1-native evaluate --headless \
  --run runs/h1-native-resumed --threads 2 --steps 500 --velocity 0.5 0 0 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --record-motion --output runs/h1-native-eval
```

上述命令只评估前进。站立、原地转向和前进转向应分别更换 `--velocity` 并使用新的输出目录独立验收，例如原地左转 `--velocity 0 0 0.5`、右转 `--velocity 0 0 -0.5`；前进通过不能代表转向合格。

每控制步 0.02 秒，500 步为 10 秒。报告包含存活率、平面与转向速度 RMSE、平均速度、回报和最后子步实际关节力矩峰值。
只统计每行的首次试验，包括终止帧；阈值不通过时保存报告并退出 2。
未指定任何阈值时 `accepted=null`，执行成功不等于行为验收通过。
误差阈值应结合指令幅值选择：例如 `--velocity 0 0 0.25` 配合 `--max-yaw-rmse 0.3` 时，不转动也可能通过 yaw 阈值。请同时检查 `mean_velocity` 与实际动作；固定绝对误差阈值不代表相对跟踪精度。
默认评估从确定性的标称姿态开始，只有一个环境；不加 `--randomized-reset` 时增加环境数会重复同一初始状态。
使用 `--randomized-reset` 可按训练的 reset 分布采样每行的初始状态和物理参数，同时保持固定速度指令、关闭观测噪声。该选项仅用于评估，不改变训练或默认评估行为：

```bash
python -m embodiedforge h1-native evaluate --headless \
  --run runs/h1-native-resumed --randomized-reset --num-envs 32 \
  --seed 9701 --threads 2 --steps 1000 --velocity 0 0 0.5 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/h1-native-randomized-left
```

采样范围为滑动摩擦 0.6–0.9，躯干质量/惯量比例 0.8–1.25（对数均匀），根节点 x/y 为 ±0.5 m、yaw 为 ±π rad，初始线速度分量为 ±0.5 m/s、角速度分量为 ±0.5 rad/s。相同模型、环境数和 seed 可复现同一批初始状态；更换 seed 可采样另一批。这覆盖训练内的初始条件变化，不包含运行中推力、观测噪声或新地形。
各行仍只计首次试验；提前跌倒的行不会因自动重置后的存活而提高成绩。报告的 `identical_initial_states` 和 `reset_protocol` 区分两种协议。全局 RMSE 按已观察帧汇总，提前失败会缩短统计时长，需同时查看 `survival_fraction` 和逐行 `observed_seconds`。同一权重的多个初始状态不等于多个独立训练种子。
`survived`、`planar_rmse_per_env` 和 `yaw_rmse_per_env` 按环境索引保存首次试验的存活结果及误差，可与 `observed_seconds` 配对分析。最后一步跌倒仍记为未存活，不能仅凭观察时长判断。现有验收阈值仍作用于全局指标；逐行 RMSE 的算术平均与按帧汇总的全局 RMSE 不同。
既有报告不会被改写；使用原权重和新的输出目录重新评估，才能获得这些逐环境字段。

`termination_reasons` 同样按环境索引记录首次试验的终止条件；完整存活的行为空列表。一行可能同时触发多项条件，所以各原因数量相加不一定等于跌倒数。`torso_contact` 表示躯干 touch 传感器超过 1，`base_height` 表示根高度低于 0.45 m，`base_tilt` 表示躯干姿态矩阵的 z-z 分量低于 0.2；如发生环境时间截断则记录 `time_limit`。正常完成请求步数不记为终止。它们是终止控制步实际触发的判据，不包含此前积分子步的接触历史，也不等同于物理根因分析。重置后的再次跌倒不会覆盖首次记录。

`action_clip_fraction_per_env` 记录各环境首次试验中，原始策略输出超出 ±5 的动作分量比例（分母为实际观测步数 × 19 个关节）。`action_clip_fraction` 按所有环境实际观测步数加权汇总，包含终止帧、排除重置后的动作；恰好等于边界不计为截断。它描述动作限幅，不代表关节达到机械行程边界，也不作为验收门槛。早期失败行对全局比例的权重较小，定位问题时应同时查看逐环境比例与 `observed_seconds`。

运动记录默认保存环境 0 的首次试验，最多 10000 步。搭配 `--record-motion --record-env 7` 可记录环境 7，索引从 0 开始且必须小于 `--num-envs`；`--record-env` 必须与 `--record-motion` 一起使用。复现某一行时保持相同权重、环境数、seed 和随机化选项，不能把批量缩为 1 后仍认为是同一个起点。记录只包含所选行的首次试验，包括终止帧，不混入重置后的动作。运动文件元数据保存环境索引、批量大小、seed、随机化模式及模型和权重的 SHA256。

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
- 当前训练采样前进速度 0–1 m/s、横向速度 0、转向速度 ±1 rad/s，并将 2% 指令置为站立；接口可接收横向命令，但该配方没有训练横移能力。原地转向也应单独评估，不能从前进转向的成绩推断。
- PPO 为三层 128 单元 ELU、固定学习率；没有上游 adaptive-KL 学习率调度、分布式训练或 GPU 批量状态。

## 本机验证：2026-09-15

更新：2026-09-25 完成了六组 5000 轮训练，较低学习率在三个训练种子的标称起点行走和转向评估中均完成 10 秒试验；数据与限制见 [学习率对照](rl-training-study-20260925.md)。以下保留早期 500 轮结果。

`runs/h1-native-validation-20260915` 保存训练、续训、评估和 RTX 回放证据。
128 环境 × 24 步 × 500 轮完成 1,536,000 条环境转换，4 个 CPU 线程下训练循环约 75 秒。
500 轮策略在标称起点的 10 秒前进评估中未倒，但平均前进速度约 0.0065 m/s，平面 RMSE 约 0.4996 m/s，**没有通过 0.5 m/s 行走跟踪验收**。
这证明独立训练链路能够执行，不代表已有合格的行走策略。

测试在禁止导入 `isaaclab*`、`isaacsim*`、`omni*`、`rsl_rl*` 的进程中执行训练、续训和评估；另验证关节顺序、选行隔离、随机化常量、实际力矩与单环境 MuJoCo 的一致性，以及记录的独立 FK 回放。
