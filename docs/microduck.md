# Microduck RL

按步骤执行安装、训练、续训、评估和运行，见 [训练与部署命令](training-deployment.md)（[English](training-deployment.en.md)）。

EmbodiedForge 提供独立进程入口，运行 [Microduck 上游](https://github.com/pollen-robotics/microduck_rl/tree/53b8971b61baf5b7f3c16d135dd7cac37623de4b) 的平地行走 PPO 配方。机器人 MJCF、接触、BAM XL330 执行器、奖励、域随机化和网络均来自上游，未移植到当前 `VectorEnv` 或 Newton 点质量适配器。

固定版本为 `53b8971b61baf5b7f3c16d135dd7cac37623de4b`。上游要求 Python 3.12、mjlab 1.3.0、Warp 1.12.0、PyTorch 2.9.1；安装严格使用它的 `uv.lock`，包括 BAM 的 Git 提交及依赖覆盖规则。现有 `ef-viewer` 使用 Warp 1.17.0，因此训练使用独立环境，不添加到 EmbodiedForge 的安装 extras。

## 安装与检查

从 EmbodiedForge 根目录执行；需要 `uv`、Git 和支持 CUDA 的 NVIDIA GPU。`--repo` 指向用户已有的干净源码检出，不会自动 checkout 或覆盖它。

```bash
conda activate ef
python -m embodiedforge.microduck setup \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl
python -m embodiedforge.microduck check \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl
```

默认环境位于当前目录的 `.cache/microduck-venv`；从其他目录执行时，为所有命令添加相同的绝对 `--env-dir`。首次安装需要下载 Python 和数 GiB CUDA/PyTorch 依赖。`check` 验证 Python/SDK 版本、CUDA 可用性及实际 GPU 矩阵运算，并输出环境包版本。入口本身不在调用者 Python 中导入训练 SDK。

## 先运行小规模验证

```bash
python -m embodiedforge.microduck smoke \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --output runs/microduck-smoke
```

该命令依次执行 CUDA 检查、64 个环境的 5 次 PPO 迭代、TensorBoard 指标检查、上游官方 ONNX 导出，以及 ONNX Runtime CPU 推理检查。指标检查要求完成指定次数的更新、标量全部有限、NaN 状态终止数为零，结果保存在 `metrics.validation.json`。导出必须包含上游观测归一化；验证 actor 输入为 61 维、动作输出为 14 维，且零输入和随机输入都产生有限值。启用上游 NaN guard。首次运行需要编译 Warp 内核。

输出目录必须不存在，避免混入旧 checkpoint。`run.json` 记录源码提交、锁文件散列、执行参数、状态和 checkpoint 路径；`runtime.json` 记录实际环境；`logs/rsl_rl/microduck/` 保存上游参数 YAML、TensorBoard 事件和模型。成功的 smoke 另外生成 `policy.onnx` 与 `policy.validation.json`。发生错误或 Ctrl+C 时保留已产生的模型，并更新运行状态。

这只验证训练/导出链路，5 次迭代不能证明已经学会稳定行走，也不能证明实机部署效果。

## 训练、回放与导出

smoke 通过后，再设置正式训练预算，例如：

```bash
python -m embodiedforge.microduck train \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --num-envs 4096 --iterations 4000 \
  --output runs/microduck-walk
```

`--iterations` 必填，避免意外使用上游 50,000 次迭代默认值。使用本地 TensorBoard logger；未启用 W&B 上传或 Hugging Face Jobs。GPU 选择沿用上游约定与 `CUDA_VISIBLE_DEVICES`。环境数需要按可用显存调整，4096 只是训练配置示例。

训练期间可从另一个终端读取已落盘进度：

```bash
python -m embodiedforge.microduck progress \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --run runs/microduck-walk
```

该命令在隔离环境中读取 TensorBoard 事件，不加载模型或使用 CUDA。输出本次已记录的更新数、最新迭代编号、最近 checkpoint、标量有限性、NaN 状态、奖励和吞吐。剩余时间按最近 20 次已记录迭代的采样与学习耗时中位数估计，不包括启动和最终导出。`run_status` 来自运行记录；应结合 `seconds_since_last_event` 判断日志是否仍在更新，它不是进程存活检查。事件尚未写入时，无法判断的指标为 `null`。

从 `run.json` 读取实际 checkpoint 路径，替换下面的 `PATH_TO_MODEL.pt`：

```bash
python -m embodiedforge.microduck play \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --viewer native

python -m embodiedforge.microduck export \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --output runs/microduck-policy.onnx
```

回放使用 mjlab 的 MuJoCo 原生查看器，也支持 `--viewer viser`。它与 EmbodiedForge 的点质量 RTX 示例是不同的入口。导出调用上游 `mjlab_microduck.export`，随后执行相同的 ONNX 推理验证；拒绝覆盖已有文件。

原生回放支持按步数结束，便于检查模型和窗口启动：

```bash
python -m embodiedforge.microduck play \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --viewer native --steps 250 --seed 0
```

`--steps` 是仿真步数停止阈值，暂停时不会增加；退出时打印实际步数和是否被中断。不传该选项则持续回放到关闭窗口或 Ctrl+C。原生回放现在与无窗口评估共用模型加载和环境清理逻辑，默认 seed 为 0。Viser 保持上游入口及 checkpoint 浏览/切换功能，不接受 `--steps` 或非默认 `--seed`。

MuJoCo 3.10.0 的 passive viewer 在守护线程内渲染，`Handle.close()` 只请求退出。本项目的局部子类等待本次创建的渲染线程结束，再释放环境并退出进程，避免 GLFW 的进程退出清理与窗口销毁并发。该适配使用固定版本的私有线程入口标识，启动前检查 MuJoCo 版本；没有修改上游安装文件或屏蔽警告。

Linux 启动器为子任务创建独立进程组：第一次 Ctrl+C 转发 SIGINT，并给予最多 10 秒清理时间；再次 Ctrl+C 或清理超时才强制结束。中断退出码保持 130，训练/评估运行记录保持 `interrupted`。

## 断点续训

```bash
python -m embodiedforge.microduck train \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --resume PATH_TO_MODEL.pt --num-envs 4096 --iterations 1000 \
  --output runs/microduck-resumed
```

`--iterations` 表示**追加**的 PPO 更新数。入口在新运行目录复制输入 checkpoint，记录 SHA-256、原始路径、课程计数与学习率；原训练目录不被覆盖。仅支持当前上游格式的完整训练 checkpoint，需要 actor、critic、优化器和环境课程计数。

上游恢复网络、归一化、优化器及 `common_step_counter`。本入口额外从优化器读取学习率，写入上游 `TrainConfig.agent.algorithm.learning_rate`，使 RSL-RL 的独立自适应学习率变量也从保存值启动。指标验证按实际起始编号检查：上游从保存的编号重新编号，例如 `model_4.pt` 追加 2 次更新，日志编号为 4、5，最终保存 `model_5.pt`，课程计数仍增加 48 个控制步。不要仅按文件名推断累计更新次数。

续训还会通过上游 `startup` 事件在首次 reset 前恢复课程计数。原入口的顺序是先创建 `RslRlVecEnvWrapper`（立即 reset），再加载模型；仅等待模型加载恢复计数，会让首个 episode 使用训练初期的课程配置。本入口仍调用上游训练 launcher、runner 和模型加载，只在本次配置中增加计数恢复与首次 reset 记录事件，不修改上游源码或安装文件。

`resume.initialization.json` 保存首次 reset 在课程计算之后实际使用的计数、动作变化惩罚权重、站立指令采样比例、头部指令范围及 checkpoint 散列；其中站立比例是配置概率，不是本次采样中实际站立的环境占比。后续 reset 不再回写计数。运行结束时会校验记录与输入 checkpoint 一致；缺失或不一致会将本次运行判为失败，成功记录也写入 `run.json.resume_initialization`。

这是上游的 checkpoint 续训语义：重新建立仿真环境，不恢复每个环境的物理状态或随机数状态，不能视为原进程的逐步精确续接。

## 无窗口评估

```bash
python -m embodiedforge.microduck evaluate \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --num-envs 16 --steps 250 --seed 0 \
  --output runs/microduck-evaluation
```

评估按指定步数自动结束，不需要桌面窗口。`evaluation.json` 包含 checkpoint 散列、每步平均奖励、完成的 episode 数和平均长度、仍未结束的 episode 长度，以及各终止原因的计数。始终检查观测、动作和奖励是否有限；NaN 状态导致失败。`run.json` 和 `runtime.json` 记录运行状态与环境。

默认评估采用上游 `play=True` 配方，保留它的随机指令和推扰，策略使用确定性动作并应用观测归一化。各终止原因可能同时触发，计数不能相加后当作 episode 数；未结束的 episode 也不计入完成 episode 的平均长度。

固定前进指令并检查导出模型：

```bash
python -m embodiedforge.microduck evaluate \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --onnx PATH_TO_POLICY.onnx \
  --velocity 0.2 0 0 --no-pushes \
  --num-envs 16 --steps 250 --seed 0 \
  --output runs/microduck-fixed-evaluation
```

`--velocity VX VY WZ` 使用机体坐标系，单位依次为 m/s、m/s、rad/s；头部与身体姿态指令设为相对 HOME/标称站姿的零偏移。环境重置和定时采样都保持这些值，关闭会改写指令的课程、随机站立/转向/世界坐标系模式；每步及末状态检查实际指令。`--no-pushes` 单独控制外部推扰。BAM、其他域随机化和奖励课程仍采用上游配置，不代表无随机性的理想环境。

`--onnx` 每步选一个环境（步号对环境数取余），将策略收到的原始 61 维观测同时送入 ONNX CPU 推理，与 PyTorch 裁剪前的 14 维确定性动作比较。这能发现导出模型错配、归一化遗漏等问题。对照模式关闭 TF32，使用完整 FP32，逐元素容差为 `abs(onnx - torch) <= 1e-4 + 1e-4 * abs(torch)`。仿真仍由 PyTorch 动作驱动；这是实际观测抽样对照，不能替代 ONNX 独立闭环或实机验收。

报告新增 `conditions`（指令、推扰、计算精度、指令检查次数）及 `onnx_parity`（模型散列、样本数、最大绝对误差、容差）。失败退出码非零，`run.json` 标为失败，worker 的具体错误写入 `evaluation.failure.json`。比较策略时应保持条件与精度一致，覆盖多个种子；奖励权重受课程影响，不能只用平均奖励判断步态质量。相同种子也不保证 GPU 仿真逐位一致。

步态评估还会输出以下指标，无需新增参数：

- `velocity_tracking`：按 `[vx, vy, wz]` 顺序记录平均指令、平均实际速度、MAE 和 RMSE，单位为 `[m/s, m/s, rad/s]`；`planar_velocity_rmse_m_s` 为 `sqrt(mean(vx_error² + vy_error²))`，不混入角速度误差。
- `velocity_tracking.per_environment`：按环境编号分别保留上述速度指标及有符号偏差（实际均值减指令均值）。每行覆盖这个环境槽的全部控制步，包含终止步与后续自动重启的 episode；它不是首次 episode 专属统计。可据此区分所有机器人都在缓慢移动，还是只有少数移动、其他停在原地。离线验收和对比会校验逐环境样本数、数据形状，以及它们与批量均值、RMSE 的一致性；旧报告没有该字段时仍可读取。
- `initial_episodes`：每个环境首次 episode 的持续时间、是否终止、是否仅到达上游时间上限，以及持续到完整评估时长且未终止的数量。后续自动重启不会覆盖首次记录，避免反复摔倒的环境在这项统计中被重复计入。

速度取机器人 root link 的机体坐标系真实状态，使用上游 metrics 回调在物理步之后、自动重置和指令重采样之前累计。每个环境步等权，包含终止步和未结束 episode；重置后的速度与新指令不会代替终止步的数据。采样与上游奖励/指标处于同一阶段，其派生运动学量可能滞后一个物理子步。

`right_censored=true` 表示本次观察没有看到首次 episode 的失败终止（评估结束或仅触发上游时间上限），记录时间是已观察到的下界。`mean_observed_duration_seconds` 是这些有限观察时长的平均值，不是完整寿命估计。`censored_before_horizon_count` 记录提前到达上游时间上限的数量，不能将它们当作已经坚持到整个评估结束。若同一步同时超时和失败，则按失败记录。

统计在 GPU 上累计，所需存储仅随环境数增长，结束时生成报告；不保存随步数增长的全部轨迹。checkpoint 的课程计数在首次 reset 前恢复，避免首个 episode 使用训练起始阶段的课程配置。

比较不同训练阶段的 checkpoint 时，添加相同的 `--curriculum-step 0`（或其他非负整数），将两者的**评估课程起点**统一。否则，即使速度和种子相同，保存的课程计数不同仍会改变其他随机化与奖励条件。该参数在首次 reset 前生效，报告记录实际起点及其来自 checkpoint 还是显式覆盖；其他课程随后仍随评估步数推进。

添加 `--video` 可同时将环境 0 录制为评估目录中的 `policy.mp4`；多种子运行在各 `seed-N/` 中分别保存。视频为 640 × 480、按仿真控制频率播放（当前 50 fps），帧数等于 `--steps`，摄像机跟踪机器人。录像使用上游 MuJoCo 离屏渲染器和隔离环境已有的 FFmpeg，逐帧编码，默认选择 EGL；若设置了 `MUJOCO_GL` 则保留该选择。无需桌面窗口，仍需要可用的 GL 渲染环境。

`evaluation.json.video` 记录帧数、帧率、文件散列、OpenGL 厂商/设备/版本和采样方式。CUDA 与 EGL 可能选择不同的 GPU，应以这里的 OpenGL 信息判断实际渲染设备。视频在每次控制步后采样，包含自动重置后的画面；速度与终止统计仍取重置前的 metrics 回调。录像只展示环境 0，不能代替全部环境的统计；启用录像的运行耗时也不适合与纯训练吞吐直接比较。退出或中断时关闭编码器和渲染器，失败运行中保留的部分视频不代表评估已完成。

同一条件下评估多个种子：

```bash
python -m embodiedforge.microduck evaluate \
  --repo /home/ubuntu/workspace/3rdparty/microduck_rl \
  --checkpoint PATH_TO_MODEL.pt --onnx PATH_TO_POLICY.onnx \
  --velocity 0.2 0 0 --no-pushes \
  --num-envs 8 --steps 250 --seeds 0 1 2 \
  --output runs/microduck-seeds
```

`--seeds` 与 `--seed` 互斥，种子不可重复，取值为 `0..2**32-1`。每个种子独立启动、顺序执行并释放环境；总环境步为种子数 × `--num-envs` × `--steps`。每个 `seed-N/` 保存自己的报告、运行状态和运行时信息，批次根目录的 `run.json` 记录全部计划种子及已完成种子。

全部完成后生成 `summary.json`，保留逐种子指标，同时给出平均值、种子间样本标准差、最小值和最大值。一个种子的标准差为 `null`；这里的波动不是置信区间，也不把同一种子的环境当成独立训练重复实验。`pooled_velocity_rmse` 先合并各次等样本量评估的平方误差再开方，与 `metrics.vx_rmse.mean` 这种逐种子 RMSE 的平均值含义不同。首次 episode 的平均观察时间仍须结合截断标记解释。

每个速度轴同时汇总 `mean_command`、`mean_actual` 和 `bias`（实际均值减指令均值）。均值能区分站立与实际前进，但正负摆动可能相互抵消，必须与 RMSE 一起阅读，不能用较小的平均偏差代替逐步跟踪误差。

汇总要求 checkpoint 内容、任务、环境数、步数、评估条件、精度和 ONNX 对照配置一致，且每次报告的样本数完整、指标有限。批次开始时记录 checkpoint 散列，后续模型替换会导致失败。中途失败或 Ctrl+C 会保留已经完成的种子和失败诊断，根目录状态标为失败或中断，不生成完整汇总，也不会覆盖已有输出目录。

为单种子或多种子评估指定验收阈值，可添加以下参数：

```bash
--max-planar-rmse 0.1 --max-yaw-rmse 0.1 --min-survival-fraction 0.8
```

这些数值仅为用法示例，需根据任务和评估时长设定。平面速度 RMSE 上限单位为 m/s，yaw rate RMSE 上限为 rad/s；存活比例下限为每个种子中首次 episode 持续到完整评估时长且未终止的环境数除以环境总数，取值 `0..1`。提前触发上游时间上限的 episode 不算已经坚持到整个评估结束。等于阈值算通过，零阈值也会执行检查。可只指定其中一项，未指定的指标不设要求。

验收要求**每个种子满足全部已指定阈值**，不会用较好的种子抵消未达标种子。批次会先完成所有种子、保存完整 `summary.json`，再写入 `acceptance.json`，其中包含模型散列、条件、评估时长、阈值、每项实测值、单位和失败种子。单种子同样保留 `evaluation.json` 和验收报告。

退出码与根目录运行状态：

| 结果 | 退出码 | `run.json` 状态 |
| --- | --- | --- |
| 运行完成且通过指定阈值 | 0 | `complete` |
| 运行完成但策略未达标 | 3 | `rejected` |
| 运行或数据校验失败 | 1 | `failed`（若已创建目录） |
| Ctrl+C | 130 | `interrupted`（若已创建目录） |

未指定任何阈值时保持原来的运行行为，不生成 `acceptance.json`；此时退出码 0 只说明流程完成，不表示策略步态达标。非法阈值会在创建输出目录和启动 GPU 工作之前拒绝。

已有评估结果可以离线重新验收，无需再次启动仿真，也无需上游仓库、checkpoint 文件或隔离训练环境仍在原位置：

```bash
python -m embodiedforge.microduck assess \
  --run runs/microduck-seeds \
  --max-planar-rmse 0.1 --min-survival-fraction 0.8 \
  --output runs/microduck-reassessment
```

`--run` 指向单种子或批次评估目录；至少提供一个阈值。入口只接受已完成的 `complete` 或 `rejected` 评估，检查全部计划种子、子运行状态、任务/来源/配置、checkpoint 散列以及报告与记录命令的一致性。运行时 JSON 也需保留。直接读取各个 `evaluation.json` 重新计算，不采用可能过期的 `summary.json`。

新目录中保存 `acceptance.json`、本次 `run.json` 和 `inputs/` 下的原始输入字节快照；运行记录包含各输入文件的 SHA-256。原评估目录不会被修改。离线验收沿用退出码 0/3/1/130，输入不完整或阈值非法时在创建输出目录前拒绝，已有输出目录不会覆盖。

两个模型完成相同条件的评估后，可以离线生成逐种子配对对比：

```bash
python -m embodiedforge.microduck compare \
  --before runs/microduck-before --after runs/microduck-after \
  --output runs/microduck-comparison
```

两次评估须使用相同的上游提交、锁文件、任务、种子集合、环境数、时长、精度、指令、推扰和课程起点。ONNX 文件可以不同，但对照采样方式和容差须一致；种子顺序可以不同，会按编号配对。旧报告未记录课程起点时，需要重新评估。与 `assess` 一样，该命令不依赖训练 SDK 或 GPU，并保存原始输入快照。

`comparison.json` 给出各指标的 before/after 值、逐种子变化及变化的平均值与样本标准差。变化统一定义为 `after - before`：RMSE 为负通常表示误差下降，存活数为正表示增加。不推断统计显著性或自动给出总体胜负；使用 `assess` 对候选模型执行具体验收要求。

## Newton 原生移植范围

需要先补齐带关节的机器人资产、关节状态与力矩动作协议、接触传感器、BAM 执行器、GPU 张量环境和批量 reset，再逐项对齐观测、奖励与域随机化。61 维 actor 观测、14 个舵机关节顺序及归一化是训练和部署间的契约。不能仅将 `--physics` 改为 `newton` 就复现该任务。

代码和机器人资产的授权不同，分发资产前需遵循上游许可证；本接入读取外部仓库，未复制模型资产。

## 验证记录

续训首次 reset 的课程恢复修复与 64 环境实测见 [续训初始化验证](microduck-resume-initialization-20260912.md)。

中间与最终 checkpoint 的同条件对比、15 秒录像和逐环境停步诊断见 [保存阶段对比](microduck-checkpoint-analysis-20260912.md)。

本轮 2,000 次追加训练、同条件模型对比、录像及未达标原因见 [2026-09-12 优化结果](microduck-optimization-20260912.md)。

初期 smoke、续训、固定指令与多种子验收记录见 [2026-09-11 验证记录](microduck-validation-20260911.md)。
