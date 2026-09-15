# H1 行走训练：IsaacLab 接入验证

按步骤执行安装、训练、续训、评估和运行，见 [训练与部署命令](training-deployment.md)（[English](training-deployment.en.md)）。

2026-09-12 已在本机完成 H1 平地行走配方的真实 GPU 训练短测：64 个环境、5 轮 PPO 更新、7680 条环境转换，进程正常退出并保存 checkpoint。可以先通过独立 IsaacLab 进程引入训练，随后接入 EmbodiedForge 的训练管理、评估和回放接口。

现已提供 EmbodiedForge H1 训练、续训、固定指令评估与状态查询入口，由独立 IsaacLab 进程执行上游配方。累计 1000 轮训练的策略已通过四项 10 秒平地仿真评估，并完成 60 秒复测及离线骨架回放；60 秒站立项未通过，详见下文。2026-09-15 新增独立的[原生 H1 CPU 训练入口](h1-native.md)，本页继续描述 IsaacLab GPU 配方；二者 checkpoint 不兼容。

## EmbodiedForge 入口

在已有 `ef` 环境及本页所列 IsaacLab 环境中运行，无需向 `ef` 安装训练 SDK：

```bash
conda activate ef
python -m embodiedforge h1 train \
  --num-envs 64 --updates 5 --timeout 240 \
  --output runs/h1-smoke

python -m embodiedforge h1 status --run runs/h1-smoke

# 恢复模型和优化器，再执行 5 轮更新，写入新目录
python -m embodiedforge h1 train \
  --resume-run runs/h1-smoke \
  --num-envs 64 --updates 5 --timeout 240 \
  --output runs/h1-resume
```

也支持 `python -m embodiedforge.h1 ...` 和安装后的 `embodiedforge h1 ...`。`--repo`、`--environment` 可指定其他路径；默认使用本页列出的本机路径。源码必须为已验证 revision 且 checkout 干净；检查进程还会确认 IsaacLab 包实际来自所选 checkout、Python 为 3.12、RSL-RL 为 5.0.1，并记录其他依赖版本。

默认 64 环境、5 轮更新，适合检查入口；正式训练需显式设置 `--num-envs` 和 `--updates`。`--timeout` 包含训练初始化时间，不包含前后各最多 60 秒的 CPU 检查；省略后训练无额外时限。首次资产加载仍可能需要网络。输出目录必须不存在。

每次运行保留：

- `run.json`：源码 revision、入口/检查脚本 SHA256、运行参数、命令、依赖版本、状态和最终模型索引。
- `console.log`：完整上游标准输出和错误输出。运行中可用 `tail -f runs/h1-smoke/console.log` 查看。
- `runtime.json`：独立环境和导入路径检查结果。
- `verification.json`：最终 checkpoint SHA256、迭代编号、张量有限性、每轮 PPO 指标完整性及全部标量有限性。
- `logs/rsl_rl/h1_flat/`：上游模型、解析后的环境/算法配置和 TensorBoard events。

只有训练正常退出、最终模型编号正确、actor/critic 维度匹配、每轮指标齐全且所有检查数值有限时，状态才会标为 `complete`。Ctrl+C 和 SIGTERM 会转发给整个训练进程组并记录 `interrupted`；超时记录 `timed_out`，其他失败记录 `failed`。退出宽限期后会清理剩余子进程，不因上游捕获中断后返回 0 而误记成功。`status` 显示当前记录，不是 PID 存活探针；无法捕获的 SIGKILL 或主机断电可能留下 `running` 状态。

续训目前只接受该入口生成的 `complete` 运行，并核对任务、物理、源码版本和 checkpoint SHA256。输入模型复制到新运行的 `_resume_input/model.pt`，复制后再次核验。上游 RSL-RL 5.0.1 会从保存的迭代索引开始编号，例如 `model_4.pt` 续训 5 轮得到 `model_8.pt`，日志步数为 4～8；`completed_updates=5` 才是此次新增更新数。环境状态和随机数生成器状态没有恢复，因此不是逐步等价的断点恢复。

## 固定指令评估

```bash
python -m embodiedforge h1 evaluate \
  --run runs/h1-resume --output runs/h1-evaluation \
  --num-envs 32 --steps 500 --seeds 0 1 2 \
  --velocity 0.5 0 0 --min-survival-fraction 0.8
```

`--velocity` 依次为前向速度、侧向速度（m/s）和转向角速度（rad/s）。每个种子使用独立进程加载同一个 checkpoint，以确定性策略执行无窗口推理。默认 32 环境、500 步（10 秒）、种子 0/1/2，单个种子包含初始化的超时为 240 秒。每个环境只统计第一次试验；全部环境结束后提前停止。

使用上游 H1 Flat Play 配置，关闭观测噪声与外部推力，保留上游启动/重置随机化。固定速度覆盖整个试验，并关闭随机站立、随机 heading 和速度变化。环境超时设在评估窗口之后，窗口内应只出现真实终止；仍单独统计截断。它是单一指令下的仿真评估，不代表实机能力或所有速度下的行走能力。

通过上游 RecorderTerm 的 post-step 回调，在物理更新后、自动重置前获取速度和终止状态。指标包含摔倒的终止帧，忽略随后重置出的新 episode：

- `planar_rmse`：有效帧上 `sqrt(mean((vx-vx_cmd)^2 + (vy-vy_cmd)^2))`，单位 m/s。
- `yaw_rmse`：有效帧上转向角速度 RMSE，单位 rad/s。线速度在基座 yaw 对齐坐标系计算，角速度取世界 z 轴，与 H1 跟踪奖励的坐标定义一致。
- `survival_fraction`：完整窗口内没有终止或截断的环境比例；`fall_fraction` 和 `truncation_fraction` 分别记录终止和截断。
- `mean_observed_seconds`：首次试验持续到终止/截断或窗口末尾的平均时长，包含终止步，不是无限时域的预期存活时间。
- `per_env`：各环境观察步数、终止/截断标记及 RMSE。汇总 RMSE 按有效帧加权，必须结合存活率解读，避免短时摔倒产生的低误差被误认为更好。

输出包含输入模型副本 `input.pt`、模型及评估代码指纹、完整 `console.log`、每个种子的 `evaluation-seed-N.json` 和 `acceptance.json`。可选 `--max-planar-rmse`、`--max-yaw-rmse`、`--min-survival-fraction`；每个种子必须分别满足全部条件，任一失败时记录 `rejected` 并以退出码 2 结束，仍保留评估报告。不指定阈值时只报告指标，`acceptance.passed=null`，`complete` 仅表示评估执行完成。

首次实际评估：`runs/h1-evaluation-20260912`，输入为 `h1-managed-resume-20260912/model_8.pt`（模型位于该运行的日志子目录）。3 个种子共 96 次试验，前向指令 0.5 m/s、窗口 10 秒；存活率均为 0，平均首次试验时长为 1.526/1.505/1.611 秒，平面 RMSE 为 1.047/0.964/1.000 m/s。入口正确拒绝了最低存活率 80% 的验收条件。

新增 13 项评估测试覆盖终止帧、自动重置隔离、最后一步摔倒、截断、部分评估拒绝、非有限值、逐种子验收及拒绝状态落盘。本轮 `ef-viewer` 全量测试：276 通过、37 跳过；Ruff 和差异空白检查通过。

## 四项指令评估套件

```bash
python -m embodiedforge h1 evaluate \
  --run runs/h1-flat-512x300-20260912 --output runs/h1-basic \
  --suite basic --num-envs 32 --steps 500 --seeds 0 1 2 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3
```

`--suite basic` 与 `--velocity` 互斥，按固定顺序运行下列四项：

| 项目 | 前向/侧向速度（m/s） | 转向角速度（rad/s） |
| --- | --- | --- |
| `stand` | 0 / 0 | 0 |
| `forward` | 0.5 / 0 | 0 |
| `turn_left` | 0.5 / 0 | 0.5 |
| `turn_right` | 0.5 / 0 | -0.5 |

每个种子只加载一次模型和场景，各项开始前重新设置速度范围、重置随机种子并 reset 环境，指标累加器也重新初始化。四项保留同一组启动时质量/材质随机化；报告记录 `reset_protocol`。单指令评估保留原先初始 wrapper reset 的方式，不能把两种 reset 协议的结果当成严格相同条件直接比较。

每个种子的 JSON 包含 `cases` 列表；所有“种子 × 项目”分别检查阈值，不通过平均值掩盖某个项目的失败。缺项、重复项目、错误指令或模型迭代不一致会使评估标记为 `failed`。示例中的 80%、0.3 m/s 和 0.3 rad/s 是本轮选定的验收线，可按任务需求调整。

新增 `mean_velocity`（前向/侧向/转向）及逐环境平均速度，采样范围与 RMSE 一致，仅包括首次试验及其终止帧。平均值可能相互抵消，因此它用于诊断，不能替代 RMSE 或存活率。

基线输出：`runs/h1-suite-300-20260912`。300 轮模型在前进项的单环境平均前向速度中位数分别为 -0.037、-0.090、-0.066 m/s，未能稳定跟随 +0.5 m/s 指令。较高的存活率不足以证明已学会按指令行走。整套测试仅启动了三个仿真进程，完成四项 × 三种子 × 32 环境的 384 次试验。

本轮新增 7 项测试，覆盖速度统计对自动重置的隔离、均值抵消、逐项目验收、suite/velocity 互斥，以及报告缺项、错指令和重复项目拒绝。`ef-viewer` 全量测试：283 通过、37 跳过；Ruff 与 `git diff --check` 通过。

## 300 轮训练后的复测（2026-09-12）

从零训练 512 环境、300 轮（3,686,400 条环境转换），输出为 `runs/h1-flat-512x300-20260912`，最终 `model_299.pt`；300 轮指标齐全、9300 个标量样本及 checkpoint 张量均通过有限性检查。

复测使用相同版本、相同固定指令 0.5 m/s、相同种子 0/1/2、每种子 32 环境和 10 秒窗口。输出位于 `runs/h1-evaluation-512x300-20260912`，`comparison.json` 记录逐种子对比、输入报告 SHA256，并检查了任务、物理、时间步、指令、采样口径及依赖版本一致。

| 种子 | 短测模型存活率 | 300 轮模型存活率 | 平面 RMSE 前→后（m/s） | 平均首次试验时长前→后（秒） |
| --- | --- | --- | --- | --- |
| 0 | 0% | 62.5% | 1.047 → 0.631 | 1.526 → 6.956 |
| 1 | 0% | 81.25% | 0.964 → 0.606 | 1.505 → 8.447 |
| 2 | 0% | 75% | 1.000 → 0.614 | 1.611 → 8.056 |

合计 96 次试验，完整窗口存活数从 0 增至 70（72.9%），平均首次试验时长从 1.55 秒增至 7.82 秒。但种子 0 和 2 仍未达到各自 80% 的验收线，运行正确记录为 `rejected`；模型仍有摔倒和明显的速度误差，不能据此宣称稳定行走。此次没有覆盖侧向、倒退、转弯和更长时长，也没有录制图形回放。

## 累计 1000 轮训练后的四项复测（2026-09-12）

保持上游配方不变，从 300 轮模型再训练 700 轮，仍使用 512 环境。输出 `runs/h1-flat-512x1000-20260912`，最终模型位于 `logs/rsl_rl/h1_flat/2026-09-12_10-48-13_embodiedforge/model_998.pt`。文件编号沿用 RSL-RL 的恢复编号规则，累计实际更新数为 300 + 700 = 1000；本次 700 轮和 21700 个标量样本通过检查。

复测输出：`runs/h1-suite-1000-20260912`。四项 × 三种子 × 32 环境，共 384 次试验均完成 10 秒窗口，没有终止或截断。每个“项目 × 种子”均满足存活率 ≥80%、平面 RMSE ≤0.3 m/s、转向 RMSE ≤0.3 rad/s，`acceptance.json` 中 36 项检查全部通过。

下表按有效帧汇总 RMSE 和实际速度，按试验数汇总存活率。`comparison.json` 校验了两套评估的代码指纹、依赖版本、指令、时间步、种子、reset 协议和采样方式相同，并记录输入报告 SHA256。

| 项目 | 300→1000 轮存活率 | 平面 RMSE 前→后（m/s） | 转向 RMSE 前→后（rad/s） | 1000 轮实际平均前向/转向速度 |
| --- | --- | --- | --- | --- |
| 站立 | 84.4% → 100% | 0.280 → 0.090 | 0.345 → 0.080 | -0.013 m/s / -0.035 rad/s |
| 前进 | 71.9% → 100% | 0.616 → 0.149 | 0.350 → 0.119 | 0.400 m/s / 0.015 rad/s |
| 左转 | 69.8% → 100% | 0.607 → 0.150 | 0.532 → 0.182 | 0.401 m/s / 0.373 rad/s |
| 右转 | 71.9% → 100% | 0.615 → 0.149 | 0.680 → 0.206 | 0.404 m/s / -0.347 rad/s |

策略已从大部分试验未能跟随指令，改善到本轮四项验收全部通过。仍存在速度偏差：前进目标为 0.5 m/s，实际约 0.40 m/s；转向目标为 ±0.5 rad/s，实际约 +0.37/-0.35 rad/s。此处结果限于 10 秒平地窗口和启动/重置随机化。后续 60 秒结果见下节；外部推力、其他地形或实机尚未覆盖。

复现当前模型的验收：

```bash
python -m embodiedforge h1 evaluate \
  --run runs/h1-flat-512x1000-20260912 --output runs/h1-basic-1000-repeat \
  --suite basic --num-envs 32 --steps 500 --seeds 0 1 2 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3
```

## 60 秒评估和离线骨架回放（2026-09-12）

```bash
python -m embodiedforge h1 evaluate \
  --run runs/h1-flat-512x1000-20260912 --output runs/h1-long-motion \
  --suite basic --num-envs 32 --steps 3000 --seeds 0 1 2 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --record-motion --record-env 0 --timeout 360
```

实际输出在 `runs/h1-suite-60s-motion-20260912`，包括 `summary.json`、逐种子报告、12 份 NPZ 动作轨迹和 12 个 HTML 回放。三项移动任务共 288 次试验全部存活，但站立只有 53/96 次存活，三个种子的站立存活率分别为 40.625%、62.5%、62.5%。整套验收状态为 `rejected`（退出码 2），不能把 10 秒通过解释为长时稳定。

| 项目（每项 96 次） | 60 秒存活率 | 平面 RMSE（m/s） | 转向 RMSE（rad/s） |
| --- | --- | --- | --- |
| 站立 | 55.2% | 0.233 | 0.135 |
| 前进 | 100% | 0.104 | 0.098 |
| 左转 | 100% | 0.099 | 0.170 |
| 右转 | 100% | 0.101 | 0.187 |

`--record-motion` 在自动重置前记录所选环境（默认 0）的首次试验，遇终止即停止，不拼接下一次试验。NPZ 包含各刚体世界位置、`xyzw` 四元数、关节位置、时间和终止/截断标记；父子连接来自真实 Newton 模型。当前 H1 为 20 个刚体、19 条连接和 19 个关节。预计原始轨迹超过 256 MiB 会在采样前拒绝。

每份 NPZ 自动生成同名 HTML，可直接在浏览器打开，无需 GPU、仿真环境或网络。它显示刚体原点及其连接，支持播放、暂停、拖动进度、速度和视角切换，是诊断用骨架图；不包含机器人网格。也可从已有轨迹重新生成，输出文件必须不存在：

```bash
python -m embodiedforge h1 replay \
  --motion runs/h1-suite-60s-motion-20260912/motion-seed-0-forward-env-0.npz \
  --output runs/h1-forward-replay.html
```

站立失败样本 `motion-seed-0-stand-env-0.html` 在 23.24 秒终止，包含 1162 帧，骨盆高度由约 1.05 m 降至 0.145 m。前进样本 `motion-seed-0-forward-env-0.html` 含完整 3000 帧。浏览器检查覆盖播放、暂停、进度拖动和视角切换，无脚本异常；证据为同目录 `browser-check.json` 和 `replay-preview.png`。本轮测试 288 通过、37 跳过，Ruff 与差异空白检查通过。

## 已验证的配置

| 项目 | 本次配置 |
| --- | --- |
| 源码 | `/home/ubuntu/workspace/3rdparty/IsaacLab` |
| 源码版本 | `VERSION=3.0.0`，revision `2e44ddb2e19536579140496023b5ccb060bc4152` |
| 环境 | `/home/ubuntu/miniconda3/envs/isaaclab`，Python 3.12.0 |
| 依赖 | PyTorch 2.10.0+cu128、Warp 1.13.0、Newton 1.2.1、RSL-RL 5.0.1 |
| GPU | NVIDIA GeForce RTX 5090 D v2 |
| 任务 | `Isaac-Velocity-Flat-H1-v0` |
| 物理 | `physics=newton_mjwarp` |
| 可视化 | `--visualizer none` |
| 控制 | 19 维关节位置动作、69 维 policy 观测 |
| 时间步 | 物理 0.005 秒、环境 0.02 秒 |
| PPO | 每环境每轮 24 步，actor/critic 各三层 128，ELU |

本地 IsaacLab 已包含无需 Isaac Sim 的 Newton 工作流，H1 平地配置明确提供 `newton_mjwarp` 预设。本次沿该路径运行成功；无需为此先安装 Isaac Sim。保持 IsaacLab、`ef`、`ef-viewer` 和 Microduck 的 Python 环境独立，这些环境使用的 Python、Newton、Warp 和 PyTorch 版本不同。

上游也注册了 `Isaac-Velocity-Rough-H1-v0` 及 Flat/Rough 的 Play 任务。本节记录 2026-09-12 的 Flat 训练与离线骨架回放验证，未验证 Rough 任务。后续已接入 H1 的统一 Web 网格记录回放，包含 OVRTX 后端，见 [机器人回放](robot-web-replay.md)；这不代表 H1 在线策略控制已接入。

## 重现短测

在 EmbodiedForge 根目录执行。输出目录必须尚不存在；子 shell 中任何一步失败都会停止，避免误在其他目录写入训练输出。

```bash
(
  set -eu
  mkdir runs/h1-flat-smoke
  cd runs/h1-flat-smoke
  timeout --signal=INT --kill-after=15s 180s env \
    CONDA_PREFIX=/home/ubuntu/miniconda3/envs/isaaclab \
    VIRTUAL_ENV= PYTHONUNBUFFERED=1 \
    /home/ubuntu/workspace/3rdparty/IsaacLab/isaaclab.sh train \
    --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-H1-v0 \
    physics=newton_mjwarp --visualizer none \
    --num_envs 64 --max_iterations 5 --seed 0 \
    --logger tensorboard --run_name embodiedforge_probe
)
```

`CONDA_PREFIX` 选择已经配置的 IsaacLab 环境，清空 `VIRTUAL_ENV` 避免 wrapper 优先使用其他虚拟环境。H1 USD 资产来自上游远程资产库，首次加载需要网络，下载或内核编译较慢时可能超过短测的 180 秒限制；超时不算成功。

训练通过上游统一 `train` 命令分发至 `train_rsl_rl.py`，没有调用已弃用的旧 `rsl_rl/train.py`。日志相对当前工作目录生成，因此保存在 EmbodiedForge 的 `runs/` 内。后续正式训练需要另建输出目录、去掉短测超时，并调整环境数和迭代数；上游 Flat 配方默认 1000 轮，这不是收敛保证。

## 本次结果与限制

本次输出目录：`runs/h1-newton-probe-20260912/`。

- 模型：`logs/rsl_rl/h1_flat/2026-09-12_09-49-41_embodiedforge_probe/model_4.pt`，保存迭代索引为 4，即完成第 5 轮。
- 同目录保存了 TensorBoard events、环境/算法 YAML 和源码差异记录。
- 离线检查结果：`verification.json`；checkpoint 中 68 个张量均为有限值，151 个 TensorBoard 标量样本均为有限值。
- 第 3～5 轮日志吞吐约 8300～8700 transitions/s，仅为此次 64 环境短测观测，不代表正式训练性能。
- 末轮平均 reward 为 -5.61，仍有明显摔倒，不能把启动成功或有限数值解释为学会行走。完整训练后还需要固定速度命令、多随机种子评估与回放验收。

最初上游短测输出了 Hub 未找到、地面颜色回退、Newton 颜色兼容层弃用和 RSL-RL `obs_groups` 隐式推断提示；它们没有阻止训练。新增入口通过配置显式指定 actor/critic 均使用 69 维 `policy` 观测，真实运行中已不再出现观测组推断警告。其余上游提示保留在 `console.log`，没有屏蔽。

## 统一入口的验证结果（2026-09-12）

| 运行目录（均在 `runs/` 下） | 结果 |
| --- | --- |
| `h1-managed-smoke-v3-20260912` | 64 环境，从零训练 5 轮，`model_4.pt`，状态 `complete` |
| `h1-managed-resume-20260912` | 恢复上述模型，再训练 5 轮，`model_8.pt`，状态 `complete` |

两次运行均通过最终模型、每轮 PPO 指标、68 个 checkpoint 张量和 151 个标量样本的检查。续训日志确认加载了复制后的输入 checkpoint，并执行编号 4～8 的 5 轮更新。末轮平均 reward 约 -5.56，仍明显摔倒；此结果用于验证续训机制，不作为行走质量验收。

针对性测试在具有 PyTorch/TensorBoard 的环境中通过 21 项，覆盖环境隔离、超时、SIGTERM 转发、失败记录、输出目录保护、输入模型完整性以及部分/非有限训练产物拒绝。`ef-viewer` 中全量测试通过 263 项、跳过 37 项；其中 6 个新增模型/指标检查测试因该环境无 PyTorch 而跳过，已在前述针对性运行中验证。Ruff 和 `git diff --check` 通过。

## 接入 EmbodiedForge 的后续工作

第一阶段已提供训练/续训入口、版本和配置记录、进程退出处理、checkpoint 索引、TensorBoard 指标检查以及有限时长的多种子固定指令评估。已增加离线骨架回放；后续需要改善长时站立，并补充机器人网格录像。H1 的环境、奖励、随机化和 PPO 配方继续由上游维护。

第二阶段再移植到 EmbodiedForge 的任务/物理接口，需要补齐机器人资产、批量 articulation 状态、关节位置控制、足部接触、69 维观测及奖励/终止逻辑。现有点质量 Newton 适配器不能直接承担 H1 训练。上游 `scripts/sim2sim_transfer/config/newton_to_physx_h1.yaml` 还显示不同物理引擎的关节顺序不同，策略迁移必须按关节名称映射并核对动作尺度、默认姿态和控制频率。

源码依据（路径均相对上述 IsaacLab 仓库）：

- `docs/source/setup/installation/kitless_installation.rst`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/__init__.py`
- 同目录 `flat_env_cfg.py`、`rough_env_cfg.py`、`agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_assets/isaaclab_assets/robots/unitree.py`
- `scripts/reinforcement_learning/rsl_rl/train_rsl_rl.py`
