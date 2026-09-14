# Wuji UniLab 与 mjbatch 任务接入

按步骤执行安装、训练、续训、评估和运行，见 [训练与部署命令](training-deployment.md)（[English](training-deployment.en.md)）。

2026-09-12 已接入两个固定版本的本地仓库，并完成真实训练、评估和 CPU 求解验证。入口为 `python -m embodiedforge recipes`；训练 SDK 在各自独立环境执行，EmbodiedForge 主环境仍只需要原来的核心依赖。

| 任务名 | 内容 | 方法 | 执行位置 |
| --- | --- | --- | --- |
| `wuji-reorient` | 五指 Wuji Hand 持物并重定向方块 | 非对称 actor/critic PPO、观测历史、课程和域随机化 | UniLab + MuJoCo-Warp GPU |
| `wuji-reorient-light` | 同一任务，使用上游 Light 随机化课程 | PPO | UniLab + MuJoCo-Warp GPU |
| `go1-joystick` | Go1 四足速度跟踪 | PPO、镜像对称策略、观测归一化、超时 bootstrap | mjbatch CPU 仿真 + CPU 学习 |
| `cartpole-mpc` | 倒立摆起摆与稳定 | 预测采样 MPC | mjbatch CPU |
| `arm-throw-codesign` | 联合优化机械臂长度、传动比、释放时刻和控制 | CEM；与固定机械结构、只优化控制的基线比较 | mjbatch CPU |

另新增原生 `physics=mjbatch` 后端，现有 `reach` / `hold` 可直接使用 C++ 线程池。机器人任务目前通过独立流程接入，尚不是 `VectorEnv` 内的机器人任务。

2026-09-13 已将 Go1 的模型构建、reset、观测、奖励、镜像策略、归一化、rollout、GAE 和 PPO 更新移植到 `embodiedforge.locomotion`，训练入口不再导入上游 `go1_joystick.py` 或 `window.py`。Wuji、MPC 和机械臂仍是固定版本源码快照加适配层。原生 `mjbatch` 后端实现了 EmbodiedForge 的物理接口。

Go1 的任务和训练代码由本仓库维护，仍依赖 MuJoCo、mjbatch、Torch 和 Menagerie 机器人资产。受管 `recipes` 入口继续使用原来的固定版本 SDK 环境及源码完整性检查；这不等于整个 SDK 已移植，也不等于 Go1 已接入现有平面任务的 `VectorEnv`。

## Go1 移植范围与直接调用

- [go1.py](../src/embodiedforge/locomotion/go1.py)：物理模型、50 维观测、12 维动作、指令采样、十项奖励和 reset。
- [go1_ppo.py](../src/embodiedforge/locomotion/go1_ppo.py)：镜像策略、运行归一化、超时 bootstrap、GAE 和 PPO 更新，保留旧模型参数名称。
- 场景 XML 和上游 Apache-2.0 许可证随 wheel 分发；机器人模型继续由固定版本 Menagerie 提供。

在准备好的 mjbatch SDK 环境中，可以直接调用迁入实现，无需导入上游示例：

```python
import numpy as np
from embodiedforge.locomotion.go1 import Go1

env = Go1(32, seed=0, num_threads=4, episode_steps=500)
observation = env.obs()  # (32, 50), float32，返回独立数组
reward, done, fell, terms = env.step(np.zeros((32, 12), dtype=np.float32))
env.reset(np.flatnonzero(done))
```

`reset()` 重置全部环境；`reset([3, 1])` 接受乱序编号并按调用者顺序分配随机样本；`reset([])` 不改变状态或随机数流。重复、越界和非整数编号会在调用物理后端前拒绝，未选中环境保持原状态。动作必须为形状正确、有限且可表示为 float32 的实数数组。

终止处理沿用上游约定：`step` 不自动 reset，也不冻结已结束环境；调用者负责处理 `done`。PPO rollout 会及时 reset，评估则独立统计首次 episode。环境时限和线程数现在由实例参数控制，rollout 长度由函数参数控制，避免跨实例修改全局配置。

### 指令与奖励时序版本

`Go1` 默认使用 `semantics="transition-v2"`：先按动作输入观测中的指令计算当步奖励，再采样下一步指令。上游在指令到期时先换指令再计算奖励，导致动作被按尚未看到的新目标评价；`semantics="upstream-v1"` 保留这一旧行为，仅用于兼容和数值对照。两版保留相同的物理、奖励公式与观测维度。

受管训练/评估会将版本写入请求、结果及 checkpoint：新训练默认 v2；续训和评估默认继承输入训练运行，历史未标版本的模型视为 v1。可显式使用 `--go1-semantics transition-v2` 将旧模型迁入修复后的训练规则。输入 checkpoint 的版本必须与输入运行记录相符，输出版本必须与本次请求相符。固定指令评估不会触发指令切换，不能单独验证这个时序问题；专项测试会强制切换目标并核对旧目标奖励与下一步观测。

## 安装与版本

```bash
conda activate ef
python -m embodiedforge recipes list
python -m embodiedforge recipes setup --source wuji_unilab --python 3.12
python -m embodiedforge recipes setup --source mjbatch --python 3.12
```

默认读取 `/home/ubuntu/workspace/3rdparty/<source>`；`--repo` 可以更换源目录，`--cache` 可以更换缓存根目录。要求原仓库为固定 revision 且工作区干净。安装先复制 Git 跟踪文件到 `.cache/external/<source>`，再通过原始 `uv.lock` 安装，不修改上游仓库，不向 `ef`、H1 或 Microduck 环境安装这些 SDK。首次安装与首次 Go1 资产下载需要网络。默认安装超时 1200 秒，可通过 `--timeout` 调整。

| 来源 | 固定 revision | 本机独立环境 |
| --- | --- | --- |
| Wuji UniLab | `91ccfa0ec8c129b300865bd36c59dc9eed56a744` | `.cache/external/wuji_unilab/.venv`，Python 3.12，Torch 2.8.0+cu128，UniLab/UniSim/unilab-rl 1.2.0，Warp 1.16.0，RSL-RL 5.0.1 |
| mjbatch | `b84c0c20aedbdf048122cbc47f554e9b93cc4754` | `.cache/external/mjbatch/.venv`，Python 3.12，Torch 2.9.0+cu128，MuJoCo 3.11.0，Menagerie 2026.9.0 |

当前锁定环境面向 Linux x86_64。Go1 虽使用带 CUDA 的上游 Torch wheel，但本入口将物理和学习都放在 CPU，隐藏 CUDA 设备；Wuji 必须有可用 NVIDIA CUDA，不做 CPU 回退。`--threads` 默认 4，控制 CPU 线程池及 Torch 线程数。

## 训练和续训

```bash
# Wuji 入口短测，不能据此认定学会重定向
python -m embodiedforge recipes train --task wuji-reorient \
  --num-envs 32 --horizon 40 --updates 5 --timeout 300 --output runs/wuji-smoke

# 本机已验证的 Go1 配置
python -m embodiedforge recipes train --task go1-joystick \
  --num-envs 512 --updates 600 --threads 4 --timeout 1200 --output runs/go1-train

# 新目录中继续训练，恢复模型和优化器
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/go1-train --num-envs 512 --updates 100 \
  --output runs/go1-continued

python -m embodiedforge recipes status --run runs/go1-train
```

Wuji 的 `--horizon` 建议显式使用上游 40；Go1 上游为 24。入口默认 32 环境、3 轮、24 步，仅用于短测。样本数 `num-envs × horizon` 必须能分成完整小批量：Wuji 为 32 个、Go1 为 4 个，每批至少两个样本。输出目录必须不存在。

Go1 迁入实现保留上游环境、奖励、镜像网络、PPO update、GAE 和 reset 的计算；管理层负责有界训练、CPU 线程数、逐轮 JSONL 指标和模型落盘。优化器使用普通 Adam，未使用上游的 fused Adam；没有加载仓库附带的预训练 `go1_policy.pt`。观测 50 维，动作 12 维，物理时间步 0.004 秒，控制间隔 0.02 秒。新运行额外保存 `recipe-config.json` 和带 Menagerie tree ID/资产校验信息的 `assets.json`。

Wuji 直接调用上游 `wuji-train`，20 维关节动作、207 维策略观测、413 维特权价值观测，物理间隔 0.01 秒、控制间隔 0.05 秒。模型检查包含 actor/critic 维度、优化器和课程状态，以及每轮 TensorBoard loss 历史。完整解析配置保存在上游 `run_config.json`。

两者均可使用 `--resume-run`，只接受同任务同版本的 `complete` 训练运行，核对并复制输入 checkpoint。Wuji 同时通过上游恢复课程状态，其 RSL-RL 编号从保存索引重用：300 轮模型 `model_299.pt` 再更新 3 轮得到 `model_301.pt`，实际累计 303 轮。Go1 从下一索引继续，600 轮后再更新 3 轮索引为 602。`completed_updates` 表示此次更新数，`cumulative_updates` 表示累计数。仿真状态和 RNG 不恢复，所以不保证逐步等价续训。

每次运行记录源码文件 SHA256、入口和适配器指纹、依赖版本、完整 `console.log`、请求参数、结果和 checkpoint SHA256。子进程使用 `-I`，防止本地 `logging.py` 等覆盖标准库。Ctrl+C/SIGTERM 转发到子进程组；超时与失败分别记录 `timed_out` / `failed`。`complete` 表示执行和产物检查完成，策略效果由独立评估判断。`status` 读取记录，不是进程存活探针。

新建受管任务在 `implementation/embodiedforge/` 保存并执行本地 Python 模块、Go1 场景和许可证快照，记录各文件哈希并在结束后复核；`runtime.json` 记录实际 adapter 导入路径。后台任务从自己的快照加载实现，工作区后续修改不会混入训练或评估。输出目录需位于源码包之外。旧运行仍可作为输入。

若要显式迁移旧 Go1 模型的奖励时序：

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/go1-train --go1-semantics transition-v2 \
  --num-envs 512 --horizon 24 --updates 100 --output runs/go1-v2-continued
```

### Go1 跟踪奖励配置

`--go1-reward-profile tracking-v1` 只将平面速度跟踪奖励 `track` 从 1.0 提到 2.0，保留全部其他权重、物理、指令分布、观测和 PPO 参数。`original` 为原始权重，也是新训练默认值。每个环境持有独立的奖励权重副本。

`tracking-turn-v1` 在 `tracking-v1` 基础上将 `turn` 从 0.5 提到 1.0，用于降低行走时的偏航摆动。它是独立的实验配置，不改变旧配置或默认值；是否采用某份模型仍需独立评估，增加权重不保证所有训练种子均改善。

`tracking-balanced-v1` 使用 `track=3.0, turn=1.0`，其他权重保持原值。它用于对照单独提高转向权重后出现的平面跟踪退化，同样通过独立配置记录和继承。

`tracking-moderate-v1` 使用 `track=2.5, turn=1.0`，用于小幅提高平移跟踪权重的对照。它是独立的实验配置，不改变已有配置名称的含义，也不自动替换默认权重。

`tracking-strong-v1` 使用 `track=3.0, turn=1.5`，保留 `tracking-turn-v1` 的 2:1 比例，同时提高两项跟踪相对其他奖励的权重。这也是需要独立评估的实验配置。

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/go1-v2-512x600-20260913 \
  --go1-reward-profile tracking-v1 --num-envs 512 --horizon 24 --updates 300 \
  --output runs/go1-v2-tracking
```

配置会写入请求、完整权重记录、checkpoint 和结果。续训、评估默认继承输入模型配置；旧模型未记录配置时视为 `original`。输入模型元数据必须与其运行记录相符。奖励配置与任务时序版本分别记录；提高奖励权重会改变训练奖励的数值尺度，不能直接用两组总 reward 高低判断效果。

2026-09-13 的等量续训对照从同一份 v2 / 600 轮模型出发，两组均为 512 环境 × 300 次新增更新 × 24 步，共各新增 3,686,400 次转换，累计 900 轮。种子、优化器恢复、学习率和其他配置相同，唯一奖励差异为 `track: 1 → 2`。

逐步诊断在评估种子 0、前进阶段第 12–20 秒测量：原 v2 / 600 轮模型的平均前向速度偏差为 −0.0663 m/s，原奖励继续训练后的偏差为 −0.0649 m/s，提高跟踪权重后为 −0.0095 m/s。速度波动的 RMS 约为 0.02 m/s，初始主要问题是偏慢；原奖励续训也减少了波动，因此同样改善了前进的连续达标比例，不能把这部分改善全部归因于奖励调整。

开发集种子 0/1/2 中，原奖励对照组仍未通过右转的稳定跟踪比例门槛，`tracking-v1` 通过全部 84 项检查。选定 `go1-v2-tracking-900-20260913` 后固定模型，再评估未参与本轮选择的种子 3/4/5：

| 同预算候选 | 动态试验数 | 动态验收 | 右转连续达标比例 | 全阶段最差单种子达标比例 |
| --- | --- | --- | --- | --- |
| 原奖励续训至 900 轮 | 96 × 60 秒 | 4 项失败，`rejected` | 74.0% | 65.6% |
| tracking-v1 续训至 900 轮 | 96 × 60 秒 | 84/84 项通过 | 100% | 100% |

新模型另外通过同一留出种子的 384 次固定指令试验、36 项验收，全部存活。该轮通过模型为 `runs/go1-v2-tracking-900-20260913/model.pt`；原 v2 / 600 轮的失败结论仍保留。结果只覆盖这次训练种子和指定的速度/环境范围，不能推断所有训练种子均有相同改善。

同预算、同留出种子的固定指令对照存在取舍：前进平面 RMSE 从 0.0738 降至 0.0401 m/s，但前进转向 RMSE 从 0.0273 增至 0.0673 rad/s；两者均低于本次 0.3 的验收上限。动态稳定跟踪改善不代表所有误差分量都降低。

`runs/go1-tracking-audit-20260913/` 保存输入哈希和配方差异检查、逐步速度/动作数据、开发集选择记录、留出集结果及 `forward-tracking.png`。`go1-tracking-inherit-smoke-20260913` 验证新模型后续续训自动继承 `tracking-v1`，累计 901 轮；该短测模型不替代已完成评估的 900 轮候选。

后续角速度优化得到 `runs/go1-yaw-balanced-s1-1200-20260913/model.pt`：相对上述 900 轮基准，在预留种子 6/7/8 的三条基本移动指令中，角速度 RMSE 降低 60.4%，平面 RMSE 增加 2.3%；通过 105 项动态检查及十项扩展固定指令验收。扩展方向仍有平面误差取舍，见 [受控训练与复现报告](go1-yaw-study.md)。

本轮另外完成九组从零初始化训练，选出 `runs/go1-yaw-fresh-turn-s11-1200-20260913/model.pt`。它在另一组预留种子 20/21/22 中通过动态与扩展验收，三条基本固定移动指令的角速度 RMSE 降低 64.7%、平面 RMSE 降低 5.3%；动态平面误差增加约 10.8%。两个模型与训练路径的差异均记录在上述报告中。

本轮横移与急停训练完成 33 组对照，得到 `runs/go1-stop-stops-s0-1650-20260913/model.pt`。它相对 1200 轮起始模型在独立评估中降低横移平面误差 33.4%、原动态平面误差 7.9%，基本移动角速度误差增加约 4.0%；三个常规套件、五分钟横移切换和两秒阶段快速切换均通过。中间候选的失败、额外停车训练、选择规则及复现见 [完整报告](go1-response-study.md)。

后续 `maneuver-switching` 检查发现复合斜向后退跟踪不足。六组续训中，部分模型改善复合动作与横移，但未同时保留原有转弯精度；当前不提升为新的基准模型，详见 [复合指令与能力保留对照](go1-maneuver-study.md)。

### Go1 指令采样配置

`--go1-command-profile lateral-v1` 将重采样时横移轴的启用概率从 0.25 提至 0.75。前进和转向轴仍为 0.9 / 0.5，速度范围仍为 ±1.5 / ±0.8 m/s、±1.2 rad/s，重采样间隔仍服从均值 5 秒的指数分布，各轴有 50% 概率保留旧值。它用于增加横移训练覆盖，属于实验配置。

`lateral-stop-v2` 将 `lateral-stop-v1` 的额外强制完整停车概率从 10% 提高到 20%，其他分布参数不变，用于检验停车训练覆盖。完整停车会覆盖保留的旧轴值。新旧配置消耗相同数量的随机数；固定指令评估仍使用相同物理初态和响应。该配置需要独立行为验收，不自动替换 `lateral-stop-v1`。

快速切换训练另提供三个实验配置，均保留横移启用概率 0.75 和额外完整停车概率 10%：

| 配置 | 重采样间隔均值 | 每轴保留旧值概率 | 速度范围（前向 / 横向 / 偏航） |
| --- | --- | --- | --- |
| `lateral-fast-v1` | 2 秒 | 50% | ±1.5 / ±0.8 m/s / ±1.2 rad/s |
| `lateral-fast-full-v1` | 2 秒 | 0% | ±1.5 / ±0.8 m/s / ±1.2 rad/s |
| `lateral-fast-moderate-v1` | 2 秒 | 0% | ±0.8 / ±0.5 m/s / ±1.0 rad/s |

间隔服从指数分布，2 秒是均值，不是固定周期；未保留的轴也可能再次采到零。`COMMAND_KEEP` 在配置中记录实际保留概率，重置时始终全部重采样。三个配置分别用于比较重采样频率、轴保留和速度范围；它们不会自动启用，也不保证提升快速响应。强制评估指令的物理初态、随机数消费量和动作响应已作一致性检查。

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/go1-yaw-fresh-turn-s11-1200-20260913 \
  --go1-command-profile lateral-v1 --num-envs 512 --horizon 24 --updates 300 \
  --output runs/go1-lateral-continued
```

新训练默认 `original`，续训和评估自动继承输入模型配置，旧模型缺失该字段时按 `original` 处理。配置名称和实际采样参数分别写入 checkpoint / 结果和 `recipe-config.json`。评估仍强制使用套件指令；训练分布不会改变固定指令评估的物理初态和响应。不能仅凭训练 reward 判断横移质量。

`lateral-stop-v1` 在 `lateral-v1` 基础上，每次重采样额外以 10% 概率把完整指令设为零，覆盖被保留的轴，使训练包含从运动转入站立的片段；初始重置时也应用该概率。实际值写入 `COMMAND_STOP_PROBABILITY`。原有配置不额外消费随机数。

### Go1 保守续训学习率

`--go1-learning-rate 0.0001` 将这次 Go1 训练设为固定学习率，恢复模型及 Adam 状态后，在第一次更新前生效。后续续训默认继承该值，也可以显式传入另一正数；评估继承元数据并检查 checkpoint 一致性。每轮 `metrics.jsonl` 记录实际学习率，配置记录有效 `LR` / `LR_END`。

Go1 受管训练还逐轮记录 PPO 诊断：`ppo_kl` 是同一 rollout 观测上更新前策略到最终策略的精确高斯 KL 均值，`ppo_max_kl` 是最大值；`ppo_kl_before_normalization` 在更新归一化统计之前测量。`ppo_normalizer_kl` 单独比较统计更新前后的策略，两种 KL 不可相加分解。`ppo_clip_fraction` 是最终策略相对 rollout 采样概率超出 PPO 裁剪区间的比例，`ppo_mean_gradient_norm` 是各 minibatch 裁剪前梯度范数的平均值，`ppo_action_std` 是动作标准差均值。这些诊断不改变优化步骤，也不自动触发早停；它们帮助区分大幅策略更新与归一化漂移，不能替代行为评估。

受管训练使用 `rollout(..., record_policy=True)` 缓存实际采样时的动作均值和对数标准差，诊断复用这份分布，减少一次全批策略前向计算。512 环境 × 24 步的额外 float32 缓存为 589,872 字节，约 0.563 MiB。直接调用 `rollout()` 默认仍返回原有字段；没有缓存的批次继续支持现场计算。部分缺失、非有限值或形状/类型错误的缓存会在优化器更新前被拒绝。

`ActorCritic.action_mean(obs)` 提供仅策略推理路径，评估和诊断不用再执行价值网络；训练 `forward()` 仍返回动作均值和价值。两项优化分别经过 150 轮完整续训复跑，checkpoint 与原实现逐字节一致。诊断中的微小浮点差异可能来自逐步采样与全批前向计算的批大小不同，不改变 PPO 损失或更新。

可在已安装的 Go1 SDK 中复测学习器、缓存和推理耗时：

```bash
PYTHONPATH=src MENAGERIE_CACHE_DIR=.cache/external/mjbatch/assets-cache \
  .cache/external/mjbatch/.venv/bin/python benchmarks/go1_policy.py \
  --checkpoint runs/go1-preserve-balanced-s0-1800-20260914/model.pt \
  --num-envs 512 --horizon 24 --pairs 8 --output runs/go1-policy-benchmark.json
```

可用 `--reference-ppo <旧运行>/implementation/embodiedforge/locomotion/go1_ppo.py` 同时比较旧实现。基准交替执行各模式，保存每次计时、源码与模型哈希，并核对推理动作完全相同；重复使用同一真实 rollout，不包含仿真和日志 I/O，因此不能将其耗时变化直接当作端到端训练加速比例。

训练期间可用 `python -m embodiedforge recipes status --run <运行目录>` 查看 `progress`：已记录的更新数、请求更新数、完成比例、累计更新数和最新 reward / PPO 诊断。该读取不加载 Torch 或 checkpoint，最多读取 `metrics.jsonl` 尾部 64 KiB，忽略尚未写完的末行；完整行损坏或计数不符合训练预算时会报错。`progress.observed_updates` 表示已完成并记录的训练更新，checkpoint 按保存周期落盘，可能略滞后；顶层 `completed_updates` 仍以受管运行完成后的结果为准。刚启动尚无完整日志行时不输出 `progress`。

未设置覆盖值的新训练及旧模型仍使用原始累计轮次退火：从 0.001 在前 400 轮线性降到 0.0005，之后保持 0.0005。降低学习率是控制参数变化的实验选项，不保证同时改善所有速度方向。

## 评估与回放

```bash
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/go1-train --output runs/go1-eval \
  --suite basic --seeds 0 1 2 --num-envs 32 --steps 3000 --record-motion \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3

python -m embodiedforge recipes evaluate --task wuji-reorient \
  --run runs/wuji-smoke --output runs/wuji-eval \
  --seed 0 --num-trials 50 --steps 280 --min-success-rate 0.5 --max-drop-rate 0.2
```

Go1 `basic` 固定顺序测试站立 `[0,0,0]`、前进 `[0.5,0,0]`、左转 `[0.5,0,0.5]`、右转 `[0.5,0,-0.5]`。也可以用 `--velocity vx vy yaw` 指定单条命令，与 `--suite` 互斥。线速度单位 m/s，角速度单位 rad/s，均使用躯干局部传感器坐标；它与 H1 的 yaw 对齐线速度/世界 z 轴角速度口径不同，不能直接混作同一指标。

`--suite extended` 保留上述四项，再加入快速前进 `[1,0,0]`、后退 `[-0.5,0,0]`、左右横移 `[0,±0.3,0]`、左右原地旋转 `[0,0,±0.8]`，共十项固定指令。每项从新环境开始，用于检查更多速度方向；它与不中断环境的 `switching` 分别回答不同问题。扩展评估可使用更严格的独立门槛，例如：

```bash
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/go1-v2-tracking-900-20260913 --suite extended \
  --seeds 0 1 2 --num-envs 32 --steps 3000 \
  --min-survival-fraction 1 --max-planar-rmse 0.15 --max-yaw-rmse 0.2 \
  --output runs/go1-extended
```

`--suite lateral-switching` 在同一个环境中连续执行站立、左横移 `[0,0.3,0]`、右横移 `[0,-0.3,0]`、后退 `[-0.5,0,0]`、斜向前进 `[0.5,0.3,0]`、停车六个等长阶段。默认 3000 步，共 60 秒。它复用 `switching` 的首回合、跌倒后不重入、连续 0.5 秒达标和全时段达标比例口径，接受 `--min-settled-fraction` / `--min-tracking-fraction`，并可输出带指令阶段标签的回放。横移固定指令通过不代表横移反向切换也通过。

`--suite maneuver-switching` 补充复合指令：站立、斜向后退 `[-0.5,-0.3,0]`、左横移同时左转 `[0,0.3,0.5]`、右横移同时右转 `[0,-0.3,-0.5]`、原地左转 `[0,0,0.8]`、停车。时长、指标、验收参数和回放方式与其他切换套件一致。指令依次表示机体坐标系的前向速度、侧向速度（m/s）和偏航角速度（rad/s）；组合动作能暴露单轴测试未覆盖的跟踪误差。

`--suite maneuver-switching-mirrored` 将上述复合套件的侧向速度与偏航角速度同时取反，保持前向速度和阶段时长不变：斜向后退改为 `[-0.5,0.3,0]`，横移转弯方向互换，原地旋转改为右转。它用于检查相反方向和切换顺序，使用相同的逐阶段首回合指标与验收参数。

每项每种子重新构建环境，以确定性策略运行；保留上游 reset 的姿态、速度、相位和摩擦随机化。只统计首次试验，包含终止帧，不自动重置。终止判据沿用上游 `up_z < 0`；时间上限设在评估窗口之后。采集前调用 `forward`，消除 MuJoCo 派生传感器落后一物理子步的影响。所有“项目 × 种子”分别验收，任一不通过均为 `rejected`（退出码 2），不能用平均值掩盖失败。不设阈值时 `acceptance.passed=null`。

### 动态指令与稳定响应验收

```bash
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/go1-port-512x600-20260913 --suite switching --seeds 0 1 2 \
  --num-envs 32 --steps 3000 --record-motion \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --min-settled-fraction 0.8 --output runs/go1-switching
```

`switching` 连续执行站立、前进、停止、左转、右转、转向后停止六段，使用与 `basic` 相同的速度值，中途不重置。默认 3000 步、60 秒，每段 10 秒；自定义步数需至少 300 且能被 6 整除。命令在动作推理前设置，当步测量与该命令配对；不是按每段重新构建环境的固定指令测试。

每段记录首个 episode 的 RMSE、存活率和稳定跟踪比例。稳定定义为平面速度误差 ≤0.1 m/s、转向误差 ≤0.15 rad/s，连续保持 25 个控制步（0.5 秒）；时间从该段开始到完成保持计时。比例的分母始终是初始环境数，未到达后续阶段的环境不会从分母中删掉；平均时间仅统计达标者，未达标时间为 null。若一段完全没有有效帧，RMSE 为 null，设置相应验收条件时直接不通过。

`--min-settled-fraction` 仅用于 switching。每个种子的整体结果及六个阶段分别验收，整体稳定跟踪比例取最差阶段，避免平均值掩盖失败。每个模型 3 种子 × 32 环境为 96 次连续试验；四条阈值对应 84 项检查。

还可设置 `--min-tracking-fraction 0.9`，要求每段至少 90% 的计划环境时间同时满足上述两个速度误差容差。这与“曾经连续保持半秒”的 `settled_fraction` 不同：`tracking_fraction` 统计整段，分母是初始环境数 × 计划阶段步数，包含指令切换后的响应时间和跌倒后缺失的时间，终止帧不计为达标。每环境达标步数与版本化的 `tracking_protocol` 一起落盘，整体仍取最差阶段；三种子加上此项共 105 项检查。历史没有此字段的报告仍可读取，但不能据此认定通过该新门槛。

受管 Go1 评估同时核对逐环境有效帧数、终止标记、存活率及合并 RMSE。合并 RMSE 按有效帧数加权后开平方，不直接平均各环境的 RMSE；这样短暂存活的环境不会被赋予与完整试验相同的时间权重。

2026-09-13 同协议对照：两版均未跌倒，原有 RMSE/存活门槛均通过；加入 ≥80% 稳定跟踪比例后，v1 通过 84/84 项，v2 有 12 项不通过，运行标记为 `rejected`，退出码 2。

| 阶段 | v1 稳定跟踪比例 | v2 稳定跟踪比例 | v1 达标者平均响应（秒） | v2 达标者平均响应（秒） |
| --- | --- | --- | --- | --- |
| 站立 | 100% | 100% | 0.754 | 0.644 |
| 前进 | 100% | 15.6% | 1.074 | 1.812 |
| 停止 | 100% | 100% | 0.739 | 0.932 |
| 左转 | 89.6% | 70.8% | 1.240 | 1.691 |
| 右转 | 92.7% | 37.5% | 0.834 | 1.214 |
| 转向后停止 | 100% | 100% | 0.751 | 0.935 |

这区分了代码时序修复与策略质量：v2 的奖励时序测试通过，但这份 600 轮模型的稳定速度跟踪仍需改善。只覆盖一个训练种子和这组指令/容差，不能据此推断所有 v2 训练都会退化。详细报告为 `runs/go1-switching-v1-strict-20260913/` 与 `runs/go1-switching-v2-strict-20260913/`；汇总和浏览器检查在 `runs/go1-switching-audit-20260913/`。

动态回放带有完整命令时间表，拖动进度时显示当前阶段和实际用于该帧动作的指令；首帧是第一个控制步完成后的状态。六个阶段边界已在真实 Chrome 中检查，无 JavaScript 异常。

`--record-motion` 为每项每种子的环境 0 输出 NPZ 和独立 HTML，超过预计 256 MiB 轨迹预算会拒绝。Go1 回放包含 13 个刚体原点和 4 个真实足端 site，元数据区分两者；四元数统一为 `xyzw`。浏览器可离线播放、暂停、拖动进度、调速和切换视角，无需 GPU 或仿真 SDK。它是骨架诊断图，不含机器人网格。

Wuji 使用上游顺序单环境试验：目标 SO(3) 姿态离当前姿态至少 90°，误差小于 0.2 rad 连续保持 5 步为成功；成功后沿用场景设置新目标，掉落或超时后重置。掉落定义为相对 reset 高度降低 0.15 m。`--steps 280` 对应每试验 14 秒，`--num-trials` 控制试验数量。适配器将 `--seed` 同时用于目标采样和环境 reset 配置；上游原入口只将该参数用于目标采样。Wuji 不接受速度命令、Go1 套件或并行环境数，可用 `--policy zero` 运行全零动作对照；诊断记录每次试验的实际动作、方块运动及目标误差变化。`--record-video --num-trials 1 --steps 280` 导出 640×368、20 fps 的 MP4（含目标姿态叠加），需要 PATH 中的 ffprobe，限制原始帧预算为 256 MiB。录像采用单进程离屏渲染，检查解码、帧数、尺寸和 SHA256；渲染失败会使运行失败。尚无 ONNX 导出。

## CPU 控制与联合优化任务

```bash
python -m embodiedforge recipes solve --task cartpole-mpc \
  --num-envs 1024 --horizon 25 --steps 150 --output runs/cartpole-mpc

python -m embodiedforge recipes solve --task arm-throw-codesign \
  --num-envs 512 --generations 30 --output runs/arm-codesign
```

`num-envs` 在 MPC 中表示候选轨迹数，在 CEM 中表示种群数。MPC 每个控制周期执行 4 个 0.01 秒物理步，代价、噪声缩放、候选选择和移位规则来自上游。结果同时保留零控制基线和完整轨迹。CEM 分别运行同预算的固定结构基线与联合搜索，保留设计参数、历代最优和运动轨迹。投掷距离使用上游“释放状态推算弹道落点”的目标定义，不能等同于实物投掷距离。两项输出指标，不默认附加验收阈值。

`rizon_inertia.py` 的参数辨识、G1 空翻和双杆 iLQR 仍是可参考的上游示例，本次未接入为受管理任务。

## 原生 mjbatch 后端

在已准备好的独立环境运行现有任务：

```bash
PYTHONPATH=src .cache/external/mjbatch/.venv/bin/python -m embodiedforge rollout \
  --task hold --physics mjbatch --num-envs 32 --steps 100

PYTHONPATH=src .cache/external/mjbatch/.venv/bin/python benchmarks/mjbatch_cpu.py \
  --num-envs 1024 --steps 200 --threads 4 --repeats 5 --output runs/mjbatch-benchmark.json
```

也提供可选依赖 `embodiedforge[mjbatch]`（mjbatch 0.1.0、MuJoCo 3.11.0）。后端复用顺序 MuJoCo 的同一平面模型，支持乱序局部 reset、active mask、终止后冻结和拥有独立数组的 snapshot。`Batch` 默认使用逻辑 CPU 数并按环境数限制；直接构造 `MjbatchPhysics(num_threads=4)` 可以限制线程。本机 benchmark 显式使用 4 个线程，包含控制写入、4 个物理子步和 snapshot，排除任务逻辑、渲染和初始化。

## 本机验证结果

CPU：Ryzen 9 9950X，GPU：RTX 5090 D v2。以下都是本机固定版本与指定种子下的仿真结果。

| 运行（`runs/` 下） | 结果 |
| --- | --- |
| `go1-512x600-20260912` | 512 环境 × 600 轮 × 24 步，7,372,800 次转换；约 220 秒总运行，学习循环约 218 秒，模型/优化器有限性通过 |
| `go1-suite-motion-v2-20260912` | 四项 × 三种子 × 32 环境，384 次 60 秒试验全部存活；36 项验收全部通过，12 份回放通过轨迹检查，浏览器交互检查通过 |
| `go1-v2-eval-20260913` | 新版 600 轮模型完成 384 次 60 秒试验，全部存活、36 项验收通过，生成 12 份运动回放 |
| `go1-v2-512x600-20260913` | transition-v2 从零训练 512 环境 × 600 轮 × 24 步，7,372,800 次转换，训练产物检查通过 |
| `go1-v1-inherit-20260913` / `go1-v2-migrate-20260913` / `go1-v2-inherit-20260913` | 旧版默认继承、显式迁入 v2、v2 后续继承均通过；旧版续训模型及优化器仍与原结果逐张量一致 |
| `go1-port-512x600-20260913` | 迁入 Go1 实现从零训练 600 轮；使用 upstream-v1，全部逐轮指标（耗时除外）、最终模型和优化器张量与上游版本完全一致 |
| `go1-port-old-model-20260913` | 旧模型在迁入代码下的 384 次 60 秒试验、全部逐环境指标与原评估完全一致 |
| `go1-port-eval-20260913` | 迁入实现新训练模型：384 次 60 秒试验全部存活，36 项验收通过，生成 12 份运动回放 |
| `go1-port-resume-20260913` | 旧实现的 600 轮模型在迁入代码下恢复模型和优化器，再更新 3 轮，累计 603 轮通过检查 |
| `wuji-512x300-20260912` | 512 环境 × 300 轮 × 40 步，6,144,000 次转换；训练产物检查通过 |
| `wuji-eval-300-20260912` | 50 次重定向试验：成功 0%、掉落 0%、超时 100%，验收 `rejected`；尚未学会目标重定向 |
| `wuji-light-smoke-20260912` | Light 变体 32 环境、3 轮，模型和指标检查通过 |
| `go1-resume-smoke-20260912` / `wuji-resume-smoke-20260912` | 两者各新增 3 轮，恢复及产物检查通过 |
| `cartpole-mpc-20260912` | 平均代价 0.297，对照零控制 2.0；最终倾角 0.0013 rad，最后一秒最大倾角 0.0168 rad |
| `arm-codesign-20260912` | 同为 512 候选、30 代：固定结构预测距离 5.964 m，联合优化 9.360 m |
| `integration-20260912/mjbatch-benchmark.json` | 1024 环境、5 次交替测量中位数：顺序 MuJoCo 118,685 transitions/s，mjbatch 837,755，约 7.06 倍；最终状态一致 |

Go1 60 秒结果按试验汇总，所有试验均完成窗口，因此 RMSE 按等量有效帧合并：

| 指令 | 存活率 | 平面 RMSE（m/s） | 转向 RMSE（rad/s） | 实际前向/转向均值 |
| --- | --- | --- | --- | --- |
| 站立 | 100% | 0.0103 | 0.0048 | -0.0006 m/s / 约 0 rad/s |
| 前进 | 100% | 0.0477 | 0.0276 | 0.4703 m/s / 0.0003 rad/s |
| 左转 | 100% | 0.0473 | 0.0753 | 0.4708 m/s / 0.4372 rad/s |
| 右转 | 100% | 0.0485 | 0.0752 | 0.4695 m/s / -0.4370 rad/s |

同样使用 512 环境、600 轮、训练种子 0 的 v2 模型，在相同评估种子与固定指令下：

| 指令 | v1 平面 RMSE（m/s） | v2 平面 RMSE（m/s） | v1 转向 RMSE（rad/s） | v2 转向 RMSE（rad/s） |
| --- | --- | --- | --- | --- |
| 站立 | 0.0103 | 0.0080 | 0.0048 | 0.0040 |
| 前进 | 0.0477 | 0.0795 | 0.0276 | 0.0272 |
| 左转 | 0.0473 | 0.0750 | 0.0753 | 0.0465 |
| 右转 | 0.0485 | 0.0833 | 0.0752 | 0.0491 |

两版均完成全部窗口，RMSE 按等量有效帧合并。v2 的站立和转向误差改善，移动时平面速度误差上升；该结果只覆盖一个训练种子，不证明性能全面提升。奖励时序的正确性由强制指令切换专项测试验证。版本继承和结果对照保存在 `runs/go1-semantics-audit-20260913/`。

3 轮 Go1 短测模型的同种子前进评估虽然有 90.6% 存活率，但实际前向速度约 -0.0016 m/s、平面 RMSE 0.503，未通过验收。600 轮后的改善有独立速度指标支持。当前结果不覆盖全速度范围、外部推力和其他地形。

Wuji 上游默认规模为 8192 环境 × 5000 轮。本机保持 512 环境、40 步 horizon，从 300 轮续训到累计 1000 轮（20,480,000 次转换，`wuji-512x1000-20260912`），训练产物检查通过。

2026-09-13 对照复测使用相同的 20 组初始位置、姿态和目标，每组 14 秒：300 轮策略、1000 轮策略和全零动作均为 0% 成功、0% 掉落。平均最终姿态误差减少量分别为 0.118、0.194、-0.190 rad（正数表示改善）。策略有一定目标跟踪改善，但仍未学会重定向；全零动作也不掉落，所以不能把零掉落归功于训练。对照证据见 `runs/wuji-comparison-20260913.json`。

`runs/wuji-video-final-20260913/evaluation.mp4` 已验证 281 帧、20 fps、640×368，校验信息保存在同目录 `video-validation.json`。当前主机的上游多进程渲染曾出现工作进程退出，适配层改用单进程渲染并验证实际产物；早期失败试验保留在运行目录中，不计为通过。

验证检查：`ef-viewer` 核心回归 333 项通过、50 项因可选依赖跳过（其中 11 项 Go1 专项测试已在独立 SDK 环境全部通过）；独立 mjbatch 环境的 2 项物理契约测试通过；含 Torch 的环境另外通过 10 项训练/模型检查。Ruff 与差异空白检查通过。浏览器证据保存在 Go1 回放目录的 `browser-check.json` 和 `replay-preview.png`。

PPO 直接调用也检查批次边界：样本数需能分为四个完整 minibatch，且每批至少两个样本；观测、动作、log-probability、advantage 和 return 必须为形状一致的有限 CPU float32 张量。GAE 检查时间/环境维度和终止标记，非有限 loss 或梯度会中止更新。已完成的 minibatch 不会因后续 minibatch 失败而自动回滚。

2026-09-13 重置与批次边界修复后，`go1-reset-fix-resume-20260913` 的三轮续训模型及优化器仍与修复前完全一致；`go1-reset-fix-eval-20260913` 的 128 次 60 秒试验通过全部 12 项验收，逐环境指标保持一致。摘要见 `runs/go1-reset-fix-audit-20260913.json`。

Go1 移植的独立验证包括 100 步物理/观测/十项奖励及部分重置对照、含超时重置的 rollout/GAE/PPO 更新对照、镜像等变性和不同实例的时限隔离。可在固定版本 SDK 环境复现：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src:tests \
  MENAGERIE_CACHE_DIR=.cache/external/mjbatch/assets-cache CUDA_VISIBLE_DEVICES= \
  EF_MJBATCH_REFERENCE=.cache/external/mjbatch \
  .cache/external/mjbatch/.venv/bin/python -m unittest test_go1_port -v
```

`runs/go1-port-audit-20260913/` 保存完整 600 轮训练对照、旧模型评估对照以及 wheel 安装包检查。安装包检查在临时解压目录运行，并禁止导入 `go1_joystick` / `window`，验证场景、许可证和一次 PPO 更新；它复用 SDK 依赖和已缓存的机器人资产。

## 源码与协议参考

两个上游仓库均为 Apache-2.0，保持其源码与资产许可。Go1 迁入文件标注了来源版本和修改说明，并随包保留 mjbatch 许可证；其余任务仍由上游实现，EmbodiedForge 提供流程管理、检查、评估和回放适配。

- Wuji：`src/wuji_unilab/cli.py`、`eval.py`、`conf/ppo/task/wuji_reorient/mjwarp.yaml`、`rl/runtime.py`；任务及机器人资产进一步来源于 Wuji Technology 的 wuji-mjlab。
- mjbatch：`src/mjbatch/_bindings.pyi`、`examples/go1_joystick.py`、`examples/cartpole_mpc.py`、`examples/arm_throw.py`。
- EmbodiedForge：[入口](../src/embodiedforge/recipes.py)、[Wuji 适配](../src/embodiedforge/_wuji_recipe.py)、[mjbatch 任务适配](../src/embodiedforge/_mjbatch_recipe.py)、[动态响应指标](../src/embodiedforge/_go1_metrics.py)、[原生后端](../src/embodiedforge/backends/mjbatch.py)。
