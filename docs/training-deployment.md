# 训练与部署命令

**简体中文** | [English](training-deployment.en.md) · [返回 README](../README.md)

按任务选择一节执行，无需安装全部 SDK。以下命令从 EmbodiedForge 仓库根目录运行；`/path/to/...` 必须替换为本机路径，所有新输出目录或导出文件必须尚不存在。示例预算用于复现流程，不承诺训练后达到行为验收要求。

[环境准备](#setup) · [核心 PPO](#core) · [Go1](#go1) · [Microduck](#microduck) · [H1](#h1) · [Wuji](#wuji) · [MPC / CEM](#solvers) · [Web 运行与远程访问](#serving) · [安装包部署](#package) · [产物与限制](#artifacts)

| 任务 | 训练 / 求解位置 | 当前运行或导出入口 |
| --- | --- | --- |
| `reach` / `hold` | 本仓库 VectorEnv + CPU PPO；NumPy / MuJoCo 等物理后端 | checkpoint 无窗口评估、数据记录 |
| Go1 | 本仓库 Go1 环境 + PPO；已安装依赖或固定 SDK 中的 CPU MuJoCo/mjbatch | 统一 Web 在线策略、运动回放 |
| Microduck | 外部 Microduck/mjlab 任务与训练实现，CUDA PPO | 原生 / Viser 策略运行、ONNX 导出与对照 |
| H1 原生 | 项目内 MuJoCo/mjbatch CPU PPO，无 IsaacLab 依赖 | 固定指令评估、统一 Web 运动回放 |
| H1 | 外部 IsaacLab 环境 + RSL-RL；Newton / MuJoCo-Warp GPU 物理 | 固定指令评估、离线 HTML / Web 记录回放 |
| Wuji / Wuji Light | 外部 Wuji/UniLab 环境与训练实现，GPU PPO | 顺序试验评估、视频记录 |
| Cartpole / 机械臂投掷 | 外部 mjbatch example，适配 CPU MPC / CEM | 求解指标与轨迹 |

这里的部署指仿真策略运行、查看器服务和模型导出。当前没有统一实机部署命令，也没有 Go1/H1 的通用 ONNX 导出入口。

**训练实际使用哪份实现？** `train` 使用核心 `VectorEnv`。Go1 和 `h1-native` 使用本仓库维护的机器人环境与 PPO，但尚未接入核心 `VectorEnv`；MuJoCo/mjbatch 仍负责底层物理计算。Go1 可使用 `--standalone` 从已安装依赖启动；默认模式仍使用固定版本的 mjbatch SDK 检出。`h1` 命令使用 IsaacLab，`h1-native` 不使用。统一 Web 查看器是独立可视化入口，不是训练环境。

新启动的核心 PPO、Go1 PPO 和原生 H1 训练会在输出目录写入 `training-runtime.json`，Go1/H1 续训也会单独记录。文件包含实际加载的环境、任务、训练函数和物理适配器、模块及文件路径、Python 源文件哈希、依赖版本、解释器、CPU 执行位置和是否使用核心 `VectorEnv`。Go1 的路径指向该次运行的实现快照。这是入口来源记录，不是全部间接依赖清单、checkpoint 兼容锁或策略质量证明；旧运行和外部训练流程不会补写此文件。

<a id="setup"></a>

## 环境准备

主环境只安装调度入口和需要的查看器依赖。使用已有 Python 3.11 环境，或按 README 创建 `ef-viewer`：

```bash
python -m pip install -e '.[viz-robot]' 'mujoco==3.11.0'
python -m embodiedforge recipes list
```

训练 SDK 保持隔离；`recipes setup` 和 Microduck setup 需要 Git、`uv` 与网络。只运行所需来源的安装命令。源码必须为干净的固定版本检出，命令不会替你切换用户仓库的版本。

```bash
python -m embodiedforge recipes setup --source mjbatch \
  --repo /path/to/mjbatch --python 3.12 --timeout 1200
python -m embodiedforge recipes setup --source wuji_unilab \
  --repo /path/to/wuji_unilab --python 3.12 --timeout 1200
python -m embodiedforge.microduck setup --repo /path/to/microduck_rl
python -m embodiedforge.microduck check --repo /path/to/microduck_rl
```

| 来源 | 固定 revision | 环境位置 |
| --- | --- | --- |
| mjbatch | `b84c0c20aedbdf048122cbc47f554e9b93cc4754` | `.cache/external/mjbatch/.venv` |
| Wuji UniLab | `91ccfa0ec8c129b300865bd36c59dc9eed56a744` | `.cache/external/wuji_unilab/.venv` |
| Microduck | `53b8971b61baf5b7f3c16d135dd7cac37623de4b` | `.cache/microduck-venv` |
| IsaacLab H1 | `2e44ddb2e19536579140496023b5ccb060bc4152` | 已有环境，通过 `--environment` 指定 |

H1 没有自动 `setup` 子命令，需要先准备 [H1 文档](h1-isaaclab.md) 中的 Python 3.12 / RSL-RL 5.0.1 独立环境。`--environment` 接收环境根目录，不是 Python 可执行文件。Wuji、Microduck 和 IsaacLab H1 需要对应 CUDA 环境；Go1 与两个求解任务使用 CPU。自定义缓存时，每一步需传入相同的 `--cache` 或 Microduck `--env-dir`。

<a id="core"></a>

## 核心 PPO：reach / hold

在当前主环境安装 Torch 与 MuJoCo；下面分别训练两个点任务，评估训练策略，并记录 reach 评估数据：

```bash
python -m pip install -e '.[train,mujoco]'
python -m embodiedforge train --task reach --physics mujoco \
  --num-envs 32 --updates 100 --output runs/commands-reach
python -m embodiedforge evaluate \
  --checkpoint runs/commands-reach/checkpoint.pt \
  --num-envs 32 --steps 300 --output runs/commands-reach-eval
python -m embodiedforge inspect --dataset runs/commands-reach-eval
python -m embodiedforge train --task hold --physics numpy \
  --num-envs 32 --updates 100 --output runs/commands-hold
python -m embodiedforge evaluate \
  --checkpoint runs/commands-hold/checkpoint.pt --num-envs 32 --steps 300
```

核心 PPO 使用 proprio，不训练图像策略。评估从 checkpoint 恢复配置；不要随意更换任务或物理。核心 CLI 当前没有续训或 ONNX 导出命令；点任务 `visualization` 运行示范控制器，不加载这些 PPO checkpoint。

要比较训练前后的策略，可在安装了核心训练依赖的环境执行以下命令。脚本沿用 checkpoint 的任务、物理、控制周期和回合长度；初始策略使用相同网络维度，两个策略使用相同评估种子。输出包含模型 SHA256 和环境配置，便于核对比较条件。

```bash
python benchmarks/check_learning.py runs/commands-reach/checkpoint.pt --num-envs 128 --seeds 1001 1002 1003
python benchmarks/check_learning.py runs/commands-hold/checkpoint.pt --num-envs 128 --seeds 1001 1002 1003
```

<a id="go1"></a>

## Go1：训练 → 续训 → 验收 → 在线运行

### 已安装依赖模式：无需外部仓库检出

Go1 训练、续训和评估加上 `--standalone`，使用当前 Python 解释器中的 MuJoCo 3.11.0、mjbatch 0.1.0、PyTorch 2.9.0（CPU/CUDA wheel 均可）和 Menagerie 2026.9.0，物理与学习都在 CPU 上执行。无需 `recipes setup`、IsaacLab、上游 example 或 SDK 源码检出；该模式不使用 `--cache`。机器人资产仍来自 Menagerie，首次使用可能下载；可通过 `MENAGERIE_CACHE_DIR` 复用现有资产缓存。

```bash
conda create -n ef-go1-native python=3.12 pip -y
conda activate ef-go1-native
python -m pip install -e '.[go1-native]'
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --num-envs 128 --horizon 24 --updates 1000 --threads 4 \
  --timeout 1200 --output runs/commands-go1-native
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --resume-run runs/commands-go1-native --num-envs 128 --horizon 24 \
  --updates 100 --threads 4 --timeout 1200 --output runs/commands-go1-native-resumed
python -m embodiedforge recipes evaluate --task go1-joystick --standalone \
  --run runs/commands-go1-native-resumed --velocity 0.5 0 0 \
  --num-envs 8 --steps 500 --record-motion --timeout 1200 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-go1-native-eval
conda activate ef-viewer
python -m embodiedforge live --run runs/commands-go1-native-resumed \
  --render-backend rtx --port 8081
```

续训和评估可读取两种启动模式下已完成的 Go1 训练，保留 checkpoint 哈希与任务/配置检查。在已安装依赖的环境中继续传 `--standalone`。Web 在线控制自动选用运行记录中的训练解释器；查看器环境仍需安装渲染依赖。策略训练不足时评估门槛会拒绝通过，应先检查报告再在线诊断。输出目录不可覆盖。

每次运行仍保存本仓库实现快照，以及实际依赖版本和加载路径。`run.json` 用 `source.kind=installed_packages` 标识该模式，不会虚构已检查某个上游 Git revision。这次仅改变启动方式，不改变任务、奖励或 PPO，也没有把 Go1 合并到核心 `VectorEnv`。

Go1 Web 在线控制从训练目录的 `implementation` 快照加载任务、配置、策略网络和场景，启动前逐文件核对 SHA256 并复制到会话私有目录。通信与资产校验使用当前版本；快照损坏会拒绝启动，完全没有快照记录的旧训练目录会明确警告后使用当前代码。启动日志及策略元数据标明实现来源。请保留完整训练目录；固定这些输入不保证不同硬件、随机种子或回合设置下轨迹一致。

Go1 评估同样加载训练快照，并核对训练记录内的依赖版本。评估目录中的 `input-implementation/` 保存校验后的训练代码副本，`evaluation-implementation.json` 和 `result.implementation` 记录实际加载的任务、策略及场景路径与哈希；`implementation/embodiedforge` 则保存本次使用的评估适配器和指标代码。缺少整个快照记录的旧产物会警告后使用当前代码，已记录但损坏的快照会拒绝执行。评估仍按命令中的速度、种子和回合长度运行，默认继承训练的任务及配置。

续训继续使用当前仓库的任务和 PPO，并为新运行保存新的实现快照，便于应用训练修复；它不承诺沿用旧训练代码或与未中断训练产生相同轨迹。续训完成后，评估和在线控制使用该次续训保存的快照。

Go1 续训会保存来源运行记录的副本 `input-run.json`，并在模型及优化器加载成功后写入 `resume-report.json`。报告列出新旧快照哈希的文件差异、环境数/rollout 长度/种子/线程数、任务语义、奖励与指令配置、学习率覆盖值，以及实际依赖版本的变化；缺少历史信息时标为 `unknown`。代码对比基于记录中的哈希，不重新验证或执行旧源码，也不推断代码变化是否影响行为。报告及来源记录的哈希会被校验，完成日志给出变化摘要。报告表示已加载续训状态，训练是否完成仍以 `run.json.status` 为准。

恢复的状态包含策略参数、归一化统计和优化器；仿真、回合及随机数生成器重新初始化，学习率按续训迭代号和当前覆盖参数计算。即使报告显示输入未变，也不表示与未中断训练等价。

Go1 续训、评估与在线控制使用同一 checkpoint 校验：哈希对应实际反序列化的字节，权重内的迭代号须为与运行记录一致的非负整数，任务配置须匹配来源记录。归一化均值和方差必须为有限浮点向量，方差不可为负，计数须为正的有限浮点标量。无效输入在创建仿真环境前拒绝；合法的零方差及旧版缺省配置仍受支持。多种子/多指令评估共用一次读取并校验后的权重，各用例仍单独创建策略和环境。

续训还会在创建仿真前检查 Adam 状态与当前参数是否匹配，包括参数组、完整且无重复的参数映射、动量形状/类型、非负二阶矩、有效步数和超参数。不兼容的优化器模式会被拒绝，避免加载时悄悄切换算法。保存仍每 25 次更新及训练结束时进行：先校验优化器状态和张量有限性，再写临时文件并替换 checkpoint；校验或写入失败时保留上一份已保存文件，运行按失败记录。

新 Go1 训练每次成功保存时还会提交 `recovery.json`，其中包含独立 checkpoint 的哈希、保存进度、配置、依赖和资产记录。`checkpoints/` 保留最近两份快照，`model.pt` 继续作为完成运行的标准产物。正常停止后，状态为 `interrupted`、`timed_out` 或 `failed` 的目录可直接通过 `--resume-run` 恢复；启动时核对恢复记录、请求、实现快照和权重哈希。续训从已保存的迭代继续，未保存的后续更新不会计入。原运行状态保持不变，续训报告标明恢复来源；新目录的 `input-run.json` 会附带已验证的保存点结果和 `checkpoint_origin` 标记。

例如，上面的独立训练因 Ctrl+C 或超时停止后，先检查 `recoverable_checkpoint`，再写入新的续训目录：

```bash
python -m embodiedforge recipes status --run runs/commands-go1-native
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --resume-run runs/commands-go1-native --num-envs 128 --horizon 24 \
  --updates 100 --threads 4 --timeout 1200 --output runs/commands-go1-native-recovered
```

没有成功保存过 checkpoint、缺少新恢复记录的旧中断运行，以及仍标为 `running` 的目录不会自动恢复。`status` 会为已停止但无法恢复的运行给出 `recovery_error`。评估和在线控制仍要求完成的训练目录；恢复续训成功后可照常使用新目录。

Go1 训练、续训、评估和在线策略启动都会校验已发布的资产缓存；运行记录包含资产信息时，还会核对机器人、tree 和 archive 标识。未设置 `MENAGERIE_CACHE_DIR` 时，独立模式的续训/评估及在线控制会复用仍存在的训练缓存；显式指定的缓存会保留并接受校验。旧版 `assets.json` 记录仍可读取。`MENAGERIE_ROOT` 本地检出覆盖会被拒绝，因为缓存校验器不能验证那个独立目录；请用 `MENAGERIE_CACHE_DIR` 选择已发布资产。这只核对机器人资产，不保证更换代码或依赖后仿真逐步等价。

### 固定 SDK 模式

先完成 mjbatch setup。续训写入新目录，`--updates` 为本次追加更新数；以下超时均为秒，可按机器速度增加。

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --num-envs 512 --horizon 24 --updates 600 --threads 4 \
  --timeout 1200 --output runs/commands-go1
python -m embodiedforge recipes status --run runs/commands-go1
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/commands-go1 --num-envs 512 --horizon 24 \
  --updates 100 --threads 4 --timeout 1200 --output runs/commands-go1-resumed
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/commands-go1-resumed --suite basic --seeds 0 1 2 \
  --num-envs 32 --steps 3000 --record-motion --timeout 1200 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-go1-eval
python -m embodiedforge live --run runs/commands-go1-resumed \
  --render-backend mujoco --num-envs 2 \
  --width 960 --height 540 --fps 30 --port 8080
```

评估门槛是示例要求，任一项目/种子未达标会返回 `rejected`，保留报告；应查看结果后决定是否进行在线诊断。在线命令在查看器环境执行，自动启动训练环境中的策略进程。打开 `http://127.0.0.1:8080`，先“前进”再“继续”。也可用 `--worker-python /path/to/venv/bin/python` 显式指定兼容策略环境。

固定指令通过后，可以单独验证连续指令切换；这是一项新的评估，使用新输出目录：

```bash
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/commands-go1-resumed --suite switching --seeds 0 1 2 \
  --num-envs 32 --steps 3000 --record-motion --timeout 1200 \
  --output runs/commands-go1-switching
```

该切换示例仅记录指标，未设置验收门槛。更多 `extended`、横移及复合切换套件见 [任务配方](recipes.md)。运动记录使用匹配 MJCF，通过后文 Web 回放命令查看。

<a id="microduck"></a>

## Microduck：训练、导出与策略运行

先完成 Microduck setup/check。smoke 检查 64 环境、5 次更新及 ONNX 导出链路；正式训练另起目录。

```bash
python -m embodiedforge.microduck smoke --repo /path/to/microduck_rl \
  --output runs/commands-microduck-smoke
python -m embodiedforge.microduck train --repo /path/to/microduck_rl \
  --num-envs 4096 --iterations 4000 --output runs/commands-microduck
python -m embodiedforge.microduck progress --repo /path/to/microduck_rl \
  --run runs/commands-microduck
```

从 `runs/commands-microduck/run.json` 的 `checkpoint` 字段读取实际模型路径，替换下面的 `/path/to/microduck/model.pt`。不要按训练轮数猜测文件名。续训、原生运行和 Viser 运行可按需要分别执行：

```bash
python -m embodiedforge.microduck train --repo /path/to/microduck_rl \
  --resume /path/to/microduck/model.pt --num-envs 4096 --iterations 1000 \
  --output runs/commands-microduck-resumed
python -m embodiedforge.microduck play --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --viewer native --steps 500 --seed 0
python -m embodiedforge.microduck play --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --viewer viser
```

Viser 使用上游页面，不是统一 Go1/H1 Web 页面；不支持上述原生模式的 `--steps`。下面导出同一个 checkpoint，再在实际观测上对照 ONNX 与 Torch 动作：

```bash
python -m embodiedforge.microduck export --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --output runs/commands-microduck.onnx
python -m embodiedforge.microduck evaluate --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --onnx runs/commands-microduck.onnx \
  --velocity 0.2 0 0 --no-pushes --num-envs 16 --steps 500 --seeds 0 1 2 \
  --output runs/commands-microduck-eval
```

导出包括观测归一化；输入为 61 维、输出为 14 维。对照仍由 Torch 动作推进仿真，不能当作 ONNX 独立闭环或实机部署验证。这里的评估未设行为门槛；可按 [Microduck 文档](microduck.md) 添加 RMSE / 存活率要求。

<a id="h1"></a>

## H1：IsaacLab 训练与回放

### 原生 CPU 训练

新增 `h1-native` 入口，不依赖 IsaacLab、RSL-RL 或上游训练器。可在独立 Python 环境安装以下依赖；原 GPU 配方继续保留在下方。

```bash
python -m pip install -e '.[h1-native]'
python -m embodiedforge h1-native train \
  --model /path/to/unitree_h1/h1.xml --num-envs 128 --threads 4 --updates 500 \
  --output runs/commands-h1-native
python -m embodiedforge h1-native train \
  --resume runs/commands-h1-native --num-envs 128 --threads 4 --updates 1000 \
  --output runs/commands-h1-native-resumed
python -m embodiedforge h1-native evaluate \
  --run runs/commands-h1-native-resumed --steps 500 --velocity 0.5 0 0 \
  --record-motion --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-h1-native-eval
```

在 `ef-viewer` 中，用续训目录的 `model.mjb` 和评估目录的 `motion.npz` 运行统一 `replay` 命令。接口、力矩与验收边界见[原生 H1](h1-native.md)。原生 checkpoint 与 IsaacLab 流程不兼容。

### IsaacLab GPU 配方

使用已准备好的固定版本 IsaacLab 与独立环境。先短测，再追加训练；以下命令的 `--repo` 和 `--environment` 每次保持一致。

```bash
python -m embodiedforge h1 train \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --num-envs 64 --updates 5 --timeout 600 --output runs/commands-h1-smoke
python -m embodiedforge h1 status --run runs/commands-h1-smoke
python -m embodiedforge h1 train \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --resume-run runs/commands-h1-smoke --num-envs 512 --updates 1000 \
  --output runs/commands-h1-resumed
python -m embodiedforge h1 evaluate \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --run runs/commands-h1-resumed --suite basic --num-envs 32 \
  --steps 500 --seeds 0 1 2 --record-motion --record-env 0 --timeout 600 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-h1-eval
python -m embodiedforge h1 replay \
  --motion runs/commands-h1-eval/motion-seed-0-forward-env-0.npz \
  --output runs/commands-h1-forward.html
```

正式训练未设置额外超时，可显式加 `--timeout` 限制墙钟时间。500 步为 10 秒；长时验证可使用 `--steps 3000` 并更换输出目录。HTML 文件可离线打开，显示骨架；真实网格回放见后文。H1 目前没有统一 Web 在线指令入口，不能使用 `live --run` 加载 H1 训练目录。

<a id="wuji"></a>

## Wuji / Wuji Light：重定向训练

先完成 wuji_unilab setup。标准与 Light 是不同任务，使用不同运行目录，续训和评估必须保持相同 `--task`。以下小规模预算用于入口验证：

```bash
python -m embodiedforge recipes train --task wuji-reorient \
  --num-envs 32 --horizon 40 --updates 5 --timeout 600 \
  --output runs/commands-wuji
python -m embodiedforge recipes train --task wuji-reorient-light \
  --num-envs 32 --horizon 40 --updates 5 --timeout 600 \
  --output runs/commands-wuji-light
python -m embodiedforge recipes train --task wuji-reorient-light \
  --resume-run runs/commands-wuji-light --num-envs 32 --horizon 40 \
  --updates 100 --timeout 1800 --output runs/commands-wuji-light-resumed
python -m embodiedforge recipes status --run runs/commands-wuji-light-resumed
python -m embodiedforge recipes evaluate --task wuji-reorient-light \
  --run runs/commands-wuji-light-resumed --num-envs 1 --seed 0 \
  --num-trials 50 --steps 280 --timeout 1200 \
  --min-success-rate 0.5 --max-drop-rate 0.2 --output runs/commands-wuji-eval
```

评估逐个执行单环境试验，不能用 Go1 的 `--suite` / `--seeds` / `--record-motion`。标准版评估将上面 `--task` 改为 `wuji-reorient`，`--run` 改为 `runs/commands-wuji`，并使用新的输出目录。可另录一个小样本视频，环境需要 `ffprobe`：

```bash
python -m embodiedforge recipes evaluate --task wuji-reorient-light \
  --run runs/commands-wuji-light-resumed --num-envs 1 --seed 0 \
  --num-trials 1 --steps 280 --record-video --timeout 600 \
  --output runs/commands-wuji-video
```

视频用于检查行为，不代表完整验收；Wuji 当前没有统一 Web 在线控制或模型导出命令。

<a id="solvers"></a>

## CPU MPC 与 CEM 求解

复用 mjbatch setup。MPC 的 `--num-envs` 是候选轨迹数，CEM 中是种群数；两者均不产生 PPO checkpoint。

```bash
python -m embodiedforge recipes solve --task cartpole-mpc \
  --num-envs 1024 --horizon 25 --steps 150 --threads 4 --timeout 600 \
  --output runs/commands-cartpole
python -m embodiedforge recipes solve --task arm-throw-codesign \
  --num-envs 512 --generations 30 --threads 4 --timeout 600 \
  --output runs/commands-arm
python -m embodiedforge recipes status --run runs/commands-arm
```

输出求解指标、基线与轨迹；不能将求解目录传给 Go1 `live`。更多目标函数和轨迹语义见 [任务配方](recipes.md)。

<a id="serving"></a>

## Web 运行与远程访问

在查看器环境安装所需渲染器；下列 OVRTX 安装使用已验证约束，需要兼容 RTX GPU 和驱动。安装后 Go1 在线画面也可改为 `--render-backend rtx`。

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com \
  -c configs/viewer-constraints.txt -e '.[viz-robot,viz-rtx]' 'mujoco==3.11.0'
python -m embodiedforge.visualization --viewer web --render-backend rtx --check
```

`--check` 只检查安装元数据。下面两个回放示例使用前文生成的记录和本机匹配 MJCF；模型目录需包含引用的网格与材质。分别运行，或在不同终端使用不同端口：

```bash
python -m embodiedforge replay --model /path/to/unitree_go1/go1.xml \
  --motion runs/commands-go1-switching/motion-seed-0-switching.npz \
  --render-backend mujoco --width 960 --height 540 --fps 30 --port 8081
python -m embodiedforge replay --model /path/to/unitree_h1/h1.xml \
  --motion runs/commands-h1-eval/motion-seed-0-forward-env-0.npz \
  --render-backend rtx --width 960 --height 540 --fps 30 --port 8082
```

服务默认监听回环地址。在客户端另开终端转发 Go1 在线服务和两个回放端口，随后访问对应本地地址：

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 \
  -L 8082:127.0.0.1:8082 user@server
```

打开 `http://127.0.0.1:8080`、`:8081` 或 `:8082`，对应服务需保持运行。页面可调分辨率与 1–60 FPS；Go1 在线默认暂停，回放只读取记录。停止按钮/Ctrl+C 结束整个会话，多个浏览器共享控制。当前这些命令是前台进程；没有附带系统服务安装或自动重启配置。详细生命周期与远程访问见 [查看器](viewers.md)。

出现 EGL 警告或怀疑使用了错误 GPU 时，可实际渲染测试帧并查看 `graphics.vendor`、`graphics.renderer` 和保留的驱动警告：

```bash
python -m embodiedforge doctor --graphics egl
```

该诊断在独立进程运行，不修改系统驱动配置；成功出帧不代表选中了 NVIDIA。系统驱动包修复与本机缺失注册文件的案例见 [EGL 诊断](graphics-diagnostics.md)。

<a id="package"></a>

## 安装包部署：在另一台机器运行

在源码仓库构建 wheel。该命令只打包 EmbodiedForge，不把训练 SDK、模型或缓存装进包里：

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

将生成的 `dist/embodiedforge-0.1.0-py3-none-any.whl` 和完整 Go1 运行目录复制到目标机器。目标机器先准备兼容的独立 mjbatch SDK 环境，再在 Python 3.11 查看器环境中执行（替换所有占位路径）：

```bash
python -m pip install '/path/to/embodiedforge-0.1.0-py3-none-any.whl[viz-robot]' 'mujoco==3.11.0'
python -m embodiedforge live --run /path/to/runs/go1-trained \
  --worker-python /path/to/mjbatch-venv/bin/python \
  --render-backend mujoco --port 8080
```

安装后 CLI 不依赖源码工作目录；Web 的 HTML/JavaScript、Go1 场景 XML 与移植许可证随 wheel 分发。`--worker-python` 必须保留虚拟环境入口路径，不要解析成底层通用 Python。wheel 只包含 EmbodiedForge 代码；各任务的外部 SDK、固定版本来源记录与机器人资产仍需按前文准备。OVRTX 还需要相应查看器依赖与驱动。

开发与示例复现仍建议使用源码检出；源码分发包还包含中英文文档、配图、配置、示例和基准脚本，训练产物和缓存不打包。版本变更后，将上面的 wheel 文件名换成实际构建产物。

<a id="artifacts"></a>

## 产物、续训与验收

| 流程 | 应保留的输入 / 产物 | 后续用途 |
| --- | --- | --- |
| 核心 PPO | `checkpoint.pt` 及训练目录 | `evaluate --checkpoint` |
| Go1 / H1 / Wuji 托管训练 | 完整运行目录及 `run.json`，包括配置、模型与版本记录 | 同任务续训与评估；仅 Go1 支持 `live --run` |
| Microduck | `run.json`、实际 checkpoint、配置和运行环境记录 | `--resume`、`play`、`evaluate`、`export` |
| Microduck 导出 | ONNX 文件及验证报告 | 推理接口检查与动作对照 |
| Go1 / H1 运动记录 | NPZ 与匹配的 MJCF/资产目录 | 统一 Web 回放；H1 还支持离线骨架 HTML |
| MPC / CEM | 完整求解目录及结果、轨迹 | 结果分析，不作为 PPO 策略加载 |

训练期间从另一个终端执行 `status` 或 Microduck `progress`；状态文件不是进程存活保证。续训仅恢复任务支持的模型、优化器、归一化及课程等状态，不恢复每个环境的物理状态和全部 RNG，不能视为逐步等价续接。托管 Go1/H1/Wuji 续训要求输入为同任务兼容的 `complete` 运行。

模型换机器运行时，保留完整训练目录并重建兼容 SDK 和机器人资产。Go1 可通过 `--worker-python` 更换解释器路径，但仍会校验 SDK 与资产版本；不能通过复制单个 `model.pt` 绕过这些要求。评估 `complete` 仅说明执行结束；只有设置了门槛并逐项通过，才能按该评估条件判断验收结果。
