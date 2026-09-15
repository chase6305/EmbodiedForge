# EmbodiedForge

**简体中文** | [English](README.en.md)

![EmbodiedForge 工作流：机器人与任务 → 原生训练 → 评估与记录 → Web 回放](docs/assets/embodiedforge-overview.png)

*工作流示意图：Go1 / H1 原生训练使用 MuJoCo、mjbatch 与 PyTorch，不依赖 IsaacLab；评估后可记录运动并在 Web 中回放。在线策略交互目前支持 Go1，H1 行走策略仍需训练与验收。外部 SDK 任务使用独立入口，详见[能力范围](#capabilities)。*

面向 RL 与 VLA 实验的模块化机器人仿真、数据与训练平台。任务、物理、渲染、传感器和策略通过独立接口组合；统一 Web 页面用于查看仿真、回放运动记录和控制在线策略。

当前包含 CPU `VectorEnv` 参考运行时、本仓库维护的 Go1 与原生 H1 任务及 PPO，以及 Microduck、IsaacLab H1、Wuji 等任务的独立训练入口。各部分的实现范围见下表，长期设计见 [架构文档](docs/architecture.md)。

**快速导航：** [快速运行](#quick-start) · [训练与部署命令](docs/training-deployment.md) · [Web 交互](#web-viewer) · [Go1 在线](#go1-live) · [运动回放](#robot-replay) · [机器人训练](#robot-training) · [常见问题](#troubleshooting) · [文档索引](#documentation)

<a id="quick-start"></a>

## 快速运行：无需 GPU

以下命令从仓库根目录执行，环境创建示例面向 Linux。核心包支持 Python 3.10+；完整查看器的本机验证环境为 Python 3.11。已有 `ef-viewer` 环境可直接激活并跳过创建步骤。

```bash
conda create -n ef-viewer python=3.11 -y
conda activate ef-viewer
python -m pip install -e '.[viz-web]'
python -m embodiedforge.visualization --physics numpy --port 8080
```

打开 **http://127.0.0.1:8080**，即可查看点质量任务并操作暂停、单步、重置与相机控件。默认查看器为 Web，默认画面为 CPU Raster，默认目标帧率为 **30 FPS**。Raster 使用正交视图，不支持旋转相机。

只使用环境和数据接口时，安装 `python -m pip install -e .` 即可，不需要 Pillow、Torch 或图形 SDK。

<a id="capabilities"></a>

## 当前能力

| 组件 | 已实现 | 范围 |
| --- | --- | --- |
| 核心运行时 | `reach` / `hold`、局部重置、终止冻结、相机采样、最终观测 | CPU 点质量参考任务 |
| 物理后端 | NumPy、MuJoCo、mjbatch 线程池、Newton **1.6.0rc1** | 核心后端使用 CPU 状态快照 |
| 核心训练与数据 | PyTorch PPO、终止/超时 GAE、checkpoint 评估、episode 记录与窗口读取 | 核心 PPO 使用 proprio；训练与数据验证可独立运行 |
| Go1 行走 | 任务、奖励、镜像策略、归一化和 PPO 已移植到本仓库；支持训练、续训、评估与在线控制 | 仍依赖 mjbatch、MuJoCo、Torch 和 Menagerie 资产；尚未纳入核心 `VectorEnv` |
| H1 原生训练 | MJCF 模型、关节分组、批量命令、随机化、重置、力矩数据与 PPO | CPU MuJoCo/mjbatch，不依赖 IsaacLab；初步策略尚未通过行走跟踪验收。[说明](docs/h1-native.md) |
| 其他机器人任务 | Microduck、IsaacLab H1、Wuji 重定向、Cartpole MPC、机械臂投掷联合优化 | 独立 SDK 环境中的上游流程与适配层，非完整移植 |
| 统一 Web | Raster / MuJoCo / OpenGL / OVRTX、Go1 在线策略、Go1/H1 网格回放 | 查看器与物理后端独立选择；Web 画面不作为训练相机观测 |
| VLA 接口 | 图像、语言、proprio 数据窗口及 action chunk 执行器 | 尚无预训练 VLA 模型接入或微调器 |

<a id="web-viewer"></a>

## 统一 Web 可视化

点任务、Go1 在线控制和 Go1/H1 回放共用灰蓝色场景风格：低对比度网格地板、方向光和补光，MuJoCo 使用渐变天空，RTX 使用柔和棚拍光照。地板纹理固定在世界坐标中，并按机器人尺寸选用 0.5 m 或 1 m 网格，便于判断移动距离。Raster 使用简化网格背景。这些样式仅用于查看器，不改变训练观测或物理参数；各渲染器的色调和阴影仍有差异。

浏览器与原生窗口使用不同参数：

| 想要的入口 | 命令 |
| --- | --- |
| Web + MuJoCo 画面 | `python -m embodiedforge.visualization --viewer web --render-backend mujoco` |
| Web + OVRTX 画面 | `python -m embodiedforge.visualization --viewer web --render-backend rtx` |
| OVRTX 原生窗口 | `python -m embodiedforge.visualization --viewer rtx` |

这些命令默认使用 NumPy 点任务物理；机器人请使用下方的在线策略或回放入口。Web 入口无需 `--headless`，也无需 Node 或 ImGui；OpenGL 后端仍需要可用的图形上下文。

### 安装其他渲染器

MuJoCo 网格查看器无需 RTX GPU。下面固定本机 Go1 训练环境使用的 MuJoCo 版本；在线查看器必须与策略进程使用相同 MuJoCo 版本。

```bash
python -m pip install -e '.[viz-robot]' 'mujoco==3.11.0'
python -m embodiedforge.visualization --physics numpy --render-backend mujoco
```

OVRTX 需要兼容的 NVIDIA RTX GPU 和驱动。下列命令使用本机验证过的查看器约束，安装后也具备 OpenGL 查看能力：

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com \
  -c configs/viewer-constraints.txt -e '.[viz-robot,viz-rtx]' 'mujoco==3.11.0'
python -m embodiedforge.visualization --physics numpy --render-backend rtx
```

仅使用 OpenGL 可安装 `.[viz-robot,viz-gl]`。`--physics numpy|mujoco|mjbatch|newton` 选择点任务物理，`--render-backend raster|mujoco|gl|rtx` 选择画面；安装对应物理依赖后可自由组合。例如 `pip install -e '.[newton]'` 后，使用 `--physics newton --render-backend mujoco`。

### 页面交互

| 操作 | 行为 |
| --- | --- |
| 暂停 / 继续 / 单步 | 控制整个会话；重置仅作用于选中环境 |
| 切换渲染器 | 验证候选首帧后替换，失败保留旧画面 |
| 分辨率 | 360p、540p、720p、1080p；调整时保留仿真状态，渲染器重建期间短暂等待 |
| 目标帧率 | 支持 1–60 FPS，默认 30；页面显示实际出帧率、尺寸和单帧大小 |
| 相机与全屏 | 左键拖动旋转、连续滚轮缩放、默认视角、跟随机器人、全屏及截图；按后端能力启用 |
| Go1 速度指令 | 按钮或数值输入，支持 Enter 提交；草稿按环境保存，显示等待和生效状态 |
| 机器人回放 | 时间轴定位、切换记录、保留末帧、回到开头 |

画面尺寸以页面显示的实际值为准：Raster 输出边长为 `min(width, height)` 的正方形，其他三个后端支持上述宽屏预设。实际 FPS 统计服务端出帧，不代表浏览器刷新率；暂停时显示“静止”属于正常行为。

Go1 快捷键：**W/S** 前后、**A/D** 左右、**Q/E** 转向、**X** 指令归零。每次按键设置持续指令，松开不会归零，也不会自动开始仿真。**Space** 暂停/继续、**N** 单步、**R** 重置、**F** 默认视角；输入框内不会触发这些快捷键。

暂停且画面未变时复用 JPEG，不重复渲染和传输；控制状态仍更新。多个浏览器共享会话。断线后禁用操作、取消浏览器中尚未发送的操作并自动重连，不会自动暂停服务端仿真。已经发出的请求可能仍会完成，请在重连后确认当前状态，再重新执行被取消的操作。停止按钮或 Ctrl+C 释放会话资源。

远程访问可在客户端执行 `ssh -L 8080:127.0.0.1:8080 user@server`，再打开本地地址。不同会话使用不同端口。详细依赖、接口和帧率语义见 [可视化文档](docs/viewers.md)。

<a id="go1-live"></a>

### Go1 在线策略

需要一个已完成的 `go1-joystick` 托管训练目录，不能仅传入任意 `model.pt`。下面使用本机已训练产物；全新检出不包含这些 `runs/` 数据，请替换为自己的兼容训练目录。

```bash
python -m embodiedforge live \
  --run runs/go1-preserve-balanced-s0-1800-20260914 \
  --render-backend rtx --num-envs 2 \
  --width 960 --height 540 --fps 30 --port 8080
```

启动时默认暂停、指令归零。先点击“前进”，再点击“继续”。没有安装 OVRTX 时使用 `--render-backend mujoco`。

策略和 mjbatch 仿真在训练记录指定的独立 Python 环境中执行，查看器无需导入 Torch。可用 `--worker-python /path/to/venv/bin/python` 指定兼容环境。加载时检查模型哈希、配置及 SDK 版本；在线执行不更新策略权重。任一环境结束后暂停并保留末态，重置对应环境后继续。详见 [Go1 在线控制](docs/go1-live-web.md)。

<a id="robot-replay"></a>

### Go1 / H1 运动回放

回放需要匹配的本地 MJCF 与 NPZ 运动记录，不重新执行策略或物理。模型与记录逐帧校验；H1 当前在统一 Web 中支持记录回放，尚无在线速度控制。

```bash
python -m embodiedforge replay \
  --model /path/to/unitree_h1/h1.xml \
  --motion /path/to/motion-seed-0-forward-env-0.npz \
  --render-backend mujoco --fps 30 --port 8081
```

将占位路径替换为实际模型和记录后，打开 **http://127.0.0.1:8081**。本机 Go1/H1 完整示例、记录生成方式和材质支持范围见 [机器人回放](docs/robot-web-replay.md)。

<a id="core-rl"></a>

## 核心 RL 与数据流程

核心 PPO 示例训练 `reach` 或 `hold` 点任务，和 Go1 等机器人配方使用不同入口。所有输出目录均需使用尚不存在的新路径。

```bash
python -m pip install -e '.[train,mujoco,test]'

# 检查配置和物理依赖
python -m embodiedforge plan --config configs/reach.json
python -m embodiedforge doctor --physics mujoco

# 记录示范数据并检查完整 episode
python -m embodiedforge rollout --config configs/reach.json \
  --physics mujoco --steps 300 --output runs/reach-demo
python -m embodiedforge inspect --dataset runs/reach-demo

# 无图像 PPO 训练与评估
python -m embodiedforge train --task reach --num-envs 32 \
  --updates 100 --output runs/reach-ppo
python -m embodiedforge evaluate --checkpoint runs/reach-ppo/checkpoint.pt \
  --num-envs 32 --steps 300
```

Python 接口示例：

```python
import numpy as np
from embodiedforge import Config, VectorEnv

with VectorEnv(Config(physics="numpy", render="raster", channels=("rgb",))) as env:
    observation = env.observe()
    result = env.step(np.zeros((env.config.num_envs, 2)))
    done_ids = np.flatnonzero(result.terminated | result.truncated)
    env.reset(done_ids)
```

核心任务动作是范围 `[-1, 1]` 的 XY 力（牛顿）；`reach` 为六维 proprio，`hold` 为四维。`result.observation` 保留终止帧，调用方显式重置。RGB 带环境和相机维度；具体规格通过 `env.spec` 查询。接口、数据窗口和扩展方式见 [接口文档](docs/interfaces.md) 与 [实现说明](docs/implementation.md)。

<a id="robot-training"></a>

## 机器人训练与上游接入

按任务复制安装、训练、续训、评估、模型导出和远程 Web 运行命令，见 [训练与部署命令手册](docs/training-deployment.md)。覆盖 reach/hold、Go1、Microduck、H1、Wuji/Light、Cartpole MPC 和机械臂 CEM；列明各任务实际支持的运行与部署方式。

机器人训练 SDK 使用独立环境，避免不同 Warp、Torch、MuJoCo 版本相互覆盖。安装前按任务文档准备固定 revision 的源码检出；本机路径、缓存和 GPU 要求均在对应文档中说明。

| 入口 | 任务与方法 | 实现方式与文档 |
| --- | --- | --- |
| `python -m embodiedforge recipes` | Go1 PPO、Wuji / Wuji Light 重定向 PPO、Cartpole MPC、机械臂投掷 CEM | Go1 已移植；其他为固定源码快照加适配层。[任务配方](docs/recipes.md) |
| `python -m embodiedforge.microduck` | Microduck 平地行走 PPO、评估、回放和 ONNX 导出 | 隔离运行上游 mjlab 流程。[Microduck](docs/microduck.md) |
| `python -m embodiedforge h1-native` | H1 原生训练、续训、固定指令评估与运动记录 | 项目内 CPU 任务与 PPO。[原生 H1](docs/h1-native.md) |
| `python -m embodiedforge h1` | H1 平地行走训练、续训、多种子固定指令评估 | 独立 IsaacLab / Newton / MuJoCo-Warp 配方。[H1](docs/h1-isaaclab.md) |

Go1 托管训练示例（先将源码路径替换为文档要求的干净检出）：

```bash
python -m embodiedforge recipes list
python -m embodiedforge recipes setup --source mjbatch \
  --repo /path/to/mjbatch --python 3.12
python -m embodiedforge recipes train --task go1-joystick \
  --num-envs 512 --updates 600 --threads 4 --timeout 1200 \
  --output runs/go1-train
python -m embodiedforge recipes status --run runs/go1-train
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/go1-train --suite basic --num-envs 32 \
  --seeds 0 1 2 --steps 500 --output runs/go1-evaluation
```

运行完成与策略通过行为验收是两件事。验收阈值、训练预算、续训语义和运动记录选项见各任务文档。历史结果包含尚未通过验收的 Wuji 重定向、H1 长时间站立及部分 Go1 复合指令候选，不能将短测成功视为所有行为均稳定。

<a id="troubleshooting"></a>

## 常见问题

先检查当前环境中的查看器依赖，检查本身不会打开窗口：

```bash
python -m embodiedforge.visualization --viewer web --render-backend mujoco --check
```

`--check` 仅验证安装元数据，不验证 GPU、驱动或实际渲染。检查 OVRTX 时将 `mujoco` 换成 `rtx`。

| 现象 | 处理方式 |
| --- | --- |
| EGL 警告或使用了错误 GPU | 运行 `python -m embodiedforge doctor --graphics egl` 查看实际厂商、设备和原始警告，见 [EGL 诊断](docs/graphics-diagnostics.md) |
| 想用浏览器，却打开了独立窗口 | 使用 `--viewer web --render-backend rtx`；`--viewer rtx` 是原生窗口入口 |
| 页面有画面，但 Go1 不动 | 默认暂停且指令为零；设置“前进”再“继续”。如果回合已结束，先重置所有已终止的环境 |
| 设置 60 FPS 后达不到目标 | 查看实际 FPS、尺寸与单帧大小；先降到 540p / 30 FPS，再逐项提高。渲染、编码及网络都会影响观看效果 |
| 远程浏览器连接不上 | 保持服务运行，确认端口匹配，并使用上文 SSH 转发；默认只监听服务器回环地址 |
| Go1 提示训练目录、模型或 SDK 不兼容 | 使用已完成的托管训练目录及对应环境；按报错核对版本，必要时指定 `--worker-python`，见 [加载要求](docs/go1-live-web.md) |

图形初始化失败、驱动诊断与原生窗口依赖见 [查看器安装说明](docs/viewers.md)。

<a id="validation"></a>

## 验证与当前边界

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check src tests
```

缺少可选 SDK 时对应测试会显式跳过。2026-09-14 的 `ef-viewer` 回归结果为 **496 passed / 55 skipped**；该结果不代表所有训练 SDK 组合都已验收。另有 **22 项真实浏览器检查**覆盖分辨率、帧率、三种渲染器、状态保留及运行中调整。

本机 RTX 5090 D v2、960×540 的 Go1 在线短测中，目标 60 FPS 时实际约 59 FPS，物理推进约 50 步/秒。更高显示 FPS 不产生额外策略动作，也不保证其他模型、分辨率或机器达到相同速度。方法及完整结果见 [帧率验证](docs/viewers.md)。

核心运行时尚未提供 GPU 状态传输、通用机器人资产导入、真实训练相机管线、视觉 PPO、分布式调度或实机控制。机器人查看器的专用 MJCF 桥接不等于通用场景导入；机器人配方也尚未统一到核心 `VectorEnv`。统一 Web 当前不提供训练任务管理控制台。

<a id="documentation"></a>

## 文档与研究记录

详细文档目前以中文为主。英文主页覆盖同样的主要能力、安装方式与运行命令。

| 主题 | 文档 |
| --- | --- |
| 训练与部署命令 | [按任务运行手册](docs/training-deployment.md) · [English](docs/training-deployment.en.md) |
| 架构与接口 | [目标架构](docs/architecture.md) · [接口与模块](docs/interfaces.md) · [实现与扩展](docs/implementation.md) |
| 查看器 | [安装与后端](docs/viewers.md) · [机器人回放](docs/robot-web-replay.md) · [Go1 在线控制](docs/go1-live-web.md) |
| 训练入口 | [任务配方](docs/recipes.md) · [Microduck](docs/microduck.md) · [IsaacLab H1](docs/h1-isaaclab.md) |
| Go1 实验 | [转向精度](docs/go1-yaw-study.md) · [横移与急停](docs/go1-response-study.md) · [复合指令](docs/go1-maneuver-study.md) · [能力保留](docs/go1-preservation-study.md) · [响应与计算优化](docs/go1-fast-response-study.md) |
| 工程说明 | [Newton 兼容范围](docs/newton.md) · [日志设计](docs/logging.md) |

依赖、机器人资产和移植代码遵循各自许可证；Go1 移植部分保留了 [mjbatch 许可证](src/embodiedforge/locomotion/LICENSE.mjbatch)。
