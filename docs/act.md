# LeRobot ACT 模仿学习

ACT 是可选后端，核心包不导入 LeRobot。当前支持核心 `VectorEnv` 的
`reach` / `hold` 示范记录、LeRobotDataset 导出、上游 ACT 离线训练和仿真闭环评估。
机械臂、实机与 Go1/H1 的专用环境尚未接入这条数据路径。

## 实现归属与仓库依赖

当前采用 **LeRobot 包依赖 + EmbodiedForge 适配层**，尚未移植 ACT 模型或训练器。

| 部分 | 实际维护与执行来源 |
| --- | --- |
| ACT Transformer/CVAE、损失计算、策略 checkpoint | LeRobot `policies.act` |
| 训练循环、优化器调度、训练 checkpoint | LeRobot `scripts.lerobot_train` |
| LeRobotDataset、episode padding、归一化处理器 | LeRobot |
| 示范记录、字段映射、数据导出入口 | EmbodiedForge |
| 独立进程调度、仿真评估、动作块执行和来源记录 | EmbodiedForge |

最初验证使用 `.cache/act-lerobot-source` 的 editable 安装，运行时需要该源码副本。
现在本机 `.cache/act-venv` 已改成 **wheel 安装**：ACT 从环境的 `site-packages/lerobot`
加载，运行无需原始仓库、缓存源码副本或 wheel 文件；仍需安装 LeRobot 及其依赖。
仓库地址只用于构建与来源追踪，代码未硬编码 `/home/ubuntu/workspace/3rdparty/lerobot`。

可从 `ef` 查询当前真正加载的实现：

```bash
python -m embodiedforge act --python .cache/act-venv/bin/python doctor
```

报告包含 `implementation=lerobot`、`native_act_implementation=false`、安装模式，
以及模型、训练器、数据集、处理器的实际路径和源码 SHA256。
`installation.mode=installed_package` 表示这些入口都位于安装包目录；
`external_source` 表示至少一个入口来自目录外，应检查 editable 安装或 `PYTHONPATH`。
`build_origin` 中的本地 wheel 路径只代表构建输入，不是运行依赖。
这是入口来源检查，不覆盖每一项间接依赖。

### 是否需要移植

当前建议保留上游 ACT 实现，继续完善 EmbodiedForge 的实验接口。
移植会增加模型、损失、数据归一化、checkpoint 和训练状态的维护工作；
目前没有测得必须通过移植解决的性能问题。

| 实际需求 | 建议 |
| --- | --- |
| 离线运行、无需仓库检出 | 使用现有 wheel 安装，已验证 |
| 使用 `ef` 发起训练与评估 | 保留独立 ACT 进程，已支持 |
| 续训、定期保存、学习率实验 | 扩展适配入口，已支持 |
| 部署端严格只能使用 Python 3.10，且禁止独立进程 | 再评估提取推理模型及归一化逻辑 |
| 长期修改 ACT 网络或损失且无法通过上游扩展实现 | 再评估局部移植或维护固定 fork |

若将来提取推理实现，应先固定 checkpoint 格式，并对同一观测的归一化结果、
动作块、重置语义和闭环成功率做一致性验证。现阶段没有原生 ACT 实现。

## 独立环境

接入针对本地 LeRobot **0.6.2** 源码 revision
`89236ea0f4f81a81ca566081e20dd1ff5f823cbe`，要求 Python 3.12+。
在独立环境安装本地仓库，避免改变现有查看器或 RL 环境。
以下命令从 EmbodiedForge 根目录执行；仓库路径可替换。
主入口使用现有 `ef` 环境（本机 Python 3.10），ACT SDK 在独立 Python 3.12
子进程中运行。无需升级 `ef`。

```bash
conda activate ef
uv venv --python 3.12 .cache/act-venv
# 本机 CPU 验证组合；使用 GPU 时自行选择匹配的 CUDA wheels，跳过 CPU 约束。
uv pip install --python .cache/act-venv/bin/python \
  'torch==2.9.0+cpu' 'torchvision==0.24.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .cache/act-venv/bin/python \
  -c configs/act-cpu-constraints.txt \
  '/home/ubuntu/workspace/3rdparty/lerobot[training]'
```

Torch、torchvision、torchcodec 的版本需要相互兼容；GPU 训练应在该环境安装
匹配的 CUDA 版 Torch。核心环境通过 `act --python` 调用独立解释器，
无需在核心环境安装 LeRobot，也无需在独立环境安装 EmbodiedForge。
`--python` 放在 `doctor/export/train/resume/evaluate/benchmark/checkpoints/compare/bundle` 子命令前。
其中 `checkpoints`、`compare` 和 `bundle` 仅处理本地文件，可直接在 `ef` 执行，无需指定独立解释器。
安装命令使用普通安装，源码仅在构建时需要；如改用 `-e`，运行将依赖对应源码目录。
原始第三方仓库未修改。本机保存的 wheel 为
`.cache/act-wheels/lerobot-0.6.2-py3-none-any.whl`，旁边的 `build-provenance.json`
记录构建来源 revision、wheel SHA256 和关键源码哈希。
CPU 约束还固定了 multiprocess，避免其新版在本机 Python 3.12.0 退出时的异常。

## 示范 → 数据集 → 训练 → 评估

所有输出目录必须是新目录。`local/reach-act` 只是数据集标识，不会创建 Hub 仓库。

```bash
# 点任务自带的专家控制器采集示范；默认 control_hz=50。
python -m embodiedforge rollout --task reach --num-envs 16 \
  --steps 1000 --output runs/act-demos

python -m embodiedforge act --python .cache/act-venv/bin/python export \
  --source runs/act-demos --output runs/act-dataset --repo-id local/reach-act

python -m embodiedforge act --python .cache/act-venv/bin/python train \
  --dataset runs/act-dataset --output runs/act-train \
  --device cpu --steps 10000 --batch-size 8 --chunk-size 20 --action-steps 5 \
  --save-freq 1000 --learning-rate 0.00001

python -m embodiedforge act --python .cache/act-venv/bin/python evaluate \
  --run runs/act-train --num-envs 32 --steps 300 --seed 2001 \
  --output runs/act-train/evaluation-2001.json
```

先验证接口时，训练使用 `--small-model --steps 5`，并设置
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 限制 CPU 线程。
`--small-model` 缩小 Transformer 并禁用 ImageNet 权重下载，只用于流程验证。
短训练完成不代表策略已学会任务。正式训练可选择 `--device cuda`。
训练关闭 Hub 上传及 W&B；图像策略默认可能下载 ResNet 的 ImageNet 初始化权重。
当前封装仅训练由此导出器生成的本地数据集。
省略 `--learning-rate` 时沿用上游 ACT 优化器预设；该参数调整主参数组，
图像 backbone 仍沿用上游独立学习率。`--save-freq` 默认 1000，
设为 0 则只在结束时保存；最后一步始终保存。

## 续训

使用 LeRobot 的训练状态恢复，包括模型、优化器、随机数状态及上游保存的训练进度；
参见[上游续训说明](https://huggingface.co/docs/lerobot/il_robots)。
输出为新目录，原始运行保持不变。

```bash
python -m embodiedforge act --python .cache/act-venv/bin/python resume \
  --run runs/act-train --output runs/act-train-continued --steps 20000
```

`--steps` 是累计目标步数。例如原 checkpoint 为 10,000 步，以上命令再训练
10,000 步。模型结构、batch size、学习率和保存频率继承 checkpoint；
可用 `--device cpu|cuda` 选择执行设备，默认继承原设备。
跨设备不承诺逐位复现。

源运行必须有有效的运行记录和可用的 checkpoint，默认选择 `checkpoints/last`；支持有完整 checkpoint
的已完成、失败及强制中断运行。缺少优化器/RNG 文件、数据集 manifest 或已记录的数据内容改变、
目标步数不大于已保存步数、输出目录已存在时会拒绝续训。
续训报告的 `resumed_from` 记录来源目录、实际 checkpoint、原步数和模型 SHA256。
Hub 上传与远程作业保持关闭。

### 续训的数据一致性

新训练启动前会对 `embodiedforge.json`、`meta/`、`data/` 以及存在的
`images/`、`videos/` 逐文件计算 SHA256，保留相对路径和字节数。
运行记录及旁边的启动记录均包含 `dataset_fingerprint`，强制中断后也可核验。
续训重新读取文件内容：数据分片、归一化统计或媒体文件被修改、增加、删除、重命名时，
会在启动训练前拒绝，并列出变化的路径。仅改时间戳不影响内容指纹。

根目录的 README、日志和 `.cache/` 不参与指纹；受检目录中的符号链接会被拒绝，
以免漏检链接指向的数据。哈希按块读取，不会把整个数据文件加载到内存，
但启动前需要完整读取受检文件，耗时随数据量增加。
这是启动前的一致性检查，不是数据快照或训练期间的写锁，训练期间仍应保持数据集不变。

旧运行没有历史指纹时保留续训兼容性，终端会提示无法核验历史数据，
并在 `resumed_from.dataset_verification.status` 中记录 `unverified_legacy`。
该次续训以当前数据建立新基线；后续匹配时记录 `verified`，
这只证明与直接来源运行的基线一致，不能追溯证明旧训练使用的数据未变。
正常的新运行续训在此字段记录 `verified` 及数据指纹。

### 查看与选择 checkpoint

先在 `ef` 中列出已停止运行的保存记录，再选择历史步数进行评估或续训：

```bash
python -m embodiedforge act checkpoints --run runs/act-train

python -m embodiedforge act --python .cache/act-venv/bin/python evaluate \
  --run runs/act-train --checkpoint 5000 --num-envs 32 --steps 300 --seed 2001

python -m embodiedforge act --python .cache/act-venv/bin/python resume \
  --run runs/act-train --checkpoint latest \
  --output runs/act-train-recovered --steps 20000
```

`evaluate`、`benchmark` 和 `resume` 均接受 `--checkpoint`：

| 取值 | 行为 |
| --- | --- |
| `last`（默认） | 严格使用 `checkpoints/last`，文件缺失或检查失败时报错 |
| 步数，例如 `5000` | 使用对应数字目录，兼容 `005000` 这样的前导零 |
| `latest` | 按步数从大到小选择通过检查的目录，报告记录跳过目录及原因 |

`latest` 也可用于 `last` 链接缺失的情况，不会重写链接。它表示最近可用的保存步数，
不是成功率最高的模型。评估报告的 `checkpoint_selection` 和续训报告的
`resumed_from.selection` 保留选择方式、实际步数及跳过原因。

列表区分 `inference_ready` 和 `resume_ready`：推理检查模型、配置、前后处理器及其
引用的归一化文件；续训额外检查训练配置、步数、优化器参数组、优化器状态与 RNG。
检查覆盖 JSON、文件存在性及 safetensors 头部/字节区间，可发现缺失文件或截断保存。
字节布局依据 [safetensors 格式](https://github.com/safetensors/safetensors#format)；
这里不读取张量值，也不完整验证模型参数、dtype/shape 与上游配置语义，实际加载仍可能失败。

列表、评估与续训都要求源训练已停止。默认 `last` 评估继续要求运行正常完成；
显式指定步数或 `latest` 可以评估失败或中断运行中已经保存的模型。

## 独立推理包

将选定的 checkpoint 导出为可移动的目录，可单独复制到另一台已配置推理环境的机器：

```bash
# 在 ef 中打包，无需加载 Torch 或 LeRobot。
python -m embodiedforge act bundle \
  --run runs/act-reach-5000-20260916 --checkpoint 5000 \
  --output runs/act-reach-inference

python -m embodiedforge act --python .cache/act-venv/bin/python evaluate \
  --bundle runs/act-reach-inference --num-envs 32 --steps 300 --seed 2001 \
  --output runs/act-reach-inference/evaluation.json

python -m embodiedforge act --python .cache/act-venv/bin/python benchmark \
  --bundle runs/act-reach-inference --num-envs 64 --seeds 2001 2002 2003 \
  --output runs/act-reach-inference/benchmark.json
```

目录内包含 `act-bundle.json` 和 `pretrained_model/`。元数据保留环境配置、字段映射、
来源与推理指纹；模型目录仅复制权重、策略配置、前后处理器及它们引用的状态文件。
复制得到普通文件，不依赖 `checkpoints/last` 链接，也不携带优化器、RNG 或训练数据。
`bundle` 接受与评估相同的 `--checkpoint last|latest|步数` 选择方式和源运行状态检查。

导出目录必须尚不存在；文件先复制到临时目录，通过指纹核验后再发布，复制失败不留下
可用输出。移动时复制整个推理包目录。评估前会核验包内文件，发现指纹不匹配就拒绝加载。
评估报告保留推理指纹及包 manifest 的 SHA256。包内历史来源路径仅用于追踪，
不会被用于加载原训练目录或数据集。

`evaluate` / `benchmark` 的 `--bundle` 与 `--run` 二选一；保存步数在导出时确定，
使用包时不再选择 checkpoint。独立包用于推理，续训仍使用原训练状态及数据集。
目标机器仍需 EmbodiedForge 和兼容的 LeRobot 推理环境；这是文件打包，不包含 Python 依赖，
也没有移植 ACT 实现。当前闭环入口仍适用于核心 `reach` / `hold`，不是实机部署接口。

## 强制中断恢复

Linux 训练启动前会在输出目录旁保存两份辅助文件。例如输出为 `runs/act-train`：

- `runs/.act-train.act-run.json`：启动记录，含数据映射、命令和实现来源。
- `runs/.act-train.act.lock`：启动器与上游训练进程共同持有的文件锁。

LeRobot 要求新训练的输出目录尚不存在，因此启动记录放在目录旁。
正常退出时，记录同时写入输出目录内的 `act-run.json`；即使上游在创建目录前失败，
旁边的失败记录仍会保留。已使用过的输出名称不会自动覆盖，请使用新目录。

如果启动器遭到强制终止，`resume` 会读取旁边的启动记录。
只有锁已释放、训练进程确实退出且 checkpoint 文件齐全时才允许恢复；
仅结束启动器而上游训练仍在运行时，锁继续生效，续训会明确拒绝。
遗留的 `running` 状态在新续训记录中记为 `resumed_from.source_status=interrupted`，
不会把原运行标成训练完成，也不会修改原 checkpoint。
若要移动已停止但尚未正常收尾的运行，请同时保留上述两个辅助文件；
缺少锁文件时无法验证训练是否退出，会拒绝自动恢复。
该保护适用于经 EmbodiedForge 入口启动的训练。

## RGB 路径

采集时增加 `--render raster`，导出时增加 `--images`；其余流程一致。
图像来自训练传感器，不是 Web 查看器截图。当前核心运行时只有一个相机。

| 记录字段 | ACT 数据字段 |
| --- | --- |
| 纯状态 `obs/proprio` | `observation.state` 和 `observation.environment_state` |
| 含图像时 `obs/proprio` | `observation.state` |
| `obs/rgb[:, camera]` | `observation.images.camera_0` 等 |
| `action` | `action`，保持任务定义的物理单位 |

导出逐帧使用动作执行前的观测，不使用 `next/*`。仅导出完整 episode；
未完成尾段仍由原记录器的 `run.json` 标记。采样频率等于控制频率，
低频相机帧保持到下一次相机采样，不进行插值。图像以 PNG 特征存储；
大规模数据的视频编码优化留待后续。
动作窗口及 episode 尾部 padding 由 LeRobot 处理。
点任务 proprio 包含任务状态；纯状态 ACT 基线不等同于视觉机械臂策略。
标准 ACT 不使用数据集中的语言指令作为模型输入。
此 LeRobot revision 的 ACT 内部仍会访问 `observation.state`，因此纯状态模式
将完整 proprio 同时提供给两个输入；后续机械臂适配应明确区分机器人与环境状态。

## 推理与验收

`ACTAdapter` 加载 checkpoint 的预处理与后处理，完成图像 HWC→CHW、
0–255→0–1、状态归一化和动作反归一化。
每次预测 `chunk_size` 步，仅执行前 `action_steps` 步，再重新观察规划。
`ActionExecutor` 管理每个环境的动作缓存，在 episode 重置时丢弃旧动作。
适配器直接调用 `predict_action_chunk()`，不使用 LeRobot 的内部动作队列。
暂不支持 temporal ensembling，加载此类 checkpoint 会明确报错。

预测动作显式裁剪至任务动作范围；评估报告包含裁剪比例、成功率、平均步奖励、
评估种子及 checkpoint SHA256。较高裁剪比例应作为策略质量问题检查。
评估恢复采集时的物理、相机与控制配置，仅允许改变环境数量和种子，
并检查任务、场景、动作维度、单位和控制周期。应使用多个未参与采集的种子验收。

### 评估的模型身份与复现信息

`checkpoint_sha256` 保留原有的权重文件哈希。新增 `checkpoint_fingerprint`
包含 `config.json`、`model.safetensors`、前后处理器 JSON 及它们引用的状态文件，
逐文件记录 SHA256，并生成整体指纹。归一化统计、动作执行步数或处理器配置变化时，
即使权重未变，也会得到不同的推理指纹。仅移动保存目录不会改变指纹；
训练状态、优化器、未被引用的文件不参与此指纹。

适配器在加载前后各计算一次指纹，检测到变化就拒绝加载；评估报告使用加载时保存的
指纹，避免在评估结束后读到被替换的权重而错误标记已完成的推理。
哈希按块读取；前后检查不是文件快照或写锁，加载期间仍应保持 checkpoint 不变。

`evaluate.environment_config` 和 `benchmark.per_seed[].environment_config`
记录实际使用的环境配置（包含评估种子与环境数量）。两种报告的 `inference_settings`
记录实际设备、参数 dtype、动作块长度、执行步数、Torch/NumPy 版本、线程数，
以及确定性算法和 cuDNN 设置；禁用预训练 backbone 下载等运行时覆盖也会记录。
这些信息用于定位评估差异，不承诺跨硬件、依赖版本或 CPU/CUDA 的逐位一致。

### 固定回合对照评估

比较策略与专家时，推荐使用 `benchmark`：每个种子下，每个环境只完成一回合，
结束后冻结，不再生成替代回合。ACT 与专家使用相同初始观测，报告核对其 SHA256，
并按原环境行顺序保留逐回合成功标记、回报和长度。结束的环境不再进入策略推理。

```bash
python -m embodiedforge act --python .cache/act-venv/bin/python benchmark \
  --run runs/act-reach-5000-20260916 --num-envs 64 --seeds 2001 2002 2003 \
  --output runs/act-reach-5000-20260916/benchmark.json
```

此命令对 ACT 和专家各评估 192 个对应回合，最长运行到采集配置中的 `max_steps`，
汇总成功数、成功率和平均 episode 回报。种子必须非负且不重复。
输出 JSON 包含 `per_seed` 明细、模型哈希、实现来源及预测动作裁剪比例。
当前要求任务提供 `expert_action`；适用于现有核心 `reach` / `hold`。

原 `evaluate --steps` 保留固定步数、结束后重置的语义，适合持续运行检查。
两种口径的样本数和初始状态集合不同，结果应分别比较。

### 比较两次训练的逐回合变化

对基线与候选模型分别运行 `benchmark`，保持 `--num-envs` 与种子集合一致，
再在 `ef` 中离线比较报告，无需加载模型或 LeRobot：

```bash
python -m embodiedforge act compare \
  --baseline runs/act-state-smoke-20260916/benchmark-comparable-20260916.json \
  --candidate runs/act-reach-5000-20260916/benchmark-comparable-20260916.json \
  --output runs/act-reach-5000-20260916/comparison.json
```

比较按 `(seed, env_id)` 配对，允许两份报告中的种子顺序不同。先核对完整环境配置、
环境数量、步数上限、初始观测哈希与专家逐回合成功/回报/长度；任一条件不一致就拒绝。
`evaluate` 的固定步数报告不能用于此命令。旧 benchmark 缺少逐种子的完整环境配置时，
需重新运行 benchmark，原报告不会被改写。

统计直接从 `episode_success/return/length` 重算，不使用输入报告中的汇总值。
差值方向始终是 **候选减基线**；输出包括成功率变化（百分点）、平均回报/长度差值，
以及四类回合数：`improved`（仅候选成功）、`regressed`（仅基线成功）、
`both_succeeded`、`both_failed`。逐种子报告保留每个 `env_id` 的成功标记和差值，
可定位总体成功率掩盖的局部退步。回合更短不一定代表策略更好，需要结合成功标记判断。

输出记录两份输入报告的 SHA256、模型指纹及推理设置，已有输出文件不会覆盖。
结果描述的是这些回合的观测差异，不提供显著性检验，也不能把多项训练条件同时变化
导致的差异归因于某一个参数；正式选择模型仍应保留独立验收种子。

数据集中的 `embodiedforge.json` 保留来源 run ID 和字段映射；训练目录的
`act-run.json` 记录命令、Python、依赖版本、ACT 源文件路径及哈希。
新训练记录与评估报告还包含 `runtime`，明确标识实现归属、安装模式和四个上游入口。
已有历史报告保持原样；对旧 checkpoint 重新评估会写入当前推理环境的来源。
这是来源记录，不是全部间接依赖锁或策略质量证明。

```bash
# 核心环境运行时 SDK 测试会跳过；完整验证需使用 ACT 环境。
uv pip install --python .cache/act-venv/bin/python pytest
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .cache/act-venv/bin/python -m pytest -q \
  tests/test_act.py tests/test_act_checkpoints.py tests/test_act_dataset.py tests/test_rollout.py \
  tests/test_data_policy.py tests/test_dataset_reads.py tests/test_act_compare.py \
  tests/test_act_bundle.py
```

## 本机验证结果（2026-09-16）

主入口为 `ef`，训练使用独立 CPU 环境与上述约束。相关测试共 117 项通过，
包括状态/RGB 数据往返、episode 尾部 padding、保存后重新加载归一化统计、
子集环境推理、真实训练及闭环评估、失败导出不发布和环境单位不匹配拒绝。
测试使用 wheel 安装并设置 `HF_HUB_OFFLINE=1`；此前也已验证缓存源码副本不可用时
数据导出、训练和推理正常。
来源诊断测试还核对了实际加载模块及其文件哈希。
续训扩展验证包括定期 checkpoint、状态/RGB 模型从 2 步恢复至 4 步、
优化器步数恢复、原模型不变，以及无效续训请求拒绝。
固定回合测试检查不同时间结束的环境冻结、推理时排除结束行、动作块顺序、
超时计数、相同初始观测、逐回合统计及重复种子拒绝。
中断测试实际强制结束启动器，确认仍在运行的上游训练进程持有锁、拒绝并发续训；
随后结束训练进程，从已保存 checkpoint 继续两步，并核对优化器步数。
还覆盖创建输出目录前失败时保留记录，以及锁文件缺失时拒绝恢复。
checkpoint 选择测试覆盖模型截断时显式回退、推理/续训文件要求差异、归一化文件缺失、
步数不一致，以及状态/RGB 模型选择历史第 1 步评估并续训至第 3 步。
中断运行可显式选择 `latest` 评估；默认 `last` 仍保持严格检查。
真实 5,000 步模型通过 `ef` 使用 `--checkpoint 5000` 重新评估后，seed 2001 的
模型哈希、99 个完整回合、69.7% 成功率、奖励及动作裁剪比例均与原报告一致。
数据指纹测试覆盖状态/RGB 数据分片、归一化统计、媒体内容、文件增删和重命名，
其中等大小修改并恢复 mtime 仍被拒绝；修改数据后的续训不会产生新的启动记录。
真实旧运行从 5,000 步续训到 5,002 步时记录 `unverified_legacy`，
再从新基线续训到 5,003 步时记录 `verified`，两次均通过 `ef` 入口完成。
推理指纹测试验证状态/RGB 模型在权重相同但反归一化统计改变时，动作和整体指纹
都会变化；加载中修改配置会被拒绝。状态/RGB 的重复评估及固定回合对照报告一致，
报告写入 JSON 后与内存结果一致。
真实 5,000 步模型经 `ef` 在两个独立进程中重复评估，完整报告一致，
seed 2001 的指标也与原报告一致；没有改变策略行为。
比较测试覆盖相同总成功率下的逐回合退步、种子重排、原始回合重算、输入来源哈希、
配置/初始观测/专家不一致拒绝、非法回合数据拒绝，以及 `ef` CLI 输出不覆盖。
推理包测试将状态/RGB 包移到新目录，并暂时移走原训练目录和数据集；包内策略的
评估指标、逐回合对照结果及推理指纹均保持一致。还验证复制失败不发布、文件修改后
在模型加载前拒绝、已有输出和悬空链接不覆盖。
真实 5,000 步模型的包内模型目录有 6 个文件、共 509,171 字节；
从包评估 seed 2001 得到相同的 99 个完整回合及 69.7% 成功率。

纯状态 `reach`、NumPy 物理、每个种子 32 个环境运行 300 步：

| 策略 | seed 2001 | seed 2002 | seed 2003 |
| --- | --- | --- | --- |
| ACT 5 步，16 条示范 | 0% | 0% | 0% |
| ACT 3,000 步，16 条示范 | 51.5% | 47.5% | 59.0% |
| ACT 5,000 步，189 条示范 | 69.7% | 62.0% | 76.2% |
| 专家控制器 | 75.0% | 67.3% | 78.4% |

这些是**完整 episode 成功率**；不同策略的完成 episode 数可能不同。
训练种子为 1001，采集种子为 42。小模型使用 `chunk_size=5`、
`action_steps=2`；3,000/5,000 步训练的 batch size 为 32，5 步为 8。
数据量与训练预算同时变化，因此此表验证流程和学习效果，不是单变量对照实验。
5,000 步模型的预测动作裁剪比例为 13.6%–14.5%，仍需继续检查边界处的拟合。

新增的固定回合评估使用 seeds 2001/2002/2003，每个种子 64 个环境：

| 策略 | 成功回合 / 总回合 | 成功率 |
| --- | --- | --- |
| ACT 5 步 | 3 / 192 | 1.56% |
| ACT 5,000 步 | 132 / 192 | 68.75% |
| 专家控制器 | 139 / 192 | 72.40% |

三个策略的初始观测哈希逐种子一致。该结果使用一回合后冻结的口径，
应与上面的 300 步运行结果分开解读。当前仍仅验证了 NumPy 点任务状态策略。
用当前入口重新生成两份 benchmark 并执行 `compare` 后，5,000 步模型相对 5 步基线：
192 个配对回合中有 129 个从失败变成功、0 个从成功变失败、3 个共同成功、60 个共同失败；
成功率增加 67.1875 个百分点。这里比较的是已有模型，本轮没有重新训练策略。

本机产物（新检出不包含）：

- 189 条示范、15,371 帧：`runs/act-reach-dataset-20260916`。
- 训练与 checkpoint：`runs/act-reach-5000-20260916`。
- 三份报告：该训练目录内 `evaluation-2001.json`、`evaluation-2002.json`、
  `evaluation-2003.json`。
- 含完整推理指纹的重复评估：该训练目录内 `evaluation-inference-c.json`、
  `evaluation-inference-d.json`。
- 续训验证：`runs/act-reach-resume-5010-20260916`，从原 5,000 步恢复到 5,010 步。
- 数据指纹兼容验证：`runs/act-reach-fingerprint-5002-20260916` 与
  `runs/act-reach-fingerprint-5003-20260916`。
- 固定回合报告：`runs/act-reach-5000-20260916/benchmark-paired-20260916.json`；
  5 步基线报告位于 `runs/act-state-smoke-20260916/benchmark-paired-20260916.json`。
- 新比较输入：上述两个训练目录下的 `benchmark-comparable-20260916.json`；
  比较结果：`runs/act-reach-5000-20260916/comparison-vs-5-step-20260916.json`。
- 独立推理包：`runs/act-reach-inference-20260916`，包含 `evaluation-2001.json` 验证报告。

该结果仅说明点任务状态模仿学习有效。RGB 路径完成了接口与短训练验证，
尚未验证视觉学习效果；没有进行机械臂或实机验收。
