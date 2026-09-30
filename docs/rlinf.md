# RLinf 接入：ManiSkill PickCube + MLP + SAC

EmbodiedForge 增加了可选的 `rlinf` 入口。`ef` 环境负责启动、输入快照与结果归档；独立 SDK 环境执行 RLinf 原生训练和评估。实现依赖外部 RLinf 仓库，没有移植或修改其算法源码，也没有把 Ray、ManiSkill、Torch 加入 EmbodiedForge 的基础依赖。

当前固定使用本地已检查的提交 `67864d67c086a111de9921c5f4591eb5882f74ec`，配置为 `examples/embodiment/config/maniskill_sac_mlp.yaml`。启动时检查提交及已跟踪文件；下载资产、虚拟环境等未跟踪文件可以保留。RLinf 使用 Apache-2.0 许可证，源码及许可证仍在外部仓库维护。

当前范围是单机、单张 NVIDIA GPU、Panda 机械臂的 `PickCube-v1`，支持从零训练、完整 SAC 状态续训、独立仿真评估，以及确定性 actor 的 CPU ONNX 导出与仿真闭环评估。训练和评估默认 headless，不打开交互窗口，不保存视频；仍需要 ManiSkill/SAPIEN 所需的 GPU 驱动和图形运行库。导出在 CPU 上执行，不启动 Ray 或仿真。当前入口不包含 VLA 后训练、多机调度、Go1/Microduck 任务或真机部署。

## 环境准备

主环境继续使用 `ef`。RLinf 安装在独立虚拟环境，不要在 `ef` 中安装其依赖。以下使用上游环境安装器，包含 ManiSkill/Libero 配套依赖与资产，可能需要系统依赖安装权限：

```bash
cd /home/ubuntu/workspace/3rdparty/RLinf
git rev-parse HEAD
# 应为 67864d67c086a111de9921c5f4591eb5882f74ec
bash requirements/install.sh embodied \
  --env maniskill_libero \
  --venv .venv-ef-sac
```

安装器支持不指定大模型；不需要为了 MLP 安装 OpenVLA 权重。按该固定提交的安装脚本准备 SDK；其默认 Python 为 3.11.14，要求 `hydra-core<1.4.0.dev8`。不要复用当前安装了 Hydra 1.4.0.dev8 的 IsaacLab 环境，那个版本不兼容上游的 `version_base="1.1"`。系统依赖已经齐全时可以使用上游的 `--no-root`。

下列训练命令均回到 EmbodiedForge 目录执行。入口的 `--python` 必须指向实际 SDK Python；适配器会保留虚拟环境的 Python 符号链接路径。对带有 `pyvenv.cfg` 的环境，会在子进程中先加载该环境的 `bin/activate`，使上游安装器添加的 Vulkan/动态库设置生效；不会改变调用者的 `ef` 环境。激活失败时不会继续启动工作进程。

本机本轮验证的 SDK 位于 `.cache/rlinf-sac-venv/bin/python`，下列命令的 `--python` 可替换为这个路径。它在独立虚拟环境中只读继承现有 Torch/Ray，并在自己的目录安装匹配的 ManiSkill、SAPIEN 和 Hydra；不直接使用原 IsaacLab 环境，也未改动 `ef`。这是本机依赖叠加环境，不是通用安装锁文件，实际版本和训练结果见 [GPU 实验记录](rl-training-study-20260926.md)。

SDK 从运行目录的 `implementation/embodiedforge` 加载本地适配代码快照，并加载指定的上游源码；NumPy、Torch 等依赖由 SDK 自己提供。从 wheel 安装 EmbodiedForge 时，也不会把主环境整个 `site-packages` 加入 SDK 的导入路径。该隔离同样适用于 SDK 派生的 Python 子进程。`request.json` 保存 Python、XML 和许可证文件的 SHA256，SDK 启动前校验这些文件；运行期间继续编辑工作区不会改变本次任务的适配代码。历史运行不会自动补充快照。

## 训练

`--seed` 同时设置训练/评估环境、回放采样及各 worker 的 Python、NumPy、Torch
随机流。适配器通过上游的 worker 扩展入口，在模型构造前按 `seed + rank` 初始化；
当前单卡配置各组均为 rank 0。仅写入上游 `actor.seed` 不会自动固定 MLP 初始权重
和 rollout 探索动作，因此这里显式补齐进程内初始化。该设置不会强制 GPU 算子确定性，
也不表示续训恢复了仿真状态或 rollout RNG；下文仍保留续训的复现边界。

```bash
conda activate ef
cd /home/ubuntu/workspace/chase/EmbodiedForge

python -m embodiedforge rlinf train \
  --repo /home/ubuntu/workspace/3rdparty/RLinf \
  --python /home/ubuntu/workspace/3rdparty/RLinf/.venv-ef-sac/bin/python \
  --output runs/rlinf-pickcube-sac \
  --gpu 0 --headless --quiet \
  --num-envs 32 --seed 1234 \
  --iterations 8000 --save-interval 200
```

`--iterations` 是 RLinf 的采样/训练循环次数，不是环境步数，也不是单次梯度更新数。保留上游的 `algorithm.update_epoch=32`、训练每轮 2 个环境步、batch size 1024 等设置。每 200 次循环以及最后一次循环进行评估并保存完整检查点；训练中的评估使用 16 个并行环境，每轮 50 步。默认评估间隔跟随 `--save-interval`，也可显式指定 `--eval-interval`。例如 `--save-interval 1000 --eval-interval 500` 每 500 轮评估、每 1000 轮保存，最后一轮始终评估并保存。两个间隔必须为正数，且保存间隔必须是评估间隔的整数倍，这是固定上游的约束。比较不同保存频率时应显式固定评估间隔，不能只固定训练种子。

新任务在环境 worker 的整次评估调用外保存并恢复 Torch CPU 与评估 GPU 的随机状态，避免 ManiSkill 自动重置消耗后续训练使用的 CUDA 随机流。请求记录 `evaluation_rng=isolated-torch`，训练和 PyTorch 评估结果也记录该字段。这不修改评估内部的采样、不重设种子，也不改变训练损失、奖励或网络。隔离范围针对当前同步、单卡配方；独立 ONNX 评估不与训练共用进程。

历史请求缺少此字段时保留 `upstream` 行为；通过当前 CLI 从历史检查点新建续训则采用隔离行为。因此旧版与新版的学习轨迹可能不同，不能把“算法配置相同”解释为逐步复现旧轨迹。问题复现和配对验证见[评估随机状态隔离记录](rl-evaluation-isolation-20260929.md)。

`--gpu` 指定一个物理 NVIDIA GPU 的非负整数索引，并覆盖继承的 `CUDA_VISIBLE_DEVICES`。该参数同时设置上游 actor、rollout、env 的 `component_placement`，因为 RLinf 会根据物理硬件放置重新设置工作进程的 GPU 可见性，单设驱动进程的环境变量不足以选卡。此固定版本的适配入口不接受 UUID/MIG 标识。不要用 `torchrun` 包裹此入口。每次运行建立自己的本地 Ray 实例，将实际地址传给工作进程，结束时关闭该实例；不会自动连接已有的 Ray 集群。

这个单卡入口向 Ray 声明最多 4 个 CPU 调度资源，避免按整机核心数预启动大量闲置 worker。它不限制 Torch 或仿真的实际 CPU 线程数，也不关闭 Ray 内存保护。保留上游的 dashboard 启动设置；适配器的 worker 异常清理不依赖 dashboard State API。GPU 显存有余量不代表主机内存充足；在共享机器上先串行执行训练、续训和评估。

每次 `--output` 都必须是新目录。没有冒烟、启动试训、自动缩短训练预算等模式。

训练和评估均可加 `--quiet`，将工作进程输出直接写入 `worker.log`，终端只显示日志路径和最终结果目录。指标、保存点及错误退出状态保持相同；不加时同时输出到终端和日志。长时间训练或后台运行建议加上该参数。

Ctrl+C、SIGTERM 和未被忽略的 SIGHUP 会转发为工作进程组的 SIGINT，并等待主进程清理，最长 10 秒；随后清理本次组内残留进程。主进程正常退出、失败或日志写入失败时也会清理残留进程，包括重定向输出的后台子进程。RLinf 工作进程自身的正常收尾负责关闭本次 Ray 实例。worker 异常触发上游 SIGUSR1 时，适配器保留日志中的原始错误，通过同一收尾路径关闭本次实例；不再通过可选 dashboard API 枚举 actor。清理期间忽略重复失败信号，结束后恢复之前的处理器。实际 CPU 故障与独立 Ray 作业隔离验证见 [后训练与恢复记录](rl-posttraining-study-20260926.md)。

## 续训

```bash
python -m embodiedforge rlinf train \
  --repo /home/ubuntu/workspace/3rdparty/RLinf \
  --python /home/ubuntu/workspace/3rdparty/RLinf/.venv-ef-sac/bin/python \
  --output runs/rlinf-pickcube-sac-resume \
  --gpu 0 --headless --quiet \
  --resume runs/rlinf-pickcube-sac/maniskill_sac_mlp/checkpoints/global_step_8000 \
  --iterations 2000 --save-interval 200
```

这会从 8000 追加到 10000；适配器同时调整上游 `max_steps` 和 `max_epochs`，不会被原配置的 8000 上限截断。传入整个 `global_step_N` 目录：SAC 需要模型、优化器、学习率调度器、熵温度、目标网络以及回放缓冲区。单独的 `full_weights.pt` 可用于评估和导出，不能完整续训。

续训使用本适配器生成的检查点，任务与模型保持固定。省略 `--seed` 和 `--num-envs` 时，从回放元数据继承原来的种子和并行环境数；显式指定时必须与检查点一致，避免无意改变续训设置。可选择中断前已经保存完毕的较早目录；不要选择正在写入的目录。适配器在启动 Ray 前检查策略及目标网络的键名、形状、精度和固定动作缩放、DCP 元数据引用的分片及字节范围，并确认活动回放窗口内的轨迹文件齐全。上游保留历史索引，但只保存最近 10,000 条轨迹的缓存，已淘汰轨迹的文件不属于必需输入。

正常退出、失败或中断后，`run.json` 的 `latest_checkpoint` 和 `last_saved_step` 指向本次输出目录中最近通过布局检查的保存点。上游直接写入最终目录，适配器会跳过缺文件、回放索引不完整或活动轨迹缺失的较新目录；首次保存前退出则为 `null`。这里的 `checkpoint_validation=layout_only` 不代表权重内容和 DCP 分片已校验，实际续训仍执行上述 SDK 检查。`inputs/` 中的原始保存点不参与发现。

SDK 还会在 CPU 上读取实际恢复用的 DCP 状态，确认策略张量与 `full_weights.pt` 一致、熵温度有限，以及 actor、critic 和熵温度的 Adam 参数顺序、超参数、步数和动量缓冲有效。缺失缓冲、负二阶矩或非有限数值会在启动 Ray/GPU 前被拒绝；训练结束时也执行这些检查，结果写入 `restore_validation`。

损坏状态复现、历史保存点兼容性、旧版/新版配对训练与部署结果见[运行与恢复校验记录](rl-runtime-validation-20260928.md)。

完整输入会复制到新运行的 `inputs/` 并比对文件哈希，SDK 进程加载前再次核对快照。复制包括回放数据，需要额外磁盘空间；上述检查不覆盖回放轨迹张量和调度器的完整语义。上游 FSDP 检查点保存训练进程的 RNG，但回放缓冲区的独立采样生成器会按种子重建，仿真状态也没有在这里完整恢复；续训不承诺与不中断训练逐位一致。准备快照时中断也会记录 `interrupted` 状态。

回放元数据的轨迹数量、下一个轨迹编号和样本总数必须与完整历史索引一致，每条索引的样本数必须等于其时间长度乘并行环境数。错误计数器可能使上游覆盖已有轨迹，错误样本数可能使采样越界，因此在复制和启动 SDK 前拒绝这些保存点；这项检查不读取轨迹张量，也不要求已淘汰的轨迹文件重新出现。

## 独立评估

```bash
python -m embodiedforge rlinf evaluate \
  --repo /home/ubuntu/workspace/3rdparty/RLinf \
  --python /home/ubuntu/workspace/3rdparty/RLinf/.venv-ef-sac/bin/python \
  --output runs/rlinf-pickcube-sac-eval \
  --gpu 0 --headless --quiet \
  --num-envs 16 --seed 1234 --eval-epochs 10 \
  --checkpoint runs/rlinf-pickcube-sac/maniskill_sac_mlp/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
```

评估使用上游 `EmbodiedEvalRunner`，不创建训练 actor。适配器为 `rollout.model` 填入与训练一致的完整 MLP/Q 网络配置，并加载所选权重。`--eval-epochs` 控制 50 步 rollout 的轮数；最终成功率等指标保留上游定义，不能把这个参数直接当作完成 episode 的计数。

评估权重也会复制到本次运行的 `inputs/full_weights.pt`，并比对源文件复制前后的哈希与副本哈希。复制期间权重发生变化或副本不一致时，运行会记录为失败，不启动 SDK；复制完成后修改原文件不会影响本次评估。SDK 加载前还会再次核对副本与 `request.json` 中记录的哈希。

## 导出与 CPU 推理

在 RLinf SDK 中额外安装 `onnx` 和 `onnxruntime`；不需要把它们装进 `ef`。本机实测版本是 ONNX 1.21.0、ONNX Runtime 1.24.4。使用完成保存的 `full_weights.pt`，输出目录仍须为新目录：

```bash
python -m embodiedforge rlinf export \
  --repo /home/ubuntu/workspace/3rdparty/RLinf \
  --python /home/ubuntu/workspace/3rdparty/RLinf/.venv-ef-sac/bin/python \
  --output runs/rlinf-pickcube-sac-export \
  --headless --quiet \
  --checkpoint runs/rlinf-pickcube-sac/maniskill_sac_mlp/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
```

导出复用输入快照和权重校验，剔除两个 Q 网络及探索用的 log-std head，只保留 `backbone → actor_mean → tanh`，与上游 `mode="eval"` 一致。模型有 143,620 个参数，FP32、ONNX opset 17，支持动态 batch。先检查 ONNX 图，再用 CPU Runtime 对照上游实际策略的 1、7、32 三种 batch；全部通过后才发布 `policy.onnx`。`result.json` 的 `onnx` 字段记录文件大小、哈希、输入输出、参数量及动作误差。

需要缩小文件时，可在同一导出命令中增加 `--weight-storage float16`，并使用新的输出目录。它仅把四个权重矩阵存为 FP16，再由 ONNX `Cast` 恢复为 FP32 运算；偏置、输入、输出仍是 FP32，参数量不变。当前 actor 文件约从 562 KiB 降至 284 KiB。默认值为 `float32`，不改变原始检查点；超出 FP16 有限范围的权重会拒绝导出。完整体积、动作误差、CPU 耗时与闭环结果见 [压缩、量化后训练与消融实验](rl-compression-study-20260928.md)。

也可使用 `--weight-storage int8`（另选新的输出目录），将四个矩阵按输出通道对称存为 INT8，偏置与缩放系数保留 FP32。图中用常量 `Cast` + `Mul` 还原矩阵，本机 ORT 可在加载时折叠，避免每步反量化。它不量化观测或中间激活，计算和输入输出仍为 FP32，文件约 148 KiB；运行时可能展开为 FP32 权重，因此文件大小不代表内存占用。

FP16 和 INT8 存储都会改变权重精度。此时 `validation.max_absolute_error` 检查 ONNX 与使用相同解码权重的上游策略是否一致，仍使用 `atol=rtol=1e-5`；`original_fp32_max_absolute_error` 另外记录这些校验输入上相对原始 FP32 策略的最大动作差异。数值校验不能代替压缩模型的闭环评估，也不表示文件减半后推理速度会翻倍。

部署端只需要 NumPy 和 ONNX Runtime，以下推理代码在安装了这两个包的部署 Python 中执行。输入是固定任务 `obs_mode="state"` 的原始 42 维状态，保持 ManiSkill 的字段顺序，不添加额外归一化。输出是 `pd_ee_delta_pos` 控制器的 4 维归一化输入，取值在 `[-1, 1]`；不能直接用作关节目标或真实机械臂位置。最小推理代码如下，`states` 由部署端按照同一观测定义提供：

```python
import numpy as np
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(
    "runs/rlinf-pickcube-sac-export/policy.onnx",
    options,
    providers=["CPUExecutionProvider"],
)
actions = session.run(
    ["actions"], {"states": np.asarray(states, dtype=np.float32).reshape(-1, 42)}
)[0]
```

本机 ManiSkill 3.0.0b22 的状态排列已从实际环境逐字段重建，并与上游 16 个并行观测精确核对。切片采用 Python 的左闭右开约定：

| 切片 | 上游字段 | 内容 |
| --- | --- | --- |
| `0:9` | `agent.qpos` | 7 个臂关节及 2 个手指关节的位置 |
| `9:18` | `agent.qvel` | 同一关节顺序的速度 |
| `18:19` | `extra.is_grasped` | 抓取状态，布尔值转 float32 |
| `19:26` | `extra.tcp_pose` | TCP 的 3 维位置和 4 维四元数 |
| `26:29` | `extra.goal_pos` | 目标位置 |
| `29:36` | `extra.obj_pose` | 方块位置和四元数 |
| `36:39` | `extra.tcp_to_obj_pos` | 方块位置减 TCP 位置 |
| `39:42` | `extra.obj_to_goal_pos` | 目标位置减方块位置 |

关节顺序为 `panda_joint1` 至 `panda_joint7`，然后是 `panda_finger_joint1`、`panda_finger_joint2`。位姿保持 SAPIEN 的原始约定；目标和物体状态来自模拟器。动作前 3 维由当前 Panda 控制器映射为根坐标系下各轴 `[-0.1, 0.1]` m 的末端位移，再经过逆运动学；第 4 维映射为手指位置范围 `[-0.01, 0.04]` m。这里描述的是实测 SDK 的控制器配置，推理示例仍将归一化动作交给该控制器处理。

旧 seed 1234 / 3000 轮和新 seed 4321 / 2000 轮的两份实际权重，已分别由 CPU ONNX 驱动 PickCube 仿真完成 480 条轨迹，并各在 24,000 个真实观测上与上游 PyTorch CPU 策略逐步比较；两份模型的最大绝对动作误差分别约为 `1.73e-6`、`3.07e-6`。仿真仍使用 GPU，部署端观测与控制器适配尚不包含真实机器人。完整数据见 [奖励消融与部署实验](rl-factorial-study-20260927.md)。

### 直接评估 ONNX 策略

导出后可以直接让 CPU ONNX 策略驱动同一 PickCube 仿真，无需原始 `.pt` 权重：

```bash
python -m embodiedforge rlinf evaluate \
  --repo /home/ubuntu/workspace/3rdparty/RLinf \
  --python /home/ubuntu/workspace/3rdparty/RLinf/.venv-ef-sac/bin/python \
  --output runs/rlinf-pickcube-onnx-eval \
  --onnx runs/rlinf-pickcube-sac-export/policy.onnx \
  --gpu 0 --headless --quiet \
  --num-envs 16 --seed 7001 --eval-epochs 10
```

`--onnx` 与 `--checkpoint` 二选一。ONNX 文件先复制到本次运行的 `inputs/policy.onnx` 并校验哈希；加载时检查固定任务、上游提交、观测与动作约定及动态 batch 维度，拒绝不兼容的模型。ONNX Runtime 使用单线程 CPU 推理，ManiSkill 仍使用指定 GPU；这个入口直接运行环境，不启动 Ray 或加载训练网络。

每个 epoch 有 50 个控制步，固定配方忽略提前成功终止，并在 50 步后自动重置。因此上述 16 个环境、10 个 epoch 共完成 160 条轨迹、8,000 次观测动作计算。`success_once` 统计轨迹中至少成功过一次的比例，`success_at_end` 统计最后一步仍成功的比例；统计使用上游环境提供的终局信息，不把每个控制步重复计为轨迹。

`result.json` 保存 `policy_driver=onnx_cpu`、实际 ONNX 哈希、源权重哈希元数据、两种成功的计数、动作观测数及 `eval/*` 指标；这些指标也写入 TensorBoard。此命令测量导出策略的闭环表现，不同时执行 PyTorch 动作对照；导出时的数值一致性检查仍保留。仿真结果不代表真实机械臂部署验收。

`inference` 另记录 CPU `session.run` 的调用数、batch 大小及每批动作的平均、P50、P95 毫秒耗时。计时包含第一次调用，不含观测从 GPU 传到 CPU、仿真步进或环境初始化；它用于观察本次运行的推理开销，不能直接作为完整控制周期延迟或独占机器上的吞吐基准。

## 结果与模型大小

| 文件 | 内容 |
| --- | --- |
| `request.json` | 固定源码提交、CLI 参数、输入路径与快照文件哈希 |
| `config.yaml` | Hydra 合成并解析插值后的启动配置 |
| `tensorboard/config.yaml` | 上游校验、补齐默认值后的运行配置 |
| `environment.json` | SDK Python 路径、依赖版本与缺失/不兼容依赖信息；SDK 检查失败时也保留 |
| `worker.log` / `metrics.log` | 进程输出与上游指标表 |
| `tensorboard/` | 完整指标历史 |
| `result.json` | 成功结束后的最后一条各项指标、对应日志 step、策略权重大小与 SHA-256；训练另含目标网络 `target_weights`、完整检查点大小 `checkpoint_bytes` 和文件清单 |
| `run.json` | `preparing` / `running` / `complete` / `failed` / `interrupted` 状态和成功结果 |
| `policy.onnx` | `export` 成功后生成的确定性 actor；不包含续训状态 |

CPU 导出不生成 TensorBoard 指标、`metrics.log` 或 `tensorboard/config.yaml`；导出的 `result.json` 包含输入权重统计和 ONNX 校验记录。ONNX 评估会写 TensorBoard 指标和根目录的 `config.yaml`，不生成上游 Ray runner 的 `metrics.log` 或 `tensorboard/config.yaml`。

`result.json` 中的 `weights.file_bytes` 是实际权重文件大小，`state_tensor_elements` 和 `state_tensor_bytes` 是 state dict 的元素数和张量字节数，包含 Q 网络和注册 buffer。`weights.parameters` 单独给出 `actor`、`critic` 和 `total` 参数量，使用上游实际模型结构校验和统计；检查过程不随机初始化另一套模型，也不占用 GPU。

固定基线的结构统计如下（从上游 MLP 构造器验证，不是训练结果）：

| 项目 | 数量 |
| --- | ---: |
| Actor 参数 | 144,648 |
| 确定性部署 actor 参数 | 143,620 |
| 两个 Q 网络的参数合计 | 290,818 |
| 总参数 | 435,466 |
| State dict 元素（含两个 action buffer） | 435,468 |
| FP32 state dict 张量字节数 | 1,741,872（约 1.66 MiB） |

确定性 actor 为 `42 → 256 → 256 → 256 → 4`，三个隐藏层使用 Tanh，均值输出再经过 Tanh。含偏置的参数量为 `(42+1)×256 + 2×(256+1)×256 + (256+1)×4 = 143,620`。训练时额外的 `256 → 4` log-std head 有 1,028 个参数，因此训练 actor 为 144,648 参数。

两个 Q 网络彼此独立，也不与 actor 共享参数。每个网络拼接 42 维状态和 4 维动作，结构为 `46 → 256 → 256 → 256 → 1`，三个隐藏层使用 LayerNorm 和 Tanh；含归一化的可学习缩放与偏置，每个 Q 网络为 145,409 参数，合计 290,818。

实际 `.pt` 文件还包含序列化开销，以 `file_bytes` 为准。完整 SAC 检查点还包含优化器、目标网络和回放数据，总大小直接记录为 `checkpoint_bytes`，等于 `checkpoint_files` 中所有文件的 `bytes` 之和；通常会大于单独模型文件。这是文件内容字节数，不包含文件系统分配开销或其他保存点。权重 SHA-256 与大小来自本次反序列化的同一份字节，避免文件被替换后出现内容与哈希不一致。

训练的 `final_step` 是完成的循环数；指标 `step` 保留上游从零计数的日志语义，最终训练评估日志为 `final_step - 1`，独立评估为 0。只有子进程正常退出、最终检查点与模型检查通过、评估完成了有效轨迹、指标有限并覆盖最终日志步数时，才生成成功结果。成功率可以为零；没有完成任何轨迹不算有效评估。失败和中断不会被标记为完成。

## 当前验证边界

已验证 `ef` 中的 CLI、输入快照与准备阶段中断状态、虚拟环境激活、固定上游 YAML 的 Hydra 合成及上游配置校验、非零 GPU 的上游放置结果、实际上游 MLP 的结构/精度检查，以及 Torch DCP 分片和 TensorBoard 事件读取。实际 wheel 还通过了主环境 Python 3.10 到 SDK Python 3.12 的导入隔离检查。

Worker 种子入口经过独立 Python 进程和真实本地 Ray `WorkerGroup` 验证：相同 seed/rank 的实际 MLP 初始权重、随机动作及 Python/NumPy 样本一致；改变 rank 会改变随机流，同一 worker 后续调用正常推进随机流，不会每次重置。这些检查使用 CPU 模型，未验证 GPU 算子的逐位确定性。

回放数据通过了上游保存、缓存淘汰、重新加载与采样的 CPU 验证，种子与环境数从实际序列化元数据恢复。使用上游 `Checkpoint` 类、未包裹 FSDP 的 CPU MLP 和熵温度模块，确认 DCP 保存/加载后模型、三个 Adam 优化器、调度器及 Python/NumPy/Torch CPU RNG 状态一致；这是序列化验证，没有执行策略更新或 GPU FSDP 恢复。另使用真实 CPU Ray 实例验证了上游本地启动重试、正常和异常退出：本次实例的工作进程退出，原有实例中的作业仍可响应。

另对固定上游的 SAC 损失、三个优化器更新和目标网络 EMA 做了 CPU 数值对照：使用未包裹 FSDP 的实际 MLP、固定人工 batch，单 rank 的 alpha 梯度归约按恒等操作处理。DCP 恢复后，在相同 batch 和 Torch RNG 下，未触发梯度裁剪的用例与连续更新逐位一致。触发裁剪的用例出现微小差异：上游按优化器分别清零梯度，却按整个模型计算裁剪范数，因此连续运行中另一组参数遗留的 `.grad` 会参与裁剪，而 checkpoint 不保存这些运行时梯度；补齐相同 `.grad` 的对照恢复了逐位一致。这个观察进一步限定了续训的等价性，并未改变接入的上游更新算法。该检查不是 GPU FSDP、回放采样或 PickCube 训练验证。

上述配置与放置测试使用虚拟硬件拓扑。另已在 RTX 5090 D v2 上完成实际 PickCube GPU 采样、SAC 更新、FSDP 完整状态恢复、最终保存与独立评估：训练经过主机内存压力导致的中断，从有效保存点恢复至第 2000 轮，不能称为不中断训练。恢复前后实际 DCP 的三个优化器步数、调度器、模型与回放计数均已核对。独立评估加载同一最终权重，使用新的环境种子；完整数据、失败记录、SDK 版本与模型大小见 [2026-09-26 GPU 实验](rl-training-study-20260926.md)。这验证当前单卡 PickCube 配方，不涵盖其他任务、GPU 逐位复现、多机或真机，也没有吞吐量比较结论。

worker 异常处理修复后，又从 2000 轮完整恢复并正常训练至 3000 轮。实际 DCP 的三个优化器推进到 96,000 步，回放累计 192,000 条转换；三个独立评估种子共 480 条轨迹，曾成功比例为 98.125%，结束时成功比例为 97.917%。[后训练与恢复记录](rl-posttraining-study-20260926.md#正常-gpu-训练与独立评估) 给出与 2000 轮的同条件对照、检查点大小和命令。成功率变化来自继续训练后的模型，不能归因于异常清理修复。

2026-09-27 增加了另一训练种子和新的独立评估种子。在相同 2000 轮预算下，旧轨迹曾成功率为 94.583%，新轨迹为 38.542%；旧轨迹包含恢复，新轨迹连续训练，因此不能仅归因于种子，也不能据旧模型的高成功率认定该预算下跨种子稳定收敛。两份实际策略的 CPU ONNX 导出与仿真闭环均已验证，详见 [奖励消融、训练种子与部署资源](rl-factorial-study-20260927.md)。

2026-09-28 完成两个种子的配对对照：每个种子连续训练 3000 轮，再从同一第 1000 轮保存点追加 2000 轮。10 个比较模型共完成 4800 条独立评估轨迹；3000 轮时，续训相对连续训练的曾成功率变化分别为 -2.292、+1.667 个百分点，没有一致方向。20 个保存点的模型、优化器、目标网络和回放计数通过核对，但仿真及部分随机流并未完整恢复，不能认为续训逐步重放连续轨迹。四个最终模型另经正式 ONNX 入口完成 1920 条闭环轨迹，详见 [连续训练、保存点续训与 ONNX 评估](rl-resume-study-20260928.md)。

同轮还对四个最终 actor 追加动态 INT8 压缩对照：文件均缩小 73.02%，但在相同部署评估种子下，结束成功率下降 4.375–6.042 个百分点。量化方法、每个模型的结果及转换示例见上述文章；默认导出保持 FP32。

后续 [压缩与量化后训练实验](rl-compression-study-20260928.md) 新增 seed 2026 的 3000 轮训练，覆盖三个训练种子的五条 actor 路径，共 36,000 条对照评估轨迹。正式 FP16/INT8 权重存储分别减少 49.54%/73.74% 文件大小，均保留 FP32 计算；在这五个 actor 上，结束成功率相对 FP32 的变化范围分别为 −0.625～+0.833、−0.417～+0.625 个百分点。文章另报动态激活 INT8、QAT、推理 batch、CPU 耗时及进程内存对照，避免把两类 INT8 方案混为一谈。
