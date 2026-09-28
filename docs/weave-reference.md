# Weave 接入：G1 训练、评估、导出与动作片段

本仓库参考 [Weave](https://github.com/xiaohu-art/Weave) 本地版本
`b162a35` 的片段组织、随机帧初始化和逐片段评估思路，独立实现了 NumPy
动作库与离线轨迹比较；新增 `weave` 入口调用外部 Weave 的 G1 任务、SimBa 网络和
Muon PPO。没有复制其源码、模型或资产。`motion` 模块仅依赖 NumPy，`weave` 工作进程
需要独立的 Weave / IsaacLab 环境，不改变现有 Microduck、Go1、H1、ACT 训练入口。

当前可用：Weave 风格 NPZ 导入、严格数据检查、按物体采样片段和起始帧、
不跨片段的未来帧查询、参考/实际轨迹比较。这里的比较读取已有记录，不执行策略或仿真。
`weave train/evaluate/export` 已实现训练、续训、在线逐片段评估和策略导出适配。
**完整仿真训练尚未实测，导出的策略也不是可直接上机的完整控制器。**

GMR 与 SONIC X2 的接入边界见[人形动作指南](humanoid-motion.md)。GMR 生成的机器人
参考尚不包含此任务要求的物体轨迹、接触标签和 Inspire 手部布局，不能直接传给
`weave train`。2026-09-28 复查仍缺训练动作，`smalltable.usd` 仍为 LFS 指针。

## G1 训练环境

`ef` 只负责调度，`--python` 指向独立 SDK 环境。固定版本如下：

| 组件 | 版本 |
| --- | --- |
| Weave | `b162a351ddabe302a7a3d50d1543eb83dfeff082` |
| IsaacLab | `e17312889676ed229b986d56c9e0b23a01cf0ab7` |
| SDK Python / Isaac Sim | 3.11 / 5.1.x |
| RSL-RL | `rsl-rl-lib==3.1.2` |
| Torch / NumPy | 支持 `torch.optim.Muon` 的 Torch >= 2.10 / NumPy < 2 |

版本依据本地固定提交的 `install.sh` 和 IsaacLab 的 `source/isaaclab_rl/setup.py`。
SDK 环境还需 Weave 声明的依赖；导出需 ONNX、ONNXScript。可以参考
[Weave 安装说明](https://github.com/xiaohu-art/Weave/tree/b162a351ddabe302a7a3d50d1543eb83dfeff082)，
在独立目录准备环境。其安装脚本会 checkout IsaacLab、修改 Isaac Sim 预装依赖，
不要在现有 H1/IsaacLab 工作目录直接运行。入口本身不安装依赖或修改外部源码。
二进制 Isaac Sim 安装由指定 IsaacLab 的 `_isaac_sim/setup_conda_env.sh` 设置子进程运行库路径；
pip 安装则使用指定 Python 的包。两者都核对版本和实际 IsaacLab 导入路径。
SDK 及其 Python 子进程从运行目录加载 EmbodiedForge 适配代码快照，不引入主环境的整个
`site-packages`；从 wheel 安装主入口也不会将主环境的 NumPy 等依赖混入 SDK。
启动后编辑工作区不会改变该次任务的适配代码。加载 SDK 前先校验代码快照的 SHA256；
Weave、IsaacLab 和资产仍使用外部固定仓库。历史运行记录不会自动补充代码快照。
导出会在启动模拟器前检查 ONNX、ONNXScript 是否安装，并将版本写入 `runtime.json`；
训练和评估不要求这两项导出依赖。缺包提示指向 `--python` 指定的 SDK 环境。

2026-09-20 本机检查：现有 `isaaclab` 环境为 Python 3.12 / RSL-RL 5.0.1，
IsaacLab 提交也不同；Weave 缺动作集，选用资产还存在 Git LFS 指针。
因此目前不能直接用现有 `isaaclab` 环境执行下面的训练。需要补齐 SDK、50 fps 动作和 LFS 资产。
检查失败会明确报错，不会改用其他任务或启动短训练。
资产检查针对固定任务的输入：G1 URDF 与其引用的网格、所选物体 USD、`surface.npy`、
`sdf_128.npz` 和共用的 `geometry/bps_128.npy`。未使用的地面 USD、物体源 OBJ 和
其他物体资产不再影响启动。USD 的内部依赖仍由 Isaac Sim 加载时解析；文件检查不代表
已验证物理资产或完整仿真。

## 训练、续训、在线评估与导出命令

在 EmbodiedForge 根目录运行，替换 SDK 和数据路径。`WEAVE_PYTHON` 应保留虚拟环境
`bin/python` 路径，不要替换成它最终指向的系统 Python。
训练与续训必须显式指定 `--iterations`；省略时不会自动启动 100,000 轮训练。

```bash
conda activate ef
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
WEAVE_ROOT=/home/ubuntu/workspace/3rdparty/Weave
WEAVE_LAB=/path/to/weave-sdk/IsaacLab
WEAVE_PYTHON=/path/to/weave-sdk/sim51/bin/python

weave_common=(
  --weave-root "$WEAVE_ROOT"
  --isaaclab-root "$WEAVE_LAB"
  --python "$WEAVE_PYTHON"
)

# 原始 NPZ 必须显式描述真实 G1 数据的关节/刚体顺序。
# 也可先用下文 motion import-weave 导入，之后传 ef-motion-v1 文件并省略 --layout。
python -m embodiedforge weave train "${weave_common[@]}" \
  --motions /data/weave/train/smalltable.npz \
  --layout /data/weave/g1-inspire-layout.json \
  --num-envs 4096 --iterations 100000 --save-interval 100 \
  --output runs/weave-train

# 恢复策略、优化器和迭代编号；iterations 表示本次额外训练量。
# 指定已保存的实际 model_<iteration>.pt，新结果写到新的目录。
python -m embodiedforge weave train "${weave_common[@]}" \
  --motions /data/weave/train/smalltable.npz \
  --layout /data/weave/g1-inspire-layout.json \
  --checkpoint /path/to/model_99999.pt \
  --num-envs 4096 --iterations 20000 \
  --output runs/weave-resume

# 评估显式使用留出的数据；每次只评一个物体，环境数自动等于片段数。
python -m embodiedforge weave evaluate "${weave_common[@]}" \
  --motions /data/weave/heldout/smalltable.npz \
  --layout /data/weave/g1-inspire-layout.json \
  --checkpoint /path/to/model_99999.pt \
  --output runs/weave-eval

# 导出也要构建任务以取得真实观测和动作布局，仍需要 SDK、GPU 和参考动作。
python -m embodiedforge weave export "${weave_common[@]}" \
  --motions /data/weave/train/smalltable.npz \
  --layout /data/weave/g1-inspire-layout.json \
  --checkpoint /path/to/model_99999.pt \
  --output runs/weave-export
```

全部默认 **headless**，也可显式传 `--headless`；需要窗口时使用与它互斥的 `--gui`，
设备用 `--device cuda:0` 选择。
当前入口只启动一个 SDK 工作进程，不支持通过 `torchrun` 启动 DDP；检测到
`WORLD_SIZE` 或 `LOCAL_WORLD_SIZE` 不为 1 时，在准备数据和启动 SDK 前报错，
避免 RSL-RL 等待缺失的其他进程。子进程清理分布式 rank/连接变量，保留
`CUDA_VISIBLE_DEVICES`；例如设置 `CUDA_VISIBLE_DEVICES=4` 时，`--device cuda:0`
指该进程可见的第一张 GPU。多个独立实验应使用不同输出目录和各自的设备设置。
不需要额外的冒烟启动步骤。所有输出目录必须是新目录，不覆盖旧实验。
多物体训练可在 `--motions` 后列出多个文件，所有输入使用同一机器人布局、50 fps，
片段名称在所有文件间唯一。布局会与仿真实际生成的关节/刚体名称及顺序比较。
每次训练都从固定版本的 `configs/track/train.yaml` 和注册任务配置构建，
命令行替换数据、环境数、迭代数、保存间隔、种子和设备，不使用上游默认数据路径。
训练还对齐固定版本 `scripts/rsl_rl/train.py` 的 Torch 设置：CUDA 矩阵乘法与 cuDNN
允许 TF32，cuDNN 的 `deterministic`、`benchmark` 均为 `false`。训练结果的
`torch_backends` 记录实际开关值。相较此前依赖 Torch 默认值的入口，这可能改变数值结果，
不能仅凭相同种子要求新旧运行逐位一致；评估与导出不应用这组训练设置。
外部 checkpoint 必须采用该网络配置；续训不恢复仿真内部状态或随机数状态。

训练恢复、评估和导出都会把 checkpoint 复制到本次运行的 `inputs/checkpoint.pt`，
比对源文件复制前后的 SHA256 与副本 SHA256。复制期间检查点变化或副本不一致时，
运行记录为失败，不启动 SDK。复制完成后修改原文件不会改变本次使用的权重；
SDK 加载前仍会核对快照哈希。
训练、评估和导出加载时均检查策略键名、张量形状、精度和有限性，并拒绝负的归一化
方差/标准差/计数，以及非正的策略探索标准差，避免静默精度转换或加载损坏权重。
续训要求 checkpoint 包含全部 Muon / AdamW 子优化器状态及有效迭代编号；
加载策略前检查参数组、固定优化器设置、动量/二阶矩的形状与精度、有限性及有效步数；
缺失状态、负二阶矩或不一致的子优化器学习率会直接报错，避免静默重置或部分恢复后继续计数。
加载优化器后，适配层重新绑定 Weave 组合优化器的参数组，并从保存的学习率恢复
PPO 自适应学习率状态，确保后续调度作用于实际的 Muon / AdamW 参数组。
三种操作都先在 CPU 读取 checkpoint，再将策略权重复制到目标设备。续训校验通过后，
Torch 将优化器张量恢复到对应参数所在设备；评估、导出不加载优化器状态，避免占用 GPU。
读取完整文件仍需要相应的 CPU 内存，支持与保存时不同的 GPU 编号。
固定版本 [RSL-RL 的 checkpoint](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.2/rsl_rl/runners/on_policy_runner.py)
保存的是已完成的迭代编号；适配层续训从下一编号开始。例如 `model_99.pt` 再训练
20 轮，对应编号 100–119，最终保存 `model_119.pt`。训练的 `result.json` 记录本次
首轮、末轮编号和请求的训练轮数，避免日志与模型编号重叠。
训练只有完成请求轮数、最终编号匹配且落盘策略/优化器通过上述检查后才记录成功；
找到一个较早的模型文件不足以表示本次预算完成。`result.json` 还记录 actor 与 critic
合计的 `policy_parameters`、含归一化缓冲的 `policy_state_tensor_bytes`，以及完整
checkpoint 的文件大小和哈希，区分模型张量规模与包含优化器的保存文件体积。

每次运行保存以下内容：

- `inputs/`、`request.json`：校验后的动作快照、可选 checkpoint 快照、哈希与 SDK 提交。
  原始动作读取禁用 pickle，统一四元数格式后才交给上游加载器。
  训练/评估/导出的快照将逐帧数值统一为本机字节序的连续 float32，与 Weave 的 Torch
  加载精度一致；转为 float32 后溢出的输入会被拒绝。每个快照同时记录原始输入文件的
  路径、SHA256 和快照 SHA256，原始文件不改动。独立 `motion` 导入仍保留原始数值精度。
- `env.yaml`、`agent.yaml`、`runtime.json`：任务/算法配置与实际依赖版本。
- `implementation/embodiedforge/`：本地适配包的 Python、XML 和许可证文件，
  `request.json` 记录逐文件 SHA256，训练、评估和导出均使用此快照。
- `worker.log`、`run.json`：子进程日志及准备/运行/完成/失败/中断状态；动作快照准备期间的终止信号也会记录为中断，运行期间的中断会转发给整个工作进程组。
  主工作进程失败或处理中断后，启动器清理本次进程组中的残留子进程，避免包装进程先退出而后台工作仍在运行。
  退出时依次尝试关闭训练日志（若已创建）、环境和模拟器应用；清理失败会单独记录，
  不覆盖已有错误或中断。若此前运行正常，清理失败仍使任务以失败状态退出。
  启动器最终写入状态失败时也保留原始退出原因，并向 stderr 输出记录失败的原因；
  此时 `run.json` 可能仍停留在上次成功写入的状态，应结合进程退出码和日志判断。
- 训练的 `model_*.pt` 与 TensorBoard 日志：checkpoint 原子发布，保留最近完成的保存点。
  保存失败后的临时文件删除若也失败，会单独记录残留路径和原因，保留原始保存异常或中断。
  正常退出、失败或中断后，`run.json` 的 `latest_checkpoint` 和 `last_saved_iteration`
  标出本次目录中已发布的最近保存点；首次保存前退出则为 `null`。临时文件和输入权重
  不参与选择。成功结果对应的最终保存点标为 `checkpoint_validation=policy_and_optimizer`；
  其他退出情况下，文件发现不代表已验证权重内容（`checkpoint_validation=not_performed`）。
  续训时将此路径传给 `--checkpoint`，上次保存之后未落盘的训练进度不会恢复。
- 评估的 `result.json`：逐片段状态、终止原因、完成比例和跟踪误差。
- 导出的 `policy.pt`、`policy.onnx`、可能的 `policy.onnx.data` 及 `policy.json`：
  观测组顺序/维度、动作关节顺序、缩放/偏置、mimic 映射、控制周期、参数量和文件大小/哈希。
  ONNX 与其外部权重文件必须一起部署。

在线评估从每个片段第 0 帧开始，只统计第一次 episode；在终止判定后的奖励计算阶段，
通过恒返回零的采集项记录自动 reset 前的终止步。它不增加奖励，也不启用会额外计算
观测的 recorder，避免额外消耗观测噪声随机数和重复几何查询。训练与评估原有的噪声
设置不变。指标和终止标记分别批量传回 CPU，加上帧序号，每步仅执行三次数据传输，
避免按指标、终止项逐个同步；统计仍保留原始数据精度。
失败条件与片段结束同时触发时判失败，时间限制截断不算成功。
没有成功片段时成功组指标为 `null`；未完成全部片段则保存 incomplete 报告并返回非零退出码。
评估被中断或 stepping 报错时尝试保存已采集的报告与输入哈希；不完整或非有限的当前步骤
不会部分累加到历史统计。报告保存失败会单独写入错误日志，保留原始评估异常或中断原因。
磁盘不可写或进程被强制杀死时无法保证写出报告。
在线误差沿用 Weave 的各项 error 定义，先对一个片段内步骤求均值，再对片段等权平均，
**不是**下文离线比较的 RMSE，不能混在同一指标列。

评估与导出仅为 PPO 缓冲保留一步容量，训练仍使用 32 步。
[RSL-RL 3.1.2 的 runner](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.2/rsl_rl/runners/on_policy_runner.py)
在构建时也会分配训练 rollout 缓冲；这一调整将该部分张量容量降为原来的 1/32，
不改变网络、checkpoint 加载或实际评估步数，也不代表总显存降为 1/32。

导出先在 `--device` 指定设备上计算当前策略的校验输出，再将策略移到 CPU，
避免 SDK 导出器复制网络时额外分配一份 GPU 权重。导出是该进程使用策略的最后一步。
导出包含观测归一化，执行 TorchScript、ONNX reference evaluator 与原设备策略的输出一致性检查，
同时检查 ONNX 结构。`policy.json` 保存两种导出的最大绝对输出误差。
`policy.json` 明确记录 ONNX 目标运行时尚未验证。部署端仍需构造参考动作、物体状态、
接触/几何与机器人观测，并按动作映射实现执行器控制；当前没有 G1 真机通信或闭环部署程序。

验证范围：NumPy/启动器回归检查已通过；另使用上游真实网络构建代码与固定版本导出器，
在 CPU 上验证了包含已更新观测归一化的缩小网络的 TorchScript / ONNX 输出一致性和 ONNX 结构，
并确认导出没有改变归一化统计。没有启动仿真或训练任务，
这不代替目标 SDK 上的训练、完整模型评估与部署验证。
评估采集还通过固定版本 IsaacLab 实际 step/reward 方法的 CPU 控制流检查：
终止状态在 reset 前保存，新增项对奖励为零，不额外调用带噪观测计算。
另使用 RSL-RL 3.1.2 与 Weave 的 SimBa/Muon 实现构建缩小网络，在 CPU 上确认
一步缓冲的张量字节数为 32 步的 1/32，同一 checkpoint 加载后 40 组推理输出完全一致。
续训检查使用实际 Muon / AdamW 状态：恢复后对相同梯度执行下一次更新，参数与连续执行
完全一致；空子优化器状态、错误动量形状、负二阶矩与学习率/设置不一致的 checkpoint
在修改模型前被拒绝。这不等于完整 PPO 训练验证。

## 在 ef 环境使用

从 EmbodiedForge 根目录执行；仅需核心 NumPy 依赖：

```bash
conda activate ef
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m embodiedforge motion import-weave \
  --input /path/to/smalltable.npz \
  --layout /path/to/g1-layout.json \
  --output /path/to/reference.npz

python -m embodiedforge motion inspect --input /path/to/reference.npz

python -m embodiedforge motion compare \
  --reference /path/to/reference.npz \
  --rollout /path/to/recorded-rollout.npz \
  --output /path/to/comparison.json
```

也支持 `python -m embodiedforge.motion ...`。输出必须是尚不存在的文件，父目录需已存在。
导入成功和完整评估退出码为 0，输入错误或不完整评估为 1；评估中机器人失败会降低成功率，
但不代表离线计算失败。不完整评估仍保存报告，其 `status=incomplete`，不能作为完整验收结果。

本机 Weave 检出缺少 `data/` 动作集，且有未解析的 LFS 资产；这些命令中的路径必须替换为
实际数据。本次验证使用合成轨迹覆盖格式和判定边界，没有执行 Weave 训练。

## 数据与坐标契约

导入时需显式提供布局 JSON，以下仅为两关节示例，**不是 G1 的完整关节清单**：

```json
{
  "robot": "my-robot-v1",
  "joint_names": ["hip", "knee"],
  "body_names": ["pelvis", "foot"],
  "quaternion_order": "wxyz",
  "coordinate_frame": "world"
}
```

名称顺序必须对应输入数组，不能从其他机器人或另一版 USD/MJCF 猜测。
支持输入 `wxyz` 或 `xyzw`，导入后统一为 `wxyz`；位姿和速度不做缩放或重定向。
当前关节位置/速度契约为旋转关节的 rad、rad/s，位置/线速度为 m、m/s，角速度为 rad/s。
每个片段第一帧的时间为 0，后续时间为 `frame_index / fps`。

输入 NPZ 使用以下字段。`C` 为片段数，`T=sum(motion_lengths)`，`J/B` 为关节/刚体数：

| 字段 | 形状与含义 |
| --- | --- |
| `fps` | 有限正标量 |
| `motion_lengths` | 正整数 `[C]` |
| `motion_names` | 唯一 Unicode 字符串 `[C]`，作为片段匹配键 |
| `object_names` | 非空 Unicode 字符串 `[C]`，可重复 |
| `joint_pos`, `joint_vel` | `[T,J]` |
| `body_pos_w`, `body_lin_vel_w`, `body_ang_vel_w` | `[T,B,3]` |
| `body_quat_w` | 单位四元数 `[T,B,4]` |
| `object_pos_w`, `object_lin_vel_w`, `object_ang_vel_w` | `[T,3]` |
| `object_quat_w` | 单位四元数 `[T,4]` |
| `contact_label` | `[T,B]`，`+1` 接触、`-1` 不接触、`0` 未指定 |

仅读取数值和普通字符串数组，使用 `allow_pickle=False`。如果原始 NPZ 的名称数组为
Python object dtype，需由数据提供方另存为 `np.asarray(names, dtype=str)`；入口不会自动启用 pickle。

输出保留上述字段，并增加标量字符串 `metadata`：包含 `schema=ef-motion-v1`、布局、
源文件路径与 SHA256。加载会检查长度、维度、数值有限性、单位四元数及接触标签。
单文件包含一个固定 fps；不同帧率必须先显式重采样，不能通过修改 fps 数字冒充重采样。

## 供训练适配层调用

```python
import numpy as np
from embodiedforge.motion import MotionClips

library = MotionClips.load("reference.npz")
rng = np.random.default_rng(42)
# 输入应为实际场景每个环境对应的物体名称。
clip_ids, start_frames = library.sample(["smalltable", "smalltable"], rng)
reference = library.frames(clip_ids, start_frames, offsets=[0, 5, 10, 15, 20])
joint_targets = reference["joint_pos"]  # [环境数, 未来帧数, 关节数]
```

先在指定物体的片段中均匀采样，再在该片段内均匀采样起点；不按片段长度加权。
相同物体的请求批量采样，输出仍按输入顺序排列，避免每个环境重复扫描全部片段。
相同输入和随机种子可重复；批量采样的随机序列与旧版逐项采样可能不同。
此优化作用于 NumPy 接口，`weave train` 仍使用上游 Torch 采样器。
未来帧超过片段末尾时保持末帧，不读取下一个片段。
调用方负责将数组送入其训练设备、映射关节、设置机器人与物体初态以及统一控制频率。
这只是参考运动接口，不会自动将 Go1/H1 或 ACT 策略变成 G1 搬运策略。

## 实际轨迹记录与评估

实际 rollout 使用同一格式和布局，`metadata.reference_sha256` 必须是
`MotionClips.load(reference_path).sha256`。它需要覆盖参考库中所有命名片段，文件内片段顺序可不同。
每个片段只能包含从参考第 0 帧开始的一次 episode；提前失败允许记录较短的前缀。
不允许随机起点、自动时间对齐、换物体、替换参考库或通过补帧隐藏失败。

记录器额外保存三个 bool `[T]` 数组：

- `terminated`：实际任务失败条件，包括姿态/物体越界等。
- `truncated`：时间限制等截断。
- `clip_end`：参考片段完成。仅允许在对应参考的最后一帧标记。

应记录终止前的最后状态；三个标记任一成立后，不得继续追加重置后的帧。
与部分 episode 格式不同，这里保留同时发生的标记，不预先抹掉失败：
**failed > success（clip_end）> truncated > incomplete**。
“成功”依据任务提供的标记，离线比较不会凭跟踪误差自行设置成功阈值。
多环境录制时需去掉各环境的平移偏置，使参考和实际位姿处于相同世界坐标系。

报告包含逐片段状态、完成比例、关节 RMSE、身体/物体位置及姿态 RMSE，位置误差按欧氏距离计算。
姿态误差先对齐四元数符号，再用弦长与 `atan2` 计算最短旋转角，改善接近零误差时的
数值精度；`q` 与 `-q` 仍视为同一姿态，指标单位和聚合方式不变。
聚合值为各片段 RMSE 的等权均值；提前失败只计算实际记录的前缀，所以必须结合成功率、
完成比例阅读，不能将短时失败的低误差视作较好策略。没有成功片段时 `metrics_success` 为 `null` 值字典。
报告记录参考与实际文件的 SHA256，禁止 NaN/Infinity，也不覆盖已有结果。

`motion` 离线模块的接触标签用于数据保存和检查，不计算接触奖励或接触准确率；
`weave` 在线训练则沿用上游任务的接触奖励与失败判定。
现有 Go1/H1 回放 NPZ 缺少物体状态与本契约的片段身份，不能直接替代这里的 rollout。

## 后续接入边界

下一步可在单物体抓握任务中实现实际记录适配，再评估接触奖励与几何特征的收益。
完整 Weave 训练入口仍需在补齐动作集、资产与独立 SDK 后验证；实机控制适配尚未实现。
本地 Weave 默认 train/eval 配置复用同一 smalltable 文件；开展泛化实验时必须另行划分数据。
当前模块不提供训练/测试分割工具，也不宣称已经解决数据集划分问题。
