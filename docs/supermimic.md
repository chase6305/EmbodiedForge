# SuperMimic 兼容性预检与独立复现

`embodiedforge supermimic` CLI 支持来源目录和离线预检，**该 CLI 尚未执行策略推理、训练或行为评测**。预检完成只表示前置条件与声明的元数据通过检查；`evaluation_ready` 和 `inference_executed` 始终为 `false`。它不会下载模型、导入复现项目、反序列化 checkpoint 或读取动作数组。

## 来源与固定版本

2026-10-23 检查的来源如下。目录记录是该日期的快照，不会自动判断上游是否发布了新版。

| 来源 | 固定提交 / 框架 | 接入边界 |
| --- | --- | --- |
| [官方项目](https://supermimic.github.io/)与[论文](https://arxiv.org/abs/2610.28480) | 未确认可用于本地推理的官方模型包和导出 ABI | 当前官方预检明确拒绝执行；后续需要另行授权的模型与接口说明 |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/d9be385b3eeb13a638fb33fb200d3bfd9be9839f) | `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`；mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376` | 独立复现，MIT 代码；NPZ 动作和 `policy` / `config` checkpoint |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9) | `a1508d42c9d1699a52e8233ab16d08c488441ab9`；Isaac Lab `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` | 独立复现，MIT 代码；Isaac Lab 3 kit-less / Newton / MuJoCo Warp，CSV 动作和 `pi` / `args` checkpoint |

[官方演示许可证](https://supermimic.github.io/demo/LICENSE.html)限定浏览器中的个人非商业使用，不包含下载其模型、在本地推理或用其材料评测其他模型的权限。因此本入口不提取演示资产。两个独立复现不能标记成官方实现或官方权重。代码许可证也不替代动作数据、机器人资产和权重各自的许可证；vibed 使用的 LAFAN1 数据另有 CC BY-NC-ND 4.0 条款。

## 当前 G1 接口不能直接共用 checkpoint

| 项目 | 论文描述 | CuboiLeo 固定版本 | vibed 固定版本 |
| --- | --- | --- | --- |
| 动作 | G1 的 29 个关节 | 29 | 29 |
| 单帧本体观测 | 93 | 93 | 99，增加 6 维参考相对方向 |
| 历史 | 10 帧 | term-major 展平，930 维 | time-major 展平，990 维 |
| 目标 | 10 个未来帧 | 680；`anchor_obs=true` 时 689 | 730 |
| 归一化 | 需官方导出说明 | 固定分量尺度，以 checkpoint 配置为准 | RunningNorm，以 checkpoint 状态为准 |
| 控制频率 | 50 Hz | 50 Hz | 50 Hz，原始 CSV 动作按上游逻辑重采样 |

维度相同也不足以证明接口兼容。实际适配必须核对关节顺序、默认姿态、动作尺度、PD 参数、四元数顺序、坐标系、历史更新方向和归一化状态。CuboiLeo 的锚点增强实验使用额外观测，不能直接作为论文观测条件下的结果。两个项目的数据筛选、初始化和终止规则也需要分别记录。

官方策略仍是目标入口。两个复现可用于提前确定适配边界：CuboiLeo 的默认 93 维观测和 MuJoCo 路线更接近当前平台已有的接入方式，但依赖版本需要独立环境；vibed 则应作为另一套完整 SDK 和策略接口接入。不能将 vibed 的 99 维输入裁掉 6 维，或只重排 checkpoint 字段来复用同一适配器。

这些数值来自固定版本的源码，而非运行得到的指标：CuboiLeo 的 [`nets.py`](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/policy/nets.py)；vibed 的 [`mdp.py`](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/mdp.py)、[`networks.py`](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/networks.py)。

## 使用预检

在已安装 EmbodiedForge 的环境中执行；源码开发可在命令前设置 `PYTHONPATH=src`。

```bash
python -m embodiedforge supermimic catalog
python -m embodiedforge supermimic preflight \
  --backend cuboi-reproduced \
  --root /path/to/pinned/SuperMimic-Reproduced \
  --python /path/to/dedicated-sdk/bin/python \
  --checkpoint /path/to/authorized/policy.pt \
  --contract /path/to/policy.contract.json \
  --motions /path/to/authorized/npz-motions \
  --output runs/supermimic/preflight-cuboi
```

vibed 使用 `--backend vibed`、其固定源码目录与 CSV 动作目录。框架若为 editable 安装且分发元数据没有记录提交，可提供 `--framework-root /path/to/pinned/framework`；预检要求导入位置位于该目录中，且该仓库提交匹配、已跟踪文件没有修改。输出目录必须是新目录，避免覆盖历史记录。

SDK 在独立 Python 子进程中使用 `-I` 探测，清除继承的 Python 路径污染，记录 Python/包版本、框架提交与 Torch CUDA 可用性。它不安装或升级依赖。CuboiLeo 固定版本要求 Python 3.12、Torch ≥2.14、MuJoCo / MuJoCo Warp 3.16.x、Warp ≥1.15 以及指定 mjlab 提交。vibed 以其固定 Dockerfile 和 Isaac Lab 栈为准，预检不据包版本推断整个容器已经兼容。

源码目录必须是仓库根目录。检查会清除继承的 `GIT_*` 路径和配置覆盖，核对固定提交及已跟踪文件状态，并将必需文件的字节哈希与提交中的 blob 比较。Git 对象查询分批执行，每批最多 128 个路径，路径参数总计不超过 64 KiB；文件只读取一遍，同时计算 SHA-256 和 Git blob 摘要。对 `assume-unchanged`、`skip-worktree` 隐藏的现存文件也会比较内容；两种标记组合使用时，真正缺失的非必需稀疏文件仍可保留；悬空符号链接属于现存目录项，会被拒绝。检查不修改索引标记。这是源码预检，不表示运行环境或模型张量已验证。

退出码：`0` 为本次前置检查通过；`2` 为缺少条件或命令参数无效；`1` 为探测子进程或文件处理失败。取消和超时会分别记录 `interrupted`、`timed_out`。输出包含 `run.json`、`result.json`；运行过 SDK 探测时还包含 `sdk.json` 和 `sdk.log`。缺少 checkpoint、契约或动作会返回拒绝记录，不视为已完成推理。

## 直接运行

在仓库根目录运行即可，不需要激活环境或设置环境变量：

```bash
./supermimic play                 # 浏览器查看行走跟踪
./supermimic eval                 # 评测全部40条动作
./supermimic train --hours 1      # 从默认模型继续训练，训练预算1小时
./supermimic doctor               # 检查本地源码、模型哈希与动作完整性
```

[工作区入口](../supermimic)自动使用独立 SDK、已通过原配置训练校验的 58,000 步模型和恢复后的动作数据。浏览器地址以终端输出为准；远程服务器需要转发对应端口。`play --clip dance1_subject1` 可更换动作，`play --viewer native` 使用桌面窗口。按 `Ctrl+C` 停止。

入口只使用已有本地文件，不安装依赖或下载模型；其他缓存、模型和动作目录可分别通过 `--root`、`--checkpoint`、`--motions` 指定。自定义 checkpoint 必须来自可信来源。`doctor` 不执行策略，也不表示 CUDA 推理已通过。原 `embodiedforge supermimic preflight` 继续保持离线元数据检查语义。

每次实际运行生成独立的 `runs/supermimic/<时间>-<操作>/`，保存日志和实际子进程退出状态。`play` 只复制所选动作；评测和训练复制各自需要的动作集。在加载 CUDA 前检查输入哈希与 ZIP CRC，使用独立只读副本，运行结束再次校验副本。共享源目录后续变化不会直接替换运行中的副本；只读权限不能保证免受所有存储或内存故障，末尾校验仍保留。训练上限只限制训练循环，准备和评测另计；`--iterations` 仍是累计迭代目标。

缩小网络的闭环验证可用 `./supermimic train --fresh --profile smoke --num-envs 8 --iterations 60`。它不能作为模型质量或论文规模结果。已有 smoke 模型续训时需同时指定 `--checkpoint` 和 `--profile smoke`；689维锚点训练另需 `--anchor-obs`，属于改变论文输入条件的独立实验。

运行入口、输入隔离与受控故障验证见[本轮运行记录](experiments/supermimic-runtime-20261027.json)。

## 后空翻专项训练（SuperMimic / mjlab）

后空翻训练使用 **SuperMimic 世界模型与可微想象 / BPTT 策略更新**；mjlab 承载 G1 物理仿真。[参考项目](https://github.com/liudi159/supermimic-backflip-5090/tree/5b313d75e804cdd2ff1ce9bdcfa899a98eca856b)的动作准备流程已固定版本，mjbatch 只提供目标动作，不运行它的 iLQR 控制器，也不加载外部训练好的策略。

```bash
./supermimic prepare --task backflip
./supermimic train --task backflip --hours 1.5
./supermimic eval --task backflip
./supermimic play --task backflip
```

训练首次从随机初始化开始；后续默认继续最近一次正常完成的专用训练。加 `--fresh` 可从零开始独立一轮，`--checkpoint` 可明确指定可信模型。默认种子137、512环境、世界模型batch512、策略batch256、H=32，首次500步收集回放后开始联合更新；这与参考工程的500次世界模型单独预热不同。恢复时重新填充20步回放历史。90分钟上限只限制训练循环，准备和评测另计。`--iterations` 为累计目标，默认100000。

参考原始 `flip.npz` 固定在 mjbatch 提交 `77966f85bcd8f7ef4351cb4a1a6f42e133d19725`，SHA-256为 `f9ad1275f978bf891ee141e272d8279eb56685e4209f42d5cd7dc23529bcb4d6`。取30–240帧（含端点），前补10个首帧、后补100个末帧，共321帧、50Hz；CSV哈希 `0d7dcb9ac79834bec3f3db16b5a0d8e582b71b3854435d09ad3b886631adb007` 与参考工程完全一致。转换时按当前 G1 的29个关节顺序重新计算FK，不重复按30Hz插值。评测追加250个末帧，即额外5秒保持。此动作来源为Apache-2.0的mjbatch，独立于40条LAFAN1动作。

数据写入 `runs/supermimic/tasks/backflip/data/`；训练运行仍各自保存只读副本。`current.json`仅登记最近一次正常完成训练的最后检查点，不代表后空翻成功。未有专用模型时，后空翻播放和评测会明确报错，不回退到行走模型。小配置验证可用 `train --task backflip --profile smoke --num-envs 8 --iterations 60`；它不登记为正式专用模型，也不能作为技能成果。

本路线是当前 Cuboi/mjlab 实现上的单动作移植；参考工程使用 IsaacLab3/vibed 及不同资产、观测和碰撞设置，不能等同复现其 V14。专项评测保存 50 Hz 实际策略状态、累计横轴旋转、由保存姿态重建的非脚部地面穿透、末 2 秒保持指标。还检查固定参考时间窗口 `[90, 118)` 的髋膝角误差和脚踝间距：前者需小于 25°，后者不超过参考最大间距加 0.15 米。该窗口不随实际腾空时间移动，区别于参考工程的最长实际腾空窗口验收；未完成的 200 Hz 受力、腾空旋转等严格检查仍标记为 `strict_acceptance: not_assessed`。姿态检查失败时，`quality_checks_passed` 和 `skill_success` 明确为 `false`。

2026-10-27 实测：从单独记录的 2500 步先导模型继续训练 90 分钟，最后模型累计 36900 步、14552 次更新；选择实际最后检查点，三个独立进程均正常退出。固定评测种子 0 下，三次均后翻约 358.48° 并完成额外 5 秒保持，50 Hz 几何检查未发现非脚部地面穿透。不过固定窗口髋膝误差约 46.21°，最大脚踝间距约 1.269 米，超过本轮 25° 和约 0.545 米的限制，**姿态质量未达标**。这三次检查是同一模型的进程间复测，不代表独立训练种子的鲁棒性。179 项相关测试和 63 项文档检查通过。[实际轨迹图](experiments/assets/supermimic-backflip-20261027-progress.svg)与[完整训练、评测及证据记录](experiments/supermimic-backflip-20261027.json)保留中间失败和最终限制；不能凭回报或动画宣称完成高质量后空翻复现。

本轮四小时优化已将完整 vibed SuperMimic 算法适配到现有 mjlab：保留 99 维观测、730 维目标、10 帧历史、骨盆奖励锚点、next-frame 解析奖励与终止、H=32 想象更新，以及 500 次世界模型单独预热。没有使用官方权重、外部控制器或强制动作对称补丁。两个独立随机种子各从零训练 90 分钟策略预算；这四小时还包含适配、预热、诊断与评测，并非四小时 GPU 策略训练。

| 训练种子 | 最后迭代 / 策略更新 | 固定窗口髋膝误差范围 | 最大脚踝间距范围 | 首次力审计的腾空时长 / 腾空后翻角度 | 实际最后模型评测 |
| --- | --- | --- | --- | --- | --- |
| 137 | 19695 / 19195 | 7.38–7.92 ° | 0.389–0.390 m | 0.62 s / 272.9° | 3/3 姿态评测通过；另做 1 次完整物理评测通过 |
| 138 | 21847 / 21347 | 6.10–6.27 ° | 0.383–0.383 m | 0.60 s / 267.5° | 3/3 完整物理评测通过 |

评测保留原始时间，不在失败后重置。初始状态加 570 次真实控制覆盖 571 个状态，并记录 2,280 个 200 Hz 物理子步的实际约束力，分别检查腾空、姿态、非脚部力、负接触距离、容量溢出和额外 5 秒保持。几何分析使用该次实际编译模型；接触力来自设备求解结果。两个种子分别报告，不能把同一模型的进程复测当成训练种子统计或泛化证明。

当前工作区已将后空翻短指令切换至验证后的 reference 算法族，无需设置环境变量：

```bash
./supermimic play --task backflip
./supermimic eval --task backflip
./supermimic train --task backflip
```

首次训练从零开始，后续继续这一实现最近一次正常完成的最后检查点；`--fresh` 开始独立训练，`--hours` 设置策略更新预算（默认 1.5 小时），`--iterations` 是全局迭代目标。训练与评测自动选择本地专用 SDK。`play` 使用真实策略驱动物理仿真并在片段结束后循环，`eval` 连续执行一次完整物理验收。Viser 服务及真实策略循环已验证，浏览器侧 WebGL 交互未实测。

reference 使用独立登记 `runs/supermimic/tasks/backflip/reference/current.json`；`--recipe cuboi` 可访问原实现，两套网络不能混载。登记只代表训练正常完成，后续新模型需要重新评测。物理资产、求解器和批大小仍与 IsaacLab3 参考工程不同，这是当前 mjlab 上的完整算法复现，不宣称精确 V14 等价或实机成功。两种实现的控制计数和奖励定义不同，平均回报不作跨实现比较。[实际最后模型对比图](experiments/assets/supermimic-backflip-20261027-fourhour-comparison.svg)与[完整训练、失败和物理评测记录](experiments/supermimic-backflip-20261027-fourhour.json)。

## 独立 SDK 复现入口

[研究脚本](../benchmarks/supermimic_reproduce.py)已在 CuboiLeo 独立 SDK 中执行真实 G1 / MuJoCo Warp 仿真、世界模型训练、BPTT 策略更新、checkpoint 重载和逐片段评测；它与上述离线预检是不同入口。初始结果见[2026-10-26 记录](experiments/supermimic-reproduction-20261026.json)，后续训练和锚点对照见[四小时实验](experiments/supermimic-reproduction-20261027-fourhour.json)。这些是独立复现代码训练出的本地模型，不是官方策略。

本地环境使用 Python 3.12、Torch 2.14.1/cu130、MuJoCo 3.16、Warp 1.18 和表中的固定 mjlab 提交。安装使用 `uv sync --locked --group dev`，保留提交中的 `uv.lock`；未锁定安装可能重新解析依赖。源码校验会检测 Git 的稀疏或隐藏标记下出现的锁文件差异，不能因 `git status` 看起来干净就忽略它。

数据使用 HF `lvhaidong/LAFAN1_Retargeting_Dataset` 提交 `ce1572906efe6157840e8474d5a0d7aa87481e74` 中的 40 条 G1 CSV；上游转换器将 30 Hz、根四元数 xyzw 转为 50 Hz、刚体四元数 wxyz，共约 2.451 小时。每个下载文件均核对远端内容标识。CSV、转换数组和模型保留在本地缓存/运行目录，报告仅记录来源、哈希和指标。

在仓库根目录，用安装好固定项目的 SDK Python 运行。例如以下是缩小配置的闭环检查，并非论文训练规模：

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .cache/external/supermimic-reproduced/.venv/bin/python -I \
  benchmarks/supermimic_reproduce.py \
  --root .cache/external/supermimic-reproduced \
  --motions .cache/external/supermimic-reproduced/data/npz/lafan1 \
  --output runs/supermimic/smoke --iterations 60 --num-envs 64
```

`--profile lafan` 使用全部 40 条动作、论文宽度的网络和 H=32；环境数及训练批量按本机显存缩减，配置完整记录，不能作为论文规模的结果。`--iterations` 是总迭代目标，不是额外步数；`--max-hours` 是训练循环的墙钟上限，环境准备和评测另计。通过[监督器](../benchmarks/process_supervisor.py)运行可同时记录真实进程退出，不能只依据 worker 写出的 `complete` 字段。

恢复仅接受自己生成或明确可信的 checkpoint。上游 checkpoint 不保存 replay buffer，直接沿用旧的绝对 warmup 阈值会在首次采样时报错；研究入口恢复后补充至少 20 步真实仿真以重建历史/窗口，然后才允许更新。模型、优化器和迭代继续恢复，但不声明恢复完整随机数状态或与不中断训练逐位相同。研究脚本也会为直接调用创建动作副本；可用 `--motion-manifest` 指定含 `files` 哈希映射的 JSON，核对预期输入后再启动 SDK。

`--anchor-obs` 是增加 9 维目标观测的独立实验，目标由 680 变为 689；它不再满足论文观测条件。[锚点初始化脚本](../benchmarks/supermimic_anchor_init.py)为初始策略的目标输入末尾扩展零权重列；critic 则在目标块之后、输入偏移 751 处插入（列索引从 0 开始），保留后续状态输入的位置。同时扩展对应 Adam 动量和缩放缓冲区。CPU/CUDA 固定输入验证初始化函数在数值容差内一致；另用 CPU float64、额外特征全零的真实 Adam 单步检查旧参数及优化器状态的投影一致性。这些控制不保证真实非零锚点观测下的轨迹或后续更新相同。

评测同时报告失败片段数、上游按实际存活步数归一化的回报，以及将提前失败后的剩余片段补零的完整片段回报。较高的存活步数回报可能伴随很早失败，不能单看该数值宣称跟踪成功。40 条动作同时用于训练与这里的覆盖评测，不是未见动作上的泛化实验。

本轮锚点长跑在末尾发现 `sprint1_subject2.npz` 的两个字节与初始哈希不同，ZIP CRC 校验失败，实际退出码为 1；该训练记录按失败保留，不因已生成模型而改为成功。从原 CSV 重建的文件与最初 SHA-256 完全一致，后续共同步数的推理使用逐文件核验过的只读数据副本。该推理结果与原训练的完整性状态分开报告，不能追认原训练通过。字节变化的原因尚未确定。

本轮在核验后的相同 40 条训练动作上，每组做 3 次独立进程复测，原配置的完整片段回报均值由 0.280 提高到 0.521。两组共同检查点均为 58,000 步、23,176 次优化器更新，按最大共同已保存步数选定，不按评测分数挑选。

| 模型 / 输入条件 | 全局步数 | 失败片段数范围（共40条） | 完整片段回报均值 |
| --- | ---: | ---: | ---: |
| 初始模型，680维目标 | 10,400 | 25–27 | 0.280 |
| 续训模型，680维目标 | 58,000 | 6–7 | 0.521 |
| 锚点模型，689维目标；原长跑终检失败 | 58,000 | 9–11 | 0.753 |

锚点模型的回报更高，但失败片段更多，且原长跑完整性终检失败，不能概括为整体优于原配置。恢复输入后的另一次短程训练完成 24 次更新，并通过重载及末尾哈希检查；这是新验证任务，不替代失败长跑。这里只运行一个训练种子，复测范围不构成统计显著性或泛化结论。

[训练诊断图](experiments/assets/supermimic-reproduction-20261027-fourhour-training.svg)区分训练回报与评测成绩；[共同步数评测图](experiments/assets/supermimic-reproduction-20261027-fourhour-coverage.svg)展示均值和复测范围，图中保留锚点完整性限制。[证据归档](experiments/assets/supermimic-reproduction-20261027-fourhour-evidence.tar.gz)包含源码、日志、标量与校验记录，不包含模型、动作数组或视频。

2026-10-27 检查时，上游主分支已更新到 `ba6105349069bf040f1fdad60873d13138d505a2`，并发布新的独立复现结果。本轮训练仍固定旧提交 `d9be385...`，不将新主分支或其 checkpoint 指标混入当前实验。

## checkpoint 元数据契约 v1

JSON sidecar 由模型提供方或接入方根据实际模型填写，不能通过填写期望数值来证明权重兼容。以下为 CuboiLeo 默认配置的结构示例，关节名和 SHA 必须替换为实际值；`joint_names` 必须完整包含 29 个互不重复的名称，并按策略输出顺序排列。

```json
{
  "schema": "ef-supermimic-contract-v1",
  "backend": "cuboi-reproduced",
  "source_revision": "d9be385b3eeb13a638fb33fb200d3bfd9be9839f",
  "checkpoint_sha256": "实际 checkpoint 的 SHA-256",
  "robot": "unitree-g1-29dof",
  "control_hz": 50,
  "action_dim": 29,
  "history_frames": 10,
  "future_frames": 10,
  "observation_dim": 93,
  "observation_layout": "term-major",
  "goal_dim": 680,
  "anchor_obs": false,
  "joint_names": ["此处填写完整的 29 个真实关节名"]
}
```

vibed 的 `backend`、`source_revision`、`observation_dim`、`observation_layout`、`goal_dim` 分别换为 `vibed`、上表提交、`99`、`time-major`、`730`。当前契约固定 10 帧历史和 10 帧目标；其他配置需要新增明确版本，不会自动猜测。

预检只校验声明值与 checkpoint 文件哈希。普通文件读取由公共 `runtime.hashing.snapshot_file_hashes` 实现，与运行实现快照共用；此工具只依赖标准库。哈希先确认输入是普通文件，再核对打开后的文件身份；在系统支持时使用 `O_NOFOLLOW` 和 `O_NONBLOCK`，防止最后一级路径被替换为符号链接或 FIFO 后跟随目标或阻塞。读取量不超过打开时的文件大小加一个探测字节，文件增长或截断会拒绝，避免持续追读增长中的文件。哈希从同一个打开的文件读取，并检查文件与路径的设备、inode、大小、mtime 和 ctime，检测可观察到的重写或替换；它不对文件加锁，调用方仍应保持检查期间的源码与 checkpoint 不变。它不校验张量、动作内容或上述关节顺序在机器人资产中的含义，因此报告明确保留 `tensors_verified=false` 和 `content_verified=false`。动作文件只做存在性检查：最多扫描 10,000 个目录项、记录 129 个匹配文件、进入 64 层目录，不跟随扫描到的符号链接，也不读取文件内容。结果记录扫描量和截断标记；达到上限但未发现匹配项时，不能据此断定整个目录没有动作文件。计数不是完整数据集统计。

## 动作格式与 GMR 转换边界

平台当前 GMR 回放 NPZ 包含 `metadata`、`time`、`positions`、`quaternions`、`joints` 等字段，姿态采用 `xyzw`。它不能直接作为 CuboiLeo 的训练或评测动作包使用。

CuboiLeo 的[转换器](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/data/motion_convert.py)输出包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`joint_names` 和 `body_names`。其 G1 转换器使用 29 个关节、30 个刚体、`wxyz` 刚体四元数，默认重采样至 50 Hz。后续转换必须基于匹配的 G1 模型补齐速度和刚体运动学，核对名称顺序，并明确源时间戳与首帧时刻；不能仅重命名 NPZ 字段。

vibed 的[动作加载器](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/motion.py)读取无表头 CSV：每行依次为根位置 3 维、根四元数 `xyzw` 4 维和关节角 29 维，默认源帧率为 30 Hz，再按 50 Hz 控制时钟重采样。关节序必须与它自己的 G1 运动学模型匹配。

离线预检入口不会执行这些转换，也不会验证数组字段。即使 `motion_data_present` 通过，已有 GMR NPZ 或其他同扩展名文件也不能据此判定兼容。独立研究入口已使用 CuboiLeo 的转换器处理固定来源的 LAFAN1 数据；平台 GMR 输出到对应 SDK 格式的转换与验证仍需单独实现。

## 官方推理接入与评测的验收条件

官方策略仍需在获得对应授权的模型与完整导出说明后，单独实现适配器，并先以来源项目自己的 evaluator 为对照验证固定输入的动作输出。随后校验 G1 资产、步长、历史缓冲、归一化、关节映射和 PD 控制，再进行固定动作集、多种子、逐片段的仿真评测。

最终报告应包含来源与权重哈希、数据集清单及许可证、SDK/资产版本、种子、初始化与终止规则、失败片段和耗时。只有同一数据划分和观测条件下的结果才能并列比较。官方模型、CuboiLeo 和 vibed 使用不同来源标签；离线预检记录没有行为评测指标；独立研究结果另行报告，不代表官方策略已经接入。
