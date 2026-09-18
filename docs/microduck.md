# Microduck RL

直接开始正式训练、续训和部署，见 [机械鸭快捷命令](microduck-quickstart.md)。

模型参数量、权重体积、预训练与后训练流程，以及已有结果和消融实验设计，见 [机械鸭模型与实验详解](microduck-model-training-ablation.md)。

按步骤执行安装、训练、续训、评估和运行，见 [训练与部署命令](training-deployment.md)（[English](training-deployment.en.md)）。

EmbodiedForge 现在维护 Microduck **平地行走任务的本地移植**：任务配置、奖励与观测、对称增强、BAM 执行器扩展、机器人 MJCF 和 38 个网格资产均位于 `src/embodiedforge/locomotion/microduck/`。默认运行不再导入或读取 `3rdparty/microduck_rl`。训练仍使用 mjlab / MuJoCo-Warp / BAM / RSL-RL，不属于核心 `VectorEnv` 或 Newton 点质量适配器。

移植基线是 Microduck `53b8971b61baf5b7f3c16d135dd7cac37623de4b`。保留 Apache-2.0 许可证，`NOTICE` 与 `UPSTREAM.json` 记录改动、源文件及原始散列。当前只注册平地行走任务；共享 MDP 辅助代码及其 RewardManager/PPO 扩展暂时保留，其他上游任务、HF Jobs 和实机控制程序不在此次范围内。

主入口继续使用 `ef`。Python 3.12 工作环境独立安装 mjlab 1.3.0、Warp 1.12.0、PyTorch 2.9.1 和固定提交的 BAM；仓库中的 `requirements.txt` 从基线 `uv.lock` 导出，排除了原仓库的任务插件。`check` 输出实际任务、执行器、runner、资产路径和实现散列，以区分本地实现与历史上游运行。

## 安装与检查

从 EmbodiedForge 根目录执行；需要 `uv`、Git 和支持 CUDA 的 NVIDIA GPU。默认无需 `--repo` 或上游源码检出。

```bash
conda activate ef
python -m embodiedforge.microduck setup
python -m embodiedforge.microduck check
```

默认环境位于当前目录的 `.cache/microduck-native-venv`；从其他目录执行时，为所有命令添加相同的绝对 `--env-dir`。首次安装需要下载 Python 和数 GiB CUDA/PyTorch 依赖。`check` 验证 Python/SDK 版本、CUDA 可用性及实际 GPU 矩阵运算，并输出环境包版本。入口本身不在调用者 Python 中导入训练 SDK。

## 无窗口训练（headless）

`train` 和续训 默认采用 headless，也可显式添加 `--headless`。启动器在导入训练 SDK 前清除子进程的 `DISPLAY` / `WAYLAND_DISPLAY`，设置 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`，并明确关闭训练视频。无需桌面、X11、Wayland 或 Xvfb，不启动原生窗口或 Viser 服务；父终端环境不受影响。

```bash
python -m embodiedforge.microduck train --headless --quiet --seed 0 \
  --num-envs 512 --iterations 4000 --output runs/microduck-headless
```

`run.json.execution` 记录 `headless`、查看器、训练视频及 GL 后端。`runtime.json.process_environment` 记录工作进程实际看到的显示服务变量、GL 后端和 CUDA 设备选择，`gpu_memory` 记录检查当时的空闲/总显存（不是训练峰值估计）。训练日志与 checkpoint 正常保存，可另开终端运行 `progress`。headless 保留训练所需的仿真与传感器计算，仍需要 CUDA 和足够显存；它不能解决显存不足。交互查看策略使用独立的 `play` 命令。

## 训练输出

输出目录必须不存在，避免混入旧 checkpoint。

本地训练仅保存 checkpoint，不在每次保存时自动导出 ONNX。需要部署模型时，使用独立的 `export` 命令。

本地 TensorBoard 训练保存 checkpoint 时，先写同目录临时文件，成功后原子替换正式 `model_*.pt`。常规异常或中断会清理临时文件，旧完整模型保留；SIGKILL 可能留下 `.tmp`，进度查询和模型选择不会读取它们。这不是断电持久性保证。

本地训练和评估启动前会将当前 EmbodiedForge 包的源代码与资产复制到 `implementation/embodiedforge/`，并逐文件核对启动时记录的散列；复制期间源文件变化会立即失败。后续工作进程从这份快照加载任务，因此开发期间修改工作区不会改变正在运行的任务或后续导出。多种子评估只保存一份快照；独立 SDK 依赖仍使用固定环境，不复制到运行目录。显式 `--repo` 的历史模式仍使用原先的固定检出检查。

`train` 和 `evaluate` 支持 `--quiet`：工作进程的 stdout/stderr 直接写入日志文件，终端只显示启动器的阶段信息，无需额外的日志转发线程。默认情况下，各阶段的 stdout/stderr 同时输出到终端并保存为 `00-check.log`、`01-train.log` 等文件；续训多一个 checkpoint 检查，序号会相应变化。`run.json.logs` 保存实际列表，`phase` / `active_log` 指向当前阶段与日志。失败时保存 `failed_phase`；CUDA 等运行前检查失败时额外保存 `runtime.failure.json`，并将诊断写入运行记录。`run.json` 记录移植基线提交、依赖清单与本地实现散列、执行参数、状态和 checkpoint 路径；`runtime.json` 记录实际环境；`logs/rsl_rl/microduck/` 保存上游参数 YAML、TensorBoard 事件和模型。发生错误或 Ctrl+C 时保留已产生的模型，并更新运行状态。

## 训练、回放与导出

直接设置训练预算，例如：

```bash
python -m embodiedforge.microduck train \
  --num-envs 4096 --iterations 4000 \
  --output runs/microduck-walk
```

`--iterations` 必填，避免意外使用上游 50,000 次迭代默认值。`train` 和续训 支持 `--seed`（默认 0，范围 0 到 `2**32-1`），并将其写入运行记录；续训 seed 控制重建环境的随机状态，不是恢复原进程的全部 RNG。使用本地 TensorBoard logger；未启用 W&B 上传或 Hugging Face Jobs。GPU 选择沿用上游约定与 `CUDA_VISIBLE_DEVICES`。环境数需要按可用显存调整，4096 只是训练配置示例。

训练期间可从另一个终端读取已落盘进度：

```bash
python -m embodiedforge.microduck progress \
  --run runs/microduck-walk
```

该命令在隔离环境中读取 TensorBoard 事件，不加载模型或使用 CUDA。本地工作进程的 `progress`、指标检查、checkpoint 元数据读取和 ONNX 推理检查均跳过任务注册，不再加载 mjlab、MuJoCo、Warp 或 BAM；训练和导出仍正常注册任务。输出本次已记录的更新数、最新迭代编号、最近 checkpoint、标量有限性、NaN 状态、奖励和吞吐。剩余时间按最近 20 次已记录迭代的采样与学习耗时中位数估计，不包括启动和最终导出。`run_status` 来自运行记录；应结合 `seconds_since_last_event` 判断日志是否仍在更新，它不是进程存活检查。另有 `launcher_status` 在同一 Linux 主机上结合 PID、进程启动时间和系统启动标识检查启动器；旧记录、跨主机或无法读取 `/proc` 时为未知。它只检查启动器，不保证训练正在推进；应同时看状态、最新事件时间和阶段日志。事件尚未写入时，无法判断的指标为 `null`。

`progress` 只需要运行目录，以及所选 `--env-dir` 中可用的 Python 和 TensorBoard；不检查当前任务源码、上游仓库或训练环境版本戳。源码和依赖更新后仍可查看旧日志，旧上游仓库也不必保留。

新训练在创建 `run.json` 时即保存 `num_envs` 与 `iterations`，准备阶段也能从 `progress` 输出的 `requested_updates` 查看目标更新数。续训时该数值表示本次追加的预算；旧记录仍从已保存的训练命令读取，尚无训练命令时返回 `null`。

剩余时间仅使用有对应 PPO 更新记录的耗时。迭代编号必须从本次起点连续增长且不超出预算；缺失、错位或超出预算时，仍展示已读到的更新数，但剩余时间返回 `null`。

`progress` 保留原记录的 `run_status`，另给出 `effective_status`：如果记录仍为 `running`，但本机进程身份检查已确认启动器退出，则为 `orphaned`，剩余时间返回 `null`。跨主机或无法确认进程身份时不会推断进程已死，仍需结合 `launcher_status` 判断。检查、导出、指标验证等非训练阶段不估算训练剩余时间；原记录不会被查询修改。

`latest_checkpoint` 只从运行目录内的规范 `model_<数字>.pt` 文件中选择，忽略临时文件、异常名称、同名目录及目录外链接。TensorBoard 日志与 checkpoint 必须属于同一个训练会话；即使两者各只有一个目录，只要目录不同，也会拒绝混合展示。`checkpoint_validation=not_performed` 表示只找到了候选文件，未加载验证其内容；续训入口仍会检查完整 checkpoint、学习率和课程计数。

训练完成后，`play`、`evaluate`、`export` 可直接传 `--run`，自动选择运行记录中的 checkpoint：

```bash
python -m embodiedforge.microduck play \
  --run runs/microduck-walk --viewer native

python -m embodiedforge.microduck export \
  --run runs/microduck-walk --output runs/microduck-policy.onnx
```

回放使用 mjlab 的 MuJoCo 原生查看器，也支持 `--viewer viser`。它与 EmbodiedForge 的点质量 RTX 示例是不同的入口。

独立 `export` 默认 headless，清除桌面显示变量并使用 EGL，仍需 CUDA；可加 `--quiet` 只保存工作进程日志。`--output` 必须以 `.onnx` 结尾。导出先冻结当前本地代码、资产和输入 checkpoint，再调用本地 exporter 与 mjlab runner，并验证 ONNX 的 61 维输入、14 维输出和有限推理结果。`--run` 使用该运行的 checkpoint，导出实现取本次启动时的本地代码。

关节元数据由本地 exporter 按动作管理器的实际目标顺序生成，仅包含受控关节；默认角和逐关节缩放保留完整浮点精度。任务加载不再全局替换 SDK 的元数据函数。动作转换与硬件接口边界见 [模型与实验详解](microduck-model-training-ablation.md#66-从网络动作到关节目标)。

只有验证成功后才发布 ONNX 和同名 `.validation.json`；已有模型或验证报告均拒绝覆盖，包括并发导出的同名输出。导出或验证失败、正常处理的中断不会留下最终模型，可用同一条命令重试。每次尝试在输出旁保留独立的 `.<文件名>.export-<随机串>/` 目录，内含 `run.json`、代码/模型快照、阶段日志和临时产物；启动日志打印其路径。成功报告记录模型及 checkpoint 的 SHA256，并指向这次导出记录。发布通过同一文件系统上的硬链接完成，输出目录所在文件系统需支持硬链接；强制断电或 SIGKILL 不保证回滚。

最终导出记录保存失败或此时收到中断，也会撤回本次发布的 ONNX 和验证报告，保留尝试目录供排查，并允许使用同一输出路径重试。
若文件权限等问题导致回滚删除失败，日志会列出未能删除的路径；仍会尝试清理另一个文件，并保留原始导出异常或中断。重试前需处理这些残留文件，输出覆盖保护仍然有效。

`--run` 接受已完成的 `train` 目录，并核对任务、源码 revision 和锁文件哈希。
入口只使用记录中的模型，不扫描选择最大文件名；训练仍在进行、运行失败、模型丢失或已记录
SHA256 不匹配时会拒绝。新训练记录保存 `checkpoint_relative` 和 `checkpoint_sha256`。
旧记录仍兼容，但缺少历史哈希时会提示 `unverified_legacy`，只记录当前文件的哈希。
完整训练目录搬迁后，模型从新目录内解析，不会回退到记录里的旧绝对路径。

仍可用 `--checkpoint PATH_TO_MODEL.pt` 指定历史模型；它与 `--run` 二选一。
续训对应 `--resume` 或 `--resume-run`，同样互斥。通过运行目录发起的评估和续训会保存
`source_run`，包含来源 manifest 与模型哈希；加载后的模型哈希与选定来源不一致会将运行标为失败。
这些检查不代替行走效果验收，也不改变原来的 CUDA 训练环境或实机支持范围。

原生回放支持按步数结束，便于检查模型和窗口启动：

```bash
python -m embodiedforge.microduck play \
  --checkpoint PATH_TO_MODEL.pt --viewer native --steps 250 --seed 0
```

`--steps` 是仿真步数停止阈值，暂停时不会增加；退出时打印实际步数和是否被中断。不传该选项则持续回放到关闭窗口或 Ctrl+C。原生回放现在与无窗口评估共用模型加载和环境清理逻辑，默认 seed 为 0。Viser 使用 mjlab 查看器并注册本地任务，保留 checkpoint 浏览/切换功能，不接受 `--steps` 或非默认 `--seed`。

MuJoCo 3.10.0 的 passive viewer 在守护线程内渲染，`Handle.close()` 只请求退出。本项目的局部子类等待本次创建的渲染线程结束，再释放环境并退出进程，避免 GLFW 的进程退出清理与窗口销毁并发。该适配使用固定版本的私有线程入口标识，启动前检查 MuJoCo 版本；没有修改上游安装文件或屏蔽警告。

Linux 启动器为子任务创建独立进程组：第一次 Ctrl+C 转发 SIGINT，并给予最多 10 秒清理时间；再次 Ctrl+C 或清理超时才强制结束。SIGTERM 和未被忽略的 SIGHUP 也走同一清理路径；退出码分别为 143、129，Ctrl+C 为 130。运行记录保存 `interrupted` 和 `interrupt_signal`，保留此前已写出的 checkpoint，不把未完成训练标为成功。继承的 `nohup` SIGHUP 忽略设置保持有效。

后台运行可使用：

```bash
nohup python -m embodiedforge.microduck train --headless --seed 0 \
  --num-envs 512 --iterations 4000 --output runs/microduck-background \
  > microduck-background.log 2>&1 &
echo $!  # 记录本次启动器 PID
```

停止本次任务时，对记录的启动器 PID 执行 `kill -TERM PID`。正常中断后可直接用 `--resume-run 原运行目录` 续训，也可从 `progress` 报告的 `latest_checkpoint` 通过 `--resume PATH` 显式选择模型。SIGKILL 无法执行清理，也无法保证更新运行状态；仍标为 `running` 的目录不自动恢复。

## 断点续训

```bash
python -m embodiedforge.microduck train \
  --resume-run runs/microduck-walk --num-envs 4096 --iterations 1000 \
  --output runs/microduck-resumed
```

`--iterations` 表示**追加**的 PPO 更新数。入口在新运行目录复制输入 checkpoint，记录 SHA-256、原始路径、课程计数与学习率；原训练目录不被覆盖。仅支持当前上游格式的完整训练 checkpoint，需要 actor、critic、优化器和环境课程计数。

`--resume-run` 接受已完成的训练，也接受具有 `finished_at` 的 `interrupted` / `failed` 训练。恢复中断/失败运行时，只从其 `run.json.checkpoints` 清单中按迭代编号选择最新模型，不扫描目录里的其他文件，不读取 `.tmp`；若选定模型丢失、清单跨多个会话、编号重复、路径越出运行目录，或已确认原启动器仍存活，则拒绝自动恢复。选中的 checkpoint 损坏时会失败，不静默回退到较旧模型；可用 `--resume PATH` 明确选择其他保存点。整个运行目录迁移后仍从新目录解析记录路径。

恢复来源写入新运行的 `source_run`，包含原状态、选择方式与 SHA256。已有历史哈希会被核对；没有历史哈希的中断保存点标记 `verification=recorded_at_recovery`，表示哈希在恢复时取得，不声称验证了历史内容。复制后的模型再次核对哈希，并检查完整训练状态。`play` / `evaluate` / `export` 的 `--run` 仍要求训练已完成。

上游恢复网络、归一化、优化器及 `common_step_counter`。本入口额外从优化器读取学习率，写入上游 `TrainConfig.agent.algorithm.learning_rate`，使 RSL-RL 的独立自适应学习率变量也从保存值启动。指标验证按实际起始编号检查：上游从保存的编号重新编号，例如 `model_4.pt` 追加 2 次更新，日志编号为 4、5，最终保存 `model_5.pt`，课程计数仍增加 48 个控制步。不要仅按文件名推断累计更新次数。

续训还会通过上游 `startup` 事件在首次 reset 前恢复课程计数。原入口的顺序是先创建 `RslRlVecEnvWrapper`（立即 reset），再加载模型；仅等待模型加载恢复计数，会让首个 episode 使用训练初期的课程配置。本入口仍调用上游训练 launcher、runner 和模型加载，只在本次配置中增加计数恢复与首次 reset 记录事件，不修改上游源码或安装文件。

`resume.initialization.json` 保存首次 reset 在课程计算之后实际使用的计数、动作变化惩罚权重、站立指令采样比例、头部指令范围及 checkpoint 散列；其中站立比例是配置概率，不是本次采样中实际站立的环境占比。后续 reset 不再回写计数。运行结束时会校验记录与输入 checkpoint 一致；缺失或不一致会将本次运行判为失败，成功记录也写入 `run.json.resume_initialization`。

这是上游的 checkpoint 续训语义：重新建立仿真环境，不恢复每个环境的物理状态或随机数状态，不能视为原进程的逐步精确续接。

## 无窗口评估

```bash
python -m embodiedforge.microduck evaluate \
  --checkpoint PATH_TO_MODEL.pt --num-envs 16 --steps 250 --seed 0 \
  --output runs/microduck-evaluation
```

评估按指定步数自动结束，不需要桌面窗口。`evaluation.json` 包含 checkpoint 散列、每步平均奖励、完成的 episode 数和平均长度、仍未结束的 episode 长度，以及各终止原因的计数。始终检查观测、动作和奖励是否有限；NaN 状态导致失败。`run.json` 和 `runtime.json` 记录运行状态与环境。

默认评估采用上游 `play=True` 配方，保留它的随机指令和推扰，策略使用确定性动作并应用观测归一化。各终止原因可能同时触发，计数不能相加后当作 episode 数；未结束的 episode 也不计入完成 episode 的平均长度。

固定前进指令并检查导出模型：

```bash
python -m embodiedforge.microduck evaluate \
  --checkpoint PATH_TO_MODEL.pt --onnx PATH_TO_POLICY.onnx \
  --velocity 0.2 0 0 --no-pushes \
  --num-envs 16 --steps 250 --seed 0 \
  --output runs/microduck-fixed-evaluation
```

`--velocity VX VY WZ` 使用机体坐标系，单位依次为 m/s、m/s、rad/s；头部与身体姿态指令设为相对 HOME/标称站姿的零偏移。环境重置和定时采样都保持这些值，关闭会改写指令的课程、随机站立/转向/世界坐标系模式；每步及末状态检查实际指令。`--no-pushes` 单独控制外部推扰。BAM、其他域随机化和奖励课程仍采用上游配置，不代表无随机性的理想环境。

`--onnx` 每步选一个环境（步号对环境数取余），将策略收到的原始 61 维观测同时送入 ONNX CPU 推理，与 PyTorch 裁剪前的 14 维确定性动作比较。这能发现导出模型错配、归一化遗漏等问题。对照模式关闭 TF32，使用完整 FP32，逐元素容差为 `abs(onnx - torch) <= 1e-4 + 1e-4 * abs(torch)`。仿真仍由 PyTorch 动作驱动；这是实际观测抽样对照，不能替代 ONNX 独立闭环或实机验收。

报告新增 `conditions`（指令、推扰、计算精度、指令检查次数）及 `onnx_parity`（模型散列、样本数、最大绝对误差、容差）。失败退出码非零，`run.json` 标为失败，worker 的具体错误写入 `evaluation.failure.json`。比较策略时应保持条件与精度一致，覆盖多个种子；奖励权重受课程影响，不能只用平均奖励判断步态质量。相同种子也不保证 GPU 仿真逐位一致。

若磁盘写满等问题导致收尾时无法更新 `run.json`，启动器会保留原始失败或中断，并在日志中提示记录保存失败；此时文件可能仍是上一次保存的状态。任务正常完成但最终记录写入失败时，命令仍报错退出。

ONNX 对照报告必须满足 `samples == steps_per_env`，每步按环境编号轮换采样；推理后端为 `CPUExecutionProvider`，PyTorch 策略使用完整 FP32，`atol` 和 `rtol` 均为 `1e-4`。在线完成检查、离线验收和策略对比都会拒绝缺失/不足的样本、改写的对照配置，以及非有限或负的最大绝对误差。报告中的 ONNX 路径也须与运行命令一致。最大绝对误差用于汇总，不能单独与 `atol` 比较判定成功，因为逐元素允许误差还包含相对容差项。

`evaluate` 与 `export` 均默认 headless，也接受显式 `--headless`。评估在启动 CUDA 检查和任务工作进程前清除 `DISPLAY`、`WAYLAND_DISPLAY`，覆盖继承的 `MUJOCO_GL` / `PYOPENGL_PLATFORM` 为 `egl`；无需桌面或 Xvfb。`--video` 使用离屏渲染，可与 `--headless --quiet` 同时使用，多种子评估保持相同设置。单次和批量 `run.json` 的 `execution` 记录 headless、录像与 GL 配置，各种子的 `runtime.json` 记录检查工作进程实际环境；交互式 `play` 保留查看器设置。

评估工作进程报错时，具体异常保存在 `evaluation.failure.json`，并关联到同目录 `run.json` 的 `evaluation_failure`。例如 ONNX 对照失败会保留步骤、环境编号和动作误差；多种子评估查看对应 `seed-N/`。若工作进程未能生成该文件，则查看阶段日志及记录的退出错误。

评估在 GPU 检查前将 checkpoint 和可选 ONNX 复制到运行目录的 `inputs/`，核对复制前后的 SHA256；`--run` 还核对所选训练记录的模型哈希。多种子评估只复制一次，所有种子共用同一份输入，启动后替换或删除外部源模型不会影响本次评估。各次评估开始前检查快照完整性，结束后核对报告中的 checkpoint / ONNX 哈希。`run.json.input_snapshot` 保存原路径、实际快照路径和哈希；离线 `assess` / `compare` 也检查报告与快照记录是否一致。快照会占用一份模型存储空间。

步态评估还会输出以下指标，无需新增参数：

- `velocity_tracking`：按 `[vx, vy, wz]` 顺序记录平均指令、平均实际速度、MAE 和 RMSE，单位为 `[m/s, m/s, rad/s]`；`planar_velocity_rmse_m_s` 为 `sqrt(mean(vx_error² + vy_error²))`，不混入角速度误差。
- `velocity_tracking.per_environment`：按环境编号分别保留上述速度指标及有符号偏差（实际均值减指令均值）。每行覆盖这个环境槽的全部控制步，包含终止步与后续自动重启的 episode；它不是首次 episode 专属统计。可据此区分所有机器人都在缓慢移动，还是只有少数移动、其他停在原地。离线验收和对比会校验逐环境样本数、数据形状，以及它们与批量均值、RMSE 的一致性；旧报告没有该字段时仍可读取。
- `initial_episodes`：每个环境首次 episode 的持续时间、是否终止、是否仅到达上游时间上限，以及持续到完整评估时长且未终止的数量。后续自动重启不会覆盖首次记录，避免反复摔倒的环境在这项统计中被重复计入。

速度取机器人 root link 的机体坐标系真实状态，使用上游 metrics 回调在物理步之后、自动重置和指令重采样之前累计。每个环境步等权，包含终止步和未结束 episode；重置后的速度与新指令不会代替终止步的数据。采样与上游奖励/指标处于同一阶段，其派生运动学量可能滞后一个物理子步。

`right_censored=true` 表示本次观察没有看到首次 episode 的失败终止（评估结束或仅触发上游时间上限），记录时间是已观察到的下界。`mean_observed_duration_seconds` 是这些有限观察时长的平均值，不是完整寿命估计。`censored_before_horizon_count` 记录提前到达上游时间上限的数量，不能将它们当作已经坚持到整个评估结束。若同一步同时超时和失败，则按失败记录。

统计在 GPU 上累计，所需存储仅随环境数增长，结束时生成报告；不保存随步数增长的全部轨迹。checkpoint 的课程计数在首次 reset 前恢复，避免首个 episode 使用训练起始阶段的课程配置。

比较不同训练阶段的 checkpoint 时，添加相同的 `--curriculum-step 0`（或其他非负整数），将两者的**评估课程起点**统一。否则，即使速度和种子相同，保存的课程计数不同仍会改变其他随机化与奖励条件。该参数在首次 reset 前生效，报告记录实际起点及其来自 checkpoint 还是显式覆盖；其他课程随后仍随评估步数推进。

添加 `--video` 可同时将环境 0 录制为评估目录中的 `policy.mp4`；多种子运行在各 `seed-N/` 中分别保存。视频为 640 × 480、按仿真控制频率播放（当前 50 fps），帧数等于 `--steps`，摄像机跟踪机器人。录像使用上游 MuJoCo 离屏渲染器和隔离环境已有的 FFmpeg，逐帧编码。评估入口统一设置 EGL，不继承终端中的 GLFW/GLX 选择。无需桌面窗口，仍需要可用的 GL 渲染环境。

`evaluation.json.video` 记录帧数、帧率、文件散列、OpenGL 厂商/设备/版本和采样方式。CUDA 与 EGL 可能选择不同的 GPU，应以这里的 OpenGL 信息判断实际渲染设备。视频在每次控制步后采样，包含自动重置后的画面；速度与终止统计仍取重置前的 metrics 回调。录像只展示环境 0，不能代替全部环境的统计；启用录像的运行耗时也不适合与纯训练吞吐直接比较。退出或中断时关闭编码器和渲染器，失败运行中保留的部分视频不代表评估已完成。

同一条件下评估多个种子：

```bash
python -m embodiedforge.microduck evaluate \
  --checkpoint PATH_TO_MODEL.pt --onnx PATH_TO_POLICY.onnx \
  --velocity 0.2 0 0 --no-pushes \
  --num-envs 8 --steps 250 --seeds 0 1 2 \
  --output runs/microduck-seeds
```

`--seeds` 与 `--seed` 互斥，种子不可重复，取值为 `0..2**32-1`。每个种子独立启动、顺序执行并释放环境；总环境步为种子数 × `--num-envs` × `--steps`。每个 `seed-N/` 保存自己的报告、运行状态和运行时信息，批次根目录的 `run.json` 记录全部计划种子及已完成种子。

全部完成后生成 `summary.json`，保留逐种子指标，同时给出平均值、种子间样本标准差、最小值和最大值。一个种子的标准差为 `null`；这里的波动不是置信区间，也不把同一种子的环境当成独立训练重复实验。`pooled_velocity_rmse` 先合并各次等样本量评估的平方误差再开方，与 `metrics.vx_rmse.mean` 这种逐种子 RMSE 的平均值含义不同。首次 episode 的平均观察时间仍须结合截断标记解释。

每个速度轴同时汇总 `mean_command`、`mean_actual` 和 `bias`（实际均值减指令均值）。均值能区分站立与实际前进，但正负摆动可能相互抵消，必须与 RMSE 一起阅读，不能用较小的平均偏差代替逐步跟踪误差。

汇总要求 checkpoint 内容、任务、环境数、步数、评估条件、精度和 ONNX 对照配置一致，且每次报告的样本数完整、指标有限。批次开始时冻结 checkpoint 和可选 ONNX，替换外部源模型不影响本次运行；修改运行目录内的模型快照会导致完整性检查失败。中途失败或 Ctrl+C 会保留已经完成的种子和失败诊断，根目录状态标为失败或中断，不生成完整汇总，也不会覆盖已有输出目录。

单次评估、多种子评估和离线验收共用报告校验：核对任务、checkpoint、种子、环境数、步数、速度指令、推扰、ONNX 与课程设置，并检查样本完整性和指标有限性。即使不指定行为阈值，也不会将参数不符、缺少指标、NaN/Infinity 或 JSON 数值溢出的报告标记为成功。在线校验失败时保留原始 `evaluation.json`，`run.json` 标记 `status=failed`、`failed_phase=validate`，命令退出码为 1；多种子评估不会继续执行下一组。完整但未达到行为阈值的结果仍标记 `rejected`，退出码为 3。

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

本地移植模式还要求两次评估的 `implementation_sha256` 和 `requirements_sha256` 相同，防止将任务、奖励或物理实现变化误算为模型改进。实现哈希涵盖运行时保存的整个 EmbodiedForge 包代码与资产，因此修改启动器或报告代码也会改变它；代码更新后，应在同一版本下重新评估两个 checkpoint 再比较。不同实现种类、本地实现信息缺失或哈希不同时，`compare` 在创建输出前退出并提示重新评估。`comparison.json` 和 `run.json` 的 `source_verification` 记录核对字段；本地记录为 `local_implementation_and_requirements`，两份旧上游记录为 `upstream_baseline_only`，后者不宣称核对了本地实现。

`comparison.json` 给出各指标的 before/after 值、逐种子变化及变化的平均值与样本标准差。变化统一定义为 `after - before`：RMSE 为负通常表示误差下降，存活数为正表示增加。不推断统计显著性或自动给出总体胜负；使用 `assess` 对候选模型执行具体验收要求。

## Newton 原生移植范围

需要先补齐带关节的机器人资产、关节状态与力矩动作协议、接触传感器、BAM 执行器、GPU 张量环境和批量 reset，再逐项对齐观测、奖励与域随机化。61 维 actor 观测、14 个舵机关节顺序及归一化是训练和部署间的契约。不能仅将 `--physics` 改为 `newton` 就复现该任务。

代码和机器人资产的授权不同，分发资产前需遵循上游许可证；本接入读取外部仓库，未复制模型资产。

## 验证记录

续训首次 reset 的课程恢复修复与 64 环境实测见 [续训初始化验证](microduck-resume-initialization-20260912.md)。

中间与最终 checkpoint 的同条件对比、15 秒录像和逐环境停步诊断见 [保存阶段对比](microduck-checkpoint-analysis-20260912.md)。

本轮 2,000 次追加训练、同条件模型对比、录像及未达标原因见 [2026-09-12 优化结果](microduck-optimization-20260912.md)。

早期验证、续训、固定指令与多种子验收记录见 [2026-09-11 验证记录](microduck-validation-20260911.md)。

## 迁移兼容模式

仅在需要复核历史行为时显式传入 `--repo /home/ubuntu/workspace/3rdparty/microduck_rl`。该模式继续验证原仓库的固定提交和干净状态，默认使用旧 `.cache/microduck-venv`。两种模式的依赖环境分别保存，避免 mjlab 自动加载旧任务插件并覆盖本地注册；本地 worker 会拒绝包含 `mjlab-microduck` 分发包的环境。

同一基线任务的历史 checkpoint 可直接用于本地 `--run` / `--resume-run`。这些检查核对任务与依赖基线，不要求历史 manifest 已经含有本地实现散列；新运行记录保存当前实现的实际文件散列。源码修改后无需重装依赖；修改依赖清单后必须重新 `setup`。

迁移验证记录见 [本地移植验证](microduck-port-20260916.md)。

本轮工程改进与续训对照见 [一小时优化记录](microduck-hour-optimization-20260916.md)。
