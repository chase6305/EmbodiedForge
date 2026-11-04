# 公共运行记录

训练、数据采集和 SDK 工作流继续使用各自的产物格式。新运行在原有 manifest 中增加 `run_contract`，公共身份与状态查询仅依赖 Python 标准库，不加载 NumPy、核心环境或外部 SDK。`run_identity` 构造身份，`validate_run_identity` 校验并复制已声明的身份，未知版本直接拒绝。

```json
{
  "run_contract": {
    "schema_version": 1,
    "workflow": "recipes",
    "operation": "train",
    "task": "go1-joystick"
  },
  "status": "running"
}
```

`workflow` 标识执行体系，`operation` 标识该入口执行的操作，`task` 是该工作流的任务名或 `null`。它们不要求不同任务共用仿真接口。已有 `schema`、`schema_version`、`workflow`、`recipe` 等字段继续由原工作流解释；公共 schema 的版本只作用于 `run_contract`。

## 查询

```bash
python -m embodiedforge runs inspect --run /path/to/run
```

SuperMimic 新运行的 `launcher.json` 也使用同一身份和生命周期，可由同一个查询入口读取。

Python 中使用 `embodiedforge.runtime.records.read_run(path)`。查询返回公共身份、manifest 文件名及其 SHA256、身份来源和已记录的状态，不修改目录。

SDK 与元数据子命令放在 `embodiedforge` 后的第一个参数位置，例如 `embodiedforge h1 train ...`。核心命令的参数不会被透传到其他工作流；将它们放在 SDK 子命令前会报错，避免意外运行核心 rollout。公共 CLI 使用同一份分发目录生成命令选项并选择入口。

| 公共状态 | 含义 |
| --- | --- |
| `preparing` | 准备输入、实现快照或运行环境 |
| `running` | 运行记录标记为执行中 |
| `complete` | 工作流已报告完成 |
| `rejected` | 工作流执行完毕，但未通过显式验收 |
| `failed` | 执行或产物校验失败 |
| `interrupted` | 捕获到中断并完成状态收尾 |
| `timed_out` | 工作流超时 |

核心数据集 `core-data` 与显式元数据的原生采集 `episode-data` 将原有 `recording` / `closed` 分别映射为公共 `running` / `complete`；`recorded_status` 保留原值。关闭数据集允许丢弃未完成的 episode，不代表所有环境均有完整 episode。

`terminal` 仅描述已记录状态。机器故障可能遗留 `running`，查询不会探测进程是否仍然存在；`process_liveness` 为 `not_checked`。查询也不会读取模型或验证 checkpoint，`artifact_verification` 为 `not_checked`。恢复、评估和策略验收仍需使用对应工作流入口。

## 兼容读取

有 `run_contract` 时严格使用其版本，版本不支持则报错，不回退到猜测。旧运行仅在格式可以识别时读取：核心 PPO、核心数据集、Go1/Wuji recipes、原生 H1、IsaacLab H1、Microduck、ACT、GMR 和 SONIC；旧 RLinf/Weave 状态记录需要匹配的 `request.json` 确认身份。ACT 的记录位于 `act-run.json`；LeRobot 尚未创建目录时，可查询其父目录中的启动记录。

不根据目录名称或日期推断工作流。缺失或损坏的记录会报告错误，不初始化 SDK、不自动恢复或重写旧产物。

## 公共基础设施

| 模块 | 职责 |
| --- | --- |
| `runtime.artifacts` | 有限 JSON 的原子发布、独立临时文件与失败清理 |
| `runtime.process` | 子进程日志、进程组清理、中断转发、显式文件描述符继承、SDK 包路径 |
| `runtime.environment` | 独立 conda SDK 的基础环境配置 |
| `runtime.provenance` | 实现快照与文件身份校验 |
| `runtime.hashing` | 有界读取文件 SHA256 |
| `runtime.implementation` | 记录实际加载的实现、源码散列与依赖版本 |
| `runtime.records` | 版本化运行身份、公共生命周期与只读状态视图 |

JSON 发布使用同目录临时文件和原子替换，防止读到半个 JSON；不承诺断电后的 fsync 持久性，也不为多个写入者决定业务顺序。调用方仍需协调状态更新。

`recipes.write_json`、`h1.sha256` 等既有入口仍可导入；内部进程、哈希和学习工具直接依赖 `runtime` / `learning` 中的定义，不再经私有转发模块绕行。任务专属配置、验收与 checkpoint 兼容性判断继续留在各工作流。

新工作流可通过 `runtime.records.managed_run` 发布开始、结束和异常状态。调用方独占新目录，context 不覆盖已有记录；成功时可明确标记 `rejected`，异常按失败、中断或超时记录。原生 H1 评估使用此入口，并在 `run.json` 中引用 `evaluation.json` 及其 SHA256。

已有工作流通过 `runtime.records.run_lifecycle(record, publish)` 复用同一套状态收尾，同时保留原有 manifest 路径和专属字段。核心 PPO、原生 H1 训练、IsaacLab H1、recipes、Microduck 的训练/评估/比较/导出、ACT、RLinf、Weave 和动作 SDK 启动器已接入。它们的 checkpoint 提交、恢复校验与策略验收仍由各自工作流负责；Microduck ONNX 导出复用生命周期终态发布，回滚只清理本次创建的模型和报告链接。

`record` 初始状态为 `running` 或 `preparing`。准备工作完成后，调用方将 `preparing` 改为 `running` 并按需发布；准备状态不能在成功退出时直接变成完成。`publish(record)` 在启动和收尾时调用，负责原子写入现有记录。可选 `finalize(record)` 在退出时收集 checkpoint 目录等终态元数据，之后仍会尝试发布最终记录。

生命周期覆盖启动写入、执行和收尾期间的终止信号；`SIGTERM` 和未被忽略的 `SIGHUP` 会进入正常异常收尾，记录 `interrupt_signal`。已有执行异常不会被终态元数据收集、最终写入或信号处理器恢复错误替换。如果执行成功而最终发布失败，会保留该发布异常，并尽力补写失败或中断状态；这次补写失败也不会替换原始异常。持续 I/O 故障、`SIGKILL` 或主机退出仍可能留下旧状态。

异常优先级只跟踪本次运行离开受保护代码时的失败。即使调用方在处理此前异常的 `except` 分支里启动新运行，本次收尾失败仍会报告并更新状态，不会被外层已经处理的异常吞掉。

ACT 的训练子进程现在由 `runtime.process.run_process` 持有独立进程组，并显式继承训练锁。正常退出和可捕获的中断都会清理所持有的组；启动器被强制杀死时，仍存活的训练器继续持锁，恢复入口会拒绝并发恢复。子进程异常退出属于 `failed`，启动器收到中断属于 `interrupted`，两者不能仅根据退出信号互相推断。 已观察到的子进程退出保留 `worker_exit_observed` 和原始 `worker_returncode`，由公共失败记录处理。

在 POSIX 启动器中，静默日志直接写盘；需要回显时通过非阻塞管道和专属线程保存原始字节，并以 UTF-8 替换无效序列后显示到终端。worker 退出后，日志线程排空当时已缓冲的字节并停止，不等待持有 stdout 的其他会话关闭。脱离作业组的 SDK 服务仍由其资源所有者关闭，启动器不会越过组边界清理它们。

创建 worker、注册其清理动作和启动日志读取线程期间，启动器暂存 `SIGINT`、`SIGTERM`、`SIGHUP` 中具有 Python 处理器的取消信号，接管子进程且日志读取就绪后再交还原处理器。这样可回收刚启动的 worker，保存其取消收尾日志，并避免输出堵住管道；同时保持子进程继承的信号屏蔽集和忽略行为。已存在执行、日志采集、超时或中断异常时，关闭日志的二次 I/O 错误只单独记录，不替换原始异常；独立发生的日志关闭失败仍会向调用方报告。

启动器在取消时先向所持有的作业组发送 `SIGINT`，给 worker 十秒退出宽限，然后清理整个组并回收直接子进程。Python 处理函数可能延迟执行，完整回收不要求 worker 一定完成优雅退出。 转发信号或宽限等待时再次收到取消会直接进入强制清理；清理中的取消不会覆盖首次中断或超时，最终 manifest 继续记录首次失败原因。强杀、回收和日志排空共用同一条受保护的收尾路径。启动器已等待到 worker 非零退出，或已观察到日志采集失败时，清理中的取消也不会把已知失败改成“启动器中断”；最终日志排空仍在报告错误之前完成。

正常结束的资源清理也暂存取消信号：先清理所持有的进程组、回收直接子进程并停止日志线程，再交还首次取消。如果取消连续打断临时信号处理器的安装，会持续重试尚未完成的清理，并保留首次取消；已有执行错误继续向调用方报告。这样取消不会跳过组内后代的回收，也不会用后来的取消覆盖先前的执行失败。

Linux 上可用[信号诊断基准](../benchmarks/process_signal_probe.py)观察发送调用、进程状态、独立确认管道和子进程退出码：

```bash
PYTHONPATH=src python benchmarks/process_signal_probe.py --duration 60 \
  --output runs/process-signals --worker-trace --wakeup-trace
```

`--wakeup-trace` 使用 worker 的 `signal.set_wakeup_fd` 区分 C 层收到信号与 Python 处理函数执行。普通循环仅保留首轮和强杀场景的详细记录；`--observe-late-handler` 是诊断夹具开关，在正常十秒后额外观察最多二十五秒，并保留延迟处理场景，不改变生产启动器的宽限。`--preload-sdk` 可在同一父进程预载 Torch 和 MuJoCo，检查 SDK 主机环境下的行为。

基准自己的 `summary.json` 和 `run.json` 可能在解释器最终退出前写完。需要验证退出过程时，用独立的[进程监督器](../benchmarks/process_supervisor.py)启动它：

```bash
PYTHONPATH=src python benchmarks/process_supervisor.py \
  --output runs/process-exit --timeout 180 --trace -- \
  python benchmarks/process_signal_probe.py --duration 60 \
  --output runs/process-signals --worker-trace --wakeup-trace
```

监督器只依赖标准库；`--trace` 另需 Linux `strace`。`result.json` 保存实际 `process_returncode`、负退出码对应的 `termination_signal`、超时与监督器自身中断；只有实际等待到正常退出且未超时或取消，`completed_successfully` 才为真。显式 `exit(143)` 与被 `SIGTERM` 终止分别记录为 `143` 和 `-15`。`signals.strace` 保留信号的内核来源信息、指令地址与退出系统调用，`events.jsonl` 在发送信号和等待结束时立即刷新，`command.log` 保存被监督命令的输出。跟踪会改变运行时序，应与不启用跟踪的检查一起判断。

启用 `--trace` 时，直接子进程是 `strace`，实际等待结果对应 `launched_command` 中的跟踪启动器；目标解释器的退出须结合内核记录核对。外部终止或跟踪失败时，启动器的退出码不能替代缺失的解释器退出证据。

监督器为被监督命令持有独立进程组，超时或取消时转发信号，宽限后强杀并回收直接子进程；正常结束也清理该组中的遗留进程。被监督命令仍负责关闭另建会话的后代。这里的退出证据与工作流 manifest 分开保存，不改写训练状态。

可用 `process_fault_soak.py --cleanup-cancellation` 加入清理阶段的取消检查。每轮在原有 24 个场景之外，增加 24 个场景：三个信号、两种日志模式、是否已有执行错误，以及组清理或处理器安装两个取消时机。该 Linux 夹具启用子进程收养，核对真实同组后代的退出状态并回收它们。

可另加 `--repeated-cancellation` 验证首次中断或超时后，在信号转发、强杀、回收三个时机再次取消。两种日志模式加上连续打断处理器安装、已有执行错误或 worker 退出的组合，共增加 20 个真实 worker 场景，并检查首次异常、manifest 状态、直接子进程回收和处理器恢复；与 `--cleanup-cancellation` 同用时，每轮共 68 个场景。 `--worker-failure-cancellation` 再增加 8 个场景：worker 返回 7 或被 `SIGKILL` 终止后，在组清理或处理器安装时取消，核对失败码、组内后代回收和 manifest；三个开关同用时每轮共 76 个场景。

核心 PPO 发布和读取、Go1 发布和清理高频 checkpoint 名称时使用文件系统字符串，避免 CPython 3.12.0 的 `pathlib` 路径片段驻留造成持续内存增长；保存后的 SHA256 使用有界分块读取。目录恢复仍选择 manifest 引用的已提交快照、验证同一份读取字节的散列，并在训练清理旧快照时最多重选三次。策略、Adam 状态、当前与前一次已提交快照的保留规则与恢复语义不变。

实现快照的复制与首次 SHA256 共用一次有界源读取；目标内容独立复核，全部复制后仍重新校验完整源清单。普通文件打开、大小与身份检查复用 `runtime.hashing`，与 SuperMimic 预检保持同一实现；文件锁和未发布部分输出的清理由调用方负责。已有目标拒绝覆盖。
