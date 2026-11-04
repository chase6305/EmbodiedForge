# 实验归档

这里存放历史实验报告、原始测量和证据包。使用指南见 [文档导航](../README.md)，系统边界见 [当前架构](../current-architecture.md)。日期和轮次用于区分实验，不作为公共 API 或功能版本。

## 架构与工程

[架构边界与兼容性重构](architecture-refactor.md)：公共运行工具、环境与采集接口、来源兼容、导入循环及数值一致性。

[运行恢复与模型复测](optimization-resume-20261029.json)：原生 H1 保存异常优先级修复、MorphMimic 检查点续训与配对评测，以及已有 SuperMimic 后空翻模型的物理验收。

[启动阶段资源收尾](architecture-startup-reliability.md)：记录器构造、日志线程启动与信号处理器安装失败时的状态和资源清理；按用户要求提前结束本轮。

[工作流生命周期与环境规格验证](architecture-lifecycle-20261021.json)：公共异常收尾、ACT 进程组、环境规格版本、SDK 兼容与持续采集；日志压力测试的解释器退出异常单独保留。[资源曲线](assets/architecture-lifecycle-20261021-resources.svg) · [原始证据](assets/architecture-lifecycle-20261021-evidence.tar.gz)。

[启动开销与进程日志收尾](architecture-runtime-20261021-fivehour.json)：延迟导入、CLI 分发与脱组 stdout 的日志读取；四组各四小时回归共验证 129216 个场景，解释器退出码均为 0。此前 GC 退出异常未再复现，根因仍未确定。[性能与资源曲线](assets/architecture-runtime-20261021-fivehour-performance.svg) · [原始证据](assets/architecture-runtime-20261021-fivehour-evidence.tar.gz)。

[取消与异常收尾验证](architecture-runtime-20261022-fourhour.json)：修复创建进程时取消遗留 worker、日志关闭掩盖超时或中断，以及外层已处理异常吞掉本次收尾错误的问题。六组各三小时长测验证 1442136 次进程场景，三组各两小时采集测试在两种存储方式下各验证 73439196 条转换；九组解释器均正常退出。另有 200 个独立父进程退出检查和六组真实 PPO 取消恢复检查通过；此前 GC 段错误的根因仍未确定。[验证曲线](assets/architecture-runtime-20261022-fourhour-validation.svg) · [原始证据](assets/architecture-runtime-20261022-fourhour-evidence.tar.gz)。

本轮[分发包补充证据](assets/architecture-runtime-20261022-fourhour-distribution-evidence.tar.gz)保留 SDK 主机环境中一次“必须优雅退出”的断言失败：worker 已被回收，中断状态正确，未写收尾标记的原因尚未确定。基准现分别记录优雅退出和强杀，确定性 `SIGSTOP` 回归验证强杀回收路径；修订后的 SDK 环境三分钟复测与标准库安装场景二十五分钟检查均正常退出。

[启动取消日志与 checkpoint 路径驻留](architecture-runtime-20261022-threehour.json)：修复启动取消时日志丢失与管道阻塞，移除旧 CPython 3.12.0 高频 checkpoint 名称驻留。三组候选运行时各九十分钟，共验证 249552 次场景；新旧 checkpoint 对照分别运行八十、九十分钟，四个读取进程均验证策略与 Adam 状态一致。两套完整回归分别通过 3266、3356 项；另有两项补充诊断以 143 退出，原始记录和一次退出码误记的更正均保留，原因尚未确定。历史 GC 异常也未作根因结论。[验证曲线](assets/architecture-runtime-20261022-threehour-validation.svg) · [原始证据](../../runs/experiment-evidence/architecture-runtime-20261022-threehour-evidence.tar.gz)。

[清理阶段取消与退出证据](architecture-runtime-20261022-twohour.json)：修复进程组清理及临时处理器安装被取消打断时的后代残留、原始异常覆盖，并补齐源码包遗漏的共享测试夹具。主环境与最终源码包各通过 3285 项，原生环境通过 3375 项；最终运行时的六组有效检查验证 225408 个场景。三次监督器收到外部 SIGTERM、三次旧 PID 夹具竞态，以及一次 Python 3.12.0 worker 启动 SIGSEGV 均单独保留并排除通过计数；信号发送者、该 SIGSEGV 及历史 GC 异常的根因仍未确定。[验证证据](../../runs/experiment-evidence/architecture-runtime-20261022-twohour-evidence.tar.gz)保存精简内核记录，完整跟踪文件的位置与校验和见报告。

[重复取消、错误优先级与 SuperMimic 预检](architecture-runtime-20261023-fivehour.json)：修复重复取消覆盖原始异常、已知 worker/日志失败误记为中断，以及处理器安装连续取消时的清理问题。主环境 3361 项、原生环境 3451 项，源码包内 3361 项测试通过。严格验收通过 9 组，共 892212 个进程场景（当前修复版 324976 个），另有 34339 次 GDB 启动检查单独统计。六条外层 143 与一次服务器重启后未观察的外层退出均排除，长测整体验收未通过；历史原生崩溃与外层 143 原因仍未确定。SuperMimic 当前提供[兼容性预检与规范](../supermimic.md)，尚未执行策略推理。[原始证据](../../runs/experiment-evidence/architecture-runtime-20261023-fivehour-evidence.tar.gz)包含逐文件校验清单与精简内核记录，完整原始跟踪文件的压缩副本位置与校验和见报告。

[源码钉定与有界预检](architecture-runtime-20261024-fivehour.json)：加固 SuperMimic 的 Git 隐藏修改检查、源码哈希与目录扫描上限。主环境 3376 项、原生环境 3466 项、最终源码包 3378 项通过；五项长测分别验证 535952 个进程故障场景、45244 次 GDB 启动和 2120035 项预检检查，计数单位不混加。持续查询与不查询的两组四小时纯 C 对照均正常退出，未复现历史 143；此前异常根因仍不明确。本轮仍是[离线预检](../supermimic.md)，未执行 SuperMimic 策略推理。[原始证据](../../runs/experiment-evidence/architecture-runtime-20261024-fivehour-evidence.tar.gz)包含完整压缩跟踪文件及逐文件校验清单。

[批量源码校验与原生生命周期诊断](architecture-runtime-20261024-sevenhour.json)：SuperMimic 预检改为批量 Git 查询与单次文件读取，256 文件基准的 Git 调用从 518 次降至 8 次，并加强文件身份及 ctime 变化检查。主环境 3395 项、原生环境 3485 项、源码包内 3395 项测试通过。五项长测全部通过：两组 SDK 合计 11820 次新解释器正常退出，另一个持续进程完成 11145 轮并正常退出；源码及安装包合计完成 2471616 项预检检查，计数单位分别统计。两组六个半小时纯 C 对照均正常退出，历史崩溃及外层 143 原因仍未确定。本轮仍是[离线预检](../supermimic.md)，未执行 SuperMimic 策略推理。[原始证据](assets/architecture-runtime-20261024-sevenhour-evidence.tar.gz)包含三份完整压缩跟踪及逐文件校验清单；最终七小时完成状态以报告为准。

[文件打开保护与稀疏源码检查](architecture-runtime-20261024-tenhour.json)：修复 FIFO 替换阻塞、增长文件追读和稀疏悬空链接漏检。最终版主环境 3406 项、原生环境 3496 项、源码包内 3406 项测试通过。九项分版本长测验收通过：最终版源码与安装包各运行四个半小时，完成 3152096 项检查；其他负载运行八至九个半小时。另确认 34008 次新解释器正常退出、17421 轮持续进程执行和 3210970 项哈希边界检查，单位分别统计。两组九小时纯 C 对照均正常退出；历史原生崩溃及外层 143 原因仍未确定。[原始证据](../../runs/experiment-evidence/architecture-runtime-20261024-tenhour-evidence.tar.gz)包含四份完整压缩跟踪和 1223 个文件的校验清单。本轮仍为[离线预检](../supermimic.md)，未执行 SuperMimic 策略推理；十小时完成状态以报告为准。

## 机器人策略与训练研究

| 主题 | 报告 |
| --- | --- |
| Go1 指令响应 | [转向](go1-yaw-study.md) · [横移与急停](go1-response-study.md) · [复合指令](go1-maneuver-study.md) · [能力保留](go1-preservation-study.md) · [响应与计算开销](go1-fast-response-study.md) |
| Microduck 初期接入 | [验证](microduck-validation-20260911.md) · [训练优化](microduck-optimization-20260912.md) · [checkpoint 分析](microduck-checkpoint-analysis-20260912.md) · [续训初始化](microduck-resume-initialization-20260912.md) |
| Microduck 本地实现 | [移植](microduck-port-20260916.md) · [优化](microduck-hour-optimization-20260916.md) · [模型与消融](microduck-model-training-ablation.md) |
| 训练与后训练 | [首轮](rl-training-study-20260925.md) · [训练对照](rl-training-study-20260926.md) · [后训练](rl-posttraining-study-20260926.md) · [奖励因子](rl-factorial-study-20260927.md) |
| 恢复、压缩与运行校验 | [续训](rl-resume-study-20260928.md) · [压缩](rl-compression-study-20260928.md) · [运行校验](rl-runtime-validation-20260928.md) · [评估隔离](rl-evaluation-isolation-20260929.md) |
| 原生 H1 | [保存与恢复](h1-checkpoint-publication-20260929.md) · [转向](h1-turning-study-20260929.md) · [指令采样](h1-command-mixture-20260930.md) · [终止惩罚](h1-termination-cost-20260930.md) · [探索熵](h1-entropy-study-20260930.md) |
| 外部集成 | [人形动作](humanoid-integration-study-20260928.md) · [Light Loco 接入评估](light-loco-parkour-assessment.md) |

报告同名 JSON 是对应的结构化记录；图片、绘图脚本和证据压缩包位于 [assets](assets/)。原始 JSON、图片、脚本及压缩包的内容在目录迁移中保留，Markdown 只调整链接和复现入口路径。

## 平台性能与持续验证

`platform-optimization-*.json` 保存各轮配置、测量、回归与持续验证摘要；同名 SVG 保存图形结果，`assets/platform-optimization-*-evidence.tar.gz` 保存已归档的原始证据。轮次之间的条件可能不同，不应仅按日期把数字串成性能趋势。

代表性入口：[初始测量](platform-optimization-20261014.json)、[采集与记录优化](platform-optimization-20261016-twohour.json)、[持续验证](platform-optimization-20261017-round5-sevenhour.json)、[动作元数据优化](platform-optimization-20261020-round9-twohour.json)、[有序批次测量](ordered-batches-optimization-20261016.json)。其余轮次保留在本目录。

归档中的源码路径、运行目录、散列和环境信息描述原始实验。目录迁移不改写原始证据内部的历史路径；重建图表时使用当前报告中更新后的绘图命令。

当前架构工作的后续证据见[架构边界与持续运行验证](architecture-continuation.md)，包含数据完整性、运行收尾、源码包与持续采集检查。


<!-- architecture-runtime-20261025-eighthour -->
源码快照与预检共用有界普通文件读取；复制与首次哈希合并后，内部逻辑内容读取由四遍减为三遍，目标校验和最终源清单复核仍保留。本地配对基准不构成提速结论。最终实现通过 3418 项主环境测试、3508 项原生测试及 25 个子测试，并通过源码包和安装 wheel 校验。持续验证按最终 V1 与未采用的元数据实验 V2 分别记录，未执行 SuperMimic 策略推理。[结构化报告](architecture-runtime-20261025-eighthour.json)与[完整证据](../../runs/experiment-evidence/architecture-runtime-20261025-eighthour-evidence.tar.gz)记录真实退出、失败控制与限制。


SuperMimic 已从兼容性预检推进到独立 SDK 中的真实 GPU 训练与评测：[初始闭环与恢复修复](supermimic-reproduction-20261026.json)和[四小时优化实验](supermimic-reproduction-20261027-fourhour.json)。原论文观测与增加9维锚点观测的实验分别记录，不能直接对照论文指标。 锚点长跑末尾因动作文件哈希变化实际退出1，保留失败状态；精确重建输入后的独立推理和短程训练验证另行记录，不追认原训练通过。

[共同步数评测图](assets/supermimic-reproduction-20261027-fourhour-coverage.svg)、[训练诊断图](assets/supermimic-reproduction-20261027-fourhour-training.svg)和[逐文件校验的证据归档](assets/supermimic-reproduction-20261027-fourhour-evidence.tar.gz)保留本轮对照与失败限制；模型、动作数组和示意视频仅保留本地。

SuperMimic 的简短运行入口与输入副本隔离见[运行与稳定性验证](supermimic-runtime-20261027.json)，记录真实 GPU 推理、小配置训练/续训、故意变更测试源文件后的隔离结果，以及浏览器服务受控退出。

[SuperMimic 后空翻专项接入](supermimic-backflip-20261027.json)记录相同 321 帧参考动作在 mjlab 上的单动作训练。90 分钟续训后的最后模型在三次独立进程评测中完成约 358.48° 后翻及额外 5 秒保持，但腿部姿态质量未达标；50 Hz 诊断及未完成的严格验收项均保留，不能凭回报或动画宣称完成高质量复现。

[SuperMimic 后空翻四小时优化](supermimic-backflip-20261027-fourhour.json)将完整 vibed 算法适配到现有 mjlab，保留两个独立训练种子的实际最后模型、进程复测、200 Hz 物理检查和原有失败。结果及适用范围以结构化报告为准。


超过 50 MiB 的原始跟踪证据包保留在本地，不进入 Git；对应链接使用 `runs/experiment-evidence/` 路径。文件大小、原始本地位置及 SHA256 见 [本地证据清单](local-evidence.json)，结构化报告和图表仍随仓库发布。
