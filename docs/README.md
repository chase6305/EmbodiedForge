# 文档导航

使用指南按主题维护；实验报告、测量结果和证据包位于 [实验归档](experiments/README.md)。归档文件中的日期标识实验批次，不表示当前接口版本。

| 目的 | 阅读入口 |
| --- | --- |
| 理解模块、执行路径与扩展约束 | [架构与模块边界](architecture.md) |
| 使用和扩展公共接口 | [接口与模块](interfaces.md) · [实现与扩展](implementation.md) |
| 运行、恢复和评估训练 | [训练与部署](training-deployment.md) · [English](training-deployment.en.md) |
| 查询运行身份与状态 | [公共运行记录](run-records.md) |
| 复用 Go1/H1 的评估与 transition 接口 | [环境适配与评估](environment-adapters.md) |
| 查看和回放机器人 | [查看器](viewers.md) · [机器人回放](robot-web-replay.md) · [Go1 在线策略](go1-live-web.md) |
| Go1 与外部任务配方 | [任务配方](recipes.md) · [Light Loco](light-loco-parkour.md) |
| 原生 H1 / IsaacLab H1 | [原生 H1](h1-native.md) · [English](h1-native.en.md) · [IsaacLab H1](h1-isaaclab.md) |
| Microduck | [使用指南](microduck.md) · [快速开始](microduck-quickstart.md) |
| ACT / RLinf / Weave | [ACT](act.md) · [RLinf](rlinf.md) · [Weave](weave-reference.md) |
| 人形动作与重定向 | [动作集成](humanoid-motion.md) · [Wuji](wuji-training.md) |
| 后端和排错 | [Newton](newton.md) · [图形诊断](graphics-diagnostics.md) · [日志](logging.md) |

历史报告保留当时的环境、命令、结论和失败记录；当前命令以使用指南为准。`runs/` 中的本地训练产物通常不随源码发布，历史报告链接到这些路径时需要相应实验数据。

源分发包保留源码、报告、图表和绘图脚本，便于离线查阅。历史原始证据压缩包留在源码仓库的 `docs/experiments/assets/`，不重复嵌入源分发；各报告继续记录其位置与校验信息。wheel 只包含运行时及其资源。
