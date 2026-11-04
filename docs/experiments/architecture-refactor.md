# 架构边界与兼容性重构

本轮从 2026-10-20 07:00:53 UTC 开始，按三个实际小时的工作窗口推进，基线为 `634782d`。当前结构见 [当前架构](../current-architecture.md)，操作见 [运行记录](../run-records.md) 和 [环境适配](../environment-adapters.md)。

## 完成的调整

| 优先级 | 结果 |
| --- | --- |
| P0：架构现状 | 区分核心、原生机器人和外部 SDK；修正后台写入及 PPO 恢复的过时说明 |
| P1：运行基础设施 | 进程、JSON 发布、散列、源码快照、环境与实现来源归属 `runtime`；保留旧导入 |
| P1：记录与编排 | 版本化 `run_contract`、兼容查询、H1 评估生命周期；配方目录、设置、产物、验收与进度归属 `workflows` |
| P1：依赖方向 | 查看器不导入配方 CLI；Microduck 导出显式注入操作；图形依赖检查独立 |
| P2：环境与采集 | Go1/H1 共用 CPU 评估、Episode 记录、后台写入、完整 episode 读取和动作窗口 |
| P2：学习工具 | GAE、Adam 校验与探索约束归属 `learning`；机器人策略不导入仿真或核心 trainer |
| P3：文档归档 | 123 个报告及相关文件移至 `docs/experiments`，指南保持主题名称，新增导航与链接检查 |

原生训练、控制周期、奖励、归一化、动作裁剪和 checkpoint 布局继续使用原有契约。原生适配器要求 reset done 行后再推进整个 batch。GPU 张量传输与通用机器人资产仍属于目标架构，核心继续限定 CPU 快照。

## 验证依据

- 核心 PPO、Go1、H1 各三个种子，共九组基线对照。核心每组八次更新，机器人每组三轮采样与优化；策略、Adam 状态及机器人 rollout 的数组散列全部一致。这验证短程数值兼容性，不评估收敛质量。
- 真实 MuJoCo/mjbatch 对照覆盖适配前后的物理状态、奖励、观测、done 和局部 reset，保留的终止观测不被后续操作修改。
- 本机 964 份既有运行 manifest 全部兼容读取，读取前后散列相同；查询不代替进程存活或 checkpoint 校验。
- 显式模块/符号导入循环由三组降为零。检查包括延迟导入，不推断动态字符串和隐式父包初始化。
- 迁移中的 97 个原始 JSON、图像、SVG、脚本和证据包保持字节一致，其余 Markdown 调整链接与复现路径。
- Python 3.11 图形环境的 166 项查看器检查通过；独立安装的 wheel 验证了资源、无 SDK 核心评估、工作流导入和旧记录查询。

最终全量回归、wheel 身份与源文件身份见 [结构化摘要](architecture-refactor.json)。本机详细日志、迁移清单、记录读取审计和数值对照位于 `runs/architecture-optimization/`；其他检出不必包含这些本地训练产物。

## 复核命令

```bash
ruff check src tests benchmarks examples
ruff format --check src tests benchmarks examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
python -m embodiedforge runs inspect --run /path/to/run
```

数值对照工具为每个源码树创建独立运行目录：

```bash
PYTHONPATH=src python benchmarks/architecture_parity.py \
  --reference-root /path/to/baseline-checkout \
  --output /path/to/new-comparison
```

增加 `--native --h1-model /path/to/h1.xml` 可在兼容的 MuJoCo/mjbatch/Torch 环境对照机器人。各环境的可选依赖和跳过项单独记录。
