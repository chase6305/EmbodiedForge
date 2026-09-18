# Light Loco Parkour 接入评估

2026-09-16 检查本地 `light-loco-parkour`，来源为
`https://github.com/lucidrains/light-loco-parkour`，版本 `0.0.29`，
revision `963a6ec3b8b42eb29dd6b9dfed34ededd3c64c7b`，工作区干净。
本页记录最初的代码静态评估。随后已完成第一阶段 Go1 PPO 损失与 GAE 接入，
安装、验证和使用边界见 [Go1 实验性接入](light-loco-parkour.md)；尚无跑酷行为验收。

## 结论

可以作为可选的实验性算法组件接入，尚不能当作完整机器人跑酷配方直接发布。
README 标注 WIP，作者元数据为 Phil Wang。这份检出不应被描述为论文作者的官方训练工程。

- 已有 Actor、Critic、Agent、DistillationWrapper、RewardShapingWrapper、MotionPrior、PhaseConditionalMotionPrior 等组件。
- 顶层 `LightLocoParkour` 仅在构造函数保存 agent，没有实现完整训练循环。
- `validate_pendulum.py` 包含 Gymnasium/Pendulum 验证流程，不能替代机器人跑酷验证。
- 尚未提供本项目需要的 Go1/H1 环境适配、跑酷地形、机器人观测契约、教师训练配方及完整部署闭环。
- 本地 LICENSE 标注 MIT，版权为 2026 Phil Wang；分发复制或修改的源码时保留该许可证和版权声明。其他资产和依赖需分别保留其许可信息。

## 建议接入阶段

1. 固定源码 revision，单独维护可选依赖环境，避免影响现有 Go1/Wuji Torch 版本。
2. 先做 Go1 实验适配：观测与动作形状、动作分布和缩放、奖励组、终止与截断、GAE bootstrap、归一化和历史状态重置。
3. 跑通短训、checkpoint 保存/恢复、固定指令评估和在线运行。沿用默认 headless、显式可视化的交互规则。
4. 用相同环境与预算对比现有 PPO；只有行为结果和稳定性验证通过后再扩大范围。
5. 添加多种跑酷地形、技能教师、感知输入、参考运动数据和学生蒸馏，分别设置成功率和失败条件。

主要依赖包括 `torch>=2.5`、`env-ssl-wrapper`、`x-mlps-pytorch`、`assoc-scan`、
`einx` 等。目前未验证这些版本与现有训练环境的组合兼容性。
无需先替换 EmbodiedForge 的物理层，但必须建立清晰的环境/学习器适配边界。
