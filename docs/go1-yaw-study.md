# Go1 角速度跟踪对照（2026-09-13）

本轮从已通过动态评估的 `go1-v2-tracking-900-20260913` 出发，降低行走中的偏航摆动，同时检查平面跟踪是否退化。任务和 PPO 均执行本仓库移植实现，物理使用隔离环境中的 mjbatch / MuJoCo；不是调用上游 Go1 示例训练脚本。

## 从零训练的可复用模型

另外完成三个配置 × 三个独立初始化种子（10/11/12）× 1200 轮的完整训练，与九组续训合计 **18 组主要对照、165,888,000 次训练转换**。每组仍为 512 环境、24 步 horizon。预算及实际配置差异已逐组核对，复现和短测不计入这 18 组。

从零训练候选为 `runs/go1-yaw-fresh-turn-s11-1200-20260913/model.pt`，使用 `tracking-turn-v1`、训练种子 11，SHA256：

```text
29161e00b626e60ad2adfe1cd5ad1ef4ddd5ceaa6b92c01fe35e4dff0f29a682
```

候选只根据开发种子 0/1/2、与续训相同的筛选条件确定，再使用另一组预留种子 20/21/22 评估。其三条基本固定移动指令相对 900 轮基准，角速度 RMSE 降低 **64.7%**，平面 RMSE 降低 **5.3%**：

| 模型（同用评估种子 20/21/22） | 固定移动平面 RMSE（m/s） | 固定移动角速度 RMSE（rad/s） | 动态平面 RMSE（m/s） | 动态角速度 RMSE（rad/s） |
| --- | ---: | ---: | ---: | ---: |
| tracking-v1 基准 / 900 | 0.04184 | 0.07376 | 0.04948 | 0.05743 |
| 平衡续训 / 1200 | 0.04292 | 0.02936 | 0.04968 | 0.02834 |
| 从零训练 turn / 1200 | 0.03964 | 0.02605 | 0.05482 | 0.02855 |

三者均完成整个动态窗口且所有环境达到连续保持条件。从零训练模型最差阶段达标时间占比为 94.49%，通过 105/105 项动态检查、36/36 项基本固定指令检查、90/90 项扩展检查。它的动态平面 RMSE 比基准增加约 10.8%；右横移平面误差约为 0.0980 m/s，虽低于 0.15 m/s 门槛，却高于基准。固定指令改善不能视为所有运动方向或切换过程都改善。

两条训练路径的开发集通过情况如下，每个“通过模型”都要求三个评估种子与六个动态阶段全部通过：

| 奖励配置 | 同一基准的三个续训种子 | 三个从零初始化种子 |
| --- | ---: | ---: |
| tracking-v1 | 2/3 | 3/3 |
| tracking-turn-v1 | 2/3 | 2/3 |
| tracking-balanced-v1 | 3/3 | 2/3 |

这些数量只描述本次三个训练种子，没有证明某个奖励配置普遍优于其余配置。默认配置保持不变；应区分已通过评估的具体 checkpoint 与一个训练配方的普遍可靠性。

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --go1-reward-profile tracking-turn-v1 --seed 11 \
  --num-envs 512 --horizon 24 --updates 1200 --threads 4 --timeout 1800 \
  --output runs/go1-from-scratch-repeat
```

在当前源码快照中重新从随机初始化训练完整 1200 轮后，模型与优化器的 55 个张量、元数据及 checkpoint 文件均与候选逐字节一致，验证记录见 `reproduction-verification.json`。

从零训练的模型选择与留出结果为 `fresh-selection.json`、`fresh-holdout-summary.json`；对照图为 `fresh-yaw-tracking-comparison.png` / `.svg`。回放为 `runs/go1-yaw-fresh-turn-s11-1200-20260913-fresh-holdout-switching/motion-seed-20-switching.html`，六个阶段边界均通过浏览器检查。

## 已验证的续训模型

模型：`runs/go1-yaw-balanced-s1-1200-20260913/model.pt`，累计 1200 轮，SHA256：

```text
afa021c7f7193994abe5d9288cba2a39baa489ac47b5a21922735d55c97bc151
```

开发集种子为 0/1/2，模型在 09:23:48 UTC 固定，再运行预留种子 6/7/8。三条基本移动指令（前进、左转、右转）的 RMSE 按 864,000 个有效帧合并，包含起步过程：

| 模型 | 平面 RMSE（m/s） | 角速度 RMSE（rad/s） | 最差单种子阶段连续达标比例 | 最差阶段达标时间占比 |
| --- | ---: | ---: | ---: | ---: |
| 基准 tracking-v1 / 900 | 0.04164 | 0.07372 | 100% | 95.18% |
| 同预算 tracking-v1 / 1200，续训种子 1 | 0.07744 | 0.05351 | 78.13% | 93.09% |
| tracking-balanced-v1 / 1200，续训种子 1 | 0.04259 | 0.02916 | 100% | 96.14% |

相对 900 轮基准，新模型角速度 RMSE 降低 **60.4%**，平面 RMSE 增加 **2.3%**。相对同预算、同续训种子的对照，两类误差均降低。原配置续训对照在评估种子 8 的右转阶段未达到 80% 连续达标比例，导致该阶段与整体两项检查失败；它们对应同一个失败阶段。

新模型完成 96 次 60 秒连续动态试验，105/105 项检查通过；基本固定指令完成 384 次试验、36/36 项通过。扩展固定指令完成 960 次试验、90/90 项通过，全部存活。扩展套件包含基本四项，因此不能把 384 与 960 相加作为互不重复的固定指令样本。

动态稳定定义为平面误差 ≤0.1 m/s、角速度误差 ≤0.15 rad/s，连续保持 0.5 秒。达标时间占比统计整段，分母包含全部初始环境及计划时间，终止帧和跌倒后缺失时间不算达标。整体数值取最差阶段；`settled_fraction` 说明曾连续跟上，不表示整个阶段始终没有偏差。

## 实验与选择规则

三种配置各使用续训种子 0/1/2，从同一份 900 轮 checkpoint 继续 300 次更新，均为 512 环境、24 步 horizon，各新增 3,686,400 次状态转换。学习率、PPO、指令分布、物理、观测、终止条件和奖励时序均相同：

| 配置 | track | turn | 其他奖励 |
| --- | ---: | ---: | --- |
| tracking-v1（对照） | 2 | 0.5 | 原值 |
| tracking-turn-v1 | 2 | 1 | 原值 |
| tracking-balanced-v1 | 3 | 1 | 原值 |

单独增加 turn 在三组续训中均降低角速度误差，但平面误差都高于基准，其中一组动态评估未通过。平衡配置的三组续训均通过动态门槛，仍有不同程度的分项取舍。

候选必须通过动态与固定指令验收，基本移动指令合并角速度 RMSE 至少降低 20%，合并及每条基本移动指令的平面 RMSE 退化均不超过 10%。只在满足条件的模型中选择合并角速度误差最小者。九份续训模型中，仅平衡配置的种子 1 满足全部选择条件；其余结果也保留，没有用总 reward 代替独立评估。

奖励定义集中在 [go1_config.py](../src/embodiedforge/locomotion/go1_config.py)，CLI、任务和 checkpoint 继承共用一份配置。新增版本化配置不改变旧配置含义；新训练默认仍为 `original`。通过模型不代表从随机初始化使用相同奖励就一定得到同样结果。

## 更广泛指令的取舍

扩展套件额外覆盖 1 m/s 前进、0.5 m/s 后退、±0.3 m/s 横移、±0.8 rad/s 原地旋转，门槛为全程存活、平面 RMSE ≤0.15 m/s、角速度 RMSE ≤0.2 rad/s。

相对基准，新模型在十项指令中的角速度误差均下降，但横移、后退、原地旋转的平面误差有所增加。例如右横移从 0.0647 增至 0.0815 m/s，左原地旋转从 0.0222 增至 0.0384 m/s。因此“平面误差增加 2.3%”只对应三条基本移动指令，不能推广到所有运动方向。

逐步诊断在开发种子 0 的前进阶段排除最初两秒后，32 环境的偏航波动 RMS 从 0.0672 降至 0.0157 rad/s；平均前向速度偏差从 −0.0095 变为 −0.0005 m/s。动作 RMS 从 0.403 增至 0.471；这里没有测量机械能耗，不能把较低偏航误差解释为更低能耗。

## 复现与产物

完整复现候选所用的 300 轮续训后，模型与优化器的 55 个张量、元数据及整个 checkpoint 文件均完全一致。源码快照执行前后的同配置单轮续训 checkpoint 也逐字节一致。单轮继承短测累计 1201 轮，但没有替换上述经过独立评估的 1200 轮模型。

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/go1-v2-tracking-900-20260913 \
  --go1-reward-profile tracking-balanced-v1 --seed 1 \
  --num-envs 512 --horizon 24 --updates 300 --threads 4 \
  --output runs/go1-balanced-repeat

python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/go1-yaw-balanced-s1-1200-20260913 --suite switching \
  --seeds 6 7 8 --num-envs 32 --steps 3000 --record-motion \
  --min-survival-fraction 1 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --min-settled-fraction 0.8 --min-tracking-fraction 0.9 \
  --output runs/go1-balanced-eval
```

新建受管任务会在 `implementation/embodiedforge/` 保存并执行本地源码快照，包含 Python 模块、Go1 场景与许可证；运行记录保存各文件哈希，结束后复核。`runtime.json` 记录实际 adapter 导入路径。这样后台任务不会在后续模块导入时读到工作区的新改动。旧训练运行仍可作为续训和评估输入。

原始配置、训练日志、所有失败模型、模型选择记录、预留种子结果及检查记录均保存在 `runs/go1-yaw-audit-20260913/`：

- `continuation-selection.json`：选择条件、全部候选与固定时间。
- `holdout-summary.json`：三个模型、三个评估套件的完整比较。
- `yaw-tracking-comparison.png` / `.svg`：实际速度曲线与留出集误差图。
- `reproduction-verification.json`：完整模型及优化器状态的复现核对。
- `snapshot-verification.json`：源码快照实际执行检查。
- `browser-check.json`：六个指令边界与浏览器错误检查。

离线回放：`runs/go1-yaw-balanced-s1-1200-20260913-holdout-switching/motion-seed-6-switching.html`。它展示真实刚体与足端轨迹，已检查六个阶段的指令显示且没有 JavaScript 错误。

## 工程验证与范围

本轮核心回归为 **375 passed、50 skipped**（可选依赖相关跳过），Go1 隔离 SDK 的 **11 项检查全部通过**，包含上游物理/观测/奖励/完整 PPO 更新一致性。Go1 训练与评估、Cartpole 求解、Wuji 零动作评估均通过实际源码快照执行短测；Wuji 短测只验证运行链路，不代表重定向任务成功。安装新构建的 wheel 后，实际 Go1 隔离评估也通过。Ruff 与差异空白检查通过，两个参考仓库保持未修改。

策略结果只覆盖平地、配置中的摩擦随机化、给出的速度指令与仿真窗口。运行、模型、配置和文档均保留在本地仓库，尚未创建提交。
