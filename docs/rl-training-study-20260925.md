# RL 训练对照：H1 学习率、Go1 学习器与 Microduck 续训

本轮得到一个可落地的默认参数调整：H1 原生 PPO 新训练的学习率从 `1e-3` 调整为 `3e-4`。三个训练种子、每个 5000 次更新的对照中，较低学习率的策略均完成站立、前进和左右原地转向的 10 秒标称起点评估。Go1 保留原生 PPO 默认值；Microduck 的续训机制修复通过实际训练，但新增权重没有改善本次固定前进指令表现。

[机器可读记录](rl-training-study-20260925.json) 保存各训练种子、评估结果、原始报告路径和 SHA-256。下文 `runs/` 链接是本机产物，不随 Git 分发；表格和 JSON 摘要可以随 PR 审阅。全部实验无窗口运行，没有新增启动试训模式。

## 实验口径

- Go1/H1：AMD Ryzen 9 9950X，CPU 物理与 PPO，两个线程；Torch 2.9.0、MuJoCo 3.11.0、mjbatch 0.1.0。
- Microduck：RTX 5090 D v2，Torch 2.9.1、mjlab 1.3.0、RSL-RL 5.0.1，GPU 仿真与 PPO。
- H1/Go1 的表格对各次报告的 RMSE 做算术平均，不是把所有帧重新汇总后的 RMSE。不同指令分开统计。
- 评估种子不等于训练种子。单个模型上的多个重置样本不能替代独立训练实验；没有据此构造置信区间。
- 实验有并行任务及共享硬件负载，墙钟时间仅用于描述本次成本，不能据此声称某后端更快。

## H1：5000 次更新下的学习率对照

每个学习率分别从零训练 seed 0、1、2，其他参数相同：128 环境、24 步 rollout、5000 次 PPO 更新，即每个模型 **15,360,000 条环境转换**。六个训练均正常完成，每个约 18.5–19.9 分钟。

每个模型评估四种指令：站立 `(0,0,0)`、前进 `(0.5,0,0)`、左转 `(0,0,0.5)`、右转 `(0,0,-0.5)`。每次一个环境、500 控制步、10 秒；采用确定性的标称姿态，关闭训练随机化，首次跌倒后停止计入该次试验。这里的三个样本来自不同训练种子，而非复制同一个起点来扩大样本数。

| 指令 | 学习率 | 完整存活模型数 | 平面 RMSE 均值，m/s | 转向 RMSE 均值，rad/s |
| --- | --- | ---: | ---: | ---: |
| 站立 | 1e-3 | 2/3 | 0.3629 | 3.0431 |
| 站立 | 3e-4 | 3/3 | 0.0867 | 0.0911 |
| 前进 | 1e-3 | 2/3 | 0.2982 | 3.1557 |
| 前进 | 3e-4 | 3/3 | 0.1309 | 0.0687 |
| 左转 | 1e-3 | 1/3 | 0.4050 | 2.6851 |
| 左转 | 3e-4 | 3/3 | 0.0881 | 0.0842 |
| 右转 | 1e-3 | 2/3 | 0.2124 | 4.6870 |
| 右转 | 3e-4 | 3/3 | 0.0947 | 0.0999 |

`1e-3` 的 seed 2 在四种指令下均跌倒，因而平均误差较大；seed 0、1 的转向跟踪也不足。`3e-4` 的三个模型前进平均速度分别为 **0.4851、0.4857、0.3948 m/s**，指令为 0.5 m/s。保留所有种子的结果，没有排除失败模型。提前跌倒的 RMSE 只覆盖终止前的观测时间，应与存活列一起解读。

据此调整新训练默认值。续训省略 `--learning-rate` 时继承保存的 Adam 学习率，避免升级默认参数后悄悄改变旧实验；显式传参可覆盖。训练和评估也接受明确的 `--headless`，省略时仍无窗口。

复现单个配置；更换 `--seed` 和 `--learning-rate` 即可覆盖六组实验，输出目录必须不同：

```bash
.cache/light-loco-venv/bin/python -m embodiedforge h1-native train --headless \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml \
  --num-envs 128 --threads 2 --horizon 24 --updates 5000 \
  --seed 0 --learning-rate 0.0003 --output runs/my-h1-lr0003-s0
.cache/light-loco-venv/bin/python -m embodiedforge h1-native evaluate --headless \
  --run runs/my-h1-lr0003-s0 --num-envs 1 --threads 2 \
  --steps 500 --velocity 0.5 0 0 --record-motion \
  --output runs/my-h1-lr0003-s0-forward
```

解释器路径是本机 SDK，其他机器按 [H1 环境说明](h1-native.md) 安装。完整命令及产物见 [H1 实验记录](../runs/h1-learning-study-20260925/study.json)。当前结论限于该平地模型、预算与标称起点；没有完成扰动、多地形或真机验收，也没有证明 `3e-4` 在全部预算下最优。

## Go1：原生 PPO 与 Light Loco

每个模型从零训练 2000 次更新、1024 环境、24 步，即 **49,152,000 条环境转换**。两者使用相同的本仓库环境、镜像策略、归一化与 checkpoint 格式；任务语义为 `transition-v2`，奖励和指令配置均为 `original`。学习率沿用 `1e-3 → 5e-4` 的原有调度。

完整对照采用训练 seed 0、1。每个模型使用评估 seed 1001、1002、1003，每个种子 64 环境、500 步，四种固定指令；只统计首次 episode。Go1 的左右转指令同时包含 0.5 m/s 前进，与上面 H1 的原地转向不同。

| 指令 | 原生平面 RMSE | Light Loco 平面 RMSE | 原生转向 RMSE | Light Loco 转向 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 站立 `(0,0,0)` | 0.0162 | 0.0173 | 0.0084 | 0.0081 |
| 前进 `(0.5,0,0)` | 0.0733 | 0.0797 | 0.0257 | 0.0293 |
| 左转 `(0.5,0,0.5)` | 0.0757 | 0.0830 | 0.0358 | 0.0364 |
| 右转 `(0.5,0,-0.5)` | 0.0753 | 0.0838 | 0.0350 | 0.0351 |

平面单位为 m/s，转向单位为 rad/s。上述各次评估的完整存活率均为 100%。这证明两种学习器在本任务和预算下都能得到可跟踪指令的策略；只有两个完整训练种子，不作统计显著性或普遍优劣判断，保留原生 PPO 默认值。

原计划包含第三个训练种子。原生 seed 2 完成，但 Light Loco seed 2 在整组实验的预设截止时间到达后超时，最后提交的保存点为第 450 次更新。它没有达到同等预算，**第三个种子的两侧均不纳入上表**。保存点仍可恢复；不会把一次恢复后的实验标为原来的连续训练对照。完整记录见 [Go1 实验记录](../runs/go1-backend-study-20260925/study.json)。

```bash
.cache/light-loco-venv/bin/python -m embodiedforge recipes train \
  --task go1-joystick --standalone --headless --go1-learner native \
  --num-envs 1024 --horizon 24 --updates 2000 --threads 2 --seed 0 \
  --go1-semantics transition-v2 --go1-command-profile original \
  --go1-reward-profile original --timeout 7200 --output runs/my-go1-native-s0
.cache/light-loco-venv/bin/python -m embodiedforge recipes evaluate \
  --task go1-joystick --standalone --run runs/my-go1-native-s0 \
  --num-envs 64 --steps 500 --threads 2 --suite basic \
  --seeds 1001 1002 1003 --timeout 1200 --output runs/my-go1-native-s0-eval
```

替换 `--go1-learner light-loco` 并使用新的目录即可运行另一侧。另尝试过把 Go1 原观测和镜像观测合并为一次 actor 前向；局部计时在常用的大 minibatch 上反而慢约 7–16%，因此没有采用该改动。

## Microduck：续训修复与行为结果

原生 runner 加载 checkpoint 后，现在从保存迭代号的下一轮继续。旧上游显式 `--repo` 模式和历史日志仍按旧规则读取。实际从 `model_3017.pt` 追加 1000 次更新，512 环境、24 步，共 **12,288,000 条新增转换**：

| 检查项 | 实际结果 |
| --- | --- |
| 本次日志编号 | 3018–4017，1000 次完整更新 |
| 最终模型 | `model_4017.pt` |
| 首次 reset 课程计数 | 72,480，与来源一致 |
| 最终课程计数 | 96,480，增加 1000 × 24 |
| 最终学习率 | 0.00003375 |
| TensorBoard / checkpoint | 标量、actor、critic、优化器状态均有限；NaN 状态计数为零 |

[训练记录](../runs/microduck-resume-training-20260925/run.json) 与 [指标检查](../runs/microduck-resume-training-20260925/metrics.validation.json) 保存原始证据。另用真实 CPU RSL-RL 的学习、保存、加载验证了 Adam 步数和日志编号连续，以及仅加载 actor 不推进训练编号。

续训前后采用相同评估设置：前进指令 `(0.2,0,0)`、中性头部/身体指令、关闭推力、课程起点统一为 96,480；seed 0、1、2，各 32 环境、500 步。以下为三个评估种子的均值 ± 样本标准差：

| 指标 | 续训前 | 续训后 |
| --- | ---: | ---: |
| 首次 episode 完整存活 | 96/96 | 96/96 |
| 平面 RMSE，m/s | 0.1931 ± 0.0029 | 0.1928 ± 0.0029 |
| 转向 RMSE，rad/s | 0.2300 ± 0.0142 | 0.2733 ± 0.0184 |
| 平均前进速度，m/s | 0.0324 ± 0.0088 | 0.0293 ± 0.0096 |

两者都明显低于 0.2 m/s 的前进指令。新增预算验证了续训流程，但没有证实更好的行走行为，不能仅因训练完成或不跌倒就升级部署权重。没有用此次结果修改奖励或关闭训练随机化来掩盖问题。

复现该后模型评估；前模型换成 `runs/microduck-hour-opt-training-20260916`，保持其余条件相同：

```bash
conda activate ef
python -m embodiedforge.microduck evaluate --headless --quiet \
  --run runs/microduck-resume-training-20260925 \
  --num-envs 32 --steps 500 --seeds 0 1 2 --velocity 0.2 0 0 \
  --no-pushes --curriculum-step 96480 --output runs/my-microduck-after-eval
```

原始 [续训前](../runs/microduck-resume-before-eval-20260925/summary.json) / [续训后](../runs/microduck-resume-after-eval-20260925/summary.json) 报告包含每个 seed、样本量和 checkpoint 哈希。只有一次训练续训链，三个评估种子不是三次独立后训练。

## 参数量、权重与训练文件大小

实际计数与文件大小如下，单位为字节。总可训练参数包括策略探索标准差；网络参数列不包含它。状态张量还可能包含观测归一化统计。

| 模型 | Actor 网络参数 | Critic 网络参数 | 总可训练参数 | 模型状态张量字节 | 完整训练 checkpoint 字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 44,435 | 42,113 | 86,567 | 346,268 | 1,059,603 |
| Go1，两种学习器共用 | 24,588 | 23,169 | 47,769 | 191,480 | 590,493 |
| Microduck | 197,774 | 203,777 | 401,565 | 1,607,920 | 4,847,293 |

训练 checkpoint 包含 Adam 动量及元数据，不能把其大小称作纯 actor 权重大小；表格也没有计入仿真资产、SDK、训练显存或运行时内存。本轮未测量新的 ONNX 文件大小。Microduck 的预训练、后训练和完整消融设计继续参见 [模型与实验说明](microduck-model-training-ablation.md)。本轮 H1/Go1 是从零训练的超参数/学习器对照，Microduck 是已有 RL 模型的追加训练，不包含新的模仿学习预训练。

## RLinf 与恢复可靠性

以下保留本轮当时的验证边界；后续实际 GPU 训练、恢复与评估结果见 [2026-09-26 实验](rl-training-study-20260926.md)。

RLinf 通过受支持的 worker 扩展入口，在构造模型前设置 `seed + rank`，补齐 Python、NumPy、Torch 的进程内随机种子。独立进程和实际本地 Ray WorkerGroup 均验证了相同 seed/rank 的 MLP 权重及随机动作一致，后续调用继续推进 RNG。没有改变上游 SAC 更新公式，未完成 ManiSkill/PickCube 的端到端 GPU 训练，详见 [RLinf 验证边界](rlinf.md#当前验证边界)。专用环境只做了依赖解析，没有安装新的 ManiSkill/SAPIEN 组合，也没有改动 `ef` 的依赖。

Go1 发布保存点时，临时文件删除失败现在只记录警告，不再覆盖原来的中断或发布异常。已有测试覆盖提交前后及清理失败，最后有效保存点保持可判断；这没有增加新的恢复框架。
