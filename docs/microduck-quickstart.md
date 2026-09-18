# 机械鸭：直接训练与部署

使用仓库内已移植的 Microduck 平地行走实现，无需 `3rdparty/microduck_rl`，直接开始训练。主入口使用 `ef`，CUDA 训练依赖运行在独立环境中。以下输出目录与 ONNX 文件须尚不存在；重复运行时更换名称。

网络结构、权重大小和预训练／后训练／消融实验的详细说明，见 [机械鸭模型与实验详解](microduck-model-training-ablation.md)。

## 环境

```bash
cd /home/ubuntu/workspace/chase/EmbodiedForge
conda activate ef
```

仅首次准备依赖或依赖清单更新后执行；本机已准备好的环境可跳过：

```bash
python -m embodiedforge.microduck setup
```

## 正式训练

```bash
python -m embodiedforge.microduck train \
  --headless --quiet --seed 0 \
  --num-envs 512 --iterations 4000 \
  --output runs/duck-walk
```

该命令不启动窗口或录像，仍需 CUDA。512 个并行环境是训练规模示例；显存不足可减小 `--num-envs`。4000 次更新是预算，不保证达到行走验收要求。`--quiet` 将完整工作进程输出保存在运行目录的阶段日志中。

另开同样进入仓库并激活 `ef` 的终端查看进度：

```bash
python -m embodiedforge.microduck progress --run runs/duck-walk
```

Ctrl+C 或 SIGTERM 正常停止后，可以从已记录的 checkpoint 追加训练；完整训练也可使用同一续训入口：

```bash
python -m embodiedforge.microduck train \
  --resume-run runs/duck-walk \
  --headless --quiet --seed 0 \
  --num-envs 512 --iterations 1000 \
  --output runs/duck-walk-resumed
```

仍标记为 `running` 的目录不自动恢复。若改用续训结果，下方所有 `--run runs/duck-walk` 改为 `--run runs/duck-walk-resumed`，导出与评估输出也换新名字。

## 仿真部署与 ONNX 导出

训练完成后，在带桌面的主机上回放策略：

```bash
python -m embodiedforge.microduck play \
  --run runs/duck-walk --viewer native
```

或者启动 Viser 浏览器查看器，打开终端打印的地址：

```bash
python -m embodiedforge.microduck play \
  --run runs/duck-walk --viewer viser
```

训练仅保存 checkpoint；需要 ONNX 时单独导出。模型含观测归一化，输入 61 维、输出 14 维；导出使用 CUDA，推理检查通过后才发布文件：

```bash
python -m embodiedforge.microduck export \
  --run runs/duck-walk --headless --quiet \
  --output runs/duck-policy.onnx
```

新导出的 `default_joint_pos` 元数据保留完整浮点精度，关节顺序为左腿、头颈、右腿。动作是默认关节角的偏移；具体顺序、转换公式及执行器参数说明见 [动作与关节目标](microduck-model-training-ablation.md#66-从网络动作到关节目标)。

## 无窗口评估与录像

用训练 checkpoint 驱动仿真，并在实际观测上对照 ONNX 动作；同时录制各个种子的环境 0：

```bash
python -m embodiedforge.microduck evaluate \
  --run runs/duck-walk --onnx runs/duck-policy.onnx \
  --headless --quiet --video \
  --velocity 0.2 0 0 --no-pushes \
  --num-envs 16 --steps 1000 --seeds 0 1 2 \
  --output runs/duck-eval
```

汇总见 `runs/duck-eval/summary.json`，详细指标和录像位于各 `seed-N/` 下的 `evaluation.json`、`policy.mp4`。不需要录像时去掉 `--video`。本例未设行为验收阈值，完成表示流程及数据校验通过；速度误差和存活率门槛见 [详细说明](microduck.md)。

当前“部署”包括仿真回放与模型导出；尚未移植机械鸭的实机控制程序。ONNX 动作对照仍由 PyTorch 策略驱动仿真，不是 ONNX 独立闭环或实机部署。
