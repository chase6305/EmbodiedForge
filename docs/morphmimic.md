# MorphMimic 复现

这里运行 [MorphMimic 作者代码](https://github.com/lzyang2000/morphmimic)，固定版本
`aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`。主机入口只负责数据适配、独立环境和运行记录；
世界模型、BPTT、策略、价值网络、FiLM 和潜变量对齐均调用作者实现。
它与现有 `./supermimic` 训练和检查点分开。

## 运行

从仓库根目录执行，不需要设置环境变量：

```bash
./morphmimic setup
./morphmimic doctor
./morphmimic prepare
./morphmimic train --output runs/morphmimic-shared
./morphmimic export --checkpoint runs/morphmimic-shared/training/shared --output runs/morphmimic-export
./morphmimic eval --data runs/morphmimic-data-v1 --policy runs/morphmimic-export/policy --output runs/morphmimic-eval
```

`prepare` 默认使用现有 LAFAN1 G1 的 `walk1_subject1`、`run2_subject1`、`dance1_subject1`，
各取开头 12 秒；默认数据目录为 `runs/morphmimic-data-v1`。
目录必须是新的，重复运行请使用另一个 `--output`。
自选已有 50 Hz G1 数据可传 `--clips /path/a.npz /path/b.npz`，准备和评测时使用相同片段。
源文件必须携带 `joint_names`、`body_names` 和 wxyz 身体四元数。

默认训练 G1、X2 Ultra、T2，600 次迭代，每个机器人 32 个环境、批量 32。
前 500 次为作者实现的世界模型预热，后 100 次才包含策略和两套价值网络更新。
检查点每 500 次迭代和结束时保存，保留完整网络和优化器状态。
这是更新路径验证配置，不能据此期待稳定跟踪。扩大采样量可用
`--iterations 1000 --num-envs 128 --batch 128`。继续训练时指定更大的总迭代数：

```bash
./morphmimic train --resume runs/morphmimic-shared/training/shared --iterations 2000 --output runs/morphmimic-shared-continued
```

恢复从已保存检查点的迭代数开始；日志中更晚但尚未保存的迭代需要重新训练。
作者实现恢复网络、归一化器、优化器、调度、采样器和随机状态，重新开始物理仿真并填充回放缓存。
`--output` 使用新目录，保留原运行记录。

只验证 G1 的完整更新路径：

```bash
./morphmimic train --robots g1 --iterations 520 --num-envs 64 --batch 64 --output runs/morphmimic-g1
./morphmimic export --checkpoint runs/morphmimic-g1/training/shared/latest.pt --output runs/morphmimic-g1-export
./morphmimic eval --robots g1 --data runs/morphmimic-data-v1 --policy runs/morphmimic-g1-export/policy --output runs/morphmimic-g1-eval
```

也可使用 `python -m embodiedforge morphmimic ...`。
安装目录为 `.cache/morphmimic-venv`，Python 3.11；依赖从作者 `uv.lock` 冻结导出，
通过 PEP 751 校验下载内容。镜像只改变下载地址，不改变版本和包的原始哈希。
不安装硬件遥操作所需的 GMR 或 Unitree SDK。

## 方法与当前数据边界

训练启用共享世界模型、共享策略与价值网络、125 维形态描述、完整形态 FiLM、
32 步想象轨迹、跨机器人潜变量对齐和五点关键点接口。
世界模型隐藏层仍为 `(2048, 2048, 2048)`，策略隐藏层仍为 `(2048, 1024, 512, 256)`。
G1 单机器人没有跨机器人对齐项；三机器人训练使用作者默认权重 0.05。

单张 GPU 上使用三个进程，CUDA 张量经 CPU Gloo 同步传输，保持作者原有梯度平均和
潜变量对齐调用。它会比作者的三 GPU NCCL 配置慢。当前关闭 `torch.compile`，
保留默认的损失、学习率、世界模型批次数、传感器噪声、执行延迟和域随机化。

现有 G1 片段按关节名映射，使用作者模型重算 FK 与有限差分速度。
X2、T2 通过作者 canonical joint translator 迁移关节轨迹，按身高缩放根轨迹，
将越界关节裁剪到目标硬限位，并逐帧调整根高度，使最低足部的离地高度与
缩放后的 G1 参考一致，保留原动作的腾空阶段。G1 只作片段整体的穿地修正。
所有身体四元数按作者约定保存为 wxyz 且 w ≥ 0，随后重算速度。
每次转换记录源文件哈希、关节置换、裁剪数量、根高度修正范围和输出哈希。
作者的运动学可行性检查必须通过；浮足、穿地或缺失支撑的片段会报错。
这不是论文的 SOMA 动作重定向，不能据此报告论文同口径的跨机器人结果。

目前是小数据上的方法与工程流程验证。导出使用作者 ONNX/PyTorch 数值一致性检查，
评测分别运行完整动作和五点接口的 CPU MuJoCo 仿真。
`run.json` 的 `complete` 表示命令成功退出；不表示动作跟踪达标。
实际完成帧数、误差和失败情况以评测报告为准。训练片段上的评测也不代表泛化能力。

后续接入获授权的 BONES-SEED：按作者流程转换和重定向三种机器人，生成各自的
subset YAML，再用 `--data <包含 g1.yaml、x2.yaml、t2.yaml 的目录>` 训练。
当前不下载或代为接受受限数据的访问条件，也不操作实体机器人。

## 产物

每条命令保存 `request.json`、`run.json`、`worker.log` 和 `result.json`。
运行记录包含上游版本、源文件验证及本地适配器哈希，失败和中断保留真实状态。
训练另存 `<robot>-training.json`，检查五个优化器均发生实际更新、参数有限。
三机器人导出逐项检查共享网络权重一致性，并记录每种机器人的 ONNX 数值误差。

通信验证脚本：

```bash
.cache/morphmimic-venv/bin/python -I benchmarks/morphmimic_collectives.py --output /tmp/morphmimic-collectives
```

它真实启动三个 CUDA 进程，检查 SUM/MIN/MAX、广播、聚集以及作者潜变量对齐损失和梯度。
