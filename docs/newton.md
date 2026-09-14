# Newton v1.6.0rc1 兼容说明

物理后端的兼容目标固定为用户指定的 [v1.6.0rc1 源码 ZIP](https://github.com/newton-physics/newton/archive/refs/tags/v1.6.0rc1.zip)。`pyproject.toml` 的 `newton` 可选依赖直接引用该 URL，适配器在构建前检查 SDK 版本必须为 `1.6.0rc1`；不会自动接受 main 或其他版本。

该标签声明 Python ≥3.10、Warp ≥1.17.0；本次安装与验证使用 conda `ef`、Python 3.10、Newton 1.6.0rc1、Warp 1.17.0。[该标签的依赖定义](https://github.com/newton-physics/newton/blob/v1.6.0rc1/pyproject.toml)

## 已实现的能力

| 项目 | 实现 |
| --- | --- |
| 注册名 | `physics="newton"` |
| 适配器 | [NewtonPhysics](../src/embodiedforge/backends/newton.py) |
| 场景 | 当前 `point_reach`；支持 reach 和 hold 两个任务 |
| 求解器 | Newton `SolverSemiImplicit`，实际动力学由 Newton/Warp 执行 |
| 并行实例表示 | 一个 Newton model，每个环境一个独立 world，每个 world 一个粒子 |
| 控制 | XY 力，零重力、Z 方向无力且初始位置/速度为零 |
| 状态交换 | CPU、显式 host snapshot、float32 物理状态 |
| reset/done | 选择性 reset；done 环境的位置、速度、时间和版本冻结 |
| 渲染组合 | `null` 或独立 `raster`；保持现有图像与数据协议 |
| 运行追溯 | plan 展示固定版本与 ZIP 地址；run manifest 记录实际依赖版本及安装来源元数据 |

这是现有点质量场景的真实 Newton 适配器。它不代表已实现关节机器人、接触任务、URDF/USD 导入、Newton 相机或 GPU 训练；这些能力需要各自的场景规范与适配测试。

## 版本特定的处理

适配器通过该标签的 `ModelBuilder.begin_world/add_particle/end_world/finalize` 创建模型，使用两份 `State` 交替积分。

当前任务没有接触，关闭 `particle_grid`，避免重叠环境实例产生粒子相互作用。每个控制周期将活动掩码映射到 `ParticleFlags.ACTIVE`。该标签的 inactive 粒子积分只写出位置，因此每个子步先将输入速度复制到输出缓冲区，保证冻结行不会读到旧速度。[该标签的积分实现](https://github.com/newton-physics/newton/blob/v1.6.0rc1/newton/_src/solvers/solver.py)

局部 reset 同步两份状态缓冲区，只更改指定环境的初始条件；host snapshot 随后采用实际存储的 float32 数值，避免快照与引擎状态不一致。此 CPU 适配器复用参考后端的 host 缓冲区管理，但重写全部物理积分步骤。

## 安装与验证

```bash
conda activate ef
python -m pip install -e '.[newton,mujoco,test]'

# 可选：为 Warp 的 JIT 编译指定可写缓存位置
export WARP_CACHE_PATH=/tmp/embodiedforge-warp-cache

embodiedforge plan --physics newton --task hold --render raster
embodiedforge rollout --physics newton --task hold --render raster --num-envs 4 --steps 110 --output runs/newton
embodiedforge inspect --dataset runs/newton
python -m pytest -q tests/test_newton.py
```

版本测试、两个任务 × 两种渲染的动力学对照，以及非零速度下的 done 冻结/局部 reset 均使用真实 SDK。`tests/test_newton.py` 缺少 SDK 时显式跳过，安装其他 Newton 版本时失败；只有安装正确版本并运行测试才算完成该版本的兼容验证。

本次 `ef` 验证结果：Newton 专项 7 项通过，全量 49 项通过；缺少 PyTorch 的训练模块跳过。CLI 的 Newton + hold + raster 采集生成 6 个完整 episode、337 条 transition，均通过 reader 校验；wheel 构建和 Ruff 检查通过。该结果是 CPU 参考场景验收，不包含 GPU 或关节机器人验收。

核心包仍不依赖 Newton：仅选择该物理后端时才导入 SDK。当前运行时只接受 CPU 传输；不会在用户未指定的情况下自动切换到 CUDA。
