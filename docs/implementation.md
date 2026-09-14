# 当前实现与扩展

## 语言选择

本项目先采用 Python 平台层，调用原生仿真求解器和 PyTorch 算子。MuJoCo 的官方 Python 包本身通过 C++/pybind11 提供原生库绑定，Python 接口不意味着求解器用 Python 执行。[MuJoCo 官方说明](https://mujoco.readthedocs.io/en/stable/python.html)

| 方案 | 当前决策 | 以后采用的条件 |
| --- | --- | --- |
| Python | 环境、任务、训练、后端适配、配置与数据接口的主语言 | 保持公共接口，优先调用后端批量 API |
| C++ | 不维护一套重复的平台实现 | profiling 确认 Python 调度、控制器或状态转换成为瓶颈；或需要原生实时部署时，实现独立扩展 |
| Rust | 当前不引入 | 数据压缩/流式 I/O 或独立服务出现明确性能与部署需求时，再选择性引入 |
| GPU kernels | 当前 CPU 原型没有实现 | Newton/Warp 或其他 GPU 后端需要批量控制、奖励和状态映射时使用 |

需要 C++ 时建议通过 pybind11 暴露稳定的小接口，GPU 数据通过后端支持的设备协议传递；避免先复制整个平台再维护两套逻辑。硬实时真机控制应作为独立进程或组件设计，不能把当前 Python 训练 worker 当作硬实时控制器。

## 当前实现与目标架构的差异

`docs/architecture.md` 是长期设计。当前实现有意限定在以下范围，并在能力解析时拒绝未支持的设备和场景：

公共协议、模块/类职责及本轮接口迁移见 [interfaces.md](interfaces.md)。开发和验证使用 conda `ef` 环境。

- 场景只有 `point_reach`：一个单位质量、受 XY 力控制的球体与一个目标点。不是机械臂训练环境。
- `PhysicsBackend` 提供 reset/control/step/snapshot；`RenderBackend` 通过 sync/render 消费独立状态快照，两者不共享私有对象。
- `Task` 自己持有目标等任务状态，通过 `TaskSpec` 声明观测与动作；内置 reach/hold 分别使用六维/四维 proprio。环境和 PPO 不包含目标点逻辑或固定输入维度。
- `SensorPipeline` 独立管理采样、帧龄、输出校验和缓存。无通道时跳过同步/渲染；同一物理快照复用于奖励、图像和观测。
- 运行时仅支持 CPU `host_snapshot`，状态和观测使用拥有独立内存的 NumPy 数组；尚无零复制、设备事件或 CUDA Graph。
- 当前 `SceneSpec` 不是通用资产格式。USD/URDF/MJCF 导入、stable ID 到复杂关节/刚体的映射、缓存尚待实现。
- MuJoCo 采用真实原生动力学，逐环境步进。NumPy 采用半隐式 Euler 参考积分；两者只在这个简单场景上做容差比较。
- Newton 适配器固定兼容用户指定的 v1.6.0rc1 ZIP，使用 Warp CPU 与 SolverSemiImplicit，支持相同参考任务和独立渲染组合。版本细节及限制见 [newton.md](newton.md)。
- `raster` 是独立正交俯视调试渲染器，输出 RGB、米制 depth、有效掩码和实体/语义标签，不支持 normal。它画的是平面圆盘，不模拟球体表面深度或真实相机光照。
- 相机频率必须整除控制频率，控制频率必须整除物理频率。中间步保持最近图像并报告 frame_age，done/reset 强制采样。
- Episode recorder 是同步、有容量上限的参考实现，压缩/写盘会阻塞采集。尚未实现异步写盘 worker。中途停止的 episode 不进入完整数据集，在 manifest 中列出。
- PPO 使用 proprio，未实现视觉 PPO、SAC、分布式训练或精确断点续训。checkpoint 可恢复模型用于评估，并保存优化器状态，但没有训练恢复 CLI。
- checkpoint v2 保存模型维度与环境规格，兼容读取初版 v1；evaluate 默认恢复 checkpoint 配置，检查任务与动作语义是否兼容。
- VLA 已有多模态数据窗口和策略 action chunk 接口，尚未接入 Transformers 或具体 VLA 模型。

## 添加后端

第三方 Python 包实现 `core.py` 中的协议，并在应用启动代码中显式注册：

```python
from embodiedforge.backends import register
from embodiedforge.core import Capabilities

register("physics", "my_physics", MyPhysicsBackend,
         Capabilities(scenes=frozenset({"point_reach"})))
```

`MyPhysicsBackend` 为应用导入的类。之后可以用 `Config(physics="my_physics")`，或在加载 JSON 配置前注册它。内置 CLI 的命令行覆盖选项只列出内置后端；自定义后端可使用 Python 入口或配置文件配合注册。配置不会执行任意导入字符串。

`resolve()` 在实例化前检查场景、设备、状态传输、局部 reset、活动掩码和渲染通道。声明能力是适配器作者的契约，需要通过集成测试验证，而不是运行时自动证明。缺少 SDK 在创建该后端时报告对应可选依赖。

生命周期为 `factory → build → reset/step/render → close`。构建失败时运行时仍调用 close，因此适配器必须允许对部分初始化资源进行清理。物理 step 的 `dt` 是整个控制周期，`substeps` 是该周期内的子步数。`active=False` 的环境不可被推进，快照必须拥有独立内存。

添加真正 GPU 后端时先扩展 tensor/state 传输协议与任务算子，再实现适配器；不要让 GPU 数据在每步隐式 `.cpu().numpy()`。当前 resolver 会拒绝 GPU 能力声明，避免误用 CPU 路径。

## 策略与数据

单步策略也统一返回 `[N, 1, A]`。VLA 模型适配器实现 `act(observation)` 返回 `[N, H, A]`，交给 `ActionExecutor` 消费。携带 episode_id/step_id 时执行器会自动丢弃过期 chunk；没有这些标识时，调用方在 reset 后必须执行 `executor.reset(done_ids)`。两种情况下都需要调用方清理模型自身的历史/KV cache；执行器不替模型管理缓存。

```python
from embodiedforge.data import action_windows

for sample in action_windows("runs/demo", history=2, horizon=8):
    images = sample["observation"]["rgb"]  # [history, camera, H, W, 3]
    proprio = sample["observation"]["proprio"]
    instruction = sample["instruction"]
    actions = sample["action"]             # [horizon, action_dim]
```

窗口只取长度足够的完整 episode，不做跨边界拼接，不自动 padding。当前任务只有一个固定语言指令，数据接口可传入语言，但不能用此任务衡量语言理解能力。

PPO 对真正 terminated 的 transition 不 bootstrap，对时间限制 truncated 使用 reset 之前的 next value；两类 done 都截断 GAE 递推。采样使用 tanh 压缩动作，PPO 比率在同一个未压缩样本上计算，新旧策略变换的 Jacobian 相消。

## 验证方法

`tests/` 覆盖解析失败、延迟 SDK 导入、参考受力运动、局部 reset 隔离、done 冻结、相机时序/坐标/深度、跨后端动力学与渲染、Episode 完整性、动作窗口、chunk reset、GAE 超时语义及 checkpoint 恢复。

缺少 MuJoCo 或 PyTorch 时，相应测试使用显式 skip。完整验证环境必须安装 `.[mujoco,train,test]`，并确认没有 skip。训练可运行不等于策略收敛，学习效果应单独用固定评估种子对比训练前后奖励和成功率。

初版学习验证（本轮接口重构前）使用 Python 3.10、PyTorch 2.13.0，MuJoCo 集成测试使用 3.13.0。参考 PPO 用 NumPy 后端、32 个环境、100 次更新，共 409,600 个 transition。在三个未用于训练的固定评估种子上，每个种子运行 128 个 episode，得到：

| 评估种子 | 初始策略成功率 | 训练后成功率 | 初始平均 episode return | 训练后平均 episode return |
| --- | --- | --- | --- | --- |
| 1001 | 0.00% | 64.06% | -1.77 | 8.72 |
| 1002 | 1.56% | 73.44% | -1.52 | 7.90 |
| 1003 | 0.00% | 70.31% | -1.59 | 8.03 |

这是一条训练 seed 的基础学习验证，不是多训练 seed 的统计结论，也不代表复杂机器人任务能力。以下命令可复现评估过程；不同依赖版本可能产生不同数值：

```bash
embodiedforge train --num-envs 32 --updates 100 --output runs/ppo
PYTHONPATH=src python benchmarks/check_learning.py runs/ppo/checkpoint.pt
```

学习对比脚本现从 checkpoint 恢复任务、物理和网络维度，也支持 `hold` 及 MuJoCo 训练产物。可用 `--num-envs` 与 `--seeds` 控制配对评估规模，输出包含模型 SHA256、训练配置及评估条件；原始实验数字保留为历史记录。完整命令见 [训练与部署](training-deployment.md#core)。
