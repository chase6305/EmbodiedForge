# 环境适配与共用评估

`envs.base.BatchEnvironment` 是用于 CPU 采集与评估的最小协议，核心 `VectorEnv` 直接满足该协议。`Go1Environment` 和 `H1Environment` 包装已经构造的原生环境，不改变其训练 API、奖励、控制器或物理步进。

## 公共语义

| 接口 | 约定 |
| --- | --- |
| `spec` | 环境数、观测/动作维度、控制周期、动作单位和动作范围；无界策略动作的范围为 `None` |
| `reset(env_ids=None)` | 显式重置；只返回选中行，保留调用方 ID 顺序 |
| `observe()` | 返回当前缓存观测的独立副本，不推进物理、不重新采样噪声 |
| `step(actions)` | 返回 `StepResult`，观测属于动作执行后、reset 之前 |

核心与原生适配器的 `spec` 现在声明 `schema_version: 1` 和 `done_handling`。核心为 `freeze`；Go1/H1 为 `reset_before_next_step`。`envs.validate_environment_spec` 对公共 CPU 布局进行只读校验，评估器在 reset 前执行校验，记录器在认领输出目录前校验，读取器在打开 episode 归档前校验。未声明版本的旧规格继续支持；明确声明的未知版本直接拒绝，不按旧格式猜测。

这个版本属于嵌套的环境规格，不改变 run、episode 或 checkpoint 自身的 schema。它声明观测所有权、显式 reset 与终止帧语义，不能证明仿真资产或策略 checkpoint 兼容。核心 PPO 加载带版本的环境规格时，会先校验规格再构造策略；旧 checkpoint 继续使用其原有兼容路径。

观测包含 `env_id`、`episode_id`、`step_id`、`state_version`、`time` 和 `proprio`。原生适配器的 episode/version 从包装时开始计数，属于本次包装会话；时间为本 episode 的控制步数乘以控制周期。

在原生环境完成全量 reset 后构造适配器，此后通过适配器调用 reset/step。直接修改底层环境、速度指令或调用其 obs/reset/step 可能让缓存与元数据失去一致性，需要重新建立包装边界。适配器借用环境，调用方保留资源和随机种子的管理权。

核心环境支持冻结 done 行。Go1/H1 的批量求解器仍推进整个 batch，因此适配器要求在下一步前显式 reset 所有 done 行；未完成 reset 时直接报错，避免终止后的物理状态被误认为冻结。局部 reset 不修改其他行的缓存观测。任务执行失败会令适配器失效；输入校验失败则允许修正输入后继续。

Go1 将 `fell` 作为 `terminated`，其余 done 作为 `truncated`，保留原生无界策略动作。H1 保留原生 `[-5,5]` 动作裁剪。两者的 `info.executed_action` 记录实际提交的策略空间动作，奖励项以 `reward/<名称>` 保存。动作最终经原任务的比例和默认关节姿态转换为物理控制目标。

适配器同时拒绝冲突的终止/超时标记，以及不能表示为有限 float32 的奖励，避免在类型转换后发布无穷值。输入动作错误仍可修正重试；原生执行或输出契约出错会令本次适配会话失效。

## 共用评估器

```python
import numpy as np
from embodiedforge import Config, VectorEnv
from embodiedforge.envs.evaluation import evaluate_policy

with VectorEnv(Config(num_envs=2, max_steps=20)) as env:
    result = evaluate_policy(
        env,
        lambda observation: np.zeros((2, 2), dtype=np.float32),
        steps=40,
    )
    print(result)
```

在具备原生 Go1 依赖的环境中：

```python
import numpy as np
from embodiedforge.locomotion.go1 import Go1
from embodiedforge.envs.locomotion import Go1Environment
from embodiedforge.envs.evaluation import evaluate_policy

native = Go1(num_envs=2, num_threads=1, episode_steps=20)
env = Go1Environment(native)
result = evaluate_policy(
    env,
    lambda observation: np.zeros((2, 12), dtype=np.float32),
    steps=40,
)
```

示例使用零动作检查接口，不代表有行走能力。训练好的策略可通过 `action_fn` 将 `observation["proprio"]` 转为单步 `[N,A]` 动作；策略加载仍需对应工作流的 checkpoint 校验。

`evaluate_policy` 开始时全量 reset，并在后续步骤前 reset done 行。它统计步奖励、完整 episode return、终止/超时数量，以及任务实际提供 `success` 时的成功率；未结束的 episode 不计入 episode 均值。`on_reset` 回调接收全局环境 ID，可清理策略历史。`writer.append(observation, result)` 在 reset 前调用，可接收通用 transition。

## 记录原生环境

公共 `EpisodeRecorder` 接受上述环境协议。核心环境可自动提供原有元数据；原生或其他自定义环境需要显式提供 `RecordingMetadata`，避免根据动作维度推测资产或重建配置。

```python
from embodiedforge.data import EpisodeRecorder, RecordingMetadata, read_episodes

metadata = RecordingMetadata(
    family="go1",
    max_steps=20,
    seed=0,
    instruction="Follow the velocity command in the observation",
    scene="go1-joystick:v1",
    scene_spec={"kind": "go1-joystick"},
    config={"num_envs": 2, "episode_steps": 20, "seed": 0, "threads": 1},
    source={"asset_identity": "not_recorded"},
)
with EpisodeRecorder("runs/go1-episodes", env, metadata=metadata,
                     episode_format="compact", write_workers=1,
                     verify_archive=True) as writer:
    result = evaluate_policy(
        env, lambda observation: np.zeros((2, 12), dtype=np.float32),
        steps=40, writer=writer,
    )
episodes = list(read_episodes("runs/go1-episodes"))
```

示例接续上方的 Go1 环境。记录目录必须是新路径；`max_steps` 是每个 episode 的缓存上限，需要覆盖任务自己的 episode 限制。配置、场景、标定和来源在构造记录器时复制；如需追溯具体资产，请在 `source` 中提供其来源和散列。示例明确未记录资产身份，不能单凭这份清单重建原仿真。

原生数据与核心数据使用相同的完整 episode 标记、后台写入、容量限制、读取器和 action window。原生采集标记为 `episode-data`，保留 `environment_family`。当前 ACT 导出仍只支持核心环境数据，遇到原生数据会在创建输出或导入 LeRobot 前拒绝；共用数组协议不代表已实现所有环境的 ACT 闭环。

## 学习工具边界

`learning.advantages` 保存核心/H1 共用的 NumPy GAE，`learning.checkpoints` 保存 CPU Adam 状态校验，`learning.parameters` 保存探索参数约束。H1 策略不再从核心训练入口导入数学函数；Go1/H1 的策略模块不需要导入仿真 SDK。

Go1 的 GAE 仍使用其原生 rollout 契约：超时 bootstrap 已折入奖励。核心/H1 使用显式 `next_value` 和两类 done；没有通过改写公式强制合并这两种数据语义。观测归一化、镜像策略、优化参数和 checkpoint 布局保持各任务自己的约定。

此适配层不提供 GPU 张量传输或通用机器人场景。扩展这两项能力前，应由具体任务和后端确定张量布局、同步事件、资产映射和生命周期，再通过独立契约验证。

记录器在修改缓存前检查 episode 内的字段集合、shape 和 dtype 连续性，以及与环境规格一致的 proprio 维度。动作与奖励必须为有限实数数组；布尔或复数载荷会被拒绝，读取器也会拒绝这类损坏产物。校验错误可修正后重试，写盘失败仍会终止本次记录。

构造记录器时，依赖元数据在认领新目录前解析。目录创建后，首次清单发布或后台线程池建立失败也会进入收尾路径，尽力发布 `failed` 状态并释放已建立的资源；收尾再次失败会记录日志，保留最初的异常。

读取器首先确认 manifest 属于 episode 数据集，并验证其环境布局；明确声明的其他工作流或未知身份版本不会被当作旧数据集。完整及投影读取都检查动作/proprio 的必要维度是否与清单一致。未带 `run_contract` 的旧版核心数据集继续支持。

长期采集的动态 episode 文件名在写入与读取热路径中保持为 filesystem 字符串，避免本机 CPython 3.12 的路径组件驻留增长；外部目录参数仍接受 `str` 和 `Path`。数据文件名、完成标记、原子发布次序和排序规则保持兼容。

## 可选归档完整性核验

`EpisodeRecorder(..., verify_archive=True)` 会在发布每个 NPZ 前，用有界读取计算 SHA256，再读回校验所有 ZIP entry 的 CRC。校验失败时保留临时文件供诊断，终止本次记录，不发布该 episode 的完成标记。正常完成时，标记中保存 `archive_sha256`。

```python
from embodiedforge.data import EpisodeReadOptions

options = EpisodeReadOptions(observation_keys=(), verify_archive=True)
episodes = read_episodes("runs/go1-episodes", options=options)
```

上例需要使用开启归档核验的记录器生成数据。读取器从同一个已打开文件完成散列与解码，避免路径被原子替换后读入另一份文件；缺失或不匹配的 SHA256 会在解码前报错。投影读取仍不解压未选字段，但完整压缩文件需要扫描，因此会增加 I/O。写后 CRC 校验还增加一次解压扫描；默认关闭这项额外成本，旧数据和既有参数仍可按原方式读取。

CRC 错误会包含具体归档路径并保留原始异常。核验用于发现损坏，不自动修复或忽略坏文件。记录器的载荷字节预算仍不等同于进程 RSS。
