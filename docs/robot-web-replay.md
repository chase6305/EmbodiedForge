# 机器人统一 Web 回放

按步骤执行安装、训练、续训、评估和运行，见 [训练与部署命令](training-deployment.md)（[English](training-deployment.en.md)）。

Go1 和 H1 的已有运动记录现在可以用真实机器人网格在统一页面中回放。
MuJoCo、Newton OpenGL、OVRTX 共用时间轴、播放/暂停、单帧步进、回到开头、
相机跟随、视角和截图控件。画面来自服务端实际渲染，包含资产几何和关节运动。

这条入口读取已有 NPZ 记录，不推进新物理、不执行策略、不训练模型；页面会明确
显示“记录回放”。Go1 另有 [在线策略入口](go1-live-web.md)，支持直接加载 checkpoint
并调整实时速度指令；H1 在线执行与 Web 训练管理尚未接入。
原有无 SDK 的离线骨架 HTML 回放继续可用。

## 运行

在仓库根目录使用已配置的 `ef-viewer`，或安装 Python 3.11 的相应查看器依赖：

```bash
conda activate ef-viewer
python -m pip install -e '.[viz-robot]'

# Go1 分段指令策略的记录；多个 motion 表示同一机器人不同试验
python -m embodiedforge replay \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_go1/go1.xml \
  --motion runs/go1-switching-v1-20260913/motion-seed-0-switching.npz \
           runs/go1-switching-v1-20260913/motion-seed-1-switching.npz

# H1 前向行走记录，OVRTX 画面（需 viz-rtx 依赖）
python -m embodiedforge replay \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml \
  --motion runs/h1-suite-60s-motion-20260912/motion-seed-0-forward-env-0.npz \
  --render-backend rtx
```

打开 `http://127.0.0.1:8080`，默认暂停，点击“继续”。两个示例应分别运行；
同时启动时指定不同 `--port`。也支持 `python -m embodiedforge.robot_replay`。
`--model` 接受本地匹配的 MJCF，网格等相对资源仍由模型目录解析；示例路径引用
本机已有的 MuJoCo Menagerie 资产，其他机器需改成自己的资产路径。

回放至少需要 MuJoCo 3.11 和 Pillow；`viz-robot` 提供这两个依赖。
GL/RTX 还需各自的 `viz-gl`/`viz-rtx`，安装与驱动说明见 [可视化后端](viewers.md)。
Raster 正交点渲染器不支持机器人网格，在页面中禁用。Web 不需要 ImGui。

## 交互语义

| 操作 | 行为 |
| --- | --- |
| 继续 / 暂停 | 控制当前会话内全部记录，按各自保存的时间戳播放 |
| 单步 | 每条记录前进一步；不按画面 FPS 推算、不插值 |
| 环境下拉框 | 选择 `--motion` 列表中的一条记录，编号从 0 开始 |
| 时间轴 | 跳到选中记录的指定帧，同时暂停播放 |
| 回到开头 | 只将选中记录回到第 0 帧，其他记录位置保留 |
| 相机跟随 | 以当前机器人根部位置为视点目标；关闭后保持世界中的目标位置 |
| 渲染器切换 | 当前记录、时间轴和相机保持不变；候选首帧失败时保留原渲染器 |
| 记录结束 | 保留末帧；有终止或截断标记则显示该标记，不自动循环到新试验 |

所有记录结束后暂停，使用时间轴或“回到开头”再播放。屏幕显示当前记录的
指令段、保存的速度指令、时间、帧数和根部位置；速度由相邻记录位置差分得出，
不是训练评价器的局部速度指标。记录没有 reward，因此显示 `—`。

`--fps` 默认 30，控制画面提交上限；`--width/--height` 默认 960×540。
这些画面设置可在 Web 中实时调整，目标帧率范围 1–60；分辨率变化保留当前回放位置。
常规渲染耗时计入回放的墙钟推进时间，只有渲染器重建等待会排除在回放推进之外。
`--duration 10` 从首帧成功后计时，0 表示持续运行。停止按钮或 Ctrl+C 释放资源。
首次 RTX 初始化可能较慢，切换期间保留上一张画面；子进程等待上限为 180 秒。
渲染初始化不会计入回放推进时间，较慢机器的实际播放速率会受渲染耗时约束。

## 资产与记录校验

加载记录时保留原始浮点数据，按关节名称映射 MJCF `qpos`，支持一个浮动根关节
以及其余 hinge/slide 关节。要求覆盖所有非根关节，不按两个数组的相同下标猜测映射。
记录需明确使用 `xyzw` 四元数；转换到 MuJoCo 的 `wxyz` 并检查单位范数。

随后对**每一记录帧**执行前向运动学，核对记录的刚体位置与方向。
位置误差大于 1 mm、方向误差大于 0.002 rad 时拒绝打开并报告帧号。
缺失/重复关节或刚体名、错误根部、非有限数值、非法时间戳、终止帧之后仍有记录，
也会被拒绝。Go1 中额外记录的足部 site 点不参与刚体原点对齐校验。

MuJoCo 渲染使用 MJCF 资产与材质。GL/RTX 读取编译后的网格顶点和刚体几何变换，
支持 mesh、plane、sphere、box、capsule、cylinder、ellipsoid，复用重复肢体网格。
平面显示区域在平面内跟随相机，避免长距离移动时走出显示地面。
当前 GL/RTX 传递基础颜色与透明度，未转换纹理、完整材质着色器、灯光或地形高度场；
遇到不支持的可见几何会明确报错，可继续使用 MuJoCo 渲染。
三个适配器均隐藏 group 3..5 的碰撞辅助几何；模型需遵循这一可视分组约定。

## 本机验收

2026-09-14：Go1 两条记录共 6000 帧、H1 一条记录 3000 帧完成逐帧核对；
最大刚体位置误差分别为 `4.73e-7 m` 与 `2.37e-6 m`。
两种机器人均完成 MuJoCo → GL → RTX → MuJoCo 往返切换，暂停帧及根部位置保持一致。
这些检查证明资产与记录一致，不代表重新验证了策略的训练质量或实机行走能力。

审计日志、实际渲染图片、浏览器截图和测试结果保存在
`runs/robot-web-audit-20260914/`。

最终回归 **462 passed / 54 skipped**；本轮新增 26 项测试。
Go1/H1 共 **27 项浏览器检查**通过，未发现 JavaScript 异常，覆盖播放、暂停、
单帧、跳转、相机跟随、选择记录、渲染器切换、末帧保持、窄屏布局和停止。
独立 wheel 安装后从 `/tmp` 启动回放，成功生成 Go1 JPEG、服务页面并正常退出。
