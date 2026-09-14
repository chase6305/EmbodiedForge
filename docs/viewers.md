# 可视化后端

可视化与物理、训练相机独立选择：`--physics numpy|mujoco|newton|mjbatch` 决定积分器，
`--viewer web|viser|gl|rtx` 决定交互入口。默认使用统一 Web 页面，
`--render-backend raster|mujoco|gl|rtx` 选择生成浏览器画面的后端。

| 查看器 | 当前能力 | 环境要求 |
| --- | --- | --- |
| web（默认） | 统一页面、真实渲染帧、相同控制按钮、运行时切换渲染器 | `viz-web`，另加对应渲染器依赖 |
| viser | 旧版浏览器场景、RGB/状态、暂停/单步/重置 | Python 3.10+，`viz` |
| gl | Newton ViewerGL 原生窗口、可选 ImGui 面板 | `viz-gl`，面板另加 `viz-ui` |
| rtx | Newton ViewerRTX → OVRTX 原生窗口、可选 ImGui 面板 | RTX GPU/驱动，`viz-rtx`，面板另加 `viz-ui` |

实时点任务继续消费公共 `SceneUpdate`；机器人查看器支持 Go1/H1 真实网格记录回放，
以及 Go1 checkpoint 的在线策略执行。它们共用 Web 页面和 MuJoCo/GL/OVRTX 帧适配器。
详见 [机器人回放](robot-web-replay.md) 与 [Go1 在线控制](go1-live-web.md)。
H1 在线策略、通用训练 recipe 管理与训练相机观测仍未接入。

## 统一 Web 页面

### 场景背景与光照

点任务、Go1 在线控制和 Go1/H1 回放使用统一的灰蓝色配色。MuJoCo 提供天空背景、雾化远景和相机补光；GL/RTX 提供哑光网格地板，RTX 使用环境光、柔和主光与反向补光；补光不投射第二道阴影，保留清楚的接地参照。Raster 使用低对比度二维网格。原生 Newton GL/RTX 点任务窗口也使用相同地板和光照配置。

机器人地板纹理使用世界坐标；相机跟随时只按完整纹理周期移动地板显示范围，避免地板随机器人滑动。显示用材质和光照只写入渲染器持有的模型，保留机器人材质颜色，不修改训练模型、碰撞参数或相机观测。三维后端的色彩映射、阴影和抗锯齿实现不同，因此风格一致但画面不逐像素相同。

地板采用低对比度砖面和细接缝，减少大块棋盘对画面的干扰。点任务与较小机器人使用 0.5 m 单元格，较大机器人使用 1 m 单元格；机器人按渲染模型的 `stat.extent < 1.2` 选择小网格。MuJoCo 与 GL/RTX 使用对应的纹理比例，地板跟随范围也按同一完整纹理周期移动。已有 XML/MJB 天空会重新生成方向正确的渐变立方体贴图，避免上游天空名称不同导致背景变成纯色。Raster 的固定正交画面仅提供简化网格。

此样式目前不覆盖旧版 Viser、Microduck/Wuji 等上游 SDK 自带查看器。光照和地板当前使用内置默认配置，尚未提供网页调节控件。

```bash
conda activate ef-viewer
python -m pip install -e '.[viz-web,mujoco]'
# 同一页面，分别使用 MuJoCo 和 OVRTX 实际生成画面
python examples/visualize.py --viewer web --render-backend mujoco --physics newton
python examples/visualize.py --viewer web --render-backend rtx --physics newton
# 其他组合
python examples/visualize.py --viewer web --render-backend gl --physics mujoco
python examples/visualize.py --viewer web --render-backend raster --physics numpy
# 仅检查安装元数据，不启动图形 SDK
python examples/visualize.py --viewer web --render-backend rtx --check
```

打开 `http://127.0.0.1:8080`。MuJoCo 与 OVRTX 的画面都由服务端各自的真实渲染器生成，
通过 JPEG/MJPEG 传到浏览器。浏览器只负责展示和交互，不复制一套简化 3D 场景。
浏览器界面不需要 ImGui、Node 或外部 CDN；RTX/GL 仍需安装自身 SDK。

| 控件 | 行为 |
| --- | --- |
| 暂停 / 继续 / 单步 | 控制整个 VectorEnv；每次单步请求推进一次控制步 |
| 环境选择 / 重置选中 | 切换展示行；重置只影响请求时指定的环境 |
| 运行速度 | 调整目标控制频率，物理 dt 保持不变；实际速度受渲染耗时约束 |
| 渲染器选择 | 保留物理状态并切换画面来源，创建或首帧失败则保留原渲染器并显示错误 |
| 拖动 / 滚轮 / 相机按钮 | 旋转、缩放、默认视角、俯视；Raster 固定为正交画面 |
| 下载当前画面 | 获取服务端最新 JPEG 帧 |
| 停止会话 | 结束服务并释放物理与渲染资源，影响所有连接的客户端 |

快捷键：Space 暂停/继续、N 单步、R 重置选中、F 默认视角。
`--width/--height` 设置 Web 图形后端的像素尺寸（各 64..1920，默认 960×540）；
Raster 沿用正方形调试相机，边长取二者较小值。`--fps` 设置画面更新上限，默认 30。

Web 的“相机与画面”面板可直接调整渲染分辨率和目标帧率。分辨率预设为
640×360、960×540、1280×720、1920×1080；命令行指定的其他尺寸也会显示。
目标帧率范围为 1–60 FPS，常用值通过下拉框选择。界面分别显示目标值、实际出帧率、
实际尺寸、单帧 JPEG 大小和耗时。实际出帧率按最近约一秒的服务器帧输出统计，
不是浏览器屏幕刷新率；暂停且画面未变时仍显示“静止”。

调整帧率只改变画面调度，不重建渲染器。调整分辨率时先创建并验证新尺寸的首帧，
成功后替换旧渲染器，失败则保留旧尺寸和画面并显示错误。相机、策略、当前回合和
仿真状态保留，图形初始化期间物理短暂等待。Raster 仍输出正方形，面板显示实际尺寸。

帧间隔从本帧开始计算，渲染耗时计入帧预算；不再在渲染结束后额外等待整个帧间隔。
Web 的物理步进与画面更新各自计时，短暂渲染延迟不会不断推迟后续物理步进。
严重超时跳过过期的墙钟调度，不累积突发补帧或补步，物理 dt 保持固定。
控制操作可以触发提前发布状态和画面。高分辨率、高 FPS 仍可能受同步渲染和编码耗时
限制；目标 FPS 不保证每帧都有新物理状态，Go1 在 1× 速度下最多生成 50 个控制状态/秒。

HTTP 控制支持 `{"action":"fps","value":30}` 和
`{"action":"resolution","width":1280,"height":720}`，沿用会话 token 和控制回执。
分辨率各维必须为 64–1920 的整数，帧率必须为 1–60 的整数；非法请求在入队前拒绝。
状态新增 `target_fps`、`render_width`、`render_height`、`frame_ms`、`jpeg_bytes`；
其中 `width`/`height` 仍是实际输出尺寸，`render_width`/`render_height` 是配置尺寸。
Web 本身不创建原生窗口，无需 `--headless`；MuJoCo 在 Linux 默认使用 EGL，保留显式
`MUJOCO_GL` 配置。Newton GL 的隐藏窗口仍需要可用的 OpenGL/显示上下文。

### 框架与扩展点

```text
VectorEnv / 公共 SceneUpdate
          │
          ├── WebViewer：控制队列、状态、JPEG/MJPEG HTTP
          │       └── 统一浏览器页面
          │
          └── FrameRenderer：render(snapshot, observation, env_id, camera) → RGB
                    ├── Raster（CPU，主进程）
                    ├── MuJoCo Renderer（独立进程）
                    ├── Newton ViewerGL（独立进程）
                    └── Newton ViewerRTX → OVRTX（独立进程）
```

HTTP 线程只校验并排队命令、发送已发布的不可变帧；主线程消费命令、推进物理和调度渲染。
各图形 SDK 的构造、渲染、销毁都在自身进程的主线程执行。通过 `spawn` 创建进程，
避免继承已有 GPU 上下文；实际测试发现同进程往返切换会触发 MuJoCo `EGL_BAD_ACCESS`，
因此隔离属于渲染器生命周期设计。进程间只传公开快照和 RGB 数组，不共享物理私有模型。

切换时先创建候选渲染器并验证其首帧，再替换和关闭旧渲染器。初始化期间物理推进等待，
HTTP 服务仍可响应；图形初始化/帧等待上限 180 秒，退出时对子进程有有界清理。
Web RTX 使用同步渲染，避免异步流水线显示前一仿真状态。
传输只保留最新帧，慢客户端跳过旧帧，控制队列最多 128 条。默认绑定回环地址，
控制请求带会话 token；多浏览器共享状态。远程使用 SSH 转发即可。

暂停时，如果选中环境的状态与相机没有变化，复用已发布的 JPEG，跳过图形 SDK 调用、
编码及重复帧传输。界面的指令、奖励、播放速度等状态仍持续更新，画面更新指标显示
“静止”。相机移动、环境切换、重置、回放跳转及关节姿态变化会重新渲染；运行中继续
正常渲染。这项优化保留图形进程和模型，恢复运行无需重新初始化。

`/api/state` 中的 `state_id` 标识状态发布次数，`frame_id` 只在生成新画面时递增；
`render_idle` 表示当前更新复用了画面。MJPEG 每张 JPEG 后立即带上下一部分的分隔符和
完整 `Content-Type` 头，新连接在暂停状态下也能显示首帧，无需等待下一次仿真推进。
各部分通过 MIME 分隔符划界，不再包含长度及帧号头；需要帧号的客户端使用
`/api/frame` 或 `/snapshot.jpg` 的 `X-Frame-Id` 响应头。

“停止会话”会取消仍在队列中的控制命令，即使队列已满也可接收。停止仍由主线程执行，
已经开始的 SDK 调用需要返回或超时后才能清理，不能中断正在执行的图形初始化。

Web 控制响应中的 `control_id` 是本会话单调递增的收件编号。发布状态里的同名字段
表示主线程已消费到的控制编号；浏览器以该状态为准结束操作等待，不把 HTTP 202
当作操作已经生效。连续控制可能合并或被后续请求覆盖，编号不保证每条中间状态都显示；
例如同批两次环境选择只显示最后一次。帧传输与状态发布独立，暂停时也能收到回执。

页面提供全屏、明确的操作目标与等待提示；Go1 在线输入支持按环境保留草稿、范围校验和
键盘指令，详见 [在线交互](go1-live-web.md)。断线时禁用操作、标明最后一帧并自动重连，
恢复后读取实际状态，不重发旧操作。移动端窄屏布局保留全部按钮；HTML 与 JavaScript
随 Python 包一起分发，无需前端构建服务或外部 CDN。

机器人回放已通过独立 MJCF 适配器支持命名关节、网格、常见基本几何及相机跟随；
MuJoCo 保留模型材质，GL/RTX 桥接几何与基础颜色，不转换纹理和完整着色器。
当前不是 USD/MJCF/URDF 的通用转换器，也没有物体拖拽、施力或训练任务启停控制。
MJPEG 适合本地调试；高分辨率、多用户和低延迟远程场景可后续增加视频编码传输。

EGL 警告与实际 GPU 选择可通过 `python -m embodiedforge doctor --graphics egl` 检查；该命令实际渲染测试帧，区别于元数据 `--check`。驱动注册文件缺失的定位与修复见 [图形设备诊断](graphics-diagnostics.md)。

## 安装和运行

本机已配置 `ef-viewer`（Python 3.11.16），可直接运行：

```bash
conda activate ef-viewer
python examples/visualize.py --viewer rtx --physics newton
python examples/visualize.py --viewer gl --physics numpy
```

重新创建完整原生查看器环境时，在仓库根目录执行：

```bash
conda create -n ef-viewer python=3.11 pip
conda activate ef-viewer
python -m pip install --only-binary=imgui-bundle --extra-index-url https://pypi.nvidia.com -c configs/viewer-constraints.txt -e '.[viz-rtx,viz-ui]'
```

`configs/viewer-constraints.txt` 固定实测版本；Newton 源码固定为
`pyproject.toml` 中的 v1.6.0rc1 ZIP。ImGui 明确使用 wheel，避免意外转入源码编译；
`viz-ui` 同时声明 PyOpenGL。OVRTX 包较大，NVIDIA 官方索引提供平台 wheel。

[OVRTX 官方支持 Python 3.11–3.13](https://github.com/NVIDIA-Omniverse/ovrtx#system-requirements)，
[ImGui 的 wheel 构建从 Python 3.11 起](https://github.com/pthom/imgui_bundle/blob/main/pyproject.toml)。
原 `ef` 的 Python 3.10 虽然跑通过 RTX 点场景，但不能作为完整原生查看器的标准环境；
保留已有训练环境，使用独立查看器环境。

浏览器查看器单独安装 `python -m pip install -e '.[viz]'` 后运行：

```bash
python examples/visualize.py --viewer viser --physics numpy --port 8080
```

打开 `http://127.0.0.1:8080`。远程浏览器可通过 SSH 转发访问；原生窗口需要显示服务。

## 启动与计时

可视化入口默认输出 INFO 生命周期日志：准备查看器、生成首帧、就绪耗时、停止。
应用日志写入 stderr；`--log-json` 切换应用日志格式，`--log-level WARNING` 可减少输出。
`EMBODIEDFORGE_DEBUG` 与 JSON 环境开关仍有效，显式 CLI 日志选项优先。
第三方 SDK 的原生日志不受 Python logger 格式控制，详见 [日志设计](logging.md)。

原生 RTX 在首次 `update()` 内同步取得并展示颜色帧，再进入异步流水线；Web RTX 全程同步。
只有初始状态成功提交、原生首帧完成后才报告 `Viewer ready`。
Viser 的就绪表示服务器已发布初始状态，不表示浏览器客户端已经完成绘制。
首次 OVRTX 着色器编译可能较慢；初始化提示会先于就绪日志出现。

```bash
# 首帧完成后运行 5 秒，初始化耗时不占用 duration
python examples/visualize.py --viewer rtx --physics newton --duration 5
# 无窗口模式仍实际渲染，并检查颜色输出
python examples/visualize.py --viewer rtx --physics newton --headless --duration 5
```

默认 `--duration 0` 持续运行；关闭窗口、Ctrl+C 或 duration 到期退出并释放资源。
`--fps` 控制画面提交频率；物理使用自身 control_hz。原生窗口每个控制周期处理
输入事件，暂停、单步和关闭不再等待下一个画面提交时刻；同步渲染本身较慢时，
主线程仍会等待渲染完成。
原生查看器通过 `--env-id` 固定选择环境，Web/Viser 可在浏览器中切换。
暂停/单步作用于整个 VectorEnv，重置只作用于选中环境。

## 依赖与系统诊断

```bash
python examples/visualize.py --viewer rtx --check
python -m pip check
```

`--check` 不加载 SDK、不初始化 GPU、不创建窗口。`metadata_ok` 只表示基础依赖
存在且 Newton 精确版本匹配；`python`、`environment_notes`、`optional_packages`
报告 Python 支持范围和面板缺失。它不验证所有版本约束、驱动或实际渲染，
不能把该结果当作 GPU 验收。运行时缺少基础包会在物理初始化前给出安装命令。

本机曾缺少 `libnvidia-compute-595` 包中的 CUDA 库链接和其他文件，表现为
NVTT 找不到 `libcuda.so`，而 Warp 仍能通过 `libcuda.so.1` 使用 GPU。
正式修复使用包管理器恢复同版本用户态驱动包，不手工添加系统链接，
不把 CUDA Toolkit 的 stub 库放入运行时搜索路径。

用户重装后，普通库文件已恢复；被删除的 OpenCL 配置属于 conffile，普通重装会
保留其删除状态，需要 `Dpkg::Options::=--force-confmiss` 恢复。本机最新复核中：

```bash
dpkg --verify libnvidia-compute-595:amd64 libnvidia-compute-595:i386
```

已无输出，包完整性校验通过。上述包名对应本机驱动 595.84，其他机器应先核对
自身包版本。项目不自动修改驱动、系统动态库或库搜索路径。

Hub 是可选远程缓存服务，本机未安装时上游会输出 `Hub not found`。
是否启用由应用环境决定，项目不自动覆盖 `OMNICLIENT_HUB_MODE`，参见
[NVIDIA Client Library 配置](https://docs.omniverse.nvidia.com/kit/docs/client_library/2.60.1/index.html)。
Warp 设备/内核信息和 OVRTX Client ID 属于启动信息，隐藏日志不作为修复标准。

## 适配与验证边界

`ViewerBackend` 提供 `is_running/should_step/consume_reset/update/close`。
主线程消费控制请求、推进仿真、提交快照；适配器不调用环境 step/reset。
`should_step` 每个控制 tick 最多消费一次单步请求。

两个原生入口严格核对 Newton 1.6.0rc1。场景和颜色均使用 Warp CPU 缓冲区；
GL 关闭 CUDA/OpenGL 互操作，RTX 的光线追踪与窗口展示仍使用 GPU。
GL/RTX 使用 NullRenderer，不为训练生成额外 NumPy 相机帧，窗口截图不是训练观测。

OVRTX 0.5 返回 `/Render/Vars/LdrColor`，Newton 1.6.0rc1 期望 `LdrColor`。
本项目局部适配显示与截图中的名称，并在窗口/无窗口路径检查空帧及缺少颜色输出。
一次渲染可能包含插值帧和最终帧，显示与截图统一选择指定 render product 的最后
一帧，每次 update 只刷新一次窗口。截图复制像素，避免底层缓冲区复用改变结果。
显示异常也释放已映射的颜色资源；清理失败单独记录，保留原始渲染异常。
该适配使用固定版本的内部钩子，升级 Newton
时需重新验收，不修改已安装 SDK。

当前没有通用网格、关节树、材质或相机资产桥接，原生面板中的 picking/wind
也未连接物理力反馈。扩展真实机器人需另外实现资产和控制接口。

2026-09-11 验证：Python 3.11.16、RTX 5090 D v2、驱动 595.84；
GL 窗口与 RTX 窗口均完成运行验证。RTX 首次 update 后截图非空，连续 15 帧、
ImGui 面板、程序化暂停/单步/重置检查和正常退出通过；这不等于鼠标操作了每个控件。
首次新环境初始化曾触及超时，缓存后的复测正常。依赖版本见约束文件。

最近一轮真实 RTX 测试：15 次 update 收到 30 张帧图像，实际窗口刷新次数降为
15 次；这是减少重复提交，不代表 GPU 渲染吞吐翻倍。回归覆盖多帧选择、截图
缓冲区所有权、清理异常保留以及不渲染时的输入处理。

### 统一 Web 验证（2026-09-14）

本机 Python 3.11.16 / RTX 5090 D v2 验证了 Raster、MuJoCo 3.11.0、
Newton GL 1.6.0rc1 和 OVRTX 0.5 的真实帧。Newton 物理会话中八次渲染器选择
（含往返切换）保持暂停时的 episode、step、时间、位置不变；MuJoCo 物理也完成
同一浏览器控制流程。共享相机的三个方位与 MuJoCo 实际相机位置数值一致。

回归测试 **436 passed / 54 skipped**（包含 23 个新增 Web/IPC 测试）；
跳过项主要是此环境未安装的训练、Viser、mjbatch SDK，不代表这些组合已验收。
独立 wheel 安装确认包含页面并可从仓库外启动 HTTP 服务；Ruff 和 `pip check` 通过。
真实浏览器报告、画面与运行日志保存在 `runs/web-viewer-audit-20260914/`。

### 分辨率与帧率交互验证（2026-09-14）

本机 RTX 5090 D v2、960×540、目标 30 FPS，在同一个 Go1 模型及相机下，
分别运行原先的“渲染结束再等待”与新的帧开始调度。每组采样约 4 秒，包含渲染进程
传输和 JPEG 编码，不包含物理或浏览器；实际出帧率如下：

| 渲染器 | 原调度 | 新调度 |
| --- | ---: | ---: |
| MuJoCo | 19.5 | 29.9 |
| OpenGL | 24.5 | 30.0 |
| OVRTX | 20.7 | 30.0 |

另外在两个 Go1 在线环境中、960×540 下分别测量 15/30/60 FPS 目标。
三个后端在 60 FPS 目标下均约 59.4 FPS，物理推进约 50 步/秒；每个组合采样约 2 秒，
结果属于本机短时测试，不保证其他模型、分辨率或负载下达到相同吞吐。
H1 回放在 1280×720、60 FPS 下约 59 FPS，回放时间/墙钟时间为 0.99。

496 项 Python 回归通过、55 项跳过；22 项真实浏览器检查包含各后端的四档分辨率、
运行中调整、暂停姿态保留、非法值拒绝和单步。另验证 H1 回放单步、Raster 的实际
正方形尺寸，以及 1 FPS 下控制回执仍及时发布。日志、截图和基准脚本位于
`runs/web-display-audit-20260914/`。
