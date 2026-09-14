# 图形设备与 EGL 诊断

[训练与部署](training-deployment.md) · [查看器](viewers.md)

出现 `libEGL warning: egl: failed to create dri2 screen` 时，先确认实际使用的渲染器和设备。RTX 首帧就绪不等于之后创建的 EGL 上下文也使用 NVIDIA：在统一页面切换到 MuJoCo 时，会创建另一套图形上下文。

## 两类检查

只核对查看器包的安装元数据，不创建图形上下文：

```bash
python -m embodiedforge.visualization --viewer web --render-backend mujoco --check
```

实际创建 EGL 上下文并渲染一张 64×64 测试帧：

```bash
python -m embodiedforge doctor --graphics egl
```

后一条命令需要当前 Python 环境安装 MuJoCo（例如 `pip install -e '.[viz-robot]'`）。诊断在独立子进程中运行，最长等待 30 秒；子进程失败、驱动崩溃或超时均返回结构化错误，不向调用进程导入图形 SDK。只在诊断子进程选择 EGL，不改系统文件或当前 shell 的配置。

查看输出中的 `graphics`：

| 字段 | 含义 |
| --- | --- |
| `ok` | 上下文与非空测试帧是否成功；不表示选中了 NVIDIA |
| `vendor` / `renderer` | 实际 OpenGL 厂商与设备，如 `NVIDIA Corporation` / `RTX 5090`，或 AMD 核显 |
| `gl_version` | 实际 OpenGL/驱动版本字符串 |
| `image_shape` / `image_range` | 测试帧尺寸及像素范围 |
| `vendor_files` | 当前 GLVND 搜索路径中发现的厂商注册 JSON 与库名 |
| `environment` | 会影响 EGL 选择的相关环境变量；没有输出其他环境信息 |
| `warnings` | 例如 NVIDIA EGL 库存在，但所选搜索路径缺少其注册文件 |
| `stderr` | 原始驱动警告，保留最后 8 KiB，未屏蔽 |

退出码 0 表示依赖元数据与 EGL 渲染测试成功；退出码 1 表示其中至少一项失败。成功时仍需查看设备和警告。在混合显卡机器上，AMD 渲染可以成功，但可能不是你期望使用的设备。这个检查不验证 OVRTX/Vulkan、CUDA 策略计算或完整机器人场景。

## 本机问题：NVIDIA EGL 注册文件缺失

2026-09-14 在驱动 `595.84` 的机器上发现：

- `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 缺失，目录仅有 Mesa 注册文件。
- 默认 EGL 实际选择 AMD `RAPHAEL_MENDOCINO` 核显，并输出两条 DRI2 警告。
- `dpkg -V libnvidia-gl-595:amd64` 还报告多个缺失或内容不匹配的驱动包文件。
- 在独立诊断进程临时提供 NVIDIA 注册信息后，同一测试选择 RTX 5090 D v2，且没有该警告；没有将临时配置写入系统或项目运行入口。

GLVND 通过厂商 JSON 发现 EGL 实现，搜索还会受到 `__EGL_VENDOR_LIBRARY_FILENAMES` / `__EGL_VENDOR_LIBRARY_DIRS` 影响。详见 [NVIDIA GLVND 的 ICD 搜索说明](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)。

先核对本机包版本和完整性：

```bash
dpkg-query -W libnvidia-gl-595:amd64
dpkg -V libnvidia-gl-595:amd64
apt-get --simulate --reinstall install libnvidia-gl-595:amd64=595.84-0ubuntu0.22.04.1
```

下面的重装命令仅适用于上面已经确认的 **595.84 / Ubuntu 22.04 包版本**，其他机器应使用实际已安装的驱动分支与版本：

```bash
sudo apt-get --reinstall install -y libnvidia-gl-595:amd64=595.84-0ubuntu0.22.04.1
```

本机 apt 模拟结果为重装同版本 amd64/i386 两个包，没有升级或卸载其他包。系统写入需要管理员密码。完成后重新启动查看器进程，再执行 `doctor --graphics egl`，核对实际设备、文件完整性与剩余警告。包修复恢复厂商注册；混合显卡的默认设备选择还需以实测报告为准。

不要把屏蔽 stderr、关闭 EGL 日志或向项目永久写入临时 vendor JSON 当作驱动包修复。当前项目不会自动改写系统驱动注册文件。
