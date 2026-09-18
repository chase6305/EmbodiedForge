# 灵巧手完整训练流程

Wuji 五指灵巧手持物并重定向方块。使用固定版本 Wuji/UniLab PPO 和 MuJoCo-Warp，需要 NVIDIA CUDA。标准任务为 `wuji-reorient`，Light 随机化课程为 `wuji-reorient-light`；训练、续训和评估必须保持同一任务。

## 1. 环境准备

在仓库根目录执行。已有主环境可跳过创建和安装：

```bash
conda create -n ef-viewer python=3.11 pip -y
conda activate ef-viewer
python -m pip install -e '.[viz-robot]'
nvidia-smi
```

准备 Git、uv、ffprobe（由 FFmpeg 提供）。已有匹配的训练 SDK 环境可跳过 setup：

```bash
python -m embodiedforge recipes setup --source wuji_unilab \
  --repo /path/to/wuji_unilab --python 3.12 --timeout 1200
.cache/external/wuji_unilab/.venv/bin/python -c \
  'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
ffprobe -version
```

将 `/path/to/wuji_unilab` 替换为实际源码目录；必须是干净的 `91ccfa0ec8c129b300865bd36c59dc9eed56a744` 检出。默认源码位置为 `/home/ubuntu/workspace/3rdparty/wuji_unilab`，此时可省略 `--repo`。SDK 安装在独立的 `.cache/external/wuji_unilab/.venv`。CUDA 为 False 或 `nvidia-smi` 报驱动错误时，需先修复 GPU 环境。

## 2. 短测入口

```bash
python -m embodiedforge recipes train --task wuji-reorient \
  --num-envs 32 --horizon 40 --updates 5 --timeout 600 \
  --headless --output runs/my-wuji-smoke
```

所有输出目录必须尚不存在。短测只验证训练链路，不代表学会任务。

## 3. 训练并观察动作

```bash
python -m embodiedforge recipes train --task wuji-reorient \
  --num-envs 512 --horizon 40 --updates 300 --timeout 7200 \
  --no-headless --viewer-port 8083 --viewer-fps 10 \
  --output runs/my-wuji-01
```

打开 `http://127.0.0.1:8083`。首次 rollout 后出图；首次 CUDA 编译可能需要等待。画面来自环境 0 的实际训练采样，包含探索动作和自动重置，不代表确定性评估。页面显示采样步数；学习器更新期间画面可能暂时不动。预览固定相机，显示手与方块，暂不叠加目标姿态；评估视频包含目标姿态叠加。

**默认 headless**：省略 `--no-headless` 时不创建渲染器或 HTTP 服务，也可显式使用 `--headless`。同时指定两个开关时，以最后一个为准。刷新频率范围为 1–30 FPS，限制墙钟刷新频率，不限制 PPO 步速；状态读取和同步渲染仍增加开销，长训可使用 headless。预览用 EGL 离屏渲染，无需桌面，但需要可用的 OpenGL/EGL 驱动。端口绑定或渲染失败会使此次训练失败，不会静默关闭预览。

关闭浏览器不停止训练；训练结束、失败或中断后预览服务随进程退出。目前 recipe 训练预览支持 Wuji 和 Go1。

## 4. 日志、曲线与远程访问

另开终端，在仓库目录运行：

```bash
conda activate ef-viewer
tail -f runs/my-wuji-01/console.log
```

```bash
python -m embodiedforge recipes status --run runs/my-wuji-01
.cache/external/wuji_unilab/.venv/bin/tensorboard \
  --logdir runs/my-wuji-01/logs --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006` 查看曲线。Wuji 实时学习进度以控制台和 TensorBoard 为准，`status` 主要显示托管运行状态，不是逐轮指标接口。TensorBoard 独立运行，训练结束后仍可查看已写入的曲线。

远程运行时，在自己电脑上转发两个端口，再打开上述本地地址：

```bash
ssh -N -L 8083:127.0.0.1:8083 -L 6006:127.0.0.1:6006 用户名@服务器地址
```

## 5. 评估与视频

训练成功完成后进行独立评估。下面的门槛是示例：

```bash
python -m embodiedforge recipes evaluate --task wuji-reorient \
  --run runs/my-wuji-01 --num-envs 1 --seed 0 \
  --num-trials 50 --steps 280 --timeout 1200 \
  --min-success-rate 0.5 --max-drop-rate 0.2 \
  --output runs/my-wuji-eval-01
python -m embodiedforge recipes evaluate --task wuji-reorient \
  --run runs/my-wuji-01 --num-envs 1 --seed 0 \
  --num-trials 1 --steps 280 --record-video --timeout 600 \
  --output runs/my-wuji-video-01
```

查看 `runs/my-wuji-eval-01/evaluation.json` 和 `runs/my-wuji-video-01/evaluation.mp4`。验收失败返回 `rejected` 并保留报告。视频包含目标姿态叠加。零掉落不能替代重定向成功率。

## 6. 续训

```bash
python -m embodiedforge recipes train --task wuji-reorient \
  --resume-run runs/my-wuji-01 --num-envs 512 --horizon 40 \
  --updates 1000 --timeout 14400 --headless \
  --output runs/my-wuji-02
```

`--updates` 是追加轮数。需要画面时把 `--headless` 换成 `--no-headless --viewer-port 8083 --viewer-fps 10`。Wuji 仅支持从同任务、兼容版本、状态为 `complete` 的运行续训；不能套用 Go1 的中断恢复能力。保留完整运行目录，不要只复制模型文件。

512 环境、300 轮是体验预算，不保证收敛。已有的 512 环境、累计 1000 轮实验仍未通过重定向成功率验收。上游默认规模为 8192 环境、5000 轮，不保证本机显存足够或一定达标，应依据独立评估结果调整。

相关文档：[任务配方](recipes.md) · [其他机器人训练](training-deployment.md)。
