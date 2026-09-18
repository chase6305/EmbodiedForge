# Light Loco Parkour：Go1 实验性接入

本阶段接入固定版本 `lucidrains/light-loco-parkour` 的 **Agent PPO 损失与 GAE**，使用本仓库 Go1 环境、镜像策略网络、观测归一化、动作缩放和 checkpoint 格式。物理与学习都在 CPU 上执行。

这不是完整跑酷论文复现：尚未接入上游 Actor/Critic 网络、多技能教师、感知编码、运动先验或学生蒸馏，也没有新增跑酷地形。先提供可验证的学习器接口，便于后续扩展。默认原生 PPO 不变。

## 安装

建议独立 Python 3.12 环境，在仓库根目录运行：

```bash
conda create -n ef-light-loco python=3.12 pip -y
conda activate ef-light-loco
python -m pip install -c configs/light-loco-constraints.txt \
  -e '.[go1-native,light-loco,viz-robot]'
```

`light-loco` extra 使用 Git 安装固定 revision `963a6ec3b8b42eb29dd6b9dfed34ededd3c64c7b`，需要 Git 和网络。训练启动会核对实际加载的上游 Python 文件哈希，错误版本会拒绝运行。Go1 使用 MuJoCo 3.11.0、mjbatch 0.1.0、Torch 2.9.0 和 Menagerie 2026.9.0。首次机器人资产下载需要网络。

也可安装本地固定源码，安装前核对其 revision 和工作区：

```bash
git -C /home/ubuntu/workspace/3rdparty/light-loco-parkour rev-parse HEAD
git -C /home/ubuntu/workspace/3rdparty/light-loco-parkour status --short
python -m pip install -c configs/light-loco-constraints.txt -e '.[go1-native,viz-robot]' \
  /home/ubuntu/workspace/3rdparty/light-loco-parkour
```

若本机已有项目验证用环境，可在仓库根目录用 `.cache/light-loco-venv/bin/python` 替代下方 `python`。该本地验证环境复用了已有 SDK 包路径，不是可搬移的独立发行环境；重新部署请使用上面的安装流程。

## 训练：默认 headless

```bash
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --go1-learner light-loco --headless \
  --num-envs 128 --horizon 24 --updates 1000 --threads 4 \
  --timeout 7200 --output runs/my-go1-light-loco
```

省略 `--headless` 也默认无画面，不创建渲染器或 Web 服务。输出目录必须尚不存在。预算是体验示例，不保证行走质量或收敛。`--go1-learner native` 使用原有 PPO；新训练省略该参数也默认 native。Light Loco 训练要求 `--standalone`，不修改固定 mjbatch SDK。

## 训练期间显示机器人

使用不同输出目录，或在上面的命令中将 `--headless` 换成 `--no-headless`：

```bash
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --go1-learner light-loco --no-headless --viewer-port 8083 --viewer-fps 10 \
  --num-envs 128 --horizon 24 --updates 1000 --threads 4 \
  --timeout 7200 --output runs/my-go1-light-loco-preview
```

打开 `http://127.0.0.1:8083`。显示环境 0 的训练采样，包括探索动作和重置；相机跟随机器人。学习器更新期间画面会暂时不动。默认最多 10 FPS，可设 1–30 FPS。渲染和状态复制有额外开销，不保证训练与 headless 相同速度。

`--headless` / `--no-headless` 同时出现时以后者在命令行中的最后一次设置为准。关闭浏览器不停止训练，训练结束后预览服务退出。预览需要可用的 EGL/OpenGL 驱动；端口冲突或渲染错误会使此次运行失败。该预览也适用于原生 Go1 PPO 与 Wuji 训练。

远程访问，在自己的电脑执行：

```bash
ssh -N -L 8083:127.0.0.1:8083 用户名@服务器地址
```

## 查看进度

另开终端：

```bash
python -m embodiedforge recipes status --run runs/my-go1-light-loco
tail -f runs/my-go1-light-loco/metrics.jsonl
```

指标包含奖励分项、跌倒比例、KL、动作标准差和梯度范数。目前 Go1 的指标为 JSONL，未增加 TensorBoard 日志。Wuji TensorBoard 见其独立训练文档。

## 续训

```bash
python -m embodiedforge recipes train --task go1-joystick --standalone \
  --resume-run runs/my-go1-light-loco --headless \
  --num-envs 128 --horizon 24 --updates 1000 --threads 4 \
  --timeout 7200 --output runs/my-go1-light-loco-resumed
```

自动继承 `light-loco` 后端，`--updates` 是追加更新数。显式指定不同后端会被拒绝，避免把实验比较和续训混在一起。保留完整运行目录；Go1 已提交保存点的中断恢复规则仍适用。

## 评估与在线控制

```bash
python -m embodiedforge recipes evaluate --task go1-joystick --standalone \
  --run runs/my-go1-light-loco-resumed --velocity 0.5 0 0 \
  --num-envs 8 --steps 500 --record-motion --timeout 1200 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/my-go1-light-loco-eval
python -m embodiedforge live --run runs/my-go1-light-loco-resumed \
  --render-backend mujoco --port 8080
```

打开 `http://127.0.0.1:8080`，先点击“前进”再“继续”。评估门槛为示例；失败保留报告并返回 rejected。模型沿用原生 Go1 网络，因此评估和在线推理不需要导入 Light Loco 库，但仍需兼容 Go1 的依赖和完整训练快照。

## 实现与验收边界

- `light-loco-source.json` 记录上游 revision、版本、实际路径和源码哈希。
- `training-runtime.json` 标明实际使用 `go1_light_loco.update`，并记录关键算法依赖版本。
- 运行记录与 checkpoint 包含 `learner_backend`；旧 checkpoint 缺省视为 native。
- 保留原有超时 bootstrap 修正和回合边界；GAE 转换为上游要求的 `[environment, time]`。
- 适配器把原生策略传给上游 Agent；使用高斯动作分布，关闭上游再次标准化优势，避免重复标准化。
- 上游完整高斯熵与旧实现仅保留 log-std 部分存在常数差，因此比较参数梯度而非强制损失数值相等。
- 已验证 GAE 数值、梯度、实际权重更新、短训/续训/评估/在线控制兼容；短训不构成行为成功证明。
- 2026-09-16 的 30 轮 Go1 Light Loco 训练预览短测完成，HTTP 读取到 9 个不同采样步的 640×480 JPEG 帧，图像内容随训练变化。EGL 发出驱动警告但实际出图成功；不代表使用了 NVIDIA 渲染。

本地源码评估及后续多技能计划见 [接入评估](light-loco-parkour-assessment.md)。
