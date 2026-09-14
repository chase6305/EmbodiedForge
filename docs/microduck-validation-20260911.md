# Microduck 初期接入验证（2026-09-11）

本文保留初期 smoke、续训和评估功能的历史记录。当前用法见 [Microduck RL](microduck.md)。下文测试数量和模型质量描述对应各次历史运行。


RTX 5090 D v2、驱动 595.84、Python 3.12.14 下已完成 `runs/microduck-smoke-20260911-verified`：64 环境、5 次 PPO 更新、7,680 个环境步；生成 `model_4.pt` 和 `policy.onnx`。所有 TensorBoard 标量有限，`nan_state` 终止为零；ONNX 的 61 → 14 维接口与两组 CPU 推理输入通过验证。缓存预热后的完整 smoke 约 20 秒，不能据此推断大规模训练吞吐或收敛时间。

首次接入时上游 CPU 测试：199 passed、1 skipped；EmbodiedForge 测试：137 passed、9 skipped（可选依赖）。已补充真实 TensorBoard 事件的续训编号、不完整迭代、非有限指标和 NaN 状态回归测试，以及 checkpoint 学习率和散列验证。隔离启动修复了辅助脚本目录中 `logging.py` 遮蔽标准库的问题。

后续续训验证位于 `runs/microduck-resume-20260911-verified`：从短训模型追加 2 次更新。课程计数从 120 增至 168，Adam 更新计数从 100 增至 140，验证通过。

续训阶段验证：Microduck 专项测试 19 passed（包含真实 TensorBoard 事件和 PyTorch checkpoint）；`ef-viewer` 环境全量测试 140 passed、17 skipped，其中跳过项依赖该环境未安装的可选训练/指标包。

`runs/microduck-evaluate-20260911/evaluation.json` 记录了短训模型的 16 环境 × 250 步评估：4,000 个环境步、54 次摔倒终止、NaN 终止为零；完成 episode 平均约 59.09 步（1.18 秒）。该模型尚未形成稳定步态。未进行完整步态训练或实机部署。

回放退出问题已定位并修复：原先的 60 步复现程序在退出阶段出现过段错误（139）；等待渲染线程后，同一复现和正式原生入口均正常退出（0）。正式入口的 Ctrl+C 路径也已验证，退出码 130，未再出现 GLFW `library is not initialized` 警告。本机 `:1` 显示服务仍打印 Xlib `NV-GLX` 扩展缺失提示，该提示尚未定位；尚未完成图像级验收。

共用环境加载逻辑后的复核记录位于 `runs/microduck-evaluate-lifecycle-20260911/evaluation.json`：4,000 步、57 次摔倒、NaN 终止为零。单次短时评估不能据此判断策略优劣；完整训练和跨种子步态评估仍未完成。

退出修复后的验证：Microduck 专项测试 25 passed；`ef-viewer` 全量测试 146 passed、17 skipped。新增测试覆盖异步线程清理、失败启动清理、无关线程隔离、真实 SIGINT 子进程清理、退出码保留和 Viser 入口兼容性。Ruff 与 `git diff --check` 通过。

固定指令与导出对照验证位于 `runs/microduck-fixed-parity-fp32-20260911/evaluation.json`：16 环境 × 250 步，固定前进 0.2 m/s、无推扰；251 次全环境指令检查通过，覆盖 38 次摔倒自动重置，NaN 为零。250 份实际观测上的 ONNX/PyTorch 动作最大绝对误差为 `1.1920928955078125e-7`。最初沿用上游 TF32 时曾在第 112 步超出容差；对照模式改用完整 FP32 后通过，未放宽容差。短训模型仍未形成稳定步态。

错配检查位于 `runs/microduck-parity-mismatch-20260911/evaluation.failure.json`：用追加训练后的 `model_5.pt` 对照旧 `policy.onnx`，在第 0 步检出约 `0.10175` 的动作误差，退出码 1，运行状态正确保存为失败。该 ONNX 接口与输出均有效，证明检查不止验证维度和有限值。

本轮 Microduck 专项测试 36 passed；`ef-viewer` 全量测试 156 passed、18 skipped（可选依赖），Ruff 与 `git diff --check` 通过。

另以种子 1 验证固定倒退/横移/转向组合 `(-0.2, 0.1, -0.5)`，保留推扰、不启用 ONNX 对照：`runs/microduck-fixed-turning-20260911/evaluation.json` 完成 8 环境 × 250 步，24 次自动重置，251 次指令检查通过，NaN 为零。

重置前步态指标验证位于 `runs/microduck-tracking-metrics-20260911/evaluation.json`：固定前进 0.2 m/s、无推扰、16 环境 × 250 步，包含全部 38 个终止步在内的 4,000 份速度样本。平均前向速度约 0.06672 m/s，平面速度 RMSE 约 0.18871 m/s，yaw rate RMSE 约 0.17202 rad/s。首次 episode 的平均观察时长为 2.11375 秒，其中 3/16 个到达 5 秒评估结束而未终止；这 3 个时长是下界，不能据此推断完整寿命。ONNX 对照 250 次通过，最大动作误差约 `9.69e-8`；NaN 为零。

默认随机指令与推扰路径也已验证：`runs/microduck-tracking-random-20260911/evaluation.json` 完成 8 环境 × 250 步，31 次摔倒，首次 episode 均在 5 秒前终止，NaN 为零。两次评估条件不同，不构成策略优劣对比，短训模型仍需完整训练。

步态指标与首次 reset 课程修复后的验证：Microduck 专项 46 passed；`ef-viewer` 全量 157 passed、27 skipped（含未安装 PyTorch 的指标测试，这些已在专项环境运行）。回归覆盖终止步速度与旧指令、各轴单位、重复摔倒、超时截断、重叠终止、重置前 NaN、回调漏采样及首次 reset 课程计数。

多种子验证位于 `runs/microduck-seeds-20260911/summary.json`：种子 0/1/2，每个 8 环境 × 250 步，固定前进 0.2 m/s、无推扰、完整 FP32，总计 6,000 个环境步。各次平面速度 RMSE 为 0.18805、0.18772、0.18893 m/s；首次 episode 平均观察时间为 1.9100、1.5750、1.9975 秒，持续到 5 秒且未终止的数量为 1/8、0/8、0/8。合并后的 `[vx, vy, wz]` RMSE 为 `[0.18719, 0.01979, 0.18405]`，单位为 `[m/s, m/s, rad/s]`。750 次 ONNX 对照通过，最大动作误差约 `1.49e-7`，NaN 为零；短训模型尚未形成稳定步态。

多种子功能验证：Microduck 专项 70 passed；`ef-viewer` 全量 181 passed、27 skipped（可选依赖），Ruff 与 `git diff --check` 通过。新增回归覆盖种子间统计与 RMSE 合并公式、条件/模型错配、缺失及非有限报告、重复种子、模型中途替换、批次失败和 Ctrl+C 后保留已完成结果。

显式阈值验收验证位于 `runs/microduck-acceptance-20260911/acceptance.json`：两个种子各 8 环境 × 250 步全部完成，500 次 ONNX 对照通过，随后因策略未满足示例阈值而返回退出码 3。平面 RMSE 为 0.18851/0.18718 m/s（上限 0.1），yaw rate RMSE 为 0.18052/0.16081 rad/s（上限 0.1），完整 5 秒存活比例为 0.125/0（下限 0.8）。根目录状态为 `rejected`，两个子运行均为 `complete`，完整汇总与逐种子报告均保留。验收报告由原始评估数据复核，并补充了单位、环境数和步数。

验收功能验证：Microduck 专项 89 passed；`ef-viewer` 全量 200 passed、27 skipped（可选依赖）。新增回归覆盖逐种子要求、阈值边界及零阈值、非法阈值、无效存活计数、提前超时、单次/批次拒绝后保留报告、全部种子执行完成，以及 CLI 的独立退出码 3。
