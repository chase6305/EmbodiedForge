# Native H1 training without IsaacLab

[简体中文](h1-native.md) · [Training and deployment](training-deployment.en.md)

`h1-native` implements model construction, articulation operations, observations, rewards, resets, and PPO in this repository.
It requires MuJoCo 3.11, mjbatch 0.1.0, NumPy, and PyTorch, with no runtime imports from IsaacLab, Isaac Sim, RSL-RL, or upstream training examples.
Both simulation and PPO currently run on **CPU**. This is a separate native baseline, not an equivalent replacement for the Newton/MuJoCo-Warp GPU recipe.
The existing `h1` command retains the IsaacLab workflow; its checkpoints are incompatible with `h1-native`.

## Install, train, and resume

Run from the repository root. Supply an H1 MJCF together with its referenced meshes; assets retain their original licenses.

```bash
conda create -n ef-h1-native python=3.12 pip
conda activate ef-h1-native
python -m pip install -e '.[h1-native]'
python -m embodiedforge h1-native train --headless \
  --model /home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml \
  --num-envs 128 --threads 4 --updates 1000 \
  --output runs/h1-native-first
python -m embodiedforge h1-native train --headless \
  --resume runs/h1-native-first --num-envs 128 --threads 4 --updates 1000 \
  --output runs/h1-native-resumed
```

The existing `.cache/external/mjbatch/.venv/bin/python` also works on this machine. It selects an interpreter; the native trainer does not import mjbatch examples.
Output directories must be new. `--updates` counts additional updates, with a default rollout horizon of 24.
Training and evaluation are always headless and do not initialize a renderer. Both accept an explicit `--headless`; omitting it gives the same behavior. Seeds must be in `0..2**64-1`.
Runs contain `model.mjb`, `checkpoint.pt`, `metrics.jsonl`, and `run.json`. The compiled MJB includes meshes, so resume and replay no longer require the original asset paths. Weights and MJB models load from the same bytes used for hash validation. Resumed runs save the actual loaded model, so replacing an input file cannot change the model recorded for that run.

Training attempts to update `run.json` after a failure, Ctrl+C, or SIGTERM. If that write also fails, the log includes the file path and write error while preserving the original training exception or interruption; the saved status may still be `running`. A final record write failure after otherwise successful training is raised rather than returning success.
Checkpoints are atomically saved every 50 updates and at completion. Resume and evaluation accept recorded checkpoints from `complete`, `interrupted`, or `failed` runs, checking artifact hashes, task version, joint order, and finite tensors. Recovery starts at the last saved update; unsaved progress is lost. Runs that stopped before their first save or are still marked `running` cannot be used as inputs.
Policy and Adam state are restored; environment and RNG state restart. This is not exact continuation. New training defaults to a learning rate of `0.0003`. Resume inherits the saved learning rate unless `--learning-rate` explicitly overrides it. The default follows the [three-seed learning-rate comparison](rl-training-study-20260925.md), limited to this flat-ground task and nominal-pose evaluation.
Before restoring Adam or saving a checkpoint, the validator shared with native Go1 checks parameter groups, complete state, moment shapes/dtypes, nonnegative second moments, and integer step counts. Damaged optimizer state is rejected before creating the new run directory or simulation environment.

## Evaluate and replay

```bash
python -m embodiedforge h1-native evaluate \
  --run runs/h1-native-resumed --steps 500 --velocity 0.5 0 0 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --record-motion --output runs/h1-native-eval
conda activate ef-viewer
python -m embodiedforge replay \
  --model runs/h1-native-resumed/model.mjb \
  --motion runs/h1-native-eval/motion.npz \
  --render-backend rtx --port 8081
```

Control dt is 0.02 seconds; 500 steps cover 10 seconds. Reports include survival, planar/yaw velocity RMSE, mean velocity, return, and peak applied joint torque.
Only each row's first trial is counted, including its terminal frame. Failed thresholds produce a report and exit code 2. With no thresholds, `accepted=null`.
Evaluation currently uses the deterministic nominal pose with training noise/randomization disabled. Its default is one environment; larger batches repeat identical initial states and are not independent robustness trials.
Motion recording captures row 0's first trial, capped at 10000 steps. Replay also supports `mujoco` and `gl`. The existing `live` command still supports Go1 only.

## Articulation interface

| Capability | API and contract |
| --- | --- |
| Model | `build_model(path)` compiles H1 MJCF; `ArticulationBatch` holds independent rows with shared topology |
| Joint groups | `robot.group(*patterns)` uses full regex matches in stable actuator order; missing matches fail |
| Position commands | `robot.set_joint_position_targets(values, env_ids=..., joint_ids=...)`; unit-gear position actuators required |
| Velocity commands | `env.set_commands(values, env_ids=...)`, shape `(selected_rows, 3)` for forward/lateral/yaw; fixed commands survive reset |
| Randomization | `robot.randomize(...)` changes per-row sliding friction and selected body mass/inertia, then recomputes derived constants from nominal parameters |
| Reset | `robot.reset(ids, qpos=..., qvel=...)` and task-level `env.reset(ids)` affect selected rows only; raw MuJoCo quaternions use wxyz |
| Forces and state | `robot.snapshot()` returns owned arrays for joint position/velocity, targets, body poses, actuator forces, and generalized actuator joint torques |

Forces represent the **last integration substep** of the last control step and survive subsequent FK refreshes. They are not control-step averages or the sum of passive/external forces.
H1 has 19 joints and 69 observations, with legs/feet/arms groups plus hip/torso reward groups. Standing poses are written into simulation state; the asset's `qpos0` kinematic reference is preserved.

## Differences from the IsaacLab recipe

Configuration and reward concepts reference local IsaacLab revision `2e44ddb2e19536579140496023b5ccb060bc4152`, including H1 Flat/Rough and Unitree actuator settings. BSD-3-Clause attribution and license are retained.

- MJCF assets and MuJoCo CPU contact solving replace the upstream stack. Feet/torso use touch sensors, with additional height/tilt termination guards; upstream contact-force history is not reproduced.
- Linear-velocity observations come from the torso IMU. Tracking uses floating-root world velocity rotated into the torso yaw frame, which can differ from upstream rigid-body state definitions.
- Reset randomizes sliding friction (0.6–0.9), torso mass/inertia scale (0.8–1.25), root position/yaw, and root velocity. Separate static/dynamic friction, restitution, push events, and the full event manager are not ported.
- Direct velocity commands are resampled every ten seconds, without heading control or command/terrain curricula.
- PPO uses three 128-unit ELU layers and a fixed learning rate, without adaptive-KL scheduling, distributed training, or GPU batched state.

## Local validation, 2026-09-15

Update: the [2026-09-25 learning-rate study](rl-training-study-20260925.md) completed six 5000-update runs. At the lower learning rate, all three training seeds completed the nominal ten-second walking and turning trials. The earlier 500-update result is retained below.

Evidence is in `runs/h1-native-validation-20260915`. Training completed 128 environments × 24 steps × 500 updates (1,536,000 transitions), with approximately 75 seconds in the training loop using four CPU threads.
The resulting policy survived a nominal ten-second forward trial but averaged only 0.0065 m/s forward velocity, with planar RMSE approximately 0.4996 m/s. **It failed the 0.5 m/s walking tracking criterion.** This validates the independent training workflow, not a qualified walking policy.

Tests train, resume, and evaluate while imports of `isaaclab*`, `isaacsim*`, `omni*`, and `rsl_rl*` are forbidden. Other checks cover joint ordering, selected-row isolation, randomized constants, torque agreement with direct MuJoCo, and independent FK validation of recorded motion.
