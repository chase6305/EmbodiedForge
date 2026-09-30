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
  --num-envs 128 --threads 2 --updates 5000 --seed 0 --learning-rate 0.0003 \
  --output runs/h1-native-first
python -m embodiedforge h1-native train --headless \
  --resume runs/h1-native-first --num-envs 128 --threads 2 --updates 1000 \
  --output runs/h1-native-resumed
```

The existing `.cache/external/mjbatch/.venv/bin/python` also works on this machine. It selects an interpreter; the native trainer does not import mjbatch examples.
The example uses the 128-environment, 5000-update budget from the [three-seed study](rl-training-study-20260925.md), rather than the earlier 500/1000-update examples. This is a starting budget, not a guarantee of qualified walking. Evaluate each trained or resumed policy against the stated thresholds.

Output directories must be new. `--updates` counts additional updates, with a default rollout horizon of 24.
Training and evaluation are always headless and do not initialize a renderer. Both accept an explicit `--headless`; omitting it gives the same behavior. Seeds must be in `0..2**64-1`.
Runs contain `model.mjb`, numbered weights such as `checkpoint-000000050.pt`, `metrics.jsonl`, and `run.json`. The compiled MJB includes meshes, so resume and replay no longer require the original asset paths. Weights and MJB models load from the same bytes used for hash validation. Resumed runs save the actual loaded model, so replacing an input file cannot change the model recorded for that run.

Training attempts to update `run.json` after a failure, Ctrl+C, or SIGTERM. If that write also fails, the log includes the file path and write error while preserving the original training exception or interruption; the saved status may still be `running`. A final record write failure after otherwise successful training is raised rather than returning success.
Checkpoints are atomically saved every 50 updates and at completion. Resume and evaluation accept recorded checkpoints from `complete`, `interrupted`, or `failed` runs, checking artifact hashes, task version, joint order, and finite tensors. Recovery starts at the last saved update; unsaved progress is lost. Runs that stopped before their first save or are still marked `running` cannot be used as inputs.
The new weights are published before their filename and hash are committed to `run.json`. Successful commits normally retain the current and preceding checkpoints. Failed publication can leave unreferenced weights; loading follows the manifest rather than selecting the highest numbered file. A handled failure or interruption after writing new weights therefore preserves the weights referenced by the previous record.

Legacy runs without a `checkpoint` field still load `checkpoint.pt`; new runs do not create that alias. Keep passing a run directory to `--resume` or `--run`. Compatibility is one-way: the new loader reads legacy runs, while older programs cannot directly read the numbered layout. This does not guarantee power-loss durability or automatically recover runs left `running` by SIGKILL. See the [publication, training, and recovery study](h1-checkpoint-publication-20260929.md) for measured results.

Resumed records include the SHA256 of the validated input weights in `resume_checkpoint_sha256`; fresh training records `null`. Loading the resulting run does not require its source directory to remain available. Policy and Adam state are restored; environment and RNG state restart. This is not exact continuation. New training defaults to a learning rate of `0.0003`. Resume inherits the saved learning rate unless `--learning-rate` explicitly overrides it. The default follows the [three-seed learning-rate comparison](rl-training-study-20260925.md), limited to this flat-ground task and nominal-pose evaluation.
Before restoring Adam or saving a checkpoint, the validator shared with native Go1 checks parameter groups, complete state, moment shapes/dtypes, nonnegative second moments, and integer step counts. Damaged optimizer state is rejected before creating the new run directory or simulation environment.

PPO keeps exploration `log_std` within its existing `[-5, 2]` range by projecting the parameter after every Adam step. A forward-only clamp gives zero gradients to parameters that slightly overshoot, so older checkpoints can become unable to adjust those joints' exploration noise. The fix restores updates and still rejects nonfinite parameters; it does not lower the standard-deviation cap. Legacy checkpoints and deterministic inference remain compatible, but subsequent training trajectories can change when a boundary is reached. Core `train` PPO receives the same fix; the separate Go1 PPO is unaffected.

See the [turning and exploration study (Chinese)](h1-turning-study-20260929.md) for complete-budget continuation across three training seeds, command/reward ablations, randomized initial-state evaluation, and core PPO comparisons. Fixing frozen gradients does not guarantee improved turning metrics for every training seed; the report includes regressions.

The subsequent [command-sampling study (Chinese)](h1-command-mixture-20260930.md) adds three-seed paired fresh training and continuation, a four-arm ablation, continuous 120-second command sequences, and an actual failure reproduced with `--record-env`. The candidate fails the randomized-survival screen, so the original sampling default is retained.

The [termination-cost study (Chinese)](h1-termination-cost-20260930.md) pairs −200 / −400 penalties with both command samplers and records first-trial termination predicates, continuous-command survival, and post hoc actuation/gradient diagnostics. It separates newly executed training from reused controls and retains every training seed in the screening results. Both doubled-cost candidates fail the screen, so the original sampler and −200 coefficient remain the default.

The [entropy study (Chinese)](h1-entropy-study-20260930.md) adds full-budget 0.01 / 0.001 comparisons, continuation from historical weights, paired mean/sampled action diagnostics, and model sizes. Both candidates fail the robustness screen; the original sampler, 0.01 entropy coefficient, and −200 termination coefficient remain the default.

## Evaluate and replay

```bash
python -m embodiedforge h1-native evaluate --headless \
  --run runs/h1-native-resumed --threads 2 --steps 500 --velocity 0.5 0 0 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --record-motion --output runs/h1-native-eval
conda activate ef-viewer
python -m embodiedforge replay \
  --model runs/h1-native-resumed/model.mjb \
  --motion runs/h1-native-eval/motion.npz \
  --render-backend rtx --port 8081
```

The command above evaluates forward walking only. Evaluate standing, in-place turning, and turning while walking separately with new output directories; for example, use `--velocity 0 0 0.5` or `--velocity 0 0 -0.5` for in-place turns. Passing forward evaluation does not establish turning quality.

Control dt is 0.02 seconds; 500 steps cover 10 seconds. Reports include survival, planar/yaw velocity RMSE, mean velocity, return, and peak applied joint torque sampled at the final integration substep of each control step.
Only each row's first trial is counted, including its terminal frame. Failed thresholds produce a report and exit code 2. With no thresholds, `accepted=null`.
Choose error limits relative to the command magnitude: with `--velocity 0 0 0.25` and `--max-yaw-rmse 0.3`, a policy that does not turn can still pass the yaw limit. Check `mean_velocity` and the actual motion as well; a fixed absolute error limit does not establish relative tracking accuracy.
Evaluation defaults to one deterministic nominal initial state. Without `--randomized-reset`, larger batches repeat that state.
Use `--randomized-reset` to sample each row's initial state and physics from the training reset distribution while keeping observation noise disabled and velocity commands fixed. This evaluation-only option leaves training and nominal evaluation unchanged:

```bash
python -m embodiedforge h1-native evaluate --headless \
  --run runs/h1-native-resumed --randomized-reset --num-envs 32 \
  --seed 9701 --threads 2 --steps 1000 --velocity 0 0 0.5 \
  --min-survival 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/h1-native-randomized-left
```

The distribution samples sliding friction in 0.6–0.9, torso mass/inertia scale in 0.8–1.25 (log-uniform), root x/y in ±0.5 m, yaw in ±π rad, initial linear velocity components in ±0.5 m/s, and angular velocity components in ±0.5 rad/s. The same model, batch size, and seed reproduce the same initial states; another seed samples another batch. This covers initial conditions within the training distribution, without ongoing pushes, observation noise, or new terrain.
Only each row's first trial contributes; subsequent resets cannot improve a fallen row's result. `identical_initial_states` and `reset_protocol` identify the protocol. Global RMSE is weighted by observed frames, so interpret it together with survival and per-row `observed_seconds`; early falls shorten the measurement window. Multiple initial states for one policy are not independent training seeds.
`survived`, `planar_rmse_per_env`, and `yaw_rmse_per_env` store first-trial outcomes and errors in environment-index order, aligned with `observed_seconds`. A fall on the final step still counts as a failed trial; duration alone cannot establish survival. Existing acceptance thresholds still use global metrics. The arithmetic mean of per-row RMSE differs from the global RMSE pooled over observed frames.
Existing reports are not rewritten; reevaluate the saved weights into a new output directory to obtain these per-environment fields.

`termination_reasons` records first-trial termination conditions in the same environment order; surviving rows have empty lists. Multiple conditions can fire together, so reason counts need not sum to the number of falls. `torso_contact` means the torso touch sensor exceeds 1; `base_height` means root height is below 0.45 m; `base_tilt` means the torso rotation matrix z-z component is below 0.2. An environment time truncation is labeled `time_limit`; completing the requested evaluation window is not a termination. These are predicates observed at the terminating control step, not a physical root-cause analysis or a history of contacts during earlier integration substeps. Later resets cannot overwrite the first-trial reasons.

`action_clip_fraction_per_env` reports the fraction of raw action components outside ±5 during each first trial (observed steps × 19 joints). `action_clip_fraction` pools those counts across all observed steps, including the terminating frame and excluding actions after reset. Values exactly on the boundary are not counted as clipped. This measures action clipping, not mechanical joint-limit contact, and is not an acceptance criterion. Early failures contribute fewer observations to the pooled fraction; inspect per-environment fractions together with `observed_seconds`.

Motion recording defaults to row 0's first trial, capped at 10000 steps. Use `--record-motion --record-env 7` to record row 7; indices start at zero and must be below `--num-envs`. `--record-env` requires `--record-motion`. Keep the same weights, batch size, seed, and randomization option to reproduce a row; shrinking the batch to one does not preserve its initial state. The recording ends at the selected row's first terminal frame, excluding later resets. Motion metadata stores the row index, batch size, seed, randomization mode, and model/checkpoint SHA256. Replay also supports `mujoco` and `gl`. The existing `live` command still supports Go1 only.

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
- Training samples forward velocity in 0–1 m/s, zero lateral velocity, and yaw velocity in ±1 rad/s, with 2% standing commands. The interface accepts lateral commands, but this recipe does not train lateral locomotion. Evaluate in-place turns separately from turns while walking.
- PPO uses three 128-unit ELU layers and a fixed learning rate, without adaptive-KL scheduling, distributed training, or GPU batched state.

## Local validation, 2026-09-15

Update: the [2026-09-25 learning-rate study](rl-training-study-20260925.md) completed six 5000-update runs. At the lower learning rate, all three training seeds completed the nominal ten-second walking and turning trials. The earlier 500-update result is retained below.

Evidence is in `runs/h1-native-validation-20260915`. Training completed 128 environments × 24 steps × 500 updates (1,536,000 transitions), with approximately 75 seconds in the training loop using four CPU threads.
The resulting policy survived a nominal ten-second forward trial but averaged only 0.0065 m/s forward velocity, with planar RMSE approximately 0.4996 m/s. **It failed the 0.5 m/s walking tracking criterion.** This validates the independent training workflow, not a qualified walking policy.

Tests train, resume, and evaluate while imports of `isaaclab*`, `isaacsim*`, `omni*`, and `rsl_rl*` are forbidden. Other checks cover joint ordering, selected-row isolation, randomized constants, torque agreement with direct MuJoCo, and independent FK validation of recorded motion.
