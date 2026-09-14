# Training and deployment commands

[简体中文](training-deployment.md) | **English** · [Back to README](../README.en.md)

Choose the section for your task; installing every SDK is unnecessary. Run commands from the EmbodiedForge repository root. Replace `/path/to/...` with local paths and use nonexistent paths for new output directories or exported files. Example budgets reproduce workflows without promising behavioral acceptance.

[Environment setup](#setup) · [Core PPO](#core) · [Go1](#go1) · [Microduck](#microduck) · [H1](#h1) · [Wuji](#wuji) · [MPC / CEM](#solvers) · [Web serving and remote access](#serving) · [Wheel deployment](#package) · [Artifacts and limits](#artifacts)

| Task | Training / solver runtime | Available execution or export |
| --- | --- | --- |
| `reach` / `hold` | Core CPU PPO with NumPy, MuJoCo, or other backends | Headless checkpoint evaluation and recording |
| Go1 | Isolated mjbatch environment, CPU physics and PPO | Shared Web live policy and motion replay |
| Microduck | Isolated mjlab environment, CUDA PPO | Native / Viser policy execution, ONNX export and comparison |
| H1 | Isolated IsaacLab, Newton / MuJoCo-Warp GPU PPO | Fixed-command evaluation, offline HTML / Web motion replay |
| Wuji / Wuji Light | Isolated UniLab, GPU PPO | Sequential trial evaluation and video recording |
| Cartpole / arm throwing | Isolated mjbatch, CPU MPC / CEM | Solver metrics and trajectories |

Deployment here means running policies in simulation, serving viewers, and exporting models. There is currently no unified hardware deployment command or generic Go1/H1 ONNX export entry point.

<a id="setup"></a>

## Environment setup

Install the launchers and required viewer dependencies in the main environment. Use an existing Python 3.11 environment, or create `ef-viewer` following the README:

```bash
python -m pip install -e '.[viz-robot]' 'mujoco==3.11.0'
python -m embodiedforge recipes list
```

Keep training SDKs isolated. Recipe and Microduck setup require Git, `uv`, and network access. Run only the setup commands you need. Sources must be clean checkouts at the pinned revisions; these commands do not switch the revision of your source repository.

```bash
python -m embodiedforge recipes setup --source mjbatch \
  --repo /path/to/mjbatch --python 3.12 --timeout 1200
python -m embodiedforge recipes setup --source wuji_unilab \
  --repo /path/to/wuji_unilab --python 3.12 --timeout 1200
python -m embodiedforge.microduck setup --repo /path/to/microduck_rl
python -m embodiedforge.microduck check --repo /path/to/microduck_rl
```

| Source | Pinned revision | Environment location |
| --- | --- | --- |
| mjbatch | `b84c0c20aedbdf048122cbc47f554e9b93cc4754` | `.cache/external/mjbatch/.venv` |
| Wuji UniLab | `91ccfa0ec8c129b300865bd36c59dc9eed56a744` | `.cache/external/wuji_unilab/.venv` |
| Microduck | `53b8971b61baf5b7f3c16d135dd7cac37623de4b` | `.cache/microduck-venv` |
| IsaacLab H1 | `2e44ddb2e19536579140496023b5ccb060bc4152` | Existing environment selected with `--environment` |

H1 has no automatic `setup` subcommand. Prepare the isolated Python 3.12 / RSL-RL 5.0.1 environment described in the [H1 guide](h1-isaaclab.md) first. `--environment` takes the environment root, not its Python executable. Wuji, Microduck, and H1 require their respective CUDA stacks; Go1 and both solver tasks run on CPU. When customizing caches, pass the same `--cache` or Microduck `--env-dir` at every step.

<a id="core"></a>

## Core PPO: reach / hold

Install Torch and MuJoCo in the current main environment. Train the two point tasks separately, evaluate their policies, and record reach evaluation data:

```bash
python -m pip install -e '.[train,mujoco]'
python -m embodiedforge train --task reach --physics mujoco \
  --num-envs 32 --updates 100 --output runs/commands-reach
python -m embodiedforge evaluate \
  --checkpoint runs/commands-reach/checkpoint.pt \
  --num-envs 32 --steps 300 --output runs/commands-reach-eval
python -m embodiedforge inspect --dataset runs/commands-reach-eval
python -m embodiedforge train --task hold --physics numpy \
  --num-envs 32 --updates 100 --output runs/commands-hold
python -m embodiedforge evaluate \
  --checkpoint runs/commands-hold/checkpoint.pt --num-envs 32 --steps 300
```

Core PPO uses proprioception, not images. Evaluation restores configuration from the checkpoint; avoid changing its task or physics arbitrarily. The core CLI currently has no resume or ONNX export command. Point-task `visualization` runs a demonstration controller rather than loading these PPO checkpoints.

To compare trained and untrained policies, run these commands in an environment with core training dependencies. The script preserves the checkpoint’s task, physics, control period, and episode length. Both policies use matching network dimensions and evaluation seeds. Output includes the model SHA256 and environment configuration for comparison provenance.

```bash
python benchmarks/check_learning.py runs/commands-reach/checkpoint.pt --num-envs 128 --seeds 1001 1002 1003
python benchmarks/check_learning.py runs/commands-hold/checkpoint.pt --num-envs 128 --seeds 1001 1002 1003
```

<a id="go1"></a>

## Go1: train → resume → evaluate → run live

Complete mjbatch setup first. Resume writes to a new directory, and `--updates` specifies additional updates for that invocation. Timeouts below are in seconds; increase them for slower machines.

```bash
python -m embodiedforge recipes train --task go1-joystick \
  --num-envs 512 --horizon 24 --updates 600 --threads 4 \
  --timeout 1200 --output runs/commands-go1
python -m embodiedforge recipes status --run runs/commands-go1
python -m embodiedforge recipes train --task go1-joystick \
  --resume-run runs/commands-go1 --num-envs 512 --horizon 24 \
  --updates 100 --threads 4 --timeout 1200 --output runs/commands-go1-resumed
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/commands-go1-resumed --suite basic --seeds 0 1 2 \
  --num-envs 32 --steps 3000 --record-motion --timeout 1200 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-go1-eval
python -m embodiedforge live --run runs/commands-go1-resumed \
  --render-backend mujoco --num-envs 2 \
  --width 960 --height 540 --fps 30 --port 8080
```

The evaluation thresholds are example requirements. Any failing case/seed produces `rejected` and retains reports; inspect results before deciding whether to diagnose the policy live. Run the live command in the viewer environment; it launches the policy in the training environment automatically. Open `http://127.0.0.1:8080`, click “前进” (Forward), then “继续” (Resume). Use `--worker-python /path/to/venv/bin/python` to select a compatible policy environment explicitly.

After fixed-command evaluation, test continuous command transitions separately using a new output directory:

```bash
python -m embodiedforge recipes evaluate --task go1-joystick \
  --run runs/commands-go1-resumed --suite switching --seeds 0 1 2 \
  --num-envs 32 --steps 3000 --record-motion --timeout 1200 \
  --output runs/commands-go1-switching
```

This transition example reports metrics without acceptance thresholds. See [task recipes](recipes.md) for `extended`, lateral, and compound transition suites. View recordings with a matching MJCF using the Web replay command below.

<a id="microduck"></a>

## Microduck: training, export, and policy execution

Complete Microduck setup/check first. Smoke tests 64 environments, five updates, and ONNX export. Use a separate directory for the larger training run.

```bash
python -m embodiedforge.microduck smoke --repo /path/to/microduck_rl \
  --output runs/commands-microduck-smoke
python -m embodiedforge.microduck train --repo /path/to/microduck_rl \
  --num-envs 4096 --iterations 4000 --output runs/commands-microduck
python -m embodiedforge.microduck progress --repo /path/to/microduck_rl \
  --run runs/commands-microduck
```

Read the actual model path from the `checkpoint` field in `runs/commands-microduck/run.json` and substitute it for `/path/to/microduck/model.pt`. Do not infer filenames from iteration counts. Run resume, native playback, or Viser playback as needed:

```bash
python -m embodiedforge.microduck train --repo /path/to/microduck_rl \
  --resume /path/to/microduck/model.pt --num-envs 4096 --iterations 1000 \
  --output runs/commands-microduck-resumed
python -m embodiedforge.microduck play --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --viewer native --steps 500 --seed 0
python -m embodiedforge.microduck play --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --viewer viser
```

Viser uses the upstream page, separate from the shared Go1/H1 Web viewer, and does not support the native mode’s `--steps` option. Export the same checkpoint, then compare ONNX and Torch actions on actual observations:

```bash
python -m embodiedforge.microduck export --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --output runs/commands-microduck.onnx
python -m embodiedforge.microduck evaluate --repo /path/to/microduck_rl \
  --checkpoint /path/to/microduck/model.pt --onnx runs/commands-microduck.onnx \
  --velocity 0.2 0 0 --no-pushes --num-envs 16 --steps 500 --seeds 0 1 2 \
  --output runs/commands-microduck-eval
```

Export includes observation normalization, with 61 input and 14 output dimensions. Torch actions still drive simulation during comparison; this is not independent ONNX closed-loop or hardware validation. No behavioral thresholds are set in this example; add RMSE or survival requirements as described in the [Microduck guide](microduck.md).

<a id="h1"></a>

## H1: IsaacLab training and replay

Use the prepared pinned IsaacLab checkout and isolated environment. Run a smoke test, then add training updates. Keep `--repo` and `--environment` consistent across commands.

```bash
python -m embodiedforge h1 train \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --num-envs 64 --updates 5 --timeout 600 --output runs/commands-h1-smoke
python -m embodiedforge h1 status --run runs/commands-h1-smoke
python -m embodiedforge h1 train \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --resume-run runs/commands-h1-smoke --num-envs 512 --updates 1000 \
  --output runs/commands-h1-resumed
python -m embodiedforge h1 evaluate \
  --repo /path/to/IsaacLab --environment /path/to/envs/isaaclab \
  --run runs/commands-h1-resumed --suite basic --num-envs 32 \
  --steps 500 --seeds 0 1 2 --record-motion --record-env 0 --timeout 600 \
  --min-survival-fraction 0.8 --max-planar-rmse 0.3 --max-yaw-rmse 0.3 \
  --output runs/commands-h1-eval
python -m embodiedforge h1 replay \
  --motion runs/commands-h1-eval/motion-seed-0-forward-env-0.npz \
  --output runs/commands-h1-forward.html
```

The larger training command has no extra timeout; add `--timeout` to bound wall-clock time. Five hundred steps cover ten seconds. For longer validation, use `--steps 3000` with a new output directory. Open the HTML file offline for skeleton replay; see below for mesh replay. H1 currently has no shared Web live-command entry point: `live --run` cannot load H1 training directories.

<a id="wuji"></a>

## Wuji / Wuji Light: reorientation training

Complete wuji_unilab setup first. Standard and Light are different tasks: use separate run directories and preserve `--task` for resume and evaluation. The small budgets below validate the workflow:

```bash
python -m embodiedforge recipes train --task wuji-reorient \
  --num-envs 32 --horizon 40 --updates 5 --timeout 600 \
  --output runs/commands-wuji
python -m embodiedforge recipes train --task wuji-reorient-light \
  --num-envs 32 --horizon 40 --updates 5 --timeout 600 \
  --output runs/commands-wuji-light
python -m embodiedforge recipes train --task wuji-reorient-light \
  --resume-run runs/commands-wuji-light --num-envs 32 --horizon 40 \
  --updates 100 --timeout 1800 --output runs/commands-wuji-light-resumed
python -m embodiedforge recipes status --run runs/commands-wuji-light-resumed
python -m embodiedforge recipes evaluate --task wuji-reorient-light \
  --run runs/commands-wuji-light-resumed --num-envs 1 --seed 0 \
  --num-trials 50 --steps 280 --timeout 1200 \
  --min-success-rate 0.5 --max-drop-rate 0.2 --output runs/commands-wuji-eval
```

Evaluation runs single-environment trials sequentially; Go1 options `--suite`, `--seeds`, and `--record-motion` do not apply. To evaluate the standard task, change `--task` to `wuji-reorient`, `--run` to `runs/commands-wuji`, and choose a new output directory. Record a separate small video sample if `ffprobe` is available:

```bash
python -m embodiedforge recipes evaluate --task wuji-reorient-light \
  --run runs/commands-wuji-light-resumed --num-envs 1 --seed 0 \
  --num-trials 1 --steps 280 --record-video --timeout 600 \
  --output runs/commands-wuji-video
```

Video supports behavioral inspection, not full acceptance. Wuji currently has no shared Web live-control or model-export command.

<a id="solvers"></a>

## CPU MPC and CEM solvers

Reuse mjbatch setup. `--num-envs` means candidate trajectories for MPC and population size for CEM; neither produces a PPO checkpoint.

```bash
python -m embodiedforge recipes solve --task cartpole-mpc \
  --num-envs 1024 --horizon 25 --steps 150 --threads 4 --timeout 600 \
  --output runs/commands-cartpole
python -m embodiedforge recipes solve --task arm-throw-codesign \
  --num-envs 512 --generations 30 --threads 4 --timeout 600 \
  --output runs/commands-arm
python -m embodiedforge recipes status --run runs/commands-arm
```

Outputs include solver metrics, baselines, and trajectories. Solver directories cannot be passed to Go1 `live`. See [task recipes](recipes.md) for objectives and trajectory semantics.

<a id="serving"></a>

## Web serving and remote access

Install the desired renderer in the viewer environment. The OVRTX installation below uses validated constraints and requires a compatible RTX GPU and driver. Then select `--render-backend rtx` for live Go1 rendering as well.

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com \
  -c configs/viewer-constraints.txt -e '.[viz-robot,viz-rtx]' 'mujoco==3.11.0'
python -m embodiedforge.visualization --viewer web --render-backend rtx --check
```

`--check` checks installation metadata only. These replay examples use recordings generated above and matching local MJCF models; model directories must include referenced meshes and materials. Run separately, or use different ports in different terminals:

```bash
python -m embodiedforge replay --model /path/to/unitree_go1/go1.xml \
  --motion runs/commands-go1-switching/motion-seed-0-switching.npz \
  --render-backend mujoco --width 960 --height 540 --fps 30 --port 8081
python -m embodiedforge replay --model /path/to/unitree_h1/h1.xml \
  --motion runs/commands-h1-eval/motion-seed-0-forward-env-0.npz \
  --render-backend rtx --width 960 --height 540 --fps 30 --port 8082
```

Servers bind to loopback by default. In a separate client terminal, forward the Go1 live server and both replay ports, then open their local addresses:

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 \
  -L 8082:127.0.0.1:8082 user@server
```

Open `http://127.0.0.1:8080`, `:8081`, or `:8082` while the corresponding server is running. The page supports resolution changes and 1–60 FPS. Live Go1 starts paused; replay only reads recordings. Stop/Ctrl+C ends the entire shared session. These commands run foreground processes; they do not install system services or automatic restart configuration. See [viewers](viewers.md) for lifecycle and remote-access details.

For EGL warnings or unexpected GPU selection, render a test frame and inspect `graphics.vendor`, `graphics.renderer`, and retained driver warnings:

```bash
python -m embodiedforge doctor --graphics egl
```

The diagnostic runs in an isolated process without changing system driver configuration. A successful frame does not mean NVIDIA was selected. See [EGL diagnostics](graphics-diagnostics.md) for the local missing-registration case and driver-package repair steps (Chinese).

<a id="package"></a>

## Wheel deployment: run on another machine

Build a wheel from the source checkout. This packages EmbodiedForge without bundling training SDKs, models, or caches:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

Copy `dist/embodiedforge-0.1.0-py3-none-any.whl` and the complete Go1 run directory to the target machine. Prepare a compatible isolated mjbatch SDK environment there, then run these commands in its Python 3.11 viewer environment, replacing all placeholder paths:

```bash
python -m pip install '/path/to/embodiedforge-0.1.0-py3-none-any.whl[viz-robot]' 'mujoco==3.11.0'
python -m embodiedforge live --run /path/to/runs/go1-trained \
  --worker-python /path/to/mjbatch-venv/bin/python \
  --render-backend mujoco --port 8080
```

After installation, the CLI works independently of the source working directory. The wheel includes Web HTML/JavaScript, the Go1 scene XML, and the port’s license. Keep `--worker-python` pointed at the virtual environment entry point rather than resolving its symlink to a generic Python binary. The wheel contains EmbodiedForge code; external SDKs, pinned source records, and robot assets must still be prepared as described above. OVRTX also requires its viewer dependencies and driver.

A source checkout remains useful for development and reproducing examples. Source distributions additionally include bilingual documentation, the illustration, configuration, examples, and benchmark scripts; training artifacts and caches are excluded. After a version change, substitute the actual built wheel filename above.

<a id="artifacts"></a>

## Artifacts, resume, and acceptance

| Workflow | Inputs / artifacts to retain | Next use |
| --- | --- | --- |
| Core PPO | `checkpoint.pt` and training directory | `evaluate --checkpoint` |
| Managed Go1 / H1 / Wuji training | Entire run directory and `run.json`, including configuration, model, and version records | Same-task resume and evaluation; only Go1 supports `live --run` |
| Microduck | `run.json`, actual checkpoint, configuration, and runtime records | `--resume`, `play`, `evaluate`, `export` |
| Microduck export | ONNX file and validation report | Inference interface checks and action comparison |
| Go1 / H1 motion recording | NPZ and matching MJCF / asset directory | Shared Web replay; H1 also supports offline skeleton HTML |
| MPC / CEM | Complete solver directory, results, and trajectories | Result analysis, not PPO policy loading |

Run `status` or Microduck `progress` from another terminal during training; a status file is not a process-liveness guarantee. Resume restores the model, optimizer, normalization, curricula, and other state supported by the task, but not each environment’s physical state and all RNG state. It is not step-equivalent continuation. Managed Go1/H1/Wuji resume requires a compatible `complete` run of the same task.

To run a model on another machine, retain the full training directory and recreate compatible SDKs and robot assets. Go1 supports a new interpreter path through `--worker-python`, but still checks SDK and asset versions; copying only `model.pt` does not bypass these requirements. An evaluation marked `complete` merely finished execution. Acceptance is meaningful under its stated conditions only when thresholds were set and every required check passed.
