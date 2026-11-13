"""Reviewed SuperMimic sources; these describe prerequisites, not policy support."""

CHECKED_ON = "2026-10-23"
PROJECT = "https://supermimic.github.io/"
PAPER = "https://arxiv.org/abs/2610.28480"
DEMO_LICENSE = "https://supermimic.github.io/demo/LICENSE.html"

PAPER_CONTRACT = {
    "robot": "unitree-g1-29dof",
    "action_dim": 29,
    "proprioception_dim": 93,
    "history_frames": 10,
    "future_frames": 10,
    "control_hz": 50,
    "physics_substeps": 4,
    "physics_timestep": 0.005,
    "official_export_abi": None,
}

REFERENCES = {
    "official": {
        "provenance": "official_project",
        "url": PROJECT,
        "license_url": DEMO_LICENSE,
        "local_policy_package": "no_authorized_release_identified",
        "demo_use": "personal_noncommercial_browser_evaluation_only",
        "export_contract": None,
    },
    "cuboi-reproduced": {
        "provenance": "independent_reproduction",
        "url": "https://github.com/CuboiLeo/SuperMimic-Reproduced",
        "revision": "d9be385b3eeb13a638fb33fb200d3bfd9be9839f",
        "code_license": "MIT",
        "python_minor": [3, 12],
        "framework": {
            "distribution": "mjlab",
            "module": "mjlab",
            "revision": "8bb2bc9e187797689ec82c78c35019afd2fd0376",
        },
        "dependencies": {
            "torch": {"minimum": [2, 14, 0]},
            "warp-lang": {"minimum": [1, 15, 0]},
            "mujoco": {"minimum": [3, 16, 0], "maximum": [3, 17, 0]},
            "mujoco-warp": {"minimum": [3, 16, 0], "maximum": [3, 17, 0]},
            "mjlab": {},
            "numpy": {},
            "scipy": {"minimum": [1, 15, 0]},
            "tyro": {"minimum": [1, 0, 1]},
        },
        "source_files": [
            "LICENSE",
            "pyproject.toml",
            "scripts/eval_benchmark.py",
            "src/supermimic/config.py",
            "src/supermimic/policy/nets.py",
            "src/supermimic/eval/benchmark.py",
        ],
        "motion_suffix": ".npz",
        "evaluation_entry": "scripts/eval_benchmark.py",
        "checkpoint_keys": ["policy", "config"],
        "observation_dim": 93,
        "observation_layout": "term-major",
        "goal_dim": 680,
        "anchor_goal_dim": 689,
        "notes": [
            "anchor_obs adds nine goal features; those runs have a different observation protocol.",
            "Checkpoint configuration determines normalization and goal layout.",
        ],
    },
    "vibed": {
        "provenance": "independent_reproduction",
        "url": "https://github.com/zita-ch/supermimic_vibed",
        "revision": "a1508d42c9d1699a52e8233ab16d08c488441ab9",
        "code_license": "MIT",
        "python_minor": None,
        "framework": {
            "distribution": "isaaclab",
            "module": "isaaclab",
            "revision": "46d6cc3d7b115906fad12efd372fbf69f4e3d3ed",
        },
        "dependencies": {
            "torch": {},
            "isaaclab": {},
            "newton": {},
            "warp-lang": {},
            "mujoco-warp": {},
            "numpy": {},
        },
        "source_files": [
            "LICENSE",
            "docker/Dockerfile",
            "supermimic/evaluate.py",
            "supermimic/mdp.py",
            "supermimic/networks.py",
            "supermimic/motion.py",
        ],
        "motion_suffix": ".csv",
        "evaluation_entry": "supermimic/evaluate.py",
        "checkpoint_keys": ["pi", "args"],
        "observation_dim": 99,
        "observation_layout": "time-major",
        "goal_dim": 730,
        "notes": [
            "Requires the pinned Isaac Lab 3 kit-less stack with Newton/MuJoCo Warp.",
            "Adds six reference-relative orientation features to proprioception.",
            "Uses running normalization; its checkpoints do not share the other reference's ABI.",
        ],
    },
}

PACKAGES = tuple(
    sorted(
        {name for ref in REFERENCES.values() for name in ref.get("dependencies", {})}
    )
)
