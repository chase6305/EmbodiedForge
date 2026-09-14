"""Resume initialization uses the saved curriculum before the first reset."""

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from embodiedforge import _microduck_worker as worker


def environment():
    terms = {
        "twist": SimpleNamespace(rel_standing_envs=0.15),
        "head_pose": SimpleNamespace(ranges=((-0.39, 0.39),) * 4),
    }
    return SimpleNamespace(
        common_step_counter=0,
        num_envs=64,
        command_manager=SimpleNamespace(get_term_cfg=terms.__getitem__),
        reward_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(weight=-0.6)
        ),
    )


def test_first_reset_is_audited_once_without_rewinding_later_resets(tmp_path):
    env = environment()
    path = tmp_path / "audit.json"
    args = dict(counter=24048, report_path=str(path), checkpoint_sha256="sha")
    worker.restore_training_counter(env, None, counter=24048)
    worker.audit_training_reset(env, None, **args)
    original = path.read_bytes()
    report = json.loads(original)
    assert report["counter_at_first_reset"] == 24048
    assert report["action_rate_weight"] == -0.6
    assert report["standing_env_fraction"] == 0.15
    env.common_step_counter += 48
    worker.audit_training_reset(env, None, **args)
    assert path.read_bytes() == original and env.common_step_counter == 24096


def test_wrong_counter_fails_before_training_can_claim_correct_initialization(tmp_path):
    with pytest.raises(RuntimeError, match="before first reset"):
        worker.audit_training_reset(
            environment(),
            None,
            counter=24048,
            report_path=str(tmp_path / "audit.json"),
            checkpoint_sha256="sha",
        )
    assert not (tmp_path / "audit.json").exists()


def test_resume_uses_upstream_launcher_and_preserves_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    snapshot = tmp_path / "logs/rsl_rl/microduck/resume_source/checkpoint.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.touch()
    marker = object()
    env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), events={"bam": marker})
    agent = SimpleNamespace(algorithm=SimpleNamespace(learning_rate=0.001))

    @dataclass
    class Config:
        env: object
        agent: object
        enable_nan_guard: bool = False

        @staticmethod
        def from_task(task):
            assert task == "Mjlab-Velocity-Flat-MicroDuck"
            return Config(env_cfg, agent)

    def launch(task, cfg):
        assert cfg.enable_nan_guard and cfg.env.events["bam"] is marker
        assert cfg.env.scene.num_envs == 64 and cfg.agent.max_iterations == 5
        assert cfg.agent.algorithm.learning_rate == 3e-5
        assert cfg.agent.resume and cfg.agent.logger == "tensorboard"
        assert cfg.agent.seed == 0
        startup = cfg.env.events["ef_resume_counter"]
        audit = cfg.env.events["ef_resume_audit"]
        assert startup.mode == "startup" and audit.mode == "reset"
        env = environment()
        startup.func(env, None, **startup.params)
        assert env.common_step_counter == 24048  # Before any curriculum/reset.
        audit.func(env, None, **audit.params)

    monkeypatch.setitem(
        sys.modules, "mjlab.managers", SimpleNamespace(EventTermCfg=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "mjlab.scripts.train",
        SimpleNamespace(TrainConfig=Config, launch_training=launch),
    )
    monkeypatch.setattr(
        worker,
        "checkpoint_metadata",
        lambda path: {
            "common_step_counter": 24048,
            "learning_rate": 3e-5,
            "sha256": "sha",
        },
    )
    result = worker.resume_training(snapshot, 64, 5)
    assert result["checkpoint_sha256"] == "sha"
    assert result["counter_at_first_reset"] == 24048
