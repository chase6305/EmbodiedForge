"""Own Go1 policy and mjbatch simulation in the training SDK's interpreter."""

import argparse
import socket
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np

from ._go1_implementation import policy_action_mean
from ._live_channel import JsonChannel


class Go1LiveRuntime:
    def __init__(self, request):
        import mujoco
        import torch

        from ._go1_assets import verified_go1_assets
        from ._go1_implementation import load_go1_implementation
        from ._wuji_recipe import finite_tensors
        from .recipes import sha256

        versions = {
            name: metadata.version(name)
            for name in ("mujoco", "mjbatch", "mujoco-menagerie", "torch", "numpy")
        }
        expected_versions = request["contract"].get("runtime", {}).get("versions", {})
        for name, version in versions.items():
            if name in expected_versions and expected_versions[name] != version:
                raise ValueError(
                    f"Training SDK version mismatch for {name}: {version} != {expected_versions[name]}"
                )
        if sha256(Path(request["checkpoint"])) != request["sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch in policy worker")
        torch.set_num_threads(request["threads"])
        assets = verified_go1_assets(request["contract"].get("assets"))
        Go1, ActorCritic, control_dt, implementation = load_go1_implementation(
            request.get("implementation")
        )
        checkpoint = torch.load(
            request["checkpoint"], weights_only=True, map_location="cpu"
        )
        finite_tensors(checkpoint)
        result = request["contract"]
        for name, default in [
            ("reward_profile", "original"),
            ("command_profile", "original"),
            ("task_semantics", "upstream-v1"),
            ("learning_rate_override", None),
        ]:
            if checkpoint.get(name, default) != result.get(name, default):
                raise ValueError(f"Checkpoint {name} differs from the managed run")
        if checkpoint.get("iteration") != result["checkpoint_iteration"]:
            raise ValueError("Checkpoint iteration differs from the managed run")
        self.policy = ActorCritic().eval()
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if (self.policy.var < 0).any() or self.policy.count <= 0:
            raise ValueError("Invalid policy normalization statistics")
        self.env = Go1(
            request["num_envs"],
            seed=request["seed"],
            num_threads=request["threads"],
            episode_steps=request["episode_steps"],
            semantics=result.get("task_semantics", "upstream-v1"),
            reward_profile=result.get("reward_profile", "original"),
            command_profile=result.get("command_profile", "original"),
        )
        self.dt = control_dt
        self.command = np.zeros((request["num_envs"], 3))
        self.reward = np.zeros(request["num_envs"])
        self.done = np.zeros(request["num_envs"], bool)
        self.fell = self.done.copy()
        self.episodes = np.zeros(request["num_envs"], int)
        self.env.command[:] = self.command
        mujoco.mj_saveModel(self.env.batch.model, request["model_path"])
        self.metadata = {
            "implementation": implementation,
            "assets": assets,
            "versions": versions,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "model_sha256": sha256(Path(request["model_path"])),
            "checkpoint_sha256": request["sha256"],
            "iteration": checkpoint["iteration"],
            "dt": self.dt,
            "command_limits": self.env.command_range.tolist(),
        }

    def snapshot(self):
        value = {
            "qpos": self.env.qpos.tolist(),
            "step": self.env.steps.tolist(),
            "time": (self.env.steps * self.dt).tolist(),
            "episode": self.episodes.tolist(),
            "position": self.env.trunk.tolist(),
            "velocity": self.env.vel.tolist(),
            "yaw_velocity": self.env.gyro[:, 2].tolist(),
            "reward": self.reward.tolist(),
            "command": self.command.tolist(),
            "done": self.done.tolist(),
            "fell": self.fell.tolist(),
        }
        # Channel serialization also rejects nonfinite values before publication.
        return value

    def execute(self, message):
        operation = message.get("action")
        if operation == "snapshot" and set(message) == {"action"}:
            return self.snapshot()
        if operation == "step" and set(message) == {"action"}:
            if self.done.any():
                return self.snapshot()  # Preserve terminal poses until explicit reset.
            import torch

            self.env.command[:], self.env.until[:] = self.command, 2
            observation = self.env.obs()
            if not np.isfinite(observation).all():
                raise RuntimeError("Non-finite live policy observation")
            with torch.inference_mode():
                observation = torch.as_tensor(observation)
                action = policy_action_mean(self.policy, observation).numpy()
            self.reward, self.done, self.fell, _ = self.env.step(action)
            self.env.batch.forward()  # Match the existing fixed-command evaluator.
            return self.snapshot()
        if operation in ("reset", "velocity"):
            expected = {"action", "env_id"} | (
                {"value"} if operation == "velocity" else set()
            )
            if set(message) != expected:
                raise ValueError("Unexpected control fields")
            index = message["env_id"]
            if type(index) is not int or not 0 <= index < len(self.done):
                raise ValueError("Environment index out of range")
            if operation == "reset":
                self.env.reset([index])
                self.command[index] = 0
                self.env.command[index] = 0
                self.reward[index], self.done[index], self.fell[index] = 0, False, False
                self.episodes[index] += 1
            else:
                value = message["value"]
                if (
                    not isinstance(value, list)
                    or len(value) != 3
                    or any(type(v) not in (int, float) for v in value)
                ):
                    raise ValueError("Velocity command requires three numbers")
                value = np.asarray(value, dtype=float)
                if (
                    not np.isfinite(value).all()
                    or (np.abs(value) > self.env.command_range).any()
                ):
                    raise ValueError("Velocity command outside policy limits")
                self.command[index] = value
                self.env.command[index] = value
            return self.snapshot()
        raise ValueError("Unknown live policy operation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    with socket.socket(fileno=args.fd) as sock:
        channel = JsonChannel(sock)
        try:
            runtime = Go1LiveRuntime(channel.receive())
            channel.send(
                {"ok": True, "metadata": runtime.metadata, "state": runtime.snapshot()}
            )
            while True:
                request = channel.receive()
                if request == {"action": "close"}:
                    break
                try:
                    channel.send({"ok": True, "state": runtime.execute(request)})
                except ValueError as error:
                    channel.send({"ok": False, "error": str(error)})
        except (EOFError, BrokenPipeError):
            pass
        except BaseException as error:
            traceback.print_exc()
            try:
                channel.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
            except (OSError, ValueError):
                pass
            raise


if __name__ == "__main__":
    main()
