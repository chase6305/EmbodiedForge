"""SDK-free validation and inventory for the pinned single-rank RLinf recipe."""

import json
import re
import subprocess

from embodiedforge.runtime.hashing import file_sha256 as sha256

from .rlinf_catalog import REPLAY_WINDOW, REVISION, WEIGHTS


def check_checkout(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected RLinf {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("RLinf has tracked changes; use the pinned clean checkout")


def checkpoint_step(path, *, replay_window=REPLAY_WINDOW):
    """Check the pinned, single-rank SAC checkpoint layout before resuming."""
    match = re.fullmatch(r"global_step_([0-9]+)", path.name)
    if not match or int(match[1]) < 1:
        raise ValueError("Resume requires a global_step_N checkpoint directory")
    for relative in (
        WEIGHTS,
        "actor/dcp_checkpoint/.metadata",
        "actor/sac_components/alpha/dcp_checkpoint/.metadata",
        "actor/sac_components/target_model/checkpoint_rank_0.pt",
        "actor/sac_components/replay_buffer/rank_0/metadata.json",
        "actor/sac_components/replay_buffer/rank_0/trajectory_index.json",
    ):
        if not (path / relative).is_file() or not (path / relative).stat().st_size:
            raise ValueError(f"Incomplete SAC checkpoint: missing {path / relative}")
    for relative in (
        "actor/dcp_checkpoint",
        "actor/sac_components/alpha/dcp_checkpoint",
    ):
        if not any(
            p.stat().st_size for p in (path / relative).glob("*.distcp") if p.is_file()
        ):
            raise ValueError(
                f"Incomplete SAC checkpoint: no data shards in {path / relative}"
            )
    replay = path / "actor/sac_components/replay_buffer/rank_0"
    try:
        metadata = json.loads((replay / "metadata.json").read_text())
        index = json.loads((replay / "trajectory_index.json").read_text())
        ids = index["trajectory_id_list"]
        entries = index["trajectory_index"]
        if (
            metadata["trajectory_format"] != "pt"
            or not isinstance(ids, list)
            or not ids
        ):
            raise ValueError("expected a nonempty pt replay buffer")
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid or duplicate replay trajectory IDs")
        if ids != sorted(ids) or set(entries) != {str(i) for i in ids}:
            raise ValueError("replay history and trajectory index differ")
        samples = 0
        for trajectory_id in ids:
            entry = entries[str(trajectory_id)]
            shape = entry["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) < 2
                or any(type(n) is not int or n < 1 for n in shape)
                or type(entry["num_samples"]) is not int
                or entry["num_samples"] != shape[0] * shape[1]
                or type(entry["trajectory_id"]) is not int
                or entry["trajectory_id"] != trajectory_id
            ):
                raise ValueError("invalid replay trajectory shape or sample count")
            samples += entry["num_samples"]
        for name, expected in (
            ("size", len(ids)),
            ("trajectory_counter", ids[-1] + 1),
            ("total_samples", samples),
        ):
            if type(metadata[name]) is not int or metadata[name] != expected:
                raise ValueError(f"replay {name} differs from its index")
        # Upstream retains the historical index but serializes only the active
        # cache. Requiring every historical file breaks resumes after 10k entries.
        for trajectory_id in ids[-replay_window:]:
            version = entries[str(trajectory_id)]["model_weights_id"]
            if not isinstance(version, str) or "/" in version or "\\" in version:
                raise ValueError("invalid replay model_weights_id")
            trajectory = replay / f"trajectory_{trajectory_id}_{version}.pt"
            if not trajectory.is_file() or not trajectory.stat().st_size:
                raise ValueError(f"missing active replay trajectory: {trajectory}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SAC replay checkpoint {replay}: {exc}") from exc
    return int(match[1])


def inventory(root):
    return [
        {
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def resume_settings(checkpoint):
    """Recover the saved collection seed and batch width for this single-rank recipe."""
    replay = checkpoint / "actor/sac_components/replay_buffer/rank_0"
    metadata = json.loads((replay / "metadata.json").read_text())
    index = json.loads((replay / "trajectory_index.json").read_text())
    try:
        seed = metadata["seed"]
        latest = str(index["trajectory_id_list"][-1])
        num_envs = index["trajectory_index"][latest]["shape"][1]
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or type(num_envs) is not int
            or num_envs < 1
        ):
            raise ValueError("invalid seed or environment batch width")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SAC resume settings in {replay}: {exc}") from exc
    return {"seed": seed, "num_envs": num_envs}
