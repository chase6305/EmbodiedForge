"""Named, packed reference motions; independent of robot and simulator SDKs."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np


def _names(value, name):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(x, str) or not x.strip() for x in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must contain unique nonempty names")
    return value


class MotionClips:
    """A frame buffer indexed by clip, with explicit robot and time conventions.

    Arrays use metres, radians, seconds, world-frame poses, and wxyz quaternions.
    Each clip starts at reference time zero. This format is separate from the
    existing Go1/H1 replay format and from ACT episode/action datasets.
    """

    def __init__(self, arrays, metadata):
        if not isinstance(metadata, dict) or metadata.get("schema") != "ef-motion-v1":
            raise ValueError("Expected ef-motion-v1 metadata")
        if (
            not isinstance(metadata.get("robot"), str)
            or not metadata["robot"].strip()
            or metadata.get("quaternion_order") != "wxyz"
            or metadata.get("coordinate_frame") != "world"
        ):
            raise ValueError(
                "Require robot identity, world coordinates and wxyz quaternions"
            )
        joints = _names(metadata.get("joint_names"), "joint_names")
        bodies = _names(metadata.get("body_names"), "body_names")
        self.metadata = metadata
        self.arrays = arrays
        required = {
            "fps",
            "motion_lengths",
            "motion_names",
            "object_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
            "contact_label",
        }
        if missing := required - arrays.keys():
            raise ValueError(f"Missing motion fields: {sorted(missing)}")
        fps = arrays["fps"]
        if (
            fps.size != 1
            or fps.dtype.kind not in "iuf"
            or not np.isfinite(fps).all()
            or fps.item() <= 0
        ):
            raise ValueError("fps must be a finite positive scalar")
        self.fps = float(fps.item())
        self.lengths = arrays["motion_lengths"]
        if (
            self.lengths.ndim != 1
            or self.lengths.dtype.kind not in "iu"
            or not len(self.lengths)
            or (self.lengths <= 0).any()
        ):
            raise ValueError("motion_lengths must be positive integer[C]")
        for name in ("motion_names", "object_names"):
            value = arrays[name]
            if (
                value.shape != self.lengths.shape
                or value.dtype.kind != "U"
                or any(not x.strip() for x in value)
            ):
                raise ValueError(f"{name} must be nonempty Unicode[C]")
        self.names = _names(arrays["motion_names"].tolist(), "motion_names")
        self.objects = arrays["object_names"].tolist()
        total = sum(int(n) for n in self.lengths)
        if total > np.iinfo(np.int64).max:
            raise ValueError("Motion frame count exceeds supported indexing")
        self.lengths = self.lengths.astype(np.int64, copy=False)
        self.starts = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
        shapes = {
            "joint_pos": (total, len(joints)),
            "joint_vel": (total, len(joints)),
            "body_pos_w": (total, len(bodies), 3),
            "body_quat_w": (total, len(bodies), 4),
            "body_lin_vel_w": (total, len(bodies), 3),
            "body_ang_vel_w": (total, len(bodies), 3),
            "object_pos_w": (total, 3),
            "object_quat_w": (total, 4),
            "object_lin_vel_w": (total, 3),
            "object_ang_vel_w": (total, 3),
            "contact_label": (total, len(bodies)),
        }
        self.frame_fields = tuple(shapes)
        for name, shape in shapes.items():
            value = arrays[name]
            if (
                value.shape != shape
                or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite numeric{shape}")
            if "quat" in name and not np.allclose(
                np.linalg.norm(value, axis=-1), 1, atol=1e-3, rtol=0
            ):
                raise ValueError(f"{name} must contain unit quaternions")
        if not np.isin(arrays["contact_label"], [-1, 0, 1]).all():
            raise ValueError(
                "contact_label must use -1 (no contact), 0 (unspecified), +1 (contact)"
            )
        flags = {"terminated", "truncated", "clip_end"}
        if flags & arrays.keys():
            if not flags <= arrays.keys():
                raise ValueError(
                    "Rollouts require terminated, truncated and clip_end together"
                )
            for name in flags:
                if arrays[name].shape != (total,) or arrays[name].dtype != np.bool_:
                    raise ValueError(f"{name} must be bool[frames]")
            ended = arrays["terminated"] | arrays["truncated"] | arrays["clip_end"]
            for start, end in zip(self.starts[:-1], self.starts[1:], strict=True):
                if ended[start : end - 1].any():
                    raise ValueError("Rollout contains frames after a terminal state")

    @classmethod
    def load(cls, path):
        # Load and hash one file descriptor, including during atomic replacement.
        with Path(path).open("rb") as stream:
            with np.load(stream, allow_pickle=False) as data:
                if "metadata" not in data.files:
                    raise ValueError(
                        "Expected ef-motion-v1 metadata; import raw NPZ with motion "
                        "import-weave, or pass --layout to the weave command"
                    )
                metadata = json.loads(str(data["metadata"]))
                arrays = {key: data[key] for key in data.files if key != "metadata"}
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        library = cls(arrays, metadata)
        library.sha256 = digest.hexdigest()
        return library

    def save(self, path):
        path = Path(path)
        with path.open("xb") as stream:
            try:
                np.savez_compressed(
                    stream,
                    metadata=np.array(json.dumps(self.metadata, allow_nan=False)),
                    **self.arrays,
                )
            except BaseException:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logging.getLogger(__name__).warning(
                        "Could not remove partial motion snapshot %s: %s", path, exc
                    )
                raise

    def frames(self, clip_ids, steps, offsets=(0,)):
        """Gather [batch, future, ...] without crossing a clip's final frame."""
        ids, times, future = map(np.asarray, (clip_ids, steps, offsets))
        if (
            ids.ndim != 1
            or times.shape != ids.shape
            or future.ndim != 1
            or not len(future)
            or any(v.dtype.kind not in "iu" for v in (ids, times, future))
            or any(
                int(v.max(initial=0)) > np.iinfo(np.int64).max
                for v in (ids, times, future)
            )
            or (ids < 0).any()
            or (ids >= len(self.names)).any()
            or (times < 0).any()
            or (future < 0).any()
        ):
            raise ValueError(
                "Require integer clip_ids/steps[B] and nonnegative offsets[K]"
            )
        ids, times, future = (
            v.astype(np.int64, copy=False) for v in (ids, times, future)
        )
        lengths = self.lengths[ids]
        if (times >= lengths).any():
            raise ValueError("Frame lies outside its clip")
        # Clip before adding so large offsets cannot overflow an integer index.
        local = times[:, None] + np.minimum(
            future[None, :], (lengths - 1 - times)[:, None]
        )
        indices = self.starts[ids, None] + local
        return {key: self.arrays[key][indices] for key in self.frame_fields}

    def sample(self, object_names, rng):
        """Uniform clip and frame sampling within each requested object."""
        requests = {}
        count = 0
        for index, name in enumerate(object_names):
            requests.setdefault(name, []).append(index)
            count += 1
        ids = np.empty(count, dtype=np.int64)
        for name, rows in requests.items():
            candidates = [i for i, obj in enumerate(self.objects) if obj == name]
            if not candidates:
                raise ValueError(f"Unknown object: {name}")
            ids[rows] = rng.choice(candidates, size=len(rows))
        return ids, rng.integers(self.lengths[ids])


def import_weave(path, layout):
    """Read a Weave-style packed NPZ without importing Weave or unpickling data.

    Layout must describe the input array order; names are not inferred from a
    particular robot's SDK. Numeric payloads remain in their original precision.
    """
    if not isinstance(layout, dict) or layout.get("quaternion_order") not in (
        "wxyz",
        "xyzw",
    ):
        raise ValueError("Layout must specify quaternion_order as wxyz or xyzw")
    with Path(path).open("rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files if name != "metadata"}
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if layout["quaternion_order"] == "xyzw":
        for name in ("body_quat_w", "object_quat_w"):
            if name not in arrays:
                raise ValueError(f"Missing motion field: {name}")
            arrays[name] = np.roll(arrays[name], 1, axis=-1)
    metadata = {
        **layout,
        "schema": "ef-motion-v1",
        "quaternion_order": "wxyz",
        "source": {
            "format": "weave-packed-npz",
            "path": str(Path(path).resolve()),
            "sha256": digest.hexdigest(),
        },
    }
    return MotionClips(arrays, metadata)
