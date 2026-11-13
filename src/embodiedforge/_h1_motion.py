"""Recorded rigid-body poses and an offline canvas replay, without physics SDKs."""

import html
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .runtime.artifacts import discard_partial


class MotionRecorder:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.frames = []
        self.ended = False

    def append(
        self,
        time: float,
        positions,
        quaternions,
        joints,
        terminated: bool,
        truncated: bool,
    ):
        if self.ended:
            return
        arrays = [
            np.array(value, dtype=np.float32, copy=True, order="C")
            for value in (positions, quaternions, joints)
        ]
        bodies = len(self.metadata["body_names"])
        if [a.shape for a in arrays] != [
            (bodies, 3),
            (bodies, 4),
            (len(self.metadata["joint_names"]),),
        ]:
            raise ValueError("Motion frame does not match its body/joint metadata")
        if (
            not np.isfinite(time)
            or time <= 0
            or (self.frames and time <= self.frames[-1][0])
        ):
            raise ValueError("Motion timestamps must be finite and increasing")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite motion pose")
        self.frames.append((time, *arrays, bool(terminated), bool(truncated)))
        self.ended = bool(terminated or truncated)

    def save(self, path: Path) -> None:
        """Publish complete frames and clean up only this attempt's temporary file."""
        if not self.frames:
            raise ValueError("Cannot save an empty motion")
        # File handle prevents NumPy from silently adding another extension.
        # String paths avoid retaining unique components in CPython 3.12 pathlib.
        path = os.fspath(path)
        parent, name = os.path.split(path)
        temporary = os.path.join(parent, f"{name}.{uuid4().hex}.tmp")
        created = False
        try:
            with open(temporary, "xb") as stream:
                created = True
                np.savez_compressed(
                    stream,
                    metadata=np.array(json.dumps(self.metadata)),
                    time=np.array([frame[0] for frame in self.frames]),
                    positions=np.stack([frame[1] for frame in self.frames]),
                    quaternions=np.stack([frame[2] for frame in self.frames]),
                    joints=np.stack([frame[3] for frame in self.frames]),
                    terminated=np.array([frame[4] for frame in self.frames]),
                    truncated=np.array([frame[5] for frame in self.frames]),
                )
            os.replace(temporary, path)
        except BaseException:
            if created:
                discard_partial(temporary, label="H1 motion")
            raise


def _read_motion_arrays(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Share archive validation without constructing the JSON replay payload.

    Each required member is read once. Returned arrays remain valid after the
    archive closes; the caller chooses its own representation and copy policy.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        timestamps, positions = archive["time"], archive["positions"]
        frame_count, body_count = len(timestamps), len(metadata["body_names"])
        if (
            not frame_count
            or timestamps.shape != (frame_count,)
            or positions.shape != (frame_count, body_count, 3)
        ):
            raise ValueError("Invalid motion dimensions")
        if (
            not np.isfinite(timestamps).all()
            or not np.isfinite(positions).all()
            or timestamps[0] <= 0
            or (np.diff(timestamps) <= 0).any()
        ):
            raise ValueError("Invalid motion values or timestamps")
        arrays = {"time": timestamps, "positions": positions}
        for field_name, expected_shape in (
            ("quaternions", (frame_count, body_count, 4)),
            ("joints", (frame_count, len(metadata["joint_names"]))),
        ):
            field_values = archive[field_name]
            if (
                field_values.shape != expected_shape
                or not np.isfinite(field_values).all()
            ):
                raise ValueError(f"Invalid motion {field_name}")
            arrays[field_name] = field_values
        for field_name in ("terminated", "truncated"):
            field_values = archive[field_name]
            if (
                field_values.shape != (frame_count,)
                or field_values.dtype != np.bool_
                or field_values[:-1].any()
            ):
                raise ValueError("Motion contains frames after a terminal state")
            arrays[field_name] = field_values
        for edge in metadata["edges"]:
            if len(edge) != 2 or any(
                type(index) is not int or not 0 <= index < body_count for index in edge
            ):
                raise ValueError("Invalid motion body connection")
        if "command_schedule" in metadata:
            previous_end = 0
            schedule = metadata["command_schedule"]
            if not schedule:
                raise ValueError("Empty motion command schedule")
            for segment in schedule:
                command = np.asarray(segment["command"])
                if (
                    type(segment["start_step"]) is not int
                    or type(segment["end_step"]) is not int
                    or segment["start_step"] != previous_end
                    or segment["end_step"] <= previous_end
                    or not isinstance(segment["segment"], str)
                    or command.shape != (3,)
                    or not np.isfinite(command).all()
                ):
                    raise ValueError("Invalid motion command schedule")
                previous_end = segment["end_step"]
            if previous_end < frame_count:
                raise ValueError("Motion extends past its command schedule")
    return metadata, arrays


def load_motion(path: Path) -> dict:
    """Return the validated payload consumed by the offline HTML replay."""
    metadata, arrays = _read_motion_arrays(path)
    # JSON replay omits these fields; release validation-only arrays first.
    del arrays["quaternions"], arrays["joints"]
    return {
        "metadata": metadata,
        "time": arrays["time"].tolist(),
        "positions": np.round(arrays["positions"].astype(float), 5).tolist(),
        "terminated": bool(arrays["terminated"][-1]),
        "truncated": bool(arrays["truncated"][-1]),
    }


def render_motion(path: Path, output: Path) -> None:
    motion = load_motion(path)
    # Escape HTML delimiters in embedded JSON; labels are inserted via textContent.
    payload = (
        json.dumps(motion, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    prefix, suffix = HTML.split("__MOTION_DATA__")
    title = html.escape(motion["metadata"].get("title", "机器人动作回放"))
    page = prefix.replace("__TITLE__", title) + payload + suffix
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page)


HTML = r"""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{margin:0;background:#101723;color:#e4eaf4;font:15px system-ui}main{max-width:1100px;margin:24px auto;padding:0 20px}
h1{font-size:24px;margin:0 0 8px}p{color:#a8b8cd;line-height:1.6}canvas{width:100%;height:65vh;min-height:320px;background:#141f2f;border-radius:12px;touch-action:none}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:16px 0}button,select{background:#263750;color:white;border:1px solid #476287;border-radius:5px;padding:8px}input[type=range]{flex:1;min-width:180px}#info{font-variant-numeric:tabular-nums}small{color:#a8b8cd}
</style><main><h1>__TITLE__</h1><p id="title"></p>
<canvas id="canvas" aria-label="机器人骨架回放"></canvas>
<div class="controls"><button id="play">播放</button><input id="frame" type="range" min="0" value="0" aria-label="时间轴">
<select id="speed" aria-label="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<select id="view" aria-label="视角"><option value="oblique">斜视</option><option value="side">侧视</option><option value="front">正视</option><option value="top">俯视</option></select>
<label><input id="follow" type="checkbox" checked>跟随</label><span id="info"></span></div>
<small>拖动画布旋转，滚轮缩放。蓝色为左侧，橙色为右侧。线段连接模型的刚体原点及记录的末端点。回放使用文件中的时间戳，并显示已有的终止标记。</small></main>
<script id="motion" type="application/json">__MOTION_DATA__</script><script>
'use strict';
const motion = JSON.parse(document.getElementById('motion').textContent),
  metadata = motion.metadata;
const canvas = document.getElementById('canvas'),
  context = canvas.getContext('2d'),
  slider = document.getElementById('frame'),
  play = document.getElementById('play');
const speed = document.getElementById('speed'),
  follow = document.getElementById('follow'),
  view = document.getElementById('view'),
  info = document.getElementById('info');
const labels = [metadata.case ?? metadata.title ?? '动作回放'];
if (metadata.seed !== undefined) labels.push(`seed ${metadata.seed}`);
if (metadata.env_id !== undefined) labels.push(`环境 ${metadata.env_id}`);
if (metadata.source_kind === 'retargeted_reference') labels.push('重定向参考动作');
else if (metadata.command_schedule) labels.push('分段指令');
else if (Array.isArray(metadata.velocity_command))
  labels.push('指令 ' + metadata.velocity_command.join(' / '));
labels.push(`${motion.time.length} 帧`);
document.getElementById('title').textContent = labels.join(' · ');
slider.max = motion.time.length - 1;
let index = 0,
  playing = false,
  previousTickTime = null,
  elapsed = 0,
  yaw = -0.8,
  elevation = 0.28,
  scale = metadata.view_scale ?? 145;

function project(p, origin) {
  let x = p[0] - origin[0],
    y = p[1] - origin[1],
    z = p[2] - origin[2],
    u = x * Math.cos(yaw) - y * Math.sin(yaw),
    depth = x * Math.sin(yaw) + y * Math.cos(yaw);
  return [
    canvas.width / 2 + scale * u,
    canvas.height * .55 - scale * (z * Math.cos(elevation) - depth * Math.sin(elevation))
  ];
}

function line(a, b, color, width = 2) {
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(...a);
  context.lineTo(...b);
  context.stroke();
}

function draw() {
  let box = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(box.width) || canvas.height !== Math.round(box.height)) {
    canvas.width = Math.round(box.width);
    canvas.height = Math.round(box.height);
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  const pts = motion.positions[index],
    root = pts[0],
    initial = motion.positions[0][0],
    height = metadata.view_height ?? .9,
    origin = follow.checked
      ? [root[0], root[1], height]
      : [initial[0], initial[1], height];
  const gx = Math.floor(origin[0]), gy = Math.floor(origin[1]);
  for (let k = -8; k <= 8; k++) {
    line(project([gx + k, gy - 8, 0], origin), project([gx + k, gy + 8, 0], origin), '#28394f', 1);
    line(project([gx - 8, gy + k, 0], origin), project([gx + 8, gy + k, 0], origin), '#28394f', 1);
  }
  for (let j = Math.max(1, index - 500); j <= index; j++) {
    const a = motion.positions[j - 1][0], b = motion.positions[j][0];
    line(project([a[0], a[1], .01], origin), project([b[0], b[1], .01], origin), '#5d7e65', 2);
  }
  for (const [a, b] of metadata.edges) {
    const name = metadata.body_names[b],
      color = /left|^FL|^RL/.test(name)
        ? '#61c4ff'
        : /right|^FR|^RR/.test(name) ? '#ffb777' : '#ecf3ff';
    line(project(pts[a], origin), project(pts[b], origin), color, 5);
  }
  for (let k = 0; k < pts.length; k++) {
    let p = project(pts[k], origin);
    context.fillStyle = '#e3efff';
    context.beginPath();
    context.arc(p[0], p[1], 3, 0, Math.PI * 2);
    context.fill();
  }
  const segment = metadata.command_schedule?.find(s => index >= s.start_step && index < s.end_step);
  slider.value = index;
  info.textContent = `${motion.time[index].toFixed(2)} / ${motion.time.at(-1).toFixed(2)} s · ${segment?segment.segment+' · 指令 '+segment.command.join(' / ')+' · ':''}${index===motion.time.length-1?(motion.terminated?'终止':motion.truncated?'截断':'记录结束'):''}`;
}

function tick(now) {
  if (playing) {
    if (previousTickTime !== null) elapsed += (now - previousTickTime) / 1000 * Number(speed.value);
    while (index < motion.time.length - 1 && motion.time[index + 1] - motion.time[index] <= elapsed) {
      elapsed -= motion.time[index + 1] - motion.time[index];
      index++;
    }
    if (index === motion.time.length - 1) {
      playing = false;
      play.textContent = '播放';
    }
  }
  previousTickTime = now;
  draw();
  requestAnimationFrame(tick);
}

play.onclick = () => {
  if (index === motion.time.length - 1) {
    index = 0;
    elapsed = 0;
  }
  playing = !playing;
  previousTickTime = null;
  play.textContent = playing ? '暂停' : '播放';
};
slider.oninput = () => {
  index = Number(slider.value);
  elapsed = 0;
};
view.onchange = () => {
  [yaw, elevation] = ({
    oblique: [-.8, .28],
    side: [0, 0],
    front: [Math.PI / 2, 0],
    top: [0, Math.PI / 2]
  })[view.value];
};
let drag = null;
canvas.onpointerdown = e => {
  drag = [e.clientX, e.clientY];
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointerup = () => drag = null;
canvas.onpointermove = e => {
  if (drag) {
    yaw += (e.clientX - drag[0]) * .008;
    elevation = Math.max(-1.4, Math.min(1.57, elevation + (e.clientY - drag[1]) * .008));
    drag = [e.clientX, e.clientY];
  }
};
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  scale = Math.max(25, Math.min(600, scale * Math.exp(-e.deltaY * .001)));
}, {passive: false});
requestAnimationFrame(tick);
</script></html>"""
