"""One HTTP UI and control queue shared by every frame-rendering adapter.

HTTP threads only enqueue commands or serve immutable published frames. Physics,
camera changes and renderer lifecycle remain on the simulation's main thread.
"""

import io
import json
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from embodiedforge.logging import get_logger

from .frames import (
    FRAME_BACKENDS,
    OrbitCamera,
    create_frame_renderer,
    frame_dependencies,
)

LOGGER = get_logger(__name__)


class WebViewer:
    """Shared-session browser control and latest-frame MJPEG delivery."""

    def __init__(
        self,
        config,
        scene,
        *,
        backend="raster",
        host="127.0.0.1",
        port=8080,
        env_id=0,
        width=960,
        height=540,
        fps=30,
        replay_frames=None,
        camera=None,
        velocity_limits=None,
    ):
        self._validate_resolution(width, height)
        self._validate_fps(fps)
        self.config, self.scene = config, scene
        self.backend, self.selected_env = backend, env_id
        self.width, self.height = width, height
        self.target_fps = fps
        self.next_frame_time = 0.0
        self.reconfigured = False
        self._pending_resolution = None
        self._frame_times = deque(maxlen=61)
        self.closed, self.paused, self.playback_speed = False, False, 1.0
        if replay_frames is not None and (
            len(replay_frames) != config.num_envs
            or any(type(n) is not int or n <= 0 for n in replay_frames)
        ):
            raise ValueError("Replay frame counts must match the recording count")
        self.replay_frames = replay_frames
        if velocity_limits is not None and (
            len(velocity_limits) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in velocity_limits
            )
        ):
            raise ValueError(
                "Velocity limits must contain three positive finite numbers"
            )
        self.velocity_limits = velocity_limits
        self._velocities = {}
        self.follow = replay_frames is not None or velocity_limits is not None
        self._default_camera = camera or OrbitCamera()
        self.camera = self._default_camera
        self._seeks = {}
        self._steps = 0
        self._resets = deque()
        self._commands = deque()
        self._submitted_control_id = 0
        self._processed_control_id = 0
        self._pending_backend = None
        self._stop_requested = False
        self._condition = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._state_id = 0
        self._frame_key = None
        self._frame_shape = None
        self._state = {"ready": False}
        self._error = None
        self._token = secrets.token_urlsafe(32)
        self._availability = {
            name: frame_dependencies(name, scene) for name in FRAME_BACKENDS
        }
        self.renderer = create_frame_renderer(backend, scene, config, width, height)
        self.server = None
        self.thread = None
        try:
            self.server = ThreadingHTTPServer((host, port), self._handler())
            self.server.daemon_threads = True
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                kwargs={"poll_interval": 0.1},
                daemon=True,
            )
            self.thread.start()
            self.port = self.server.server_address[1]
            self.url = (
                f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{self.port}"
            )
            LOGGER.info("Web viewer: %s (renderer=%s)", self.url, backend)
        except BaseException:
            try:
                self.close()
            except Exception:
                LOGGER.exception("Web viewer cleanup also failed during startup")
            raise

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def reply(self, status, content, kind="application/json", extra=None):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if extra:
                    for key, value in extra.items():
                        self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    page = Path(__file__).with_name("web.html").read_text()
                    self.reply(
                        200,
                        page.replace("__SESSION_TOKEN__", owner._token).encode(),
                        "text/html; charset=utf-8",
                    )
                elif path == "/viewer.js":
                    self.reply(
                        200,
                        Path(__file__).with_name("web.js").read_bytes(),
                        "text/javascript; charset=utf-8",
                    )
                elif path == "/api/state":
                    with owner._condition:
                        value = dict(owner._state)
                    self.reply(200, json.dumps(value, allow_nan=False).encode())
                elif path in ("/api/frame", "/snapshot.jpg"):
                    with owner._condition:
                        frame, frame_id = owner._frame, owner._frame_id
                    if frame is None:
                        self.reply(503, b'{"error":"Waiting for first frame"}')
                    else:
                        extra = {"X-Frame-Id": str(frame_id)}
                        if path == "/snapshot.jpg":
                            extra["Content-Disposition"] = (
                                f'attachment; filename="embodiedforge-{frame_id}.jpg"'
                            )
                        self.reply(200, frame, "image/jpeg", extra)
                elif path == "/stream.mjpg":
                    self.stream()
                else:
                    self.reply(404, b'{"error":"Not found"}')

            def stream(self):
                self.connection.settimeout(3)
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                previous = -1
                first = True
                try:
                    while True:
                        with owner._condition:
                            owner._condition.wait_for(
                                lambda previous=previous: (
                                    owner.closed
                                    or (
                                        owner._frame is not None
                                        and owner._frame_id != previous
                                    )
                                ),
                                timeout=2,
                            )
                            if owner.closed:
                                return
                            if owner._frame is None or owner._frame_id == previous:
                                continue
                            frame, previous = owner._frame, owner._frame_id
                        # Chrome needs the following part's complete headers
                        # before displaying the preceding JPEG. Send those now,
                        # even if the next payload is indefinitely paused. MIME
                        # boundaries delimit payloads; omit per-part lengths and
                        # IDs because the next frame may not exist yet.
                        header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        prefix = header if first else b""
                        self.wfile.write(prefix + frame + b"\r\n" + header)
                        first = False
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return

            def do_POST(self):
                if urlsplit(self.path).path != "/api/control":
                    self.reply(404, b'{"error":"Not found"}')
                    return
                if self.headers.get("X-Session-Token") != owner._token:
                    self.reply(403, b'{"error":"Invalid session token"}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Command size must be 1..4096 bytes")
                    self.connection.settimeout(3)
                    command = json.loads(self.rfile.read(length))
                    control_id = owner.submit(command)
                except (ValueError, TypeError, TimeoutError) as error:
                    self.reply(400, json.dumps({"error": str(error)}).encode())
                    return
                except OverflowError as error:
                    self.reply(429, json.dumps({"error": str(error)}).encode())
                    return
                self.reply(
                    202,
                    json.dumps({"accepted": True, "control_id": control_id}).encode(),
                )

        return Handler

    def submit(self, command):
        """Validate completely before enqueueing; callable without any graphics SDK."""
        if not isinstance(command, dict):
            raise ValueError("Expected a command object")
        action = command.get("action")
        fields = {
            "pause": {"paused"},
            "step": set(),
            "reset": {"env_id"},
            "select": {"env_id"},
            "speed": {"value"},
            "camera": {"azimuth", "elevation", "distance"},
            "camera_reset": set(),
            "renderer": {"backend"},
            "stop": set(),
            "seek": {"env_id", "frame"},
            "follow": {"enabled"},
            "velocity": {"env_id", "value"},
            "resolution": {"width", "height"},
            "fps": {"value"},
        }
        if (
            not isinstance(action, str)
            or action not in fields
            or set(command) != {"action", *fields[action]}
        ):
            raise ValueError("Unknown command or unexpected fields")
        if action == "resolution":
            self._validate_resolution(command["width"], command["height"])
        if action == "fps":
            self._validate_fps(command["value"])
        if action == "pause" and type(command["paused"]) is not bool:
            raise ValueError("paused must be boolean")
        if action in ("reset", "select", "seek", "velocity"):
            if (
                type(command["env_id"]) is not int
                or not 0 <= command["env_id"] < self.config.num_envs
            ):
                raise ValueError("Environment index out of range")
        if action == "seek" and self.replay_frames is None:
            raise ValueError("This control requires a recorded motion session")
        if (
            action == "follow"
            and self.replay_frames is None
            and self.velocity_limits is None
        ):
            raise ValueError(
                "This control requires a recorded motion or live policy session"
            )
        if action == "velocity":
            if self.velocity_limits is None:
                raise ValueError("Velocity control requires a live policy session")
            value = command["value"]
            if (
                not isinstance(value, list)
                or len(value) != 3
                or any(
                    type(v) not in (int, float) or not -limit <= v <= limit
                    for v, limit in zip(value, self.velocity_limits, strict=True)
                )
            ):
                raise ValueError(
                    "Velocity command must be finite and within policy limits"
                )
        if action == "seek" and (
            type(command["frame"]) is not int
            or not 0 <= command["frame"] < self.replay_frames[command["env_id"]]
        ):
            raise ValueError("Replay frame index out of range")
        if action == "follow" and type(command["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        bounds = {
            "speed": {"value": (0.1, 4)},
            "camera": {
                "azimuth": (-3600, 3600),
                "elevation": (5, 89),
                "distance": (0.3, 20),
            },
        }
        for key, (low, high) in bounds.get(action, {}).items():
            value = command[key]
            if (
                type(value) not in (int, float)
                or not low <= value <= high
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be finite in {low}..{high}")
        if action == "renderer":
            name = command["backend"]
            if (
                not isinstance(name, str)
                or name not in self._availability
                or not self._availability[name]["metadata_ok"]
                or not self._availability[name].get("supported", True)
            ):
                raise ValueError("Renderer dependencies unavailable")
        with self._condition:
            if self.closed:
                raise ValueError("Viewer is closed")
            # Stop cancels pending work, including renderer changes. It must
            # remain available even when a client has filled the queue.
            if action == "stop":
                self._commands.clear()
            elif len(self._commands) >= 128:
                raise OverflowError("Control queue is full")
            self._submitted_control_id += 1
            owned = dict(command)
            if action == "velocity":
                owned["value"] = list(command["value"])
            owned["_control_id"] = self._submitted_control_id
            self._commands.append(owned)
            return self._submitted_control_id

    def _process_commands(self):
        with self._condition:
            commands = list(self._commands)
            self._commands.clear()
        for command in commands:
            self._processed_control_id = command.pop("_control_id")
            action = command["action"]
            if action == "pause":
                self.paused = command["paused"]
                self._steps = 0
            elif action == "step":
                self.paused = True
                self._steps += 1
            elif action == "reset":
                self._seeks.pop(command["env_id"], None)
                self._velocities.pop(command["env_id"], None)
                # Coalesce repeated requests for the same row until consumed.
                if command["env_id"] not in self._resets:
                    self._resets.append(command["env_id"])
            elif action == "select":
                self.selected_env = command["env_id"]
            elif action == "speed":
                self.playback_speed = command["value"]
            elif action == "camera":
                self.camera = replace(
                    self.camera, **{k: v for k, v in command.items() if k != "action"}
                )
            elif action == "camera_reset":
                self.camera = self._default_camera
            elif action == "seek":
                self.paused, self._steps = True, 0
                self._resets = deque(i for i in self._resets if i != command["env_id"])
                self._seeks[command["env_id"]] = command["frame"]
            elif action == "follow":
                self.follow = command["enabled"]
            elif action == "velocity":
                self._velocities[command["env_id"]] = command["value"]
            elif action == "renderer":
                self._pending_backend = command["backend"]
            elif action == "resolution":
                self._pending_resolution = command["width"], command["height"]
            elif action == "fps":
                self.target_fps = command["value"]
                self._frame_times.clear()
                self.next_frame_time = 0.0
            elif action == "stop":
                self._stop_requested = True

    def is_running(self):
        self._process_commands()
        return not self.closed and not self._stop_requested

    def should_step(self):
        if self._steps:
            self._steps -= 1
            return True
        return not self.paused

    def consume_reset(self):
        return self._resets.popleft() if self._resets else None

    def consume_seek(self):
        if not self._seeks:
            return None
        index = next(iter(self._seeks))
        return index, self._seeks.pop(index)

    def consume_velocity(self):
        if not self._velocities:
            return None
        index = next(iter(self._velocities))
        return index, self._velocities.pop(index)

    @staticmethod
    def _validate_resolution(width, height):
        if any(type(v) is not int or not 64 <= v <= 1920 for v in (width, height)):
            raise ValueError("Resolution dimensions must be integers in 64..1920")

    @staticmethod
    def _validate_fps(value):
        if type(value) is not int or not 1 <= value <= 60:
            raise ValueError("Target FPS must be an integer in 1..60")

    def update_due(self, now):
        return (
            now >= self.next_frame_time
            or self._processed_control_id != self._state.get("control_id", 0)
        )

    def _visual_key(self, scene, observation):
        """Own selected-row values: callers may reuse and mutate input arrays.

        Cache only the two scene contracts supported here. Unknown scene types
        keep rendering normally, so future adapters cannot silently go stale.
        """
        from embodiedforge.core import SceneUpdate

        from .robot import RobotSceneUpdate

        index = self.selected_env

        def row(value):
            array = np.asarray(value[index])
            return array.dtype.str, array.shape, array.tobytes()

        if isinstance(scene, RobotSceneUpdate):
            values = row(scene.qpos), row(scene.time)
        elif isinstance(scene, SceneUpdate):
            values = (
                tuple(
                    (name, row(value)) for name, value in sorted(scene.entities.items())
                ),
                tuple(
                    row(getattr(scene.state, name))
                    for name in ("position", "velocity", "time", "version")
                ),
            )
        else:
            return None
        return (
            self.backend,
            self.width,
            self.height,
            index,
            self.camera,
            values,
            tuple(row(observation[name]) for name in ("episode_id", "step_id", "time")),
        )

    @staticmethod
    def _encode(pixels):
        from PIL import Image

        pixels = np.asarray(pixels)
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or 0 in pixels.shape
        ):
            raise ValueError("Frame renderer must return a nonempty uint8 RGB image")
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def update(self, scene, observation, reward, *, display_state=None):
        if self.closed:
            raise RuntimeError("Viewer is closed")
        started = time.monotonic()
        self.next_frame_time = started + 1 / self.target_fps
        self.reconfigured = False
        index = self.selected_env
        if display_state is None:
            display_state = {
                "position": scene.entities["agent"][index].tolist(),
                "velocity": float(np.linalg.norm(scene.state.velocity[index])),
                "reward": float(reward[index]),
            }
        if self.follow:
            self.camera = replace(self.camera, target=tuple(display_state["position"]))
        pixels = None
        if self._pending_backend is not None or self._pending_resolution is not None:
            name = self._pending_backend or self.backend
            width, height = self._pending_resolution or (self.width, self.height)
            self._pending_backend = self._pending_resolution = None
            if (name, width, height) != (self.backend, self.width, self.height):
                self.reconfigured = True
                self._frame_times.clear()
                candidate = None
                reconfigure_started = time.monotonic()
                LOGGER.info(
                    "Reconfiguring renderer: %s %dx%d -> %s %dx%d "
                    "(control_id=%d); preparing first frame",
                    self.backend,
                    self.width,
                    self.height,
                    name,
                    width,
                    height,
                    self._processed_control_id,
                )
                try:
                    candidate = create_frame_renderer(
                        name, self.scene, self.config, width, height
                    )
                    pixels = candidate.render(
                        scene, observation, self.selected_env, self.camera
                    )
                    frame = self._encode(pixels)
                except Exception as error:
                    if candidate is not None:
                        try:
                            candidate.close()
                        except Exception:
                            LOGGER.exception("Failed to clean up unavailable renderer")
                    pixels = None
                    self._error = f"{name} {width}x{height}: {error}"
                    LOGGER.exception(
                        "Renderer switch failed; retaining %s", self.backend
                    )
                else:
                    previous, self.renderer = self.renderer, candidate
                    self.backend, self._error = name, None
                    self.width, self.height = width, height
                    LOGGER.info(
                        "Renderer ready: %s %dx%d (%.2f s)",
                        name,
                        width,
                        height,
                        time.monotonic() - reconfigure_started,
                    )
                    try:
                        previous.close()
                    except Exception:
                        LOGGER.exception("Previous renderer cleanup failed")
        key = self._visual_key(scene, observation)
        reused = (
            pixels is None
            and self.paused
            and key is not None
            and key == self._frame_key
            and self._frame is not None
        )
        if reused:
            frame = self._frame
        elif pixels is None:
            pixels = self.renderer.render(
                scene, observation, self.selected_env, self.camera
            )
            frame = self._encode(pixels)
        now = time.monotonic()
        if reused:
            self._frame_times.clear()
        else:
            self._frame_times.append(now)
            while len(self._frame_times) > 2 and now - self._frame_times[0] > 1:
                self._frame_times.popleft()
        fps = (
            (len(self._frame_times) - 1) / max(now - self._frame_times[0], 1e-9)
            if len(self._frame_times) > 1
            else 0.0
        )
        if not reused:
            self._frame_key = key
            self._frame_shape = pixels.shape
        index = self.selected_env
        state = {
            "ready": True,
            "control_id": self._processed_control_id,
            "physics": self.config.physics,
            "task": self.config.task,
            "renderer": self.backend,
            "renderers": self._availability,
            "env_id": index,
            "num_envs": self.config.num_envs,
            "paused": self.paused,
            "speed": self.playback_speed,
            "camera": asdict(self.camera),
            "default_camera": asdict(self._default_camera),
            "camera_control": self.renderer.camera_control,
            "episode": int(observation["episode_id"][index]),
            "step": int(observation["step_id"][index]),
            "time": float(observation["time"][index]),
            **display_state,
            "follow": self.follow,
            "fps": fps,
            "target_fps": self.target_fps,
            "render_width": self.width,
            "render_height": self.height,
            "frame_ms": 0.0 if reused else (now - started) * 1000,
            "jpeg_bytes": len(frame),
            "render_idle": reused,
            "width": int(self._frame_shape[1]),
            "height": int(self._frame_shape[0]),
            "error": self._error,
        }
        with self._condition:
            if not reused:
                self._frame_id += 1
            self._state_id += 1
            state["frame_id"] = self._frame_id
            state["state_id"] = self._state_id
            self._frame, self._state = frame, state
            self._condition.notify_all()

    def close(self):
        with self._condition:
            if self.closed:
                return
            self.closed = True
            self._condition.notify_all()
        try:
            if self.server is not None:
                if self.thread is not None and self.thread.is_alive():
                    self.server.shutdown()
                self.server.server_close()
                if self.thread is not None:
                    self.thread.join(timeout=3)
        finally:
            self.renderer.close()
