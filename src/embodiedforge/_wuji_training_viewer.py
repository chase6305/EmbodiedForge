"""Opt-in, read-only preview of Wuji's actual training rollout, environment 0.

Rendering stays on the training thread; HTTP threads only read encoded images.
No renderer, HTTP service, or optional imports are created in headless training.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic

PAGE = b"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Wuji training preview</title>
<style>body{background:#17202c;color:#eef;font:16px system-ui;margin:32px}
img{max-width:100%;border-radius:12px}p{max-width:800px}</style>
<h1>Wuji training preview</h1>
<p>Live training rollout &middot; environment 0 &middot; exploration enabled.
The policy changes during training. Use evaluation to measure success.</p>
<img id="frame" width="640" height="480" alt="Waiting for a training frame">
<p id="status">Waiting for the first rollout step...</p>
<p>Closing this page does not stop training. The server stops when training ends.</p>
<script>
async function refresh(){try{
 const r=await fetch('/frame.jpg',{cache:'no-store'});
 if(r.ok){const url=URL.createObjectURL(await r.blob());
 const img=document.getElementById('frame');const old=img.src;img.src=url;
 if(old.startsWith('blob:'))URL.revokeObjectURL(old);
 document.getElementById('status').textContent='Rollout step '+r.headers.get('X-Rollout-Step');
 }else{document.getElementById('status').textContent='Waiting for a training frame...';}
 }catch(e){document.getElementById('status').textContent='Training ended or connection lost.';}
 setTimeout(refresh,100);}
refresh();</script></html>"""


class TrainingPreview:
    title = "Wuji training preview"

    def __init__(self, port=8083, fps=10):
        self.period = 1 / fps
        self.next_frame = 0.0
        self.steps = 0
        self.frame = (b"", 0)
        self.renderer = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                frame, step = owner.frame
                if self.path == "/":
                    code, body, kind = (
                        200,
                        PAGE.replace(b"Wuji training preview", owner.title.encode()),
                        "text/html; charset=utf-8",
                    )
                elif self.path == "/frame.jpg":
                    code, body, kind = (200 if frame else 503), frame, "image/jpeg"
                else:
                    code, body, kind = 404, b"Not found", "text/plain"
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Rollout-Step", str(step))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _initialize_renderer(self, env):
        import mujoco
        from unisim.backend.mujoco.playback import resolve_render_play_model_files

        with TemporaryDirectory(prefix="wuji-training-preview-") as directory:
            files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=directory)
            path = files if isinstance(files, str) else files[0]
            self.model = (
                mujoco.MjModel.from_binary_path(str(path))
                if Path(path).suffix == ".mjb"
                else mujoco.MjModel.from_xml_path(str(path))
            )
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 0.65
        self.camera.azimuth = 135
        self.camera.elevation = -30
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def update(self, env):
        self.steps += 1
        if monotonic() < self.next_frame:
            return
        import mujoco
        import numpy as np
        from PIL import Image

        if self.renderer is None:
            self._initialize_renderer(env)
        # UniSim's public snapshot contract includes mocap poses when present.
        row = self.snapshot(env)
        model, data = self.model, self.data
        base = 1 + model.nq + model.nv
        if row.shape not in ((base,), (base + 7 * model.nmocap,)):
            raise ValueError("Unexpected Wuji physics snapshot shape")
        if not np.isfinite(row).all():
            raise ValueError("Non-finite Wuji preview snapshot")
        data.time = row[0]
        data.qpos[:] = row[1 : 1 + model.nq]
        data.qvel[:] = row[1 + model.nq : base]
        if model.nmocap and row.size > base:
            data.mocap_pos[:] = row[base : base + 3 * model.nmocap].reshape(-1, 3)
            data.mocap_quat[:] = row[base + 3 * model.nmocap :].reshape(-1, 4)
        mujoco.mj_forward(model, data)
        self.aim_camera()
        self.renderer.update_scene(data, camera=self.camera)
        output = BytesIO()
        Image.fromarray(self.renderer.render()).save(output, format="JPEG", quality=85)
        self.frame = (output.getvalue(), self.steps)
        self.next_frame = monotonic() + self.period

    def snapshot(self, env):
        import numpy as np

        return np.asarray(env.get_physics_state_snapshot())[0]

    def aim_camera(self):
        import mujoco
        import numpy as np

        palm = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link"
        )
        if palm >= 0:
            self.camera.lookat[:] = self.data.xpos[palm] + np.array([0.0, 0.0, 0.06])

    def close(self):
        try:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
        finally:
            if self.renderer is not None:
                self.renderer.close()


@contextmanager
def training_preview(request):
    from wuji_unilab.rl.runtime import WujiWrapper

    # The upstream resolver returns this class directly. Restore even on errors.
    inherited = "step" not in WujiWrapper.__dict__
    original = WujiWrapper.step
    preview = TrainingPreview(request["viewer_port"], request["viewer_fps"])

    def step(wrapper, actions):
        result = original(wrapper, actions)
        preview.update(wrapper.env)
        return result

    WujiWrapper.step = step
    try:
        yield preview
    finally:
        if inherited:
            del WujiWrapper.step
        else:
            WujiWrapper.step = original
        preview.close()
