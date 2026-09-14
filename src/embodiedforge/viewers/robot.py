"""MJCF robot assets and recorded articulation states for shared frame viewers.

MuJoCo owns forward kinematics, never advances physics here. Native viewers reuse
compiled visual mesh vertices and world geom transforms, without joint reordering.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodiedforge.logging import get_logger


@dataclass(frozen=True)
class RobotSceneSpec:
    model_path: str
    kind: str = "robot_motion"


@dataclass(frozen=True)
class RobotSceneUpdate:
    qpos: np.ndarray
    time: np.ndarray


def load_robot_model(path):
    import mujoco

    from .style import add_mujoco_assets, style_mujoco_model

    if tuple(int(v) for v in mujoco.__version__.split(".")[:2]) < (3, 11):
        raise ImportError(
            "Robot replay requires MuJoCo >=3.11; install embodiedforge[viz-robot]"
        )
    if Path(path).suffix == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(Path(path).resolve(strict=True)))
        model.vis.global_.bvactive = 0
        style_mujoco_model(model)
        return model
    spec = mujoco.MjSpec.from_file(str(Path(path).resolve(strict=True)))
    if not any(geom.type == mujoco.mjtGeom.mjGEOM_PLANE for geom in spec.geoms):
        spec.worldbody.add_geom(
            name="embodiedforge_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.1),
            rgba=(1, 1, 1, 1),
        )
    spec.worldbody.add_light(
        pos=(0, -3, 5), dir=(0, 0.5, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    add_mujoco_assets(spec)
    model = spec.compile()
    model.vis.global_.bvactive = 0
    style_mujoco_model(model)
    return model


class RobotFrames:
    """One model with named articulation states, rendered by MuJoCo/GL/OVRTX."""

    camera_control = True

    def __init__(self, backend, scene, config, width, height):
        import mujoco

        self.mujoco = mujoco
        self.model = load_robot_model(scene.model_path)
        self.data = mujoco.MjData(self.model)
        self.backend, self.renderer, self.bridge = backend, None, None
        self.assets = None
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.option.sitegroup[:] = 0
        try:
            if backend == "mujoco":
                self.model.vis.global_.offwidth = width
                self.model.vis.global_.offheight = height
                self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            else:
                from embodiedforge.core import SceneSpec

                from .newton import NewtonViewer

                # Reuse the checked SDK lifecycle and RTX color compatibility,
                # while submitting robot mesh instances instead of point batches.
                self.bridge = NewtonViewer(
                    backend,
                    SceneSpec(),
                    env_id=0,
                    headless=True,
                    width=width,
                    height=height,
                    synchronous=True,
                )
        except BaseException:
            try:
                self.close()
            except Exception:
                get_logger(__name__).exception("Robot renderer cleanup also failed")
            raise

    def _geometry(self, index):
        import newton

        m, types = self.model, self.mujoco.mjtGeom
        kind, size = m.geom_type[index], m.geom_size[index]
        if kind == types.mjGEOM_MESH:
            mesh = m.geom_dataid[index]
            va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            # Compiled vertices already include mesh scale; geom_xmat/xpos also
            # include MuJoCo's mesh centering/alignment transform.
            return m.mesh_vert[va : va + vn].copy(), m.mesh_face[
                fa : fa + fn
            ].ravel().copy()
        options = {"compute_inertia": False}
        if kind == types.mjGEOM_PLANE:
            mesh = newton.Mesh.create_plane(200, 200, **options)
        elif kind == types.mjGEOM_SPHERE:
            mesh = newton.Mesh.create_sphere(size[0], **options)
        elif kind == types.mjGEOM_BOX:
            mesh = newton.Mesh.create_box(*size, **options)
        elif kind == types.mjGEOM_ELLIPSOID:
            mesh = newton.Mesh.create_ellipsoid(*size, **options)
        elif kind in (types.mjGEOM_CAPSULE, types.mjGEOM_CYLINDER):
            factory = (
                newton.Mesh.create_capsule
                if kind == types.mjGEOM_CAPSULE
                else newton.Mesh.create_cylinder
            )
            mesh = factory(size[0], size[1], up_axis=newton.Axis.Z, **options)
        else:
            raise ValueError(
                f"Unsupported visual geom type {kind}; use MuJoCo rendering"
            )
        return mesh.vertices, mesh.indices

    def _register_meshes(self):
        import warp as wp

        viewer, model = self.bridge.viewer, self.model
        groups = {}
        for i in range(model.ngeom):
            if model.geom_group[i] >= 3:
                continue
            material = model.geom_matid[i]
            rgba = model.geom_rgba[i] if material < 0 else model.mat_rgba[material]
            if model.geom_type[i] == self.mujoco.mjtGeom.mjGEOM_PLANE:
                rgba = np.ones(4)
            if rgba[3] == 0:
                continue
            key = (
                int(model.geom_type[i]),
                int(model.geom_dataid[i]),
                tuple(model.geom_size[i]),
                tuple(rgba),
            )
            groups.setdefault(key, []).append(i)
        self.assets = []
        for number, (key, ids) in enumerate(groups.items()):
            rgba = np.asarray(key[-1], dtype=np.float32)
            vertices, indices = self._geometry(ids[0])
            mesh = f"/robot/mesh_{number}"
            floor_options = {}
            if key[0] == self.mujoco.mjtGeom.mjGEOM_PLANE:
                from .style import floor_period, floor_texture

                floor_options = {
                    "uvs": wp.array(
                        vertices[:, :2] / floor_period(model),
                        dtype=wp.vec2,
                        device="cpu",
                    ),
                    "texture": floor_texture(),
                    "roughness": 0.85,
                    "metallic": 0.0,
                }
            viewer.log_mesh(
                mesh,
                wp.array(vertices, dtype=wp.vec3, device="cpu"),
                wp.array(indices, dtype=wp.int32, device="cpu"),
                hidden=True,
                color=tuple(float(x) for x in rgba[:3]),
                backface_culling=False,
                **floor_options,
            )
            self.assets.append(
                (
                    ids,
                    mesh,
                    wp.zeros(len(ids), dtype=wp.transform, device="cpu"),
                    wp.array(np.ones((len(ids), 3)), dtype=wp.vec3, device="cpu"),
                    wp.array(
                        np.tile(rgba[:3], (len(ids), 1)), dtype=wp.vec3, device="cpu"
                    ),
                    wp.array(
                        np.full(len(ids), rgba[3]), dtype=wp.float32, device="cpu"
                    ),
                    wp.array(
                        np.tile(
                            (0.85, 0, 0, 1) if floor_options else (0.65, 0, 0, 0),
                            (len(ids), 1),
                        ),
                        dtype=wp.vec4,
                        device="cpu",
                    ),
                )
            )

    def render(self, scene, observation, env_id, camera):
        self.data.qpos[:] = scene.qpos[env_id]
        self.data.time = scene.time[env_id]
        self.mujoco.mj_forward(self.model, self.data)
        if self.backend == "mujoco":
            self.camera.lookat[:] = camera.target
            self.camera.distance = camera.distance
            self.camera.azimuth = camera.azimuth + 180
            self.camera.elevation = -camera.elevation
            self.renderer.update_scene(
                self.data, camera=self.camera, scene_option=self.option
            )
            return self.renderer.render().copy()
        import warp as wp

        viewer = self.bridge.viewer
        viewer.set_camera(
            wp.vec3(*camera.position()),
            pitch=-camera.elevation,
            yaw=camera.azimuth + 180,
        )
        viewer.begin_frame(float(self.data.time))
        if self.assets is None:
            self._register_meshes()
        for ids, mesh, transforms, scales, colors, opacity, materials in self.assets:
            poses = np.empty((len(ids), 7), dtype=np.float32)
            for row, i in enumerate(ids):
                quaternion = np.empty(4)
                self.mujoco.mju_mat2Quat(quaternion, self.data.geom_xmat[i])
                position = self.data.geom_xpos[i]
                if self.model.geom_type[i] == self.mujoco.mjtGeom.mjGEOM_PLANE:
                    # An infinite plane's display patch follows the camera only
                    # within its plane, including for tilted planes.
                    basis = self.data.geom_xmat[i].reshape(3, 3)
                    delta = basis.T @ (np.asarray(camera.target) - position)
                    # Move by whole texture periods so tiles stay anchored in world space.
                    from .style import floor_period

                    period = floor_period(self.model)
                    delta[:2] = np.floor(delta[:2] / period) * period
                    delta[2] = 0
                    position = position + basis @ delta
                poses[row] = np.r_[position, np.roll(quaternion, -1)]
            transforms.assign(poses)
            viewer.log_instances(
                mesh + "_instances",
                mesh,
                transforms,
                scales,
                colors,
                materials,
                opacities=opacity,
            )
        viewer.end_frame()
        if self.backend == "rtx":
            return viewer._capture_screenshot_pixels()[..., :3].copy()
        return viewer.get_frame().numpy().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
