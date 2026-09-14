"""Display styling preserves robot dynamics and enables native floor textures."""

from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge.viewers.style import add_mujoco_assets, style_mujoco_model

mujoco = pytest.importorskip("mujoco")


def test_styling_preserves_dynamics_and_robot_materials():
    xml = """<mujoco><asset><material name="robot" rgba=".2 .3 .4 1"/></asset>
    <worldbody><geom type="plane" size="0 0 .1"/>
    <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3"
    material="robot" mass="3"/></body></worldbody></mujoco>"""
    original = mujoco.MjModel.from_xml_string(xml)
    spec = mujoco.MjSpec.from_string(xml)
    add_mujoco_assets(spec)
    styled = spec.compile()
    style_mujoco_model(styled)
    for field in ("body_mass", "body_inertia", "jnt_type", "geom_size", "qpos0"):
        np.testing.assert_array_equal(getattr(original, field), getattr(styled, field))
    np.testing.assert_array_equal(original.mat_rgba[0], styled.mat_rgba[0])
    first, second = mujoco.MjData(original), mujoco.MjData(styled)
    first.qpos[0] = second.qpos[0] = 2
    for _ in range(20):
        mujoco.mj_step(original, first)
        mujoco.mj_step(styled, second)
    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.qvel, second.qvel)


def test_compiled_floor_does_not_recolor_shared_robot_texture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><asset>
    <texture name="shared" type="2d" builtin="checker" rgb1="1 0 0"
    rgb2="0 1 0" width="16" height="16"/>
    <material name="floor" texture="shared"/>
    <material name="robot" texture="shared" rgba=".3 .4 .5 1"/>
    </asset><worldbody><geom type="plane" size="0 0 .1" material="floor"/>
    <geom type="sphere" size=".1" pos="0 0 1" material="robot"/>
    </worldbody></mujoco>""")
    texture, material = model.tex_data.copy(), model.mat_rgba[1].copy()
    style_mujoco_model(model)
    np.testing.assert_array_equal(model.tex_data, texture)
    np.testing.assert_array_equal(model.mat_rgba[1], material)


def test_native_instances_enable_floor_texture_and_preserve_robot_color():
    pytest.importorskip("newton")
    from embodiedforge.viewers.robot import RobotFrames

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom type="plane" size="0 0 .1"/><geom type="sphere" size=".1"
    rgba=".2 .3 .4 1" pos="0 0 1"/></worldbody></mujoco>""")
    meshes = []
    renderer = RobotFrames.__new__(RobotFrames)
    renderer.model, renderer.mujoco = model, mujoco
    renderer.bridge = SimpleNamespace(
        viewer=SimpleNamespace(log_mesh=lambda *args, **kw: meshes.append(kw))
    )
    renderer._register_meshes()
    assert meshes[0]["texture"].shape == (128, 128, 3)
    np.testing.assert_allclose(renderer.assets[0][-1].numpy(), [[0.85, 0, 0, 1]])
    np.testing.assert_allclose(renderer.assets[1][-1].numpy(), [[0.65, 0, 0, 0]])
    np.testing.assert_allclose(renderer.assets[1][4].numpy(), [[0.2, 0.3, 0.4]])
