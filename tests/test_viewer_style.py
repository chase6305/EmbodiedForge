"""Display styling preserves robot dynamics and enables native floor textures."""

from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge.viewers.style import (
    add_mujoco_assets,
    floor_period,
    style_mujoco_model,
    texture_pixels,
)

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


@pytest.mark.parametrize("binary", [False, True])
def test_existing_sky_keeps_gradient_in_xml_and_compiled_models(tmp_path, binary):
    from embodiedforge.viewers.robot import load_robot_model

    xml = """<mujoco><visual><headlight active="0"/></visual><asset>
    <texture name="original_sky" type="skybox" builtin="gradient"
    rgb1="1 0 0" rgb2="0 1 0" width="32" height="32"/>
    </asset><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>"""
    path = tmp_path / ("scene.mjb" if binary else "scene.xml")
    if binary:
        mujoco.mj_saveModel(mujoco.MjModel.from_xml_string(xml), str(path))
    else:
        path.write_text(xml)
    model = load_robot_model(path)
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "original_sky")
    sky = texture_pixels(model, index)
    # A renamed upstream sky used to become a single flat color after styling.
    assert np.ptp(sky[..., 0]) > 20
    assert (sky[..., 2] >= sky[..., 0]).all()
    assert model.vis.headlight.active == 1


@pytest.mark.parametrize("extent,period", [(0.8, 1.0), (2.0, 2.0)])
def test_floor_scale_matches_robot_size_without_changing_geometry(extent, period):
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
    <geom type="plane" size="0 0 .1"/>
    <body pos="0 0 1"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>""")
    add_mujoco_assets(spec)
    model = spec.compile()
    model.stat.extent = extent
    before = model.geom_size.copy()
    style_mujoco_model(model)
    assert floor_period(model) == period
    np.testing.assert_allclose(model.mat_texrepeat[model.geom_matid[0]], 2 / period)
    np.testing.assert_array_equal(model.geom_size, before)
