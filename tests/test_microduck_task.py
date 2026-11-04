"""Local task helpers and anatomical randomization selection in the real SDK."""

import re
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from embodiedforge.locomotion.microduck.robot.microduck_constants import (  # noqa: E402
    MICRODUCK_WALK_XML,
)
from embodiedforge.locomotion.microduck.tasks import mdp  # noqa: E402

_ACTION_HISTORY = [
    (
        "leg_action_rate_l2",
        list(range(5)) + list(range(9, 14)),
        "_prev_leg_actions",
        None,
    ),
    ("neck_action_rate_l2", list(range(5, 9)), "_prev_neck_actions", None),
    (
        "leg_action_acceleration_l2",
        list(range(5)) + list(range(9, 14)),
        "_prev_leg_actions_for_acc",
        "_prev_prev_leg_actions_for_acc",
    ),
    (
        "neck_action_acceleration_l2",
        list(range(5, 9)),
        "_prev_neck_actions_for_acc",
        "_prev_prev_neck_actions_for_acc",
    ),
]


def _history_actions(dtype, layout, offset=0):
    value = torch.arange(5 * 14, dtype=dtype).reshape(5, 14) / 100 + offset
    if layout == "transpose":
        value = value.T.contiguous().T
    elif layout == "strided":
        padded = torch.zeros(5, 28, dtype=dtype)
        padded[:, ::2] = value
        value = padded[:, ::2]
    return value


@pytest.mark.parametrize("name,columns,primary,secondary", _ACTION_HISTORY)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_action_history_owns_inputs_and_keeps_partial_resets_independent(
    name, columns, primary, secondary, dtype, layout
):
    action = _history_actions(dtype, layout)
    initial = action[:, columns].clone()
    env = SimpleNamespace(
        num_envs=5,
        device="cpu",
        action_manager=SimpleNamespace(action=action),
        scene={"robot": SimpleNamespace(data=SimpleNamespace())},
    )
    function = getattr(mdp, name)
    torch.testing.assert_close(function(env), torch.zeros(5))
    old_primary = getattr(env, primary)
    action.fill_(-99)
    torch.testing.assert_close(old_primary, initial)
    if secondary:
        assert getattr(env, secondary).data_ptr() != old_primary.data_ptr()
        torch.testing.assert_close(getattr(env, secondary), initial)

    action = _history_actions(dtype, layout, 0.25)
    env.action_manager.action = action
    current = action[:, columns].clone()
    torch.testing.assert_close(function(env), ((current - initial) ** 2).sum(1))
    old_primary.fill_(99)
    torch.testing.assert_close(getattr(env, primary), current)
    if secondary:
        torch.testing.assert_close(getattr(env, secondary), initial)

    env.action_manager.action = _history_actions(dtype, layout, 0.75)
    reset_values = env.action_manager.action[:, columns].clone()
    ids = torch.tensor([3, 4, 1, 0])[::2]
    mdp.reset_action_history(env, ids)
    expected_primary = current.clone()
    expected_primary[ids] = reset_values[ids]
    torch.testing.assert_close(getattr(env, primary), expected_primary)
    if secondary:
        expected_secondary = initial.clone()
        expected_secondary[ids] = reset_values[ids]
        torch.testing.assert_close(getattr(env, secondary), expected_secondary)
    env.action_manager.action.fill_(-99)
    torch.testing.assert_close(getattr(env, primary), expected_primary)
    if secondary:
        torch.testing.assert_close(getattr(env, secondary), expected_secondary)


@pytest.mark.parametrize("name,columns,primary,secondary", _ACTION_HISTORY)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_action_history_preserves_values_and_gradients_across_three_steps(
    name, columns, primary, secondary, layout
):
    actions = [
        _history_actions(torch.float64, layout, offset).detach().requires_grad_()
        for offset in (0, 0.25, -0.125)
    ]
    env = SimpleNamespace(num_envs=5, device="cpu", action_manager=SimpleNamespace())
    results = []
    for action in actions:
        env.action_manager.action = action
        results.append(getattr(mdp, name)(env))
    first, second, third = (value[:, columns] for value in actions)
    expected = ((second - first) ** 2).sum(1)
    expected += (
        (third - 2 * second + first) ** 2 if secondary else (third - second) ** 2
    ).sum(1)
    actual = results[1] + results[2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.sum(), actions, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), actions)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )


def test_reset_actuator_history_copies_selected_values_into_owned_storage():
    source = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)[:, ::2]
    cached = torch.full_like(source, -1)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(actuator_force=source))},
        _prev_actuator_forces=cached,
    )
    ids = torch.tensor([3, 4, 1, 0])[::2]
    mdp.reset_action_history(env, ids)
    expected = torch.full_like(source, -1)
    expected[ids] = source[ids]
    source.fill_(99)
    assert env._prev_actuator_forces is cached
    torch.testing.assert_close(cached, expected)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_warmstart_joint_writes_do_not_modify_source_storage(layout):
    positions = _history_actions(torch.float32, layout)
    velocities = _history_actions(torch.float32, layout, 0.5)
    originals = positions.clone(), velocities.clone()
    qpos = torch.zeros(5, 21)
    qpos[:, 3] = 1
    written = []

    def write_joint_state(position, velocity, env_ids):
        torch.testing.assert_close(position, originals[0][env_ids])
        torch.testing.assert_close(velocity[:, 1:13], originals[1][env_ids, 1:13])
        torch.testing.assert_close(velocity[:, 0], velocity[:, 13])
        written.append(env_ids.clone())
        position.fill_(-99)
        velocity.fill_(-99)

    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=positions,
            joint_vel=velocities,
            data=SimpleNamespace(qpos=qpos),
            indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
        ),
        find_joints=lambda _: ([0, 13], ["passive_left", "passive_right"]),
        write_root_link_velocity_to_sim=lambda *a, **kw: None,
        write_joint_state_to_sim=write_joint_state,
    )
    env = SimpleNamespace(scene={"robot": asset}, device="cpu", common_step_counter=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(312)
        mdp.reset_with_forward_velocity(env, torch.tensor([3, 4, 1, 0])[::2])
    assert len(written) == 1
    torch.testing.assert_close(positions, originals[0], rtol=0, atol=0)
    torch.testing.assert_close(velocities, originals[1], rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
def test_crouch_reset_keeps_defaults_and_unselected_simulator_rows(layout):
    defaults = _history_actions(torch.float32, layout)
    original_defaults = defaults.clone()
    names = list(mdp._CROUCH_ANCHOR_BY_NAME) + [f"servo_{i}" for i in range(8)]

    def find_joints(pattern):
        ids = [i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
        return ids, [names[i] for i in ids]

    asset = SimpleNamespace(
        data=SimpleNamespace(default_joint_pos=defaults), find_joints=find_joints
    )
    qpos, qvel = torch.zeros(5, 21), torch.ones(5, 20)
    env = SimpleNamespace(
        scene={"robot": asset},
        device="cpu",
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel)),
    )
    ids = torch.tensor([3, 4, 1, 0])[::2]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(312)
        mdp.set_random_crouch_state(env, ids)
    torch.testing.assert_close(defaults, original_defaults, rtol=0, atol=0)
    torch.testing.assert_close(qpos[[0, 2, 4]], torch.zeros(3, 21))
    torch.testing.assert_close(qvel[[0, 2, 4]], torch.ones(3, 20))
    torch.testing.assert_close(qvel[ids], torch.zeros(2, 20))
    torch.testing.assert_close(qpos[ids, 3:7].norm(dim=1), torch.ones(2))
    assert not torch.equal(qpos[ids, 7:], defaults[ids])


from embodiedforge.locomotion.microduck.tasks.microduck_velocity_env_cfg import (  # noqa: E402
    make_microduck_velocity_env_cfg,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relative_height_terrain_offsets_nonfinite_values_and_gradients(dtype):
    positions = torch.tensor(
        [
            [1, 2, 0.5],
            [1, 2, -0.5],
            [1, 2, float("nan")],
            [1, 2, float("inf")],
            [1, 2, -float("inf")],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    origins = torch.tensor(
        [[0, 0, 0.25], [0, 0, -0.75], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=dtype,
        requires_grad=True,
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=SimpleNamespace(env_origins=origins))
    )
    asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions))
    height = mdp._relative_root_height(env, asset)
    limit = torch.finfo(dtype).max
    torch.testing.assert_close(
        height,
        torch.tensor([0.25, 0.25, 0, limit, -limit], dtype=dtype),
        rtol=0,
        atol=0,
    )
    position_grad, origin_grad = torch.autograd.grad(height.sum(), (positions, origins))
    expected_grad = torch.zeros_like(positions)
    expected_grad[:2, 2] = 1
    torch.testing.assert_close(position_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(origin_grad, -expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("overrides", [None, {}, {0: 0.75, -1: -0.25}])
def test_pose_override_keeps_default_storage_and_unmodified_joint_gradients(
    dtype, overrides
):
    default = (
        torch.arange(24, dtype=dtype).reshape(4, 6)[:, ::2].detach().requires_grad_()
    )
    original = default.detach().clone()
    pose = mdp._pose_override(default, overrides)
    expected = original.clone()
    expected_gradient = torch.ones_like(default)
    if overrides:
        expected[:, 0] = 0.75
        expected[:, -1] = -0.25
        expected_gradient[:, [0, -1]] = 0
    torch.testing.assert_close(pose, expected, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(pose.sum(), default)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    with torch.no_grad():
        pose.fill_(-99)
    torch.testing.assert_close(default, original, rtol=0, atol=0)


def test_pose_override_preserves_tensor_override_gradient():
    default = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    value = torch.tensor(0.5, dtype=default.dtype, requires_grad=True)
    pose = mdp._pose_override(default, {2: value})
    default_gradient, value_gradient = torch.autograd.grad(pose.sum(), (default, value))
    torch.testing.assert_close(value_gradient, torch.tensor(3.0, dtype=default.dtype))
    assert torch.count_nonzero(default_gradient[:, 2]) == 0
    assert torch.count_nonzero(default.detach()) == 0


@pytest.mark.parametrize("blend", [mdp.crouch_pose_blend, mdp.phase_pose_blend])
def test_pose_phase_boundaries_values_and_gradients(blend):
    phase = torch.tensor(
        [0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    # A strided phase view exercises the same batched helper used by rewards.
    result = blend(phase.reshape(2, 4).T, 0.2, 0.5, 0.8)
    expected = torch.tensor([0, 0.5, 1, 1, 1, 0.5, 0, 0], dtype=phase.dtype)
    torch.testing.assert_close(result.T.flatten(), expected)
    (gradient,) = torch.autograd.grad(result.sum(), phase)
    torch.testing.assert_close(
        gradient,
        torch.tensor([5, 5, 0, 0, -10 / 3, -10 / 3, 0, 0], dtype=phase.dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reset_quaternion_axes_and_gradients(dtype):
    angles = torch.zeros((4, 3), dtype=dtype)
    angles[1:, :] = torch.eye(3, dtype=dtype) * (torch.pi / 2)
    roll, pitch, yaw = angles.unbind(dim=1)  # Noncontiguous batch columns.
    quat = mdp._quat_from_roll_pitch(
        roll, pitch, torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    )
    half = 2**-0.5
    expected = torch.tensor(
        [[1, 0, 0, 0], [half, half, 0, 0], [half, 0, half, 0], [half, 0, 0, half]],
        dtype=dtype,
    )
    torch.testing.assert_close(quat, expected)
    if dtype == torch.float64:
        mixed = torch.tensor([0.2, -0.6, 1.1], dtype=dtype, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda a: mdp._quat_from_roll_pitch(
                a[0:1], a[1:2], torch.cos(a[2:3] * 0.5), torch.sin(a[2:3] * 0.5)
            ),
            (mixed,),
        )


@pytest.mark.parametrize("reward", [mdp.phase_pose_track, mdp.phase_pose_track_l1])
def test_phase_pose_uses_named_joints_and_preserves_default_pose(reward):
    positions = torch.tensor([[0.3, 99.0, 0.2], [-0.2, 77.0, 1.0]])
    defaults = torch.tensor([[0.1, 0.0, 0.2], [0.4, 0.0, 0.6]])
    original = defaults.clone()
    indices = {"left": 0, "passive": 1, "right": 2}
    asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=positions, default_joint_pos=defaults),
        find_joints=lambda names: ([indices[n] for n in names], names),
    )
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": asset},
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        ),
    )
    result = reward(
        env, target_pose={"right": 1.0, "left": -0.2}, source_pose={"left": 0.3}
    )
    expected = torch.ones(2) if reward is mdp.phase_pose_track else torch.zeros(2)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(defaults, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="non-empty target_pose"):
        reward(env)


def test_head_randomization_selects_only_head_subtree():
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    cfg = make_microduck_velocity_env_cfg()
    selection = cfg.events["randomize_head_com"].params["asset_cfg"].body_names
    neck = model.body("neck").id
    selected = set()
    for pattern in selection:
        matches = {
            i for i in range(model.nbody) if re.fullmatch(pattern, model.body(i).name)
        }
        assert matches, f"Head body pattern matches nothing: {pattern}"
        selected.update(matches)
    assert len(selected) == 4
    for body in selected:
        ancestor = body
        while ancestor and ancestor != neck:
            ancestor = int(model.body_parentid[ancestor])
        assert ancestor == neck, f"{model.body(body).name} is outside the head subtree"
    assert model.body("bearing_roll").id not in selected


def test_pose_match_preserves_effective_positional_and_keyword_interface():
    positions = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.25, 0.5]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=positions,
            default_joint_pos=torch.zeros_like(positions),
        ),
        find_joints=lambda pattern: ([0, 1, 2], ["left", "middle", "right"]),
    )
    env = SimpleNamespace(scene={"robot": asset})
    overrides = {2: 0.75}
    entity = mdp.SceneEntityCfg("robot")
    expected = torch.exp(
        -(((positions[:, [1, 2]] - torch.tensor([0, 0.75])) / 0.2) ** 2)
    ).mean(-1)
    torch.testing.assert_close(
        mdp.pose_target_match(env, overrides, entity, 0.2, [1, 2]), expected
    )
    torch.testing.assert_close(
        mdp.pose_target_match(
            env,
            target_overrides=overrides,
            asset_cfg=entity,
            std=0.2,
            joint_indices=[1, 2],
        ),
        expected,
    )


@pytest.fixture
def finite_robot_state():
    root_pose = torch.zeros(4, 7)
    root_pose[:, 3] = 1
    data = SimpleNamespace(
        joint_pos=torch.zeros(4, 14),
        joint_vel=torch.zeros(4, 14),
        root_link_pose_w=root_pose,
        root_link_pos_w=root_pose[:, :3],
        root_link_quat_w=root_pose[:, 3:7],
        root_link_lin_vel_w=torch.zeros(4, 3),
        root_link_ang_vel_w=torch.zeros(4, 3),
    )
    force = torch.zeros(4, 2, 3)

    class Scene(dict):
        pass

    scene = Scene(robot=SimpleNamespace(data=data))
    scene.sensors = {"feet": SimpleNamespace(data=SimpleNamespace(force=force))}
    return SimpleNamespace(scene=scene), data, force


@pytest.mark.parametrize(
    "field",
    [
        "joint_pos",
        "joint_vel",
        "root_link_pos_w",
        "root_link_quat_w",
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
        "force",
    ],
)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_termination_flags_only_affected_rows_without_cleaning_state(
    finite_robot_state, field, nonfinite
):
    env, data, force = finite_robot_state
    target = force if field == "force" else getattr(data, field)
    target[1].flatten()[0] = nonfinite
    before = {name: value.clone() for name, value in vars(data).items()}
    force_before = force.clone()
    mask = mdp.robot_state_is_nan(env, sensor_names=("feet", "absent"))
    torch.testing.assert_close(mask, torch.tensor([False, True, False, False]))
    for name, expected in before.items():
        torch.testing.assert_close(
            getattr(data, name), expected, rtol=0, atol=0, equal_nan=True
        )
    torch.testing.assert_close(force, force_before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "sensor_case", ["missing", "missing_force", "none", "unrequested", "extra"]
)
def test_nonfinite_termination_ignores_unavailable_or_unrequested_forces(
    finite_robot_state, sensor_case
):
    env, _, force = finite_robot_state
    sensor_names = ("feet",)
    if sensor_case == "missing":
        env.scene.sensors.clear()
    elif sensor_case == "missing_force":
        env.scene.sensors["feet"].data = SimpleNamespace()
    elif sensor_case == "none":
        env.scene.sensors["feet"].data.force = None
    elif sensor_case == "unrequested":
        force.fill_(float("nan"))
        sensor_names = ()
    else:
        env.scene.sensors["other"] = SimpleNamespace(
            data=SimpleNamespace(force=torch.full_like(force, float("nan")))
        )
    mask = mdp.robot_state_is_nan(env, sensor_names=sensor_names)
    torch.testing.assert_close(mask, torch.zeros(4, dtype=torch.bool))
