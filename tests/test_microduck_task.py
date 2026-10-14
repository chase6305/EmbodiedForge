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
from embodiedforge.locomotion.microduck.tasks.microduck_velocity_env_cfg import (  # noqa: E402
    make_microduck_velocity_env_cfg,
)


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
