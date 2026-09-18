# Adapted for EmbodiedForge: retain flat-walking configuration only; see NOTICE.

import os

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg

from embodiedforge.locomotion.microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
)

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"

def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))

HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    # vin_range=(6.9, 7.9),
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    # max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)

actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
