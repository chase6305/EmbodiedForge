"""Pinned recipe identities and source versions, independent of launchers."""

SWITCHING_SUITES = (
    "switching",
    "lateral-switching",
    "maneuver-switching",
    "maneuver-switching-mirrored",
)

SOURCES = {
    "wuji_unilab": "91ccfa0ec8c129b300865bd36c59dc9eed56a744",
    "mjbatch": "b84c0c20aedbdf048122cbc47f554e9b93cc4754",
}


RECIPES = {
    "wuji-reorient": {
        "source": "wuji_unilab",
        "task": "WujiHand_Reorient",
        "methods": ["train", "evaluate"],
        "physics": "mjwarp",
        "algorithm": "PPO",
    },
    "wuji-reorient-light": {
        "source": "wuji_unilab",
        "task": "WujiHand_Reorient_Light",
        "methods": ["train", "evaluate"],
        "physics": "mjwarp",
        "algorithm": "PPO",
    },
    "go1-joystick": {
        "source": "mjbatch",
        "task": "go1_joystick",
        "methods": ["train", "evaluate"],
        "physics": "mjbatch",
        "algorithm": "PPO",
    },
    "cartpole-mpc": {
        "source": "mjbatch",
        "task": "cartpole_mpc",
        "methods": ["solve"],
        "physics": "mjbatch",
        "algorithm": "predictive_sampling",
    },
    "arm-throw-codesign": {
        "source": "mjbatch",
        "task": "arm_throw",
        "methods": ["solve"],
        "physics": "mjbatch",
        "algorithm": "CEM",
    },
}


GO1_STANDALONE_SOURCE = {
    "kind": "installed_packages",
    "implementation": "embodiedforge.locomotion.go1+go1_ppo",
    "contract_version": 1,
}
