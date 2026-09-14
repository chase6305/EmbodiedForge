# SPDX-License-Identifier: Apache-2.0
# Copyright 2026, Kevin Zakka
# Original weights derived from mjbatch/examples/go1_joystick.py at
# b84c0c20aedbdf048122cbc47f554e9b93cc4754; profiles added for EmbodiedForge.

"""Go1 reward and command profiles shared by the SDK-free launcher and native task.

Named profiles are checkpoint contracts: introduce a new name when changing
weights instead of changing the meaning of an existing model's metadata.
"""

# Original reward weights from mjbatch's Apache-2.0 Go1 joystick example.
REWARD = {
    "track": 1.0,
    "turn": 0.5,
    "gait": 2.0,
    "pose": 0.5,
    "orient": -5.0,
    "bounce": -0.5,
    "wobble": -0.05,
    "limits": -1.0,
    "rate": -0.01,
    "land": -1.0,
}
REWARD_PROFILES = {
    "original": REWARD,
    "tracking-v1": {**REWARD, "track": 2.0},
    "tracking-turn-v1": {**REWARD, "track": 2.0, "turn": 1.0},
    "tracking-moderate-v1": {**REWARD, "track": 2.5, "turn": 1.0},
    "tracking-balanced-v1": {**REWARD, "track": 3.0, "turn": 1.0},
    "tracking-strong-v1": {**REWARD, "track": 3.0, "turn": 1.5},
}

# Probabilities apply independently to the three axes when a command is redrawn.
# Keep names stable so resumed runs retain their training distribution.
COMMAND_PROFILES = {
    "original": {"range": (1.5, 0.8, 1.2), "on": (0.9, 0.25, 0.5), "seconds": 5.0},
    "lateral-v1": {"range": (1.5, 0.8, 1.2), "on": (0.9, 0.75, 0.5), "seconds": 5.0},
    "lateral-stop-v1": {
        "range": (1.5, 0.8, 1.2),
        "on": (0.9, 0.75, 0.5),
        "seconds": 5.0,
        "stop_probability": 0.1,
    },
    "lateral-stop-v2": {
        "range": (1.5, 0.8, 1.2),
        "on": (0.9, 0.75, 0.5),
        "seconds": 5.0,
        "stop_probability": 0.2,
    },
    "lateral-fast-v1": {
        "range": (1.5, 0.8, 1.2),
        "on": (0.9, 0.75, 0.5),
        "seconds": 2.0,
        "stop_probability": 0.1,
        "keep": 0.5,
    },
    "lateral-fast-full-v1": {
        "range": (1.5, 0.8, 1.2),
        "on": (0.9, 0.75, 0.5),
        "seconds": 2.0,
        "stop_probability": 0.1,
        "keep": 0.0,
    },
    "lateral-fast-moderate-v1": {
        "range": (0.8, 0.5, 1.0),
        "on": (0.9, 0.75, 0.5),
        "seconds": 2.0,
        "stop_probability": 0.1,
        "keep": 0.0,
    },
}
