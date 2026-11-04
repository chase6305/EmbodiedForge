"""Versioned Go1 setting inheritance for training, evaluation and recovery."""

import math

from embodiedforge.locomotion.go1_config import COMMAND_PROFILES as GO1_COMMAND_PROFILES
from embodiedforge.locomotion.go1_config import REWARD_PROFILES as GO1_REWARD_PROFILES


def resolve_go1_semantics(requested, previous_result=None):
    """New runs use corrected transitions; existing runs retain their contract."""
    inherited = (
        previous_result.get("task_semantics", "upstream-v1")
        if previous_result is not None
        else "transition-v2"
    )
    choices = ("transition-v2", "upstream-v1")
    if inherited not in choices or (requested is not None and requested not in choices):
        raise ValueError("Unknown Go1 task semantics")
    return inherited if requested is None else requested


def resolve_go1_reward_profile(requested, previous_result=None):
    inherited = (previous_result or {}).get("reward_profile", "original")
    choices = GO1_REWARD_PROFILES
    if inherited not in choices or (requested is not None and requested not in choices):
        raise ValueError("Unknown Go1 reward profile")
    return inherited if requested is None else requested


def resolve_go1_command_profile(requested, previous_result=None):
    inherited = (previous_result or {}).get("command_profile", "original")
    if inherited not in GO1_COMMAND_PROFILES or (
        requested is not None and requested not in GO1_COMMAND_PROFILES
    ):
        raise ValueError("Unknown Go1 command profile")
    return inherited if requested is None else requested


def resolve_go1_learning_rate(requested, previous_result=None):
    inherited = (previous_result or {}).get("learning_rate_override")
    for value in (requested, inherited):
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("Go1 learning rate must be positive and finite")
    return inherited if requested is None else requested
