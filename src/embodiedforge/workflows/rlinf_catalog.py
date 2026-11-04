"""Pinned RLinf SAC recipe identity without launcher or SDK imports."""

REVISION = "67864d67c086a111de9921c5f4591eb5882f74ec"
CONFIG = "maniskill_sac_mlp"
WEIGHTS = "actor/model_state_dict/full_weights.pt"
REPLAY_WINDOW = 10_000  # Fixed maniskill_sac_mlp cache and sampling window.
