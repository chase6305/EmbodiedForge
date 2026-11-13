"""Native H1 policy dimensions and timing, without simulation or learner imports."""

OBS_DIM, ACT_DIM = 69, 19
TIMESTEP, DECIMATION, ACTION_SCALE = 0.005, 4, 0.5
ACTION_LIMIT = 5.0
CTRL_DT = TIMESTEP * DECIMATION
VERSION = "h1-mjbatch-flat-v1"
