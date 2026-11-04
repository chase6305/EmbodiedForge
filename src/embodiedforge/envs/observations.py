"""Merge reset rows without mutating a retained terminal observation."""

import numpy as np

from embodiedforge.core import Array, Observation


def replace_reset_rows(
    observation: Observation, selected: Observation, ids: Array
) -> Observation:
    """Replace selected rows using the terminal arrays' assignment semantics.

    IDs must be unique and in range; selected rows follow their order.
    Callers validate their own environment's reset schema. A complete reset can
    reuse already-owned plain arrays when their layouts exactly match; otherwise
    copying preserves terminal dtypes, subclasses and NumPy broadcasting.
    """
    if (
        len(ids) == len(next(iter(observation.values())))
        and np.array_equal(ids, np.arange(len(ids)))
        and tuple(observation) == tuple(selected)
        and all(
            type(value) is np.ndarray
            and type(selected[key]) is np.ndarray
            and selected[key].flags.c_contiguous
            and value.shape == selected[key].shape
            and value.dtype is selected[key].dtype
            and value.dtype.kind in "biuf"
            and value.dtype.metadata is None
            for key, value in observation.items()
        )
    ):
        # flatnonzero returns canonical row order, and reset already
        # owns its arrays. Matching base-array layouts preserve the
        # replacement path's C order and custom array copy behavior.
        # dtype equality ignores metadata and native byteorder spelling.
        # Share only plain numeric dtype objects; copy custom layouts.
        return selected
    else:
        observation = {key: value.copy() for key, value in observation.items()}
        for key in observation:
            observation[key][ids] = selected[key]
    return observation
