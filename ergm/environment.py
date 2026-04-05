"""
Mock environment for testing the ERGM loop without a real physics engine.

Produces low-dimensional raw observations that should be passed through
ToolAdapter to obtain targets in R^LATENT_DIM.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_initial_state(seed: int | None = None) -> npt.NDArray[np.float64]:
    """
    Sample an initial physical-style state (e.g. position + velocity slice).

    Returns a 1D float array of shape (raw_dim,) for use with ToolAdapter.
    Here raw_dim is fixed to 8 for the mock (4 pos + 4 vel-like channels).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(8).astype(np.float64)


def simulate_action_and_observe(
    s_raw: npt.NDArray[np.float64],
    action_raw: npt.NDArray[np.float64],
    dt: float = 0.1,
) -> npt.NDArray[np.float64]:
    """
    One mock integration step: linear drift + action as force, with light noise.

    Args:
        s_raw:  Current raw state vector (length 8 in this mock).
        action_raw: Action vector (same length as s_raw for simplicity).
        dt:     Step size.

    Returns:
        Next raw observation vector to feed into ToolAdapter as ground truth.
    """
    s = np.asarray(s_raw, dtype=np.float64).ravel()
    a = np.asarray(action_raw, dtype=np.float64).ravel()
    if a.size != s.size:
        raise ValueError("action_raw must match s_raw length in mock env")
    noise = 0.01 * np.random.default_rng().standard_normal(s.shape)
    return s + dt * a + noise
