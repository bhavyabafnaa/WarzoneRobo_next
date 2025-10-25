from typing import Sequence
import os

import numpy as np
import torch


def save_model(policy, path, icm=None, rnd=None):
    """Save policy and optional exploration modules."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {"policy": policy.state_dict()}
    if icm is not None:
        checkpoint["icm"] = icm.state_dict()
    if rnd is not None:
        checkpoint["rnd"] = rnd.state_dict()
    torch.save(checkpoint, path)


def get_checkpoint_path(method: str, seed: int, best: bool = False) -> str:
    """Return the standardized checkpoint path for ``method`` and ``seed``."""
    fname = "best.pt" if best else "final.pt"
    return os.path.join("checkpoints", method, f"seed{seed}", fname)


def load_model(
        policy_class,
        input_dim,
        action_dim,
        path,
        icm_class=None,
        rnd_class=None,
        device=None):
    """Load policy and exploration modules from a checkpoint path.

    Paths now follow the pattern ``checkpoints/<method>/seed<k>/final.pt`` or
    ``best.pt``.
    """
    checkpoint = torch.load(path, map_location=device)
    policy = policy_class(input_dim, action_dim)
    policy.load_state_dict(checkpoint["policy"])
    icm = None
    rnd = None
    if icm_class is not None and "icm" in checkpoint:
        icm = icm_class(input_dim, action_dim)
        icm.load_state_dict(checkpoint["icm"])
    if rnd_class is not None and "rnd" in checkpoint:
        rnd = rnd_class(input_dim)
        rnd.load_state_dict(checkpoint["rnd"])
    return policy, icm, rnd


def count_intrinsic_spikes(
        values: Sequence[float],
        threshold_factor: float = 1.5) -> int:
    """Return the number of intrinsic reward spikes.

    A spike is any value exceeding ``threshold_factor`` times the mean of
    the sequence. Empty sequences yield zero spikes.
    """

    if not values:
        return 0

    arr = np.asarray(list(values), dtype=float)
    threshold = threshold_factor * arr.mean()
    return int(np.sum(arr > threshold))
