"""Reproducibility helpers shared by experiment entry points."""

import random
from typing import Optional

import numpy as np
import torch


def reset_episode_rng(seed: int, env: Optional[object] = None) -> int:
    """Reset simulator and policy RNGs to one fixed episode seed."""
    episode_seed = int(seed)

    if env is not None:
        env.seed(episode_seed)

    # Seed again after env.seed(): environment implementations may touch the
    # process-global generators while forwarding the seed to the simulator.
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)

    return episode_seed
