import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from reproducibility import reset_episode_rng


class DummyEnv:
    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)
        # Exercise the helper's promise to restore process-global RNGs even
        # when an environment touches them while forwarding its seed.
        random.random()
        np.random.rand()
        torch.rand(1)


class EpisodeSeedTests(unittest.TestCase):
    def test_fixed_global_seed_reproduces_all_rng_streams(self):
        env = DummyEnv()

        reset_episode_rng(1, env=env)
        first = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

        reset_episode_rng(1, env=env)
        second = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )

        self.assertEqual(first, second)
        self.assertEqual(env.seeds, [1, 1])


if __name__ == "__main__":
    unittest.main()
