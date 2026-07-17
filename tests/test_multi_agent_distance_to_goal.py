import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HABITAT_ROOT = (
    REPO_ROOT
    / "code"
    / "vendor"
    / "conavgpt"
    / "multi-robot-setting"
)
if str(HABITAT_ROOT) not in sys.path:
    sys.path.insert(0, str(HABITAT_ROOT))

from habitat.tasks.nav.nav import DistanceToGoal


class FakeMultiAgentSim:
    def __init__(self, distances):
        self.habitat_config = SimpleNamespace(NUM_AGENTS=len(distances))
        self.positions = [
            np.array([distance, 0.0, 0.0], dtype=np.float32)
            for distance in distances
        ]
        self.geodesic_calls = 0

    def get_agent_state(self, agent_id):
        return SimpleNamespace(position=self.positions[agent_id])

    def geodesic_distance(self, position, targets, episode):
        self.geodesic_calls += 1
        return float(position[0])


class MultiAgentDistanceToGoalTests(unittest.TestCase):
    def test_stationary_agent_remains_in_team_minimum(self):
        sim = FakeMultiAgentSim([0.15, 0.40])
        config = SimpleNamespace(DISTANCE_TO="POINT")
        episode = SimpleNamespace(
            goals=[SimpleNamespace(position=np.zeros(3, dtype=np.float32))]
        )
        measure = DistanceToGoal(sim=sim, config=config)

        measure.reset_metric(episode=episode)
        self.assertAlmostEqual(measure.get_metric(), 0.15, places=6)
        self.assertEqual(sim.geodesic_calls, 2)

        # Robot 0 is already inside the success radius and stays still while
        # robot 1 moves. The shared metric must retain robot 0's cached value.
        sim.positions[1] = np.array([0.35, 0.0, 0.0], dtype=np.float32)
        measure.update_metric(episode=episode)

        self.assertAlmostEqual(measure.get_metric(), 0.15, places=6)
        self.assertEqual(sim.geodesic_calls, 3)

        # Once robot 0 moves, only its cache entry is refreshed and the team
        # minimum is recomputed over both current cached distances.
        sim.positions[0] = np.array([0.25, 0.0, 0.0], dtype=np.float32)
        measure.update_metric(episode=episode)

        self.assertAlmostEqual(measure.get_metric(), 0.25, places=6)
        self.assertEqual(sim.geodesic_calls, 4)


if __name__ == "__main__":
    unittest.main()
