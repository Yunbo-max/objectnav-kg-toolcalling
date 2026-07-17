import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from mcoconav_metrics import (
    MCoCoNavMetricTracker,
    build_instance_category_map,
    normalize_category_name,
)


class _Category:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _SemanticObject:
    def __init__(self, object_id, category):
        self.id = object_id
        self.category = _Category(category)


class MCoCoNavMetricTests(unittest.TestCase):
    def test_builds_instance_map_and_normalizes_aliases(self):
        mapping = build_instance_category_map(
            [
                _SemanticObject("object_0", "wall"),
                _SemanticObject("object_7", "television"),
            ]
        )
        self.assertEqual(mapping[7], "tv_monitor")
        self.assertEqual(normalize_category_name("potted plant"), "plant")

    def test_reproduces_path_efficiency_and_team_max(self):
        tracker = MCoCoNavMetricTracker(num_agents=2, map_resolution_cm=5)
        tracker.reset(
            goal_name="chair",
            initial_world_poses=[[0.0, 0.0], [0.0, 0.0]],
            instance_categories={4: "chair"},
        )
        # Local grid scale is 20 cells/m. Robot 0 travels an L-shaped path:
        # path=40 cells, start/end displacement=sqrt(20^2+20^2).
        tracker.observe_agent(
            0, [1.0, 0.0, 0, 0, 100, 0, 100], [100, 100], False,
            np.zeros((2, 2), dtype=np.int32),
        )
        tracker.observe_agent(
            0, [1.0, 1.0, 0, 0, 100, 0, 100], [100, 100], True,
            np.array([[0, 4], [0, 0]], dtype=np.int32),
        )
        tracker.observe_agent(
            1, [0.0, 0.0, 0, 0, 100, 0, 100], [100, 100], False,
            np.zeros((2, 2), dtype=np.int32),
        )

        result = tracker.result()
        expected = np.sqrt(20 ** 2 + 20 ** 2) / (40.0 + 1e-5)
        self.assertAlmostEqual(result["spl"], expected)
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["deciding_robot"], "robot_0")
        self.assertTrue(
            result["agent_metrics"]["robot_0"]["gt_observation_available"]
        )

    def test_preserves_mcoconav_order_dependent_sr(self):
        tracker = MCoCoNavMetricTracker(num_agents=2, map_resolution_cm=5)
        tracker.reset(
            goal_name="sofa",
            initial_world_poses=[[0.0, 0.0], [0.0, 0.0]],
            instance_categories={2: "sofa"},
        )
        pose = [0.0, 0.0, 0, 0, 100, 0, 100]
        tracker.observe_agent(
            0, pose, [100, 100], True,
            np.zeros((2, 2), dtype=np.int32),
        )
        tracker.observe_agent(
            1, pose, [100, 100], True,
            np.array([[2]], dtype=np.int32),
        )

        result = tracker.result()
        self.assertEqual(result["success"], 0.0)
        self.assertEqual(result["any_agent_success"], 1.0)
        self.assertEqual(result["spl"], 1.0)
        self.assertEqual(result["deciding_robot"], "robot_0")


if __name__ == "__main__":
    unittest.main()
