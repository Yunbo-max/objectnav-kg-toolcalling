import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONAVGPT_ROOT = REPO_ROOT / "code" / "vendor" / "conavgpt"
HABITAT_ROOT = CONAVGPT_ROOT / "multi-robot-setting"
for path in (CONAVGPT_ROOT, HABITAT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.habitat.multi_agent_env import Multi_Agent_Env


class Category:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class GTSemanticMappingTests(unittest.TestCase):
    def test_tracked_mapping_covers_objectnav_goal_aliases(self):
        mapping_path = (
            REPO_ROOT / "code" / "configs"
            / "matterport_category_mappings.tsv"
        )
        mapping = {}
        for line in mapping_path.read_text().splitlines():
            fields = line.split("    ")
            if len(fields) > 3:
                mapping[fields[2]] = fields[-1]

        self.assertEqual(mapping["tv"], "tv_monitor")
        self.assertEqual(mapping["monitor"], "tv_monitor")
        self.assertEqual(mapping["couch"], "sofa")

    def test_instance_ids_map_without_in_place_id_collisions(self):
        fake_env = SimpleNamespace(
            scene=SimpleNamespace(objects=[
                SimpleNamespace(category=Category("unknown")),
                SimpleNamespace(category=Category("chair")),
                SimpleNamespace(category=Category("bed")),
            ]),
            hm3d_semantic_mapping={},
        )
        # Instance 1 becomes channel 0 while instance 0 remains unknown. An
        # in-place conversion would incorrectly remap the newly written zeros.
        instances = np.array([[1, 0], [2, 1]], dtype=np.int32)
        mapped = Multi_Agent_Env._preprocess_semantic(fake_env, instances)

        np.testing.assert_array_equal(
            mapped,
            np.array([[0, 15], [3, 0]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            instances,
            np.array([[1, 0], [2, 1]], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
