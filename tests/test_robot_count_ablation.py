import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from brain import HelicaseBrain
from brain_robot_ablation import RobotCountAblationBrain
from kg_construction import CurrentFrontierView, KGUpdater, KnowledgeGraph
from robot_count_runtime import (
    build_history_record,
    build_probe_fields,
    configure_habitat_agents,
    fuse_agent_maps,
    initialize_actions,
    validate_ablation_agent_count,
)


class CloneableNamespace(SimpleNamespace):
    def clone(self):
        return deepcopy(self)


class FakeBrain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call(self, prompt, max_tokens=300):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def frontier(index, row, column):
    return {
        "idx": index,
        "centroid": (row, column),
        "area": 10 + index,
        "nearby_objects": [],
        "room_type": "unknown",
        "room_confidence": 0.0,
        "target_prior": 0.1,
    }


def room_id(candidate):
    row, column = candidate["centroid"]
    return f"room_{int(row // 50)}_{int(column // 50)}"


def probability_response(enriched):
    rooms = sorted({room_id(candidate) for candidate in enriched})
    return json.dumps({
        "tool_calls": [{
            "name": "estimate_room_probability",
            "arguments": {
                "room_id": room,
                "target": "bed",
                "probability": 0.5,
                "confidence": 0.7,
                "evidence_ids": [room],
                "reason": "current room evidence",
            },
        } for room in rooms],
    })


def assignment_response(enriched, num_agents):
    by_index = {
        candidate["idx"]: room_id(candidate) for candidate in enriched
    }
    return json.dumps({
        "tool_calls": [{
            "name": "assign_frontiers",
            "arguments": {
                "assignments": {
                    f"robot_{index}": {
                        "frontier_id": index,
                        "room_id": by_index[index],
                        "reason": "distinct legal frontier",
                    }
                    for index in range(num_agents)
                },
                "diversity": True,
            },
        }],
    })


class RobotCountRuntimeTests(unittest.TestCase):
    def test_launcher_counts_are_strict(self):
        self.assertEqual(validate_ablation_agent_count(1), 1)
        self.assertEqual(validate_ablation_agent_count(3), 3)
        with self.assertRaises(ValueError):
            validate_ablation_agent_count(2)

    def test_habitat_configuration_for_one_and_three(self):
        template = CloneableNamespace(
            HEIGHT=0.88,
            RADIUS=0.18,
            SENSORS=["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"],
        )
        for count in (1, 3):
            config = SimpleNamespace(SIMULATOR=CloneableNamespace(
                NUM_AGENTS=2,
                AGENTS=["AGENT_0", "AGENT_1"],
                AGENT_0=template.clone(),
                AGENT_1=template.clone(),
            ))
            configure_habitat_agents(config, count)
            self.assertEqual(config.SIMULATOR.NUM_AGENTS, count)
            self.assertEqual(
                config.SIMULATOR.AGENTS,
                [f"AGENT_{index}" for index in range(count)],
            )
            if count == 3:
                self.assertEqual(
                    config.SIMULATOR.AGENT_2.SENSORS, template.SENSORS
                )
                self.assertIsNot(
                    config.SIMULATOR.AGENT_2, config.SIMULATOR.AGENT_0
                )

    def test_dynamic_map_fusion_and_two_agent_golden_equivalence(self):
        maps = [
            torch.tensor([[1.0, 0.0], [0.0, float(index)]])
            for index in range(3)
        ]
        self.assertTrue(torch.equal(fuse_agent_maps(maps[:1]), maps[0]))
        expected_three = torch.stack(maps).max(dim=0).values
        self.assertTrue(torch.equal(fuse_agent_maps(maps), expected_three))

        legacy_two = torch.max(torch.cat((
            maps[0].unsqueeze(0), maps[1].unsqueeze(0)
        ), 0), 0).values
        compatibility_two = torch.stack(maps[:2]).max(dim=0).values
        self.assertTrue(torch.equal(legacy_two, compatibility_two))

    def test_dynamic_actions_history_and_probe(self):
        self.assertEqual(initialize_actions(1), [0])
        self.assertEqual(initialize_actions(3), [0, 0, 0])
        assignments = {"robot_0": 0, "robot_1": 1, "robot_2": 2}
        history = build_history_record(25, assignments, ["bed", "chair"], 3)
        self.assertEqual(history["assignments"], assignments)
        self.assertEqual(history["objects_found"], "bed,chair")
        probe = build_probe_fields(assignments, 3)
        self.assertEqual(probe["chosen_frontiers"], [0, 1, 2])
        self.assertEqual(probe["duplicate_frontier_count"], 0)


class RobotCountBrainTests(unittest.TestCase):
    def run_decision(self, num_agents):
        enriched = [
            frontier(index, 60 + 60 * index, 60 + 60 * index)
            for index in range(num_agents)
        ]
        poses = [
            (80 + 60 * index, 400 - 60 * index, 0)
            for index in range(num_agents)
        ]
        view = CurrentFrontierView.from_enriched(enriched)
        kg = KnowledgeGraph()
        KGUpdater(kg).update(enriched, {}, poses, None)
        transport = FakeBrain([
            probability_response(enriched),
            assignment_response(enriched, num_agents),
        ])
        brain = RobotCountAblationBrain(
            transport, num_agents=num_agents, decision_history_enabled=True
        )
        assignments, _, _ = brain.decide(
            kg,
            target_name="bed",
            enriched_frontiers=enriched,
            pose_pred=poses,
            step=25,
            max_steps=500,
            decision_history=[],
            current_frontiers=view,
        )
        return assignments, transport.prompts

    def test_one_robot_prompt_and_assignment(self):
        assignments, prompts = self.run_decision(1)
        self.assertEqual(assignments, {"robot_0": 0})
        self.assertNotIn('"robot_1"', prompts[1])
        self.assertIn("Cross-robot diversity is not applicable", prompts[1])

    def test_three_robot_prompt_and_assignment(self):
        assignments, prompts = self.run_decision(3)
        self.assertEqual(
            assignments, {"robot_0": 0, "robot_1": 1, "robot_2": 2}
        )
        self.assertIn('"robot_2"', prompts[1])
        self.assertNotIn("send the other robot", prompts[1])
        self.assertNotIn("Prefer two frontiers", prompts[1])

    def test_original_two_robot_class_is_not_replaced(self):
        self.assertIsNot(RobotCountAblationBrain, HelicaseBrain)
        with self.assertRaises(ValueError):
            RobotCountAblationBrain(FakeBrain([]), num_agents=2)


if __name__ == "__main__":
    unittest.main()
