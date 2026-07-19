import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from brain import HelicaseBrain, KGUpdater as BrainKGUpdater
from kg_construction import (
    CurrentFrontierView,
    KGEdge,
    KGNode,
    KGUpdater,
    KnowledgeGraph,
    TRANSIENT_ROOM_PROPERTIES,
)


class FakeBrain:
    def __init__(self, responses):
        self.responses = (
            list(responses) if isinstance(responses, list) else [responses]
        )
        self.prompts = []
        self.call_count = 0

    def call(self, prompt, max_tokens=300):
        self.prompts.append(prompt)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[index]


def frontier(idx, y, x, prior=0.1, area=10, room_type="unknown"):
    return {
        "idx": idx,
        "centroid": (y, x),
        "area": area,
        "nearby_objects": [],
        "room_type": room_type,
        "room_confidence": 0.0,
        "target_prior": prior,
    }


def room_id_for(candidate):
    y, x = candidate["centroid"]
    return f"room_{int(y // 50)}_{int(x // 50)}"


def query_response(room_ids):
    return json.dumps({
        "tool_calls": [{
            "name": "query_room_objects",
            "arguments": {"room_id": room_id},
        } for room_id in room_ids]
    })


def probability_response(room_probabilities, target="bed",
                         evidence_by_room=None):
    evidence_by_room = evidence_by_room or {}
    return json.dumps({
        "tool_calls": [{
            "name": "estimate_room_probability",
            "arguments": {
                "room_id": room_id,
                "target": target,
                "probability": probability,
                "confidence": 0.7,
                "evidence_ids": evidence_by_room.get(room_id, [room_id]),
                "reason": f"KG evidence for {room_id}",
            },
        } for room_id, probability in room_probabilities.items()]
    })


def assignment_response(assignments, enriched):
    by_id = {candidate["idx"]: room_id_for(candidate) for candidate in enriched}
    return json.dumps({
        "tool_calls": [{
            "name": "assign_frontiers",
            "arguments": {
                "assignments": {
                    robot_id: {
                        "frontier_id": frontier_id,
                        "room_id": by_id.get(frontier_id, "room_99_99"),
                        "reason": f"LLM selected frontier {frontier_id}",
                    }
                    for robot_id, frontier_id in assignments.items()
                },
                "diversity": True,
            },
        }]
    })


def valid_pipeline(enriched, assignments=None, probabilities=None):
    room_ids = sorted({room_id_for(candidate) for candidate in enriched})
    if probabilities is None:
        probabilities = {
            room_id: round(0.8 - 0.2 * index, 2)
            for index, room_id in enumerate(room_ids)
        }
    if assignments is None:
        frontier_ids = [candidate["idx"] for candidate in enriched]
        assignments = {
            "robot_0": frontier_ids[0],
            "robot_1": frontier_ids[min(1, len(frontier_ids) - 1)],
        }
    return [
        probability_response(probabilities),
        assignment_response(assignments, enriched),
    ]


def decision_packet_from_prompt(prompt):
    marker = "CURRENT_DECISION_PACKET:\n"
    return json.loads(prompt.split(marker, 1)[1])


class CurrentFrontierViewTests(unittest.TestCase):
    def test_runtime_robot_pose_is_converted_to_map_coordinates(self):
        self.assertEqual(
            CurrentFrontierView.robot_pose_to_map_rc((120, 80, 0)),
            (400.0, 120.0),
        )

    def test_preserves_multiple_frontiers_in_one_room(self):
        view = CurrentFrontierView.from_enriched([
            frontier(0, 110, 110),
            frontier(1, 120, 130),
            frontier(2, 210, 210),
        ])
        self.assertEqual(view.room_to_frontiers["room_2_2"], (0, 1))
        self.assertEqual(set(view.frontiers), {0, 1, 2})

        next_view = CurrentFrontierView.from_enriched([
            frontier(0, 310, 310),
        ])
        self.assertEqual(next_view.room_ids, {"room_6_6"})
        self.assertNotIn("room_2_2", next_view.room_to_frontiers)

    def test_deterministic_fallback_is_current_and_distinct(self):
        view = CurrentFrontierView.from_enriched([
            frontier(0, 110, 110),
            frontier(1, 210, 210),
        ])
        assignments = view.deterministic_assignments(
            [(100, 100, 0), (220, 220, 0)],
            num_agents=2,
        )
        self.assertEqual(set(assignments), {"robot_0", "robot_1"})
        self.assertEqual(set(assignments.values()), {0, 1})

    def test_fallback_reuses_only_when_frontiers_are_fewer_than_robots(self):
        view = CurrentFrontierView.from_enriched([
            frontier(7, 110, 110),
        ])
        assignments = view.deterministic_assignments(
            [(100, 100, 0), (200, 200, 0)],
            num_agents=2,
        )
        self.assertEqual(assignments, {"robot_0": 7, "robot_1": 7})


class PersistentKnowledgeGraphTests(unittest.TestCase):
    def test_brain_module_exports_canonical_updater(self):
        self.assertIs(BrainKGUpdater, KGUpdater)

    def test_update_removes_legacy_transient_room_properties(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        kg.add_node(KGNode(
            id="room_2_2",
            node_type="room",
            name="unknown",
            certainty=0.1,
            properties={
                "frontier_idx": 99,
                "frontier_indices": [98, 99],
                "size": 40,
                "active_frontier": True,
                "target_prior": 0.9,
                "explored": False,
            },
        ))
        updater.update(
            [frontier(0, 110, 110)],
            object_list={},
            pose_pred=[(300, 300, 0), (350, 350, 0)],
            wall_list=None,
        )
        for room in kg.get_nodes_by_type("room"):
            self.assertFalse(TRANSIENT_ROOM_PROPERTIES & set(room.properties))

    def test_prompt_separates_current_rooms_from_history(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        poses = [(400, 400, 0), (420, 420, 0)]
        updater.update([frontier(0, 110, 110)], {}, poses, None)

        current = [frontier(0, 210, 210, prior=0.7)]
        view = updater.build_current_frontier_view(current)
        updater.update(current, {}, poses, None)
        prompt_text = kg.to_text(current_frontier_view=view)

        current_section, historical_section = prompt_text.split(
            "HISTORICAL ROOM CONTEXT — NOT ASSIGNABLE"
        )
        self.assertIn("CURRENT ASSIGNABLE ROOMS", current_section)
        self.assertIn("room_4_4", current_section)
        self.assertNotIn("\n  room_2_2", current_section)
        self.assertIn("room_2_2", historical_section)
        self.assertNotIn("frontier_idx", prompt_text)
        self.assertNotIn("prior=", prompt_text)

    def test_json_prompt_separates_current_rooms_from_history(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        poses = [(400, 400, 0), (420, 420, 0)]
        updater.update([frontier(0, 110, 110)], {}, poses, None)

        current = [frontier(0, 210, 210, prior=0.7)]
        view = updater.build_current_frontier_view(current)
        updater.update(current, {}, poses, None)
        payload = json.loads(kg.to_json(current_frontier_view=view))

        current_ids = {
            room["room_id"] for room in payload["current_assignable_rooms"]
        }
        historical_ids = {
            room["room_id"]
            for room in payload[
                "historical_room_context_not_assignable"
            ]
        }
        self.assertEqual(current_ids, {"room_4_4"})
        self.assertIn("room_2_2", historical_ids)
        self.assertNotIn("room_2_2", current_ids)
        self.assertNotIn("target_prior", payload)

    def test_json_serialization_preserves_display_precision(self):
        kg = KnowledgeGraph()
        kg.add_node(KGNode(
            id="room_2_2",
            node_type="room",
            name="bedroom",
            certainty=0.846,
            properties={"explored": False, "observation_count": 3},
        ))
        view = CurrentFrontierView.from_enriched([
            frontier(0, 110.4, 110.6, area=10.5),
        ])
        payload = json.loads(kg.to_json(current_frontier_view=view))
        room = payload["current_assignable_rooms"][0]

        self.assertEqual(room["type_certainty_percent"], 85)
        self.assertEqual(room["current_frontiers"][0]["centroid_rc"], [110, 111])

    def test_triples_are_three_field_facts_with_context_roles(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        poses = [(400, 400, 0), (420, 420, 0)]
        updater.update([frontier(0, 110, 110)], {}, poses, None)

        current = [frontier(0, 210, 210, prior=0.7)]
        view = updater.build_current_frontier_view(current)
        updater.update(current, {}, poses, None)
        triples = json.loads(kg.to_triples(current_frontier_view=view))

        self.assertTrue(all(len(triple) == 3 for triple in triples))
        self.assertIn(
            ["kg", "serialization_format", "mindnav_kg_triples_v1"],
            triples,
        )
        self.assertIn(
            ["room_4_4", "context_role", "current_assignable"],
            triples,
        )
        self.assertIn(
            ["room_2_2", "context_role", "historical_not_assignable"],
            triples,
        )
        self.assertNotIn("target_prior", json.dumps(triples))

    def test_repeated_room_snapshot_is_idempotent(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        enriched = [frontier(0, 110, 110, room_type="bedroom")]
        enriched[0]["nearby_objects"] = ["bed"]
        enriched[0]["room_confidence"] = 0.5

        updater.update(enriched, {}, [(100, 100, 0)], None)
        first = dict(kg.nodes["room_2_2"].properties)
        updater.update(enriched, {}, [(100, 100, 0)], None)
        second = dict(kg.nodes["room_2_2"].properties)

        self.assertEqual(first["observation_count"], second["observation_count"])
        self.assertEqual(
            first["room_evidence_signatures"],
            second["room_evidence_signatures"],
        )

    def test_observed_room_hint_can_change_with_new_evidence(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        first = [frontier(0, 110, 110, room_type="bedroom")]
        first[0]["nearby_objects"] = ["bed"]
        first[0]["room_confidence"] = 0.5
        updater.update(first, {}, [(100, 100, 0)], None)
        self.assertEqual(kg.nodes["room_2_2"].name, "bedroom")

        second = [frontier(0, 110, 110, room_type="bathroom")]
        second[0]["nearby_objects"] = ["toilet"]
        second[0]["room_confidence"] = 0.5
        updater.update(second, {}, [(100, 100, 0)], None)
        self.assertEqual(kg.nodes["room_2_2"].name, "bathroom")

    def test_object_contour_uses_opencv_xy_to_map_rc(self):
        contour = np.asarray([
            [[100, 200]], [[120, 200]], [[120, 220]], [[100, 220]],
        ])
        self.assertEqual(
            KGUpdater._object_component_position(contour),
            (210.0, 110.0),
        )

    def test_unique_object_detections_use_noisy_or(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        first_contour = np.asarray([
            [[100, 200]], [[120, 200]], [[120, 220]], [[100, 220]],
        ])
        shifted_contour = first_contour + np.asarray([[[8, 8]]])
        enriched = [frontier(0, 110, 110)]

        updater.update(
            enriched, {"chair": [first_contour]}, [(100, 100, 0)], None
        )
        obj = kg.get_nodes_by_type("object")[0]
        self.assertAlmostEqual(obj.certainty, 0.65)

        updater.update(
            enriched, {"chair": [first_contour]}, [(100, 100, 0)], None
        )
        self.assertAlmostEqual(obj.certainty, 0.65)
        self.assertEqual(obj.properties["observation_count"], 1)

        updater.update(
            enriched, {"chair": [shifted_contour]}, [(100, 100, 0)], None
        )
        self.assertAlmostEqual(obj.certainty, 1.0 - 0.35 * 0.35)
        self.assertEqual(obj.properties["observation_count"], 2)
        self.assertAlmostEqual(obj.position[0], 212.4)
        self.assertAlmostEqual(obj.position[1], 112.4)

    def test_dynamic_robot_and_topology_edges_are_rebuilt(self):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        candidates = [
            frontier(0, 110, 110),
            frontier(1, 110, 210),
        ]
        wall = np.asarray([[[175, 50, 175, 200]]])
        updater.update(candidates, {}, [(100, 100, 0)], wall)
        updater.update(candidates, {}, [(150, 150, 0)], wall)

        robot_in_edges = [
            edge for edge in kg.edges
            if edge.source == "robot_0" and edge.relation == "in"
        ]
        self.assertEqual(len(robot_in_edges), 1)
        relations = {
            edge.relation for edge in kg.edges
            if {edge.source, edge.target} == {"room_2_2", "room_2_4"}
        }
        self.assertIn("separated_by_wall", relations)
        self.assertNotIn("connected_to", relations)


class HelicaseToolCallingTests(unittest.TestCase):
    def make_runtime(self, enriched, responses):
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        poses = [(100, 100, 0), (220, 220, 0)]
        current_view = updater.build_current_frontier_view(enriched)
        updater.update(enriched, {}, poses, None)
        fake_brain = FakeBrain(responses)
        helicase = HelicaseBrain(fake_brain, num_agents=2)
        return kg, poses, current_view, fake_brain, helicase

    def decide(self, enriched, responses):
        kg, poses, view, fake_brain, helicase = self.make_runtime(
            enriched, responses
        )
        assignments, tools, method = helicase.decide(
            kg,
            target_name="bed",
            enriched_frontiers=enriched,
            pose_pred=poses,
            step=25,
            max_steps=500,
            decision_history=[],
            current_frontiers=view,
        )
        return assignments, view, fake_brain, helicase, tools, method, kg

    def test_history_off_omits_all_history_derived_packet_fields(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        kg = KnowledgeGraph()
        updater = KGUpdater(kg)
        poses = [(100, 100, 0), (220, 220, 0)]
        view = updater.build_current_frontier_view(enriched)
        updater.update(enriched, {}, poses, None)
        fake_brain = FakeBrain(valid_pipeline(enriched))
        helicase = HelicaseBrain(
            fake_brain,
            num_agents=2,
            decision_history_enabled=False,
        )

        helicase.decide(
            kg,
            target_name="bed",
            enriched_frontiers=enriched,
            pose_pred=poses,
            step=25,
            max_steps=500,
            decision_history=[{"should": "not leak"}],
            current_frontiers=view,
        )

        packet = decision_packet_from_prompt(fake_brain.prompts[1])
        self.assertNotIn("recent_assignments", packet)
        self.assertTrue(all(
            "recent_history_summary" not in room
            for room in packet["current_rooms"].values()
        ))
        self.assertNotIn("COVERAGE AND HISTORY RULES", fake_brain.prompts[1])
        self.assertFalse(
            helicase.last_decision_audit["decision_history_enabled"]
        )
        self.assertEqual(
            helicase.last_decision_audit["recent_assignment_history"], []
        )
        self.assertEqual(
            helicase.last_decision_audit["room_history_summary"], {}
        )
        self.assertEqual(helicase._recent_assignment_history, [])

    def test_equivalent_qwen_tool_syntax_is_canonicalized(self):
        canonical_list = HelicaseBrain._load_json(json.dumps([{
            "name": "estimate_room_probability",
            "arguments": {
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": [],
                "reason": "unknown room",
            },
        }]), expected_tool_name="estimate_room_probability")
        self.assertEqual(
            canonical_list["tool_calls"][0]["name"],
            "estimate_room_probability",
        )

        compact = HelicaseBrain._load_json(json.dumps([{
            "tool_call": "estimate_room_probability",
            "room_id": "room_2_2",
            "target": "bed",
        }]))
        self.assertEqual(compact, {
            "tool_calls": [{
                "name": "estimate_room_probability",
                "arguments": {"room_id": "room_2_2", "target": "bed"},
            }]
        })

        args_alias = HelicaseBrain._load_json(json.dumps({
            "tool_calls": [{
                "name": "assign_frontiers",
                "args": {"assignments": {}, "diversity": True},
            }]
        }))
        self.assertIn("arguments", args_alias["tool_calls"][0])
        self.assertNotIn("args", args_alias["tool_calls"][0])

        fenced = HelicaseBrain._load_json(
            "```json\n" + json.dumps({
                "tool_calls": [{
                    "name": "query_room_objects",
                    "arguments": {"room_id": "room_2_2"},
                }]
            }) + "\n```",
            expected_tool_name="query_room_objects",
        )
        self.assertEqual(
            fenced["tool_calls"][0]["arguments"]["room_id"],
            "room_2_2",
        )

        stage_local = HelicaseBrain._load_json(json.dumps([{
            "tool_call": {
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": [],
                "reason": "unknown room",
            }
        }]), expected_tool_name="estimate_room_probability")
        self.assertEqual(
            stage_local["tool_calls"][0]["name"],
            "estimate_room_probability",
        )

        nested_flat_named = HelicaseBrain._load_json(json.dumps([{
            "tool_call": {
                "name": "estimate_room_probability",
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": [],
                "reason": "unknown room",
            }
        }]), expected_tool_name="estimate_room_probability")
        self.assertEqual(
            nested_flat_named["tool_calls"][0],
            {
                "name": "estimate_room_probability",
                "arguments": {
                    "room_id": "room_2_2",
                    "target": "bed",
                    "probability": 0.2,
                    "confidence": 0.8,
                    "evidence_ids": [],
                    "reason": "unknown room",
                },
            },
        )

        wrapped_stage_local = HelicaseBrain._load_json(json.dumps({
            "tool_calls": [{
                "assignments": {},
                "diversity": True,
            }]
        }), expected_tool_name="assign_frontiers")
        self.assertEqual(
            wrapped_stage_local["tool_calls"][0],
            {
                "name": "assign_frontiers",
                "arguments": {"assignments": {}, "diversity": True},
            },
        )

        qwen_named_wrapper = HelicaseBrain._load_json(json.dumps({
            "tool_calls": [],
            "assign_frontiers": {
                "assignments": {},
                "diversity": True,
            },
        }), expected_tool_name="assign_frontiers")
        self.assertEqual(
            qwen_named_wrapper["tool_calls"],
            [{
                "name": "assign_frontiers",
                "arguments": {"assignments": {}, "diversity": True},
            }],
        )

        tool_calls_flat_named = HelicaseBrain._load_json(json.dumps({
            "tool_calls": [{
                "name": "estimate_room_probability",
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": ["room_2_2"],
                "reason": "unknown room",
            }],
        }), expected_tool_name="estimate_room_probability")
        self.assertEqual(
            tool_calls_flat_named["tool_calls"][0]["arguments"],
            {
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": ["room_2_2"],
                "reason": "unknown room",
            },
        )

        double_wrapped = HelicaseBrain._load_json(json.dumps([{
            "tool_calls": [{
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": [],
                "reason": "unknown room",
            }]
        }]), expected_tool_name="estimate_room_probability")
        self.assertEqual(
            double_wrapped["tool_calls"][0]["name"],
            "estimate_room_probability",
        )

        flat_named = HelicaseBrain._load_json(json.dumps([{
            "name": "estimate_room_probability",
            "room_id": "room_2_2",
            "target": "bed",
            "probability": 0.2,
            "confidence": 0.8,
            "evidence_ids": [],
            "reason": "unknown room",
        }]), expected_tool_name="estimate_room_probability")
        self.assertEqual(flat_named["tool_calls"][0], {
            "name": "estimate_room_probability",
            "arguments": {
                "room_id": "room_2_2",
                "target": "bed",
                "probability": 0.2,
                "confidence": 0.8,
                "evidence_ids": [],
                "reason": "unknown room",
            },
        })

        with self.assertRaisesRegex(ValueError, "invalid_json"):
            HelicaseBrain._load_json(
                "Here is JSON:\n```json\n{}\n```",
                expected_tool_name="query_room_objects",
            )

    def test_historical_room_is_full_kg_context_but_not_assignable(self):
        enriched = [
            frontier(0, 110, 110),
            frontier(1, 210, 210),
        ]
        kg, poses, view, fake_brain, helicase = self.make_runtime(
            enriched, valid_pipeline(enriched)
        )
        kg.add_node(KGNode(
            id="room_9_9",
            node_type="room",
            name="historical",
            certainty=0.5,
            properties={"explored": False},
        ))

        assignments, _, _ = helicase.decide(
            kg, "bed", enriched, poses, 25, 500, [], current_frontiers=view
        )
        self.assertTrue(all(idx in view.frontiers for idx in assignments.values()))
        self.assertIn("HISTORICAL ROOM CONTEXT", fake_brain.prompts[0])
        self.assertNotIn("room_9_9", fake_brain.prompts[1])
        probability_rooms = {
            call["arguments"]["room_id"]
            for call in json.loads(fake_brain.responses[0])["tool_calls"]
        }
        self.assertNotIn("room_9_9", probability_rooms)

    def test_same_room_keeps_both_llm_assigned_frontiers(self):
        enriched = [
            frontier(0, 110, 110, area=10),
            frontier(1, 120, 130, area=30),
        ]
        responses = valid_pipeline(
            enriched,
            assignments={"robot_0": 0, "robot_1": 1},
        )
        assignments, view, _, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        self.assertEqual(assignments, {"robot_0": 0, "robot_1": 1})
        self.assertEqual(view.room_to_frontiers["room_2_2"], (0, 1))
        self.assertEqual(
            helicase.last_decision_audit["accepted_llm_rooms"],
            {"robot_0": "room_2_2", "robot_1": "room_2_2"},
        )
        self.assertFalse(
            helicase.last_decision_audit["room_diversity_required"]
        )

    def test_valid_two_llm_pipeline_uses_assignment_directly(self):
        enriched = [
            frontier(0, 110, 110, prior=0.99),
            frontier(1, 210, 210, prior=0.01),
        ]
        responses = valid_pipeline(
            enriched,
            assignments={"robot_0": 1, "robot_1": 0},
            probabilities={"room_2_2": 0.2, "room_4_4": 0.8},
        )
        assignments, _, fake_brain, helicase, tools, method, _ = self.decide(
            enriched, responses
        )
        self.assertEqual(assignments, {"robot_0": 1, "robot_1": 0})
        self.assertEqual(fake_brain.call_count, 2)
        self.assertEqual(tools, [
            "query_room_objects",
            "estimate_room_probability",
            "assign_frontiers",
        ])
        self.assertEqual(helicase.last_decision_audit["llm_output_status"], "valid")
        self.assertTrue(helicase.last_decision_audit["llm_effective"])
        self.assertEqual(helicase.last_decision_audit["llm_call_count"], 2)
        self.assertEqual(
            helicase.last_decision_audit["tool_status"]["query_room_objects"],
            "executed",
        )
        self.assertIn("helicase_room_first(valid", method)

    def test_full_kg_only_enters_probability_prompt(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        _, _, fake_brain, _, _, _, _ = self.decide(
            enriched, valid_pipeline(enriched)
        )
        self.assertEqual(len(fake_brain.prompts), 2)
        for prompt in fake_brain.prompts:
            self.assertNotIn("base_target_score", prompt)
            self.assertNotIn("prior=", prompt)
        self.assertIn("SERIALIZED_KG", fake_brain.prompts[0])
        self.assertIn("CURRENT ASSIGNABLE ROOMS", fake_brain.prompts[0])
        self.assertIn("REQUIRED_CALL_IDENTITIES", fake_brain.prompts[0])
        self.assertNotIn("SERIALIZED_KG", fake_brain.prompts[1])
        self.assertNotIn("HISTORICAL ROOM CONTEXT", fake_brain.prompts[1])
        self.assertIn("CURRENT_DECISION_PACKET", fake_brain.prompts[1])
        self.assertIn("ROOM_DIVERSITY_REQUIRED: true", fake_brain.prompts[1])

    def test_json_format_only_changes_probability_kg_serialization(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        kg, poses, view, fake_brain, _ = self.make_runtime(
            enriched, valid_pipeline(enriched)
        )
        helicase = HelicaseBrain(
            fake_brain,
            num_agents=2,
            kg_serialization="json",
        )
        helicase.decide(
            kg,
            target_name="bed",
            enriched_frontiers=enriched,
            pose_pred=poses,
            step=25,
            max_steps=500,
            decision_history=[],
            current_frontiers=view,
        )

        self.assertIn(
            '"format":"mindnav_kg_json_v1"',
            fake_brain.prompts[0],
        )
        self.assertIn('"current_assignable_rooms"', fake_brain.prompts[0])
        self.assertNotIn('"format":"mindnav_kg_json_v1"', fake_brain.prompts[1])
        self.assertIn("CURRENT_DECISION_PACKET", fake_brain.prompts[1])

    def test_triples_only_change_probability_kg_serialization(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        kg, poses, view, fake_brain, _ = self.make_runtime(
            enriched, valid_pipeline(enriched)
        )
        helicase = HelicaseBrain(
            fake_brain,
            num_agents=2,
            kg_serialization="triples",
        )
        helicase.decide(
            kg,
            target_name="bed",
            enriched_frontiers=enriched,
            pose_pred=poses,
            step=25,
            max_steps=500,
            decision_history=[],
            current_frontiers=view,
        )

        marker = '["kg","serialization_format","mindnav_kg_triples_v1"]'
        self.assertIn(marker, fake_brain.prompts[0])
        self.assertIn(
            '["room_2_2","context_role","current_assignable"]',
            fake_brain.prompts[0],
        )
        self.assertNotIn(marker, fake_brain.prompts[1])
        self.assertIn("CURRENT_DECISION_PACKET", fake_brain.prompts[1])

    def test_decision_packet_groups_frontiers_without_topology_or_history(self):
        enriched = [
            frontier(0, 110, 110, area=30),
            frontier(1, 120, 130, area=20),
            frontier(2, 210, 210, area=40),
        ]
        responses = valid_pipeline(
            enriched,
            assignments={"robot_0": 0, "robot_1": 2},
            probabilities={"room_2_2": 0.8, "room_4_4": 0.2},
        )
        _, _, fake_brain, _, _, _, _ = self.decide(enriched, responses)
        packet = decision_packet_from_prompt(fake_brain.prompts[1])
        room = packet["current_rooms"]["room_2_2"]
        self.assertEqual(
            [option["frontier_id"] for option in room["frontiers"]],
            [0, 1],
        )
        self.assertEqual(
            packet["constraints"]["valid_room_frontier_pairs"],
            [
                {"room_id": "room_2_2", "frontier_id": 0},
                {"room_id": "room_2_2", "frontier_id": 1},
                {"room_id": "room_4_4", "frontier_id": 2},
            ],
        )
        self.assertIn(
            "VALID_ROOM_FRONTIER_PAIRS", fake_brain.prompts[1]
        )
        serialized = json.dumps(packet)
        for forbidden in (
                "spatial_relations", "connected_to",
                "separated_by_wall", "path_to"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(packet["recent_assignments"], [])

    def test_multiple_rooms_reject_same_room_assignment_then_repairs(self):
        enriched = [
            frontier(0, 110, 110),
            frontier(1, 120, 130),
            frontier(2, 210, 210),
        ]
        invalid_same_room = assignment_response(
            {"robot_0": 0, "robot_1": 1}, enriched
        )
        valid_distinct_rooms = assignment_response(
            {"robot_0": 0, "robot_1": 2}, enriched
        )
        responses = [
            probability_response({"room_2_2": 0.8, "room_4_4": 0.2}),
            invalid_same_room,
            valid_distinct_rooms,
        ]
        assignments, view, fake_brain, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        selected_rooms = {
            view.frontiers[frontier_id].room_id
            for frontier_id in assignments.values()
        }
        self.assertEqual(selected_rooms, {"room_2_2", "room_4_4"})
        self.assertEqual(fake_brain.call_count, 3)
        self.assertEqual(
            helicase.last_decision_audit["tool_status"]["assign_frontiers"],
            "repaired",
        )
        self.assertTrue(
            helicase.last_decision_audit["room_diversity_achieved"]
        )

    def test_assignment_extra_robot_id_is_repaired_with_specific_error(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        invalid = json.loads(assignment_response(
            {"robot_0": 0, "robot_1": 1}, enriched
        ))
        invalid["tool_calls"][0]["arguments"]["assignments"][
            "robot_0"
        ]["robot_id"] = "robot_0"
        responses = [
            probability_response({"room_2_2": 0.8, "room_4_4": 0.2}),
            json.dumps(invalid),
            assignment_response({"robot_0": 0, "robot_1": 1}, enriched),
        ]
        _, _, fake_brain, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        self.assertEqual(fake_brain.call_count, 3)
        self.assertEqual(
            helicase.last_decision_audit["llm_output_status"], "repaired"
        )
        self.assertIn("extra=['robot_id']", fake_brain.prompts[2])

    def test_recent_history_uses_stable_geometry_and_resets_at_step_zero(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        responses = valid_pipeline(enriched) * 3
        kg, poses, view, fake_brain, helicase = self.make_runtime(
            enriched, responses
        )
        helicase.decide(
            kg, "bed", enriched, poses, 25, 500, [], current_frontiers=view
        )
        kg.add_node(KGNode(
            id="bed_history",
            node_type="object",
            name="bed",
            certainty=0.9,
            position=(115, 115),
            properties={"observation_count": 1},
        ))
        kg.add_edge(KGEdge("room_2_2", "bed_history", "contains"))
        kg.nodes["room_2_2"].properties["observation_count"] += 1

        helicase.decide(
            kg, "bed", enriched, poses, 50, 500, [], current_frontiers=view
        )
        packet = decision_packet_from_prompt(fake_brain.prompts[3])
        history = packet["recent_assignments"]
        self.assertEqual(len(history), 1)
        self.assertEqual(
            history[0]["robots"]["robot_0"],
            {
                "room_id": "room_2_2",
                "frontier_centroid_rc": [110.0, 110.0],
            },
        )
        self.assertEqual(
            history[0]["new_object_ids_by_room"],
            {"room_2_2": ["bed_history"]},
        )
        self.assertTrue(
            history[0]["new_room_evidence_by_room"]["room_2_2"]
        )
        self.assertNotIn("frontier_id", json.dumps(history))

        helicase.decide(
            kg, "bed", enriched, poses, 0, 500, [], current_frontiers=view
        )
        reset_packet = decision_packet_from_prompt(fake_brain.prompts[5])
        self.assertEqual(reset_packet["recent_assignments"], [])

    def test_recent_history_resets_on_kg_reset_at_same_step(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        kg, poses, view, fake_brain, helicase = self.make_runtime(
            enriched, valid_pipeline(enriched) * 2
        )
        helicase.decide(
            kg, "bed", enriched, poses, 25, 500, [],
            current_frontiers=view,
        )
        self.assertEqual(len(helicase._recent_assignment_history), 1)

        kg.reset()
        KGUpdater(kg).update(enriched, {}, poses, None)
        helicase.decide(
            kg, "bed", enriched, poses, 25, 500, [],
            current_frontiers=view,
        )
        reset_packet = decision_packet_from_prompt(fake_brain.prompts[3])
        self.assertEqual(reset_packet["recent_assignments"], [])

    def test_invalid_probability_json_is_repaired_once(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        rooms = sorted({room_id_for(candidate) for candidate in enriched})
        responses = [
            "not json",
            probability_response({rooms[0]: 0.8, rooms[1]: 0.2}),
            assignment_response({"robot_0": 0, "robot_1": 1}, enriched),
        ]
        _, _, fake_brain, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        self.assertEqual(fake_brain.call_count, 3)
        self.assertEqual(
            helicase.last_decision_audit["tool_status"][
                "estimate_room_probability"
            ],
            "repaired",
        )
        self.assertEqual(
            helicase.last_decision_audit["llm_output_status"], "repaired"
        )
        self.assertIn("complete corrected JSON replacement", fake_brain.prompts[1])
        self.assertIn("must not contain only the missing", fake_brain.prompts[1])

    def test_invalid_probability_after_repair_uses_current_fallback(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        assignments, view, fake_brain, helicase, tools, _, _ = self.decide(
            enriched, ["bad", "still bad"]
        )
        self.assertEqual(fake_brain.call_count, 2)
        self.assertIn("current_frontier_fallback", tools)
        self.assertEqual(set(assignments.values()), set(view.frontiers))
        self.assertEqual(
            helicase.last_decision_audit["fallback_reason"],
            "invalid_estimate_room_probability",
        )

    def test_out_of_range_probability_rejects_probability_stage(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        rooms = sorted({room_id_for(candidate) for candidate in enriched})
        invalid_probability = probability_response({
            rooms[0]: 1.4,
            rooms[1]: 0.2,
        })
        assignments, view, fake_brain, helicase, _, _, _ = self.decide(
            enriched,
            [invalid_probability, invalid_probability],
        )
        self.assertEqual(fake_brain.call_count, 2)
        self.assertTrue(all(idx in view.frontiers for idx in assignments.values()))
        self.assertEqual(
            helicase.last_decision_audit["fallback_reason"],
            "invalid_estimate_room_probability",
        )

    def test_probability_evidence_must_belong_to_queried_room(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        rooms = sorted({room_id_for(candidate) for candidate in enriched})
        invalid_probability = probability_response(
            {rooms[0]: 0.8, rooms[1]: 0.2},
            evidence_by_room={rooms[0]: [rooms[1]], rooms[1]: [rooms[1]]},
        )
        _, _, _, helicase, _, _, _ = self.decide(
            enriched,
            [invalid_probability, invalid_probability],
        )
        self.assertTrue(any(
            "unknown evidence" in error
            for error in helicase.last_decision_audit["validation_errors"]
        ))

    def test_probability_citations_are_stable_room_and_object_nodes_only(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        _, _, _, helicase, _, _, kg = self.decide(
            enriched, valid_pipeline(enriched)
        )
        kg.add_edge(KGEdge(
            "room_2_2", "room_4_4", "connected_to", distance=10
        ))
        view = CurrentFrontierView.from_enriched(enriched)
        results, allowed = helicase._query_room_objects(
            kg, view, sorted(view.room_ids)
        )
        self.assertNotIn("spatial_relations", results[0])
        for room_id, evidence_ids in allowed.items():
            self.assertIn(room_id, evidence_ids)
            self.assertFalse(any(
                evidence_id.startswith((
                    "connected_to:", "separated_by_wall:", "frontier_"
                ))
                for evidence_id in evidence_ids
            ))

    def test_invalid_assignment_uses_valid_llm_probabilities_for_fallback(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        rooms = sorted({room_id_for(candidate) for candidate in enriched})
        invalid_assignment = assignment_response(
            {"robot_0": 99, "robot_1": 0}, enriched
        )
        responses = [
            probability_response({rooms[0]: 0.2, rooms[1]: 0.9}),
            invalid_assignment,
            invalid_assignment,
        ]
        assignments, view, fake_brain, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        self.assertEqual(fake_brain.call_count, 3)
        self.assertEqual(set(assignments.values()), set(view.frontiers))
        selected_rooms = {
            view.frontiers[frontier_id].room_id
            for frontier_id in assignments.values()
        }
        self.assertEqual(selected_rooms, set(rooms))
        self.assertEqual(
            helicase.last_decision_audit["fallback_reason"],
            "invalid_assign_frontiers",
        )
        self.assertEqual(
            helicase.last_decision_audit["selection_source_by_robot"]["robot_0"],
            "llm_probability_fallback",
        )

    def test_probability_fallback_selects_top_distinct_rooms(self):
        enriched = [
            frontier(0, 110, 110),
            frontier(1, 210, 210),
            frontier(2, 310, 310),
        ]
        invalid_assignment = assignment_response(
            {"robot_0": 99, "robot_1": 99}, enriched
        )
        responses = [
            probability_response({
                "room_2_2": 0.9,
                "room_4_4": 0.8,
                "room_6_6": 0.1,
            }),
            invalid_assignment,
            invalid_assignment,
        ]
        assignments, view, _, helicase, _, _, _ = self.decide(
            enriched, responses
        )
        selected_rooms = {
            view.frontiers[frontier_id].room_id
            for frontier_id in assignments.values()
        }
        self.assertEqual(selected_rooms, {"room_2_2", "room_4_4"})
        self.assertEqual(
            helicase.last_decision_audit["fallback_reason"],
            "invalid_assign_frontiers",
        )

    def test_geometry_fallback_uses_distinct_rooms_when_available(self):
        enriched = [
            frontier(0, 110, 110),
            frontier(1, 120, 130),
            frontier(2, 210, 210),
        ]
        assignments, view, _, helicase, _, _, _ = self.decide(
            enriched, ["bad probability", "still bad"]
        )
        selected_rooms = {
            view.frontiers[frontier_id].room_id
            for frontier_id in assignments.values()
        }
        self.assertEqual(selected_rooms, {"room_2_2", "room_4_4"})
        self.assertEqual(
            helicase.last_decision_audit["selection_source_by_robot"][
                "robot_0"
            ],
            "geometry_fallback",
        )

    def test_transport_fallback_is_current_and_distinct(self):
        enriched = [frontier(0, 110, 110), frontier(1, 210, 210)]
        _, poses, view, _, helicase = self.make_runtime(enriched, [])
        assignments, tools, _ = helicase.deterministic_fallback(
            view, poses, reason="TimeoutError"
        )
        self.assertEqual(set(assignments.values()), set(view.frontiers))
        self.assertEqual(tools, ["current_frontier_fallback"])
        self.assertEqual(
            helicase.last_decision_audit["llm_output_status"],
            "transport_fallback",
        )


if __name__ == "__main__":
    unittest.main()
