"""MindNav KG reasoning brain with room-first frontier allocation.

Each planning decision has three logical operations but only two LLM calls:

1. Python executes ``query_room_objects`` for every current room.
2. The LLM reads the full serialized KG and estimates room probabilities.
3. The LLM reads a current-only decision packet and assigns rooms/frontiers.

Python validates the feasible action space and provides explicit fallbacks.
It does not compute a target prior or replace a valid LLM assignment.
"""

import copy
import itertools
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from kg_construction import (
    CurrentFrontierView,
    KGEdge,
    KGNode,
    KGUpdater,
    KnowledgeGraph,
)


class HelicaseBrain:
    """Central LLM that reasons over the KG and assigns current frontiers."""

    def __init__(self, brain, num_agents=2):
        self.brain = brain
        self.num_agents = int(num_agents)
        self.reflexion_memory: List[str] = []
        self.last_decision_audit: Dict = {}
        self._recent_assignment_history: List[Dict] = []
        self._history_snapshot: Optional[Dict] = None
        self._history_last_step: Optional[int] = None
        self._history_kg_generation: Optional[int] = None

    @staticmethod
    def _reject_json_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    @classmethod
    def _load_json(cls, response, expected_tool_name=None):
        try:
            response_text = response.strip()
        except AttributeError as error:
            raise ValueError(f"invalid_json: {error}") from error

        # Accept one exact enclosing Markdown JSON fence.  Do not search
        # explanatory prose for an embedded object.
        response_lines = response_text.splitlines()
        if (len(response_lines) >= 3
                and response_lines[0].strip().lower() in {"```", "```json"}
                and response_lines[-1].strip() == "```"):
            response_text = "\n".join(response_lines[1:-1]).strip()
        try:
            payload = json.loads(
                response_text,
                parse_constant=cls._reject_json_constant,
            )
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid_json: {error}") from error
        # Qwen commonly emits the semantically equivalent compact form
        # ``[{"tool_call": name, ...arguments...}]``.  Canonicalize only this
        # syntax; all tool names, argument fields, values, and coverage remain
        # subject to the strict stage validators below.
        if (isinstance(payload, list)
                and len(payload) == 1
                and isinstance(payload[0], dict)
                and set(payload[0]) == {"tool_calls"}
                and isinstance(payload[0]["tool_calls"], list)):
            payload = payload[0]
        if isinstance(payload, list):
            canonical_calls = []
            for compact_call in payload:
                if not isinstance(compact_call, dict):
                    raise ValueError(
                        "schema_error: compact tool call must be an object"
                    )
                tool_call = compact_call.get("tool_call")
                if isinstance(tool_call, str):
                    canonical_calls.append({
                        "name": tool_call,
                        "arguments": {
                            key: value for key, value in compact_call.items()
                            if key != "tool_call"
                        },
                    })
                elif (isinstance(compact_call.get("name"), str)
                      and "arguments" not in compact_call
                      and "args" not in compact_call
                      and "tool_call" not in compact_call):
                    canonical_calls.append({
                        "name": compact_call["name"],
                        "arguments": {
                            key: value for key, value in compact_call.items()
                            if key != "name"
                        },
                    })
                elif (isinstance(tool_call, dict)
                      and isinstance(expected_tool_name, str)
                      and set(compact_call) == {"tool_call"}):
                    nested_name = tool_call.get(
                        "name", expected_tool_name
                    )
                    if "arguments" in tool_call:
                        if set(tool_call) != {"name", "arguments"}:
                            raise ValueError(
                                "schema_error: nested tool_call with arguments "
                                "must contain only name and arguments"
                            )
                        nested_arguments = tool_call["arguments"]
                    else:
                        nested_arguments = {
                            key: value for key, value in tool_call.items()
                            if key != "name"
                        }
                    canonical_calls.append({
                        "name": nested_name,
                        "arguments": nested_arguments,
                    })
                else:
                    raise ValueError(
                        "schema_error: compact tool call must name the tool "
                        "or contain stage-local arguments"
                    )
            payload = {"tool_calls": canonical_calls}
        elif (isinstance(payload, dict)
              and isinstance(payload.get("tool_call"), str)):
            payload = {
                "tool_calls": [{
                    "name": payload["tool_call"],
                    "arguments": {
                        key: value for key, value in payload.items()
                        if key != "tool_call"
                    },
                }]
            }
        elif (isinstance(payload, dict)
              and isinstance(payload.get("tool_call"), dict)
              and isinstance(expected_tool_name, str)
              and set(payload) == {"tool_call"}):
            nested_tool_call = payload["tool_call"]
            nested_name = nested_tool_call.get(
                "name", expected_tool_name
            )
            if "arguments" in nested_tool_call:
                if set(nested_tool_call) != {"name", "arguments"}:
                    raise ValueError(
                        "schema_error: nested tool_call with arguments must "
                        "contain only name and arguments"
                    )
                nested_arguments = nested_tool_call["arguments"]
            else:
                nested_arguments = {
                    key: value for key, value in nested_tool_call.items()
                    if key != "name"
                }
            payload = {
                "tool_calls": [{
                    "name": nested_name,
                    "arguments": nested_arguments,
                }]
            }
        if not isinstance(payload, dict):
            raise ValueError("schema_error: top level must be an object")
        # Qwen may place the stage name beside an empty ``tool_calls`` list,
        # while keeping the complete stage arguments under that name.  This
        # exact, unambiguous wrapper is safe to canonicalize: the strict stage
        # validator below still checks every field, value, and required item.
        if (isinstance(expected_tool_name, str)
                and set(payload) == {"tool_calls", expected_tool_name}
                and payload["tool_calls"] == []
                and isinstance(payload[expected_tool_name], dict)):
            payload = {
                "tool_calls": [{
                    "name": expected_tool_name,
                    "arguments": payload[expected_tool_name],
                }]
            }
        calls = payload.get("tool_calls")
        if isinstance(calls, list):
            normalized_calls = []
            for call in calls:
                if (isinstance(call, dict)
                        and set(call) == {"name", "args"}):
                    normalized_calls.append({
                        "name": call["name"],
                        "arguments": call["args"],
                    })
                elif (isinstance(call, dict)
                      and isinstance(call.get("name"), str)
                      and "arguments" not in call
                      and "args" not in call
                      and "tool_call" not in call):
                    # Equivalent Qwen form inside an explicit tool_calls list:
                    # {"name": tool, ...flat stage arguments...}.
                    normalized_calls.append({
                        "name": call["name"],
                        "arguments": {
                            key: value for key, value in call.items()
                            if key != "name"
                        },
                    })
                elif (isinstance(call, dict)
                      and isinstance(expected_tool_name, str)
                      and not ({"name", "arguments", "args", "tool_call"}
                               & set(call))):
                    # A stage-local call may omit both wrappers, e.g.
                    # {"tool_calls": [{"assignments": ..., "diversity":
                    # true}]}.  The active stage supplies only the tool name;
                    # its validator still enforces the exact argument schema.
                    normalized_calls.append({
                        "name": expected_tool_name,
                        "arguments": call,
                    })
                else:
                    normalized_calls.append(call)
            payload = dict(payload)
            payload["tool_calls"] = normalized_calls
        return payload

    def _call_json_stage(self, stage_name, prompt, validator, max_tokens):
        """Call one tool stage, allowing exactly one schema repair."""
        errors = []
        responses = []

        response = self.brain.call(prompt, max_tokens=max_tokens)
        responses.append(response)
        try:
            return validator(
                self._load_json(response, expected_tool_name=stage_name)
            ), "valid", errors, responses
        except ValueError as first_error:
            errors.append(f"{stage_name}: {first_error}")

        repair_prompt = (
            f"The previous {stage_name} tool call was invalid. Re-run the "
            "same tool stage from its original stage context and return one "
            "complete corrected JSON replacement. The replacement must cover "
            "EVERY required current room or robot from ORIGINAL_TOOL_TASK; it "
            "must not contain only the missing item(s). Return pure JSON and "
            "do not explain the repair. For assign_frontiers, copy each "
            "room_id/frontier_id pair intact from VALID_ROOM_FRONTIER_PAIRS "
            "in ORIGINAL_TOOL_TASK; never combine values from different "
            "pairs. For estimate_room_probability, every evidence_id must "
            "come from that SAME room's allowed_evidence_ids in "
            "REQUIRED_CALL_IDENTITIES. Delete every ID named as unknown by "
            "VALIDATION_ERROR; an empty evidence_ids list is valid when no "
            "grounded ID is needed.\n\n"
            f"VALIDATION_ERROR:\n{errors[-1]}\n\n"
            f"INVALID_RESPONSE:\n{response}\n\n"
            f"ORIGINAL_TOOL_TASK:\n{prompt}"
        )
        repaired_response = self.brain.call(
            repair_prompt,
            max_tokens=max_tokens,
        )
        responses.append(repaired_response)
        try:
            result = validator(self._load_json(
                repaired_response,
                expected_tool_name=stage_name,
            ))
            return result, "repaired", errors, responses
        except ValueError as repair_error:
            errors.append(f"{stage_name}: {repair_error}")
            return None, "invalid_after_repair", errors, responses

    @staticmethod
    def _frontier_options(current_frontiers, pose_pred, num_agents):
        robot_positions = {
            f"robot_{robot_index}": (
                CurrentFrontierView.robot_pose_to_map_rc(
                    pose_pred[robot_index]
                ) if robot_index < len(pose_pred) else None
            )
            for robot_index in range(num_agents)
        }
        options = []
        for frontier_id in sorted(current_frontiers.frontiers):
            frontier = current_frontiers.frontiers[frontier_id]
            distances = {}
            for robot_id, position in robot_positions.items():
                distances[robot_id] = (
                    round(float(np.hypot(
                        frontier.centroid[0] - position[0],
                        frontier.centroid[1] - position[1],
                    )), 1)
                    if position is not None else None
                )
            options.append({
                "frontier_id": int(frontier.idx),
                "room_id": frontier.room_id,
                "centroid_rc": [
                    round(float(frontier.centroid[0]), 1),
                    round(float(frontier.centroid[1]), 1),
                ],
                "area_px": round(float(frontier.area), 1),
                "distance_by_robot_px": distances,
            })
        return options

    @staticmethod
    def _query_room_objects(kg, current_frontiers, room_ids):
        """Execute query_room_objects against the live persistent KG."""
        results = []
        allowed_evidence = {}
        for room_id in room_ids:
            room = kg.nodes.get(room_id)
            objects = sorted(
                kg.get_objects_in_room(room_id),
                key=lambda obj: (-float(obj.certainty), obj.id),
            )
            object_records = [{
                "evidence_id": obj.id,
                "category": obj.name,
                "certainty": round(float(obj.certainty), 3),
                "position_rc": [
                    round(float(obj.position[0]), 1),
                    round(float(obj.position[1]), 1),
                ],
                "observation_count": int(
                    obj.properties.get("observation_count", 1)
                ),
            } for obj in objects]

            evidence_ids = {room_id}
            evidence_ids.update(obj.id for obj in objects)

            frontier_ids = list(
                current_frontiers.room_to_frontiers.get(room_id, ())
            )
            results.append({
                "room_id": room_id,
                "observed_type": (
                    room.name if room is not None else "unknown"
                ),
                "type_certainty": round(
                    float(room.certainty if room is not None else 0.0), 3
                ),
                "explored": bool(
                    room is not None
                    and room.properties.get("explored", False)
                ),
                "observation_count": int(
                    room.properties.get("observation_count", 0)
                    if room is not None else 0
                ),
                "objects": object_records,
                "current_frontier_ids": frontier_ids,
                "allowed_evidence_ids": sorted(evidence_ids),
            })
            allowed_evidence[room_id] = evidence_ids
        return results, allowed_evidence

    @staticmethod
    def _kg_history_snapshot(kg):
        """Capture stable evidence identifiers for progress accounting."""
        objects_by_room = {}
        room_observation_counts = {}
        for room in kg.get_nodes_by_type("room"):
            objects_by_room[room.id] = {
                obj.id for obj in kg.get_objects_in_room(room.id)
            }
            room_observation_counts[room.id] = int(
                room.properties.get("observation_count", 0)
            )
        return {
            "objects_by_room": objects_by_room,
            "room_observation_counts": room_observation_counts,
        }

    def _prepare_recent_history(self, kg, step):
        """Finalize prior-decision progress and return JSON-safe history."""
        step = int(step)
        kg_generation = int(getattr(kg, "reset_generation", 0))
        new_episode = (
            bool(self._recent_assignment_history)
            and (
                self._history_kg_generation != kg_generation
                or (
                    self._history_last_step is not None
                    and (step < self._history_last_step or step == 0)
                )
            )
        )
        if new_episode:
            self._recent_assignment_history = []
            self._history_snapshot = None
            self._history_last_step = None

        self._history_kg_generation = kg_generation
        current_snapshot = self._kg_history_snapshot(kg)
        if (self._history_snapshot is not None
                and self._recent_assignment_history
                and step != self._recent_assignment_history[-1]["step"]):
            prior_snapshot = self._history_snapshot
            prior_record = self._recent_assignment_history[-1]
            selected_rooms = {
                assignment["room_id"]
                for assignment in prior_record["robots"].values()
            }
            new_objects = {}
            observation_deltas = {}
            evidence_by_room = {}
            for room_id in sorted(selected_rooms):
                current_objects = current_snapshot["objects_by_room"].get(
                    room_id, set()
                )
                prior_objects = prior_snapshot["objects_by_room"].get(
                    room_id, set()
                )
                added = sorted(current_objects - prior_objects)
                if added:
                    new_objects[room_id] = added

                delta = max(
                    0,
                    current_snapshot["room_observation_counts"].get(
                        room_id, 0
                    ) - prior_snapshot["room_observation_counts"].get(
                        room_id, 0
                    ),
                )
                if delta:
                    observation_deltas[room_id] = int(delta)
                evidence_by_room[room_id] = bool(added or delta)

            prior_record["new_object_ids_by_room"] = new_objects
            prior_record["room_observation_count_delta"] = observation_deltas
            prior_record["new_room_evidence_by_room"] = evidence_by_room
            prior_record["new_room_evidence"] = any(
                evidence_by_room.values()
            )
            prior_record["progress_evaluated_at_step"] = step

        self._history_snapshot = current_snapshot
        return copy.deepcopy(self._recent_assignment_history[-4:])

    def _record_assignment_history(self, current_frontiers, assignments,
                                   step):
        """Store stable room/centroid assignments, never transient IDs."""
        robots = {}
        for robot_id, frontier_id in sorted(assignments.items()):
            frontier = current_frontiers.frontiers[frontier_id]
            robots[robot_id] = {
                "room_id": frontier.room_id,
                "frontier_centroid_rc": [
                    round(float(frontier.centroid[0]), 1),
                    round(float(frontier.centroid[1]), 1),
                ],
            }
        record = {
            "step": int(step),
            "robots": robots,
            "new_object_ids_by_room": {},
            "room_observation_count_delta": {},
            "new_room_evidence_by_room": {},
            "new_room_evidence": None,
        }
        if (self._recent_assignment_history
                and self._recent_assignment_history[-1]["step"] == int(step)):
            self._recent_assignment_history[-1] = record
        else:
            self._recent_assignment_history.append(record)
        self._recent_assignment_history = self._recent_assignment_history[-4:]
        self._history_last_step = int(step)

    @staticmethod
    def _room_history_summary(current_room_ids, recent_history):
        summaries = {}
        for room_id in current_room_ids:
            recent_count = 0
            last_by_robot = {}
            for record in recent_history:
                for robot_id, assignment in record["robots"].items():
                    if assignment["room_id"] == room_id:
                        recent_count += 1
                        last_by_robot[robot_id] = int(record["step"])

            no_evidence_streak = 0
            for record in reversed(recent_history):
                assigned_here = any(
                    assignment["room_id"] == room_id
                    for assignment in record["robots"].values()
                )
                if not assigned_here:
                    break
                evidence = record.get(
                    "new_room_evidence_by_room", {}
                ).get(room_id)
                if evidence is False:
                    no_evidence_streak += 1
                else:
                    break

            summaries[room_id] = {
                "recent_assignment_count": int(recent_count),
                "consecutive_assignments_without_new_evidence": int(
                    no_evidence_streak
                ),
                "last_assigned_step_by_robot": last_by_robot,
            }
        return summaries

    def _build_current_decision_packet(
            self, query_results, probability_records, frontier_options,
            pose_pred, current_room_ids, recent_history):
        """Build the current-only KG projection used for final allocation."""
        query_by_room = {
            record["room_id"]: record for record in query_results
        }
        frontiers_by_room = {room_id: [] for room_id in current_room_ids}
        for option in frontier_options:
            room_id = option["room_id"]
            frontiers_by_room.setdefault(room_id, []).append({
                key: value for key, value in option.items()
                if key != "room_id"
            })

        history_summary = self._room_history_summary(
            current_room_ids, recent_history
        )
        current_rooms = {}
        for room_id in current_room_ids:
            query = query_by_room[room_id]
            probability = probability_records[room_id]
            object_ids = {
                obj["evidence_id"] for obj in query["objects"]
            }
            current_rooms[room_id] = {
                "observed_type": query["observed_type"],
                "type_certainty": query["type_certainty"],
                "target_probability": probability["probability"],
                "probability_confidence": probability["confidence"],
                "object_evidence_ids": [
                    evidence_id
                    for evidence_id in probability["evidence_ids"]
                    if evidence_id in object_ids
                ],
                "probability_reason": probability["reason"],
                "objects": [{
                    "id": obj["evidence_id"],
                    "name": obj["category"],
                    "certainty": obj["certainty"],
                    "position_rc": obj["position_rc"],
                    "observation_count": obj["observation_count"],
                } for obj in query["objects"]],
                "explored": query["explored"],
                "observation_count": query["observation_count"],
                "recent_history_summary": history_summary[room_id],
                "frontiers": sorted(
                    frontiers_by_room.get(room_id, []),
                    key=lambda option: option["frontier_id"],
                ),
            }

        robot_states = {}
        for robot_index in range(self.num_agents):
            robot_id = f"robot_{robot_index}"
            position = (
                CurrentFrontierView.robot_pose_to_map_rc(
                    pose_pred[robot_index]
                ) if robot_index < len(pose_pred) else None
            )
            robot_states[robot_id] = {
                "position_rc": (
                    [round(float(position[0]), 1),
                     round(float(position[1]), 1)]
                    if position is not None else None
                )
            }

        return {
            "robots": robot_states,
            "current_rooms": current_rooms,
            "recent_assignments": recent_history,
            "constraints": {
                "distinct_rooms_required": (
                    len(current_room_ids) >= self.num_agents
                ),
                "distinct_frontiers_required": (
                    sum(len(room["frontiers"])
                        for room in current_rooms.values())
                    >= self.num_agents
                ),
                "valid_room_frontier_pairs": [
                    {
                        "room_id": option["room_id"],
                        "frontier_id": option["frontier_id"],
                    }
                    for option in sorted(
                        frontier_options,
                        key=lambda item: item["frontier_id"],
                    )
                ],
            },
        }, history_summary

    @staticmethod
    def _validate_probability_calls(payload, expected_room_ids, target_name,
                                    allowed_evidence):
        if set(payload) != {"tool_calls"}:
            raise ValueError("top level must contain only tool_calls")
        calls = payload["tool_calls"]
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")

        records = {}
        required_arguments = {
            "room_id",
            "target",
            "probability",
            "confidence",
            "evidence_ids",
            "reason",
        }
        for call in calls:
            if (not isinstance(call, dict)
                    or set(call) != {"name", "arguments"}
                    or call.get("name") != "estimate_room_probability"):
                raise ValueError(
                    "every call must be estimate_room_probability"
                )
            arguments = call["arguments"]
            if (not isinstance(arguments, dict)
                    or set(arguments) != required_arguments):
                raise ValueError(
                    "estimate_room_probability arguments have wrong schema"
                )
            room_id = arguments["room_id"]
            if not isinstance(room_id, str):
                raise ValueError("probability room_id must be a string")
            if room_id in records:
                raise ValueError(f"duplicate probability room_id: {room_id}")
            if arguments["target"] != target_name:
                raise ValueError(
                    f"probability target must be exactly {target_name}"
                )

            for field_name in ("probability", "confidence"):
                value = arguments[field_name]
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0):
                    raise ValueError(
                        f"invalid {field_name} for room {room_id}"
                    )
            evidence_ids = arguments["evidence_ids"]
            if (not isinstance(evidence_ids, list)
                    or any(not isinstance(item, str) for item in evidence_ids)
                    or len(evidence_ids) != len(set(evidence_ids))):
                raise ValueError(
                    f"invalid evidence_ids for room {room_id}"
                )
            unknown_evidence = sorted(
                set(evidence_ids) - set(allowed_evidence.get(room_id, ()))
            )
            if unknown_evidence:
                raise ValueError(
                    f"unknown evidence for room {room_id}: "
                    f"{unknown_evidence}; allowed evidence is "
                    f"{sorted(allowed_evidence.get(room_id, ()))}"
                )
            if (not isinstance(arguments["reason"], str)
                    or not arguments["reason"].strip()):
                raise ValueError(f"missing reason for room {room_id}")

            records[room_id] = {
                "probability": float(arguments["probability"]),
                "confidence": float(arguments["confidence"]),
                "evidence_ids": list(evidence_ids),
                "reason": arguments["reason"].strip(),
            }

        expected = set(expected_room_ids)
        if set(records) != expected:
            raise ValueError(
                "probability coverage mismatch: "
                f"missing={sorted(expected - set(records))}, "
                f"extra={sorted(set(records) - expected)}"
            )
        return records

    def _validate_assignment_calls(self, payload, current_frontiers):
        if set(payload) != {"tool_calls"}:
            raise ValueError("top level must contain only tool_calls")
        calls = payload["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("assign_frontiers requires exactly one tool call")
        call = calls[0]
        if (not isinstance(call, dict)
                or set(call) != {"name", "arguments"}
                or call.get("name") != "assign_frontiers"):
            raise ValueError("final call must be assign_frontiers")
        arguments = call["arguments"]
        if (not isinstance(arguments, dict)
                or set(arguments) != {"assignments", "diversity"}
                or arguments.get("diversity") is not True):
            raise ValueError(
                "assign_frontiers arguments require assignments and "
                "diversity=true"
            )
        assignments_payload = arguments["assignments"]
        expected_robots = {
            f"robot_{robot_index}" for robot_index in range(self.num_agents)
        }
        if (not isinstance(assignments_payload, dict)
                or set(assignments_payload) != expected_robots):
            raise ValueError(
                f"assignments must cover exactly {sorted(expected_robots)}"
            )

        assignments = {}
        rooms = {}
        reasons = {}
        for robot_id, assignment in assignments_payload.items():
            expected_assignment_fields = {
                "frontier_id", "room_id", "reason"
            }
            if not isinstance(assignment, dict):
                raise ValueError(
                    f"assignment for {robot_id} must be an object"
                )
            actual_assignment_fields = set(assignment)
            if actual_assignment_fields != expected_assignment_fields:
                raise ValueError(
                    f"invalid assignment schema for {robot_id}: expected "
                    f"only {sorted(expected_assignment_fields)}, missing="
                    f"{sorted(expected_assignment_fields - actual_assignment_fields)}, "
                    f"extra={sorted(actual_assignment_fields - expected_assignment_fields)}"
                )
            frontier_id = assignment["frontier_id"]
            if (isinstance(frontier_id, bool)
                    or not isinstance(frontier_id, int)
                    or frontier_id not in current_frontiers.frontiers):
                raise ValueError(
                    f"non-current frontier for {robot_id}: {frontier_id}"
                )
            expected_room = current_frontiers.frontiers[frontier_id].room_id
            if assignment["room_id"] != expected_room:
                raise ValueError(
                    f"room/frontier mismatch for {robot_id}: expected "
                    f"{expected_room}, got {assignment['room_id']}"
                )
            if (not isinstance(assignment["reason"], str)
                    or not assignment["reason"].strip()):
                raise ValueError(f"missing assignment reason for {robot_id}")
            assignments[robot_id] = frontier_id
            rooms[robot_id] = expected_room
            reasons[robot_id] = assignment["reason"].strip()

        if (len(current_frontiers.frontiers) >= self.num_agents
                and len(set(assignments.values())) != self.num_agents):
            raise ValueError(
                "duplicate frontiers despite enough current candidates"
            )
        if (len(current_frontiers.room_ids) >= self.num_agents
                and len(set(rooms.values())) != self.num_agents):
            raise ValueError(
                "duplicate rooms despite enough current room choices"
            )
        return assignments, rooms, reasons

    def _fallback_assignments(self, current_frontiers, pose_pred,
                              probability_records=None):
        """Room-first current-only fallback with lexicographic ranking."""
        if not current_frontiers.frontiers:
            return {}
        probability_records = probability_records or {}
        use_probabilities = bool(probability_records)
        frontier_ids = sorted(current_frontiers.frontiers)
        require_distinct_frontiers = (
            len(frontier_ids) >= self.num_agents
        )
        require_distinct_rooms = (
            len(current_frontiers.room_ids) >= self.num_agents
        )
        robot_positions = []
        for robot_index in range(self.num_agents):
            raw_position = (
                pose_pred[robot_index]
                if robot_index < len(pose_pred) else None
            )
            robot_positions.append(
                CurrentFrontierView.robot_pose_to_map_rc(raw_position)
            )

        feasible = []
        for selected_ids in itertools.product(
                frontier_ids, repeat=self.num_agents):
            if (require_distinct_frontiers
                    and len(set(selected_ids)) != self.num_agents):
                continue
            selected_rooms = [
                current_frontiers.frontiers[frontier_id].room_id
                for frontier_id in selected_ids
            ]
            if (require_distinct_rooms
                    and len(set(selected_rooms)) != self.num_agents):
                continue

            probability_sum = sum(
                float(probability_records.get(room_id, {}).get(
                    "probability", 0.0
                ))
                for room_id in selected_rooms
            )
            distance_sum = 0.0
            area_sum = 0.0
            for robot_index, frontier_id in enumerate(selected_ids):
                frontier = current_frontiers.frontiers[frontier_id]
                position = robot_positions[robot_index]
                if position is not None:
                    distance_sum += float(np.hypot(
                        frontier.centroid[0] - position[0],
                        frontier.centroid[1] - position[1],
                    ))
                area_sum += float(frontier.area)
            rank = (
                -probability_sum if use_probabilities else 0.0,
                distance_sum,
                -area_sum,
                tuple(int(frontier_id) for frontier_id in selected_ids),
            )
            feasible.append((rank, selected_ids))

        if not feasible:
            return {}
        _, selected_ids = min(feasible, key=lambda item: item[0])
        return {
            f"robot_{robot_index}": int(frontier_id)
            for robot_index, frontier_id in enumerate(selected_ids)
        }

    def _store_audit(self, current_frontiers, assignments,
                     probability_records, assigned_rooms,
                     assignment_reasons, selection_source, output_status,
                     stage_statuses, validation_errors=None,
                     fallback_reason=None, tool_trace=None,
                     llm_call_count=0, recent_history=None,
                     room_history_summary=None):
        selected_rooms = {
            robot_id: current_frontiers.frontiers[frontier_id].room_id
            for robot_id, frontier_id in assignments.items()
        }
        probabilities = {
            room_id: record["probability"]
            for room_id, record in probability_records.items()
        }
        probability_values = list(probabilities.values())
        probability_spread = (
            max(probability_values) - min(probability_values)
            if probability_values else 0.0
        )
        fallback = selection_source != "llm_tool_assignment"
        room_diversity_required = (
            len(current_frontiers.room_ids) >= self.num_agents
        )
        selected_room_values = list(selected_rooms.values())

        frontier_details = {}
        for frontier_id, frontier in current_frontiers.frontiers.items():
            record = probability_records.get(frontier.room_id, {})
            selected_by = sorted(
                robot_id
                for robot_id, selected_id in assignments.items()
                if selected_id == frontier_id
            )
            frontier_details[str(frontier_id)] = {
                "room_id": frontier.room_id,
                "llm_room_probability": record.get("probability"),
                "llm_confidence": record.get("confidence"),
                "selected_by": selected_by,
            }

        self.last_decision_audit = {
            "allowed_room_ids": sorted(current_frontiers.room_ids),
            "allowed_frontier_ids": sorted(current_frontiers.frontiers),
            "raw_room_probabilities": dict(probabilities),
            "accepted_room_probabilities": dict(probabilities),
            "raw_room_assignments": dict(assigned_rooms),
            "accepted_llm_rooms": (
                dict(assigned_rooms) if not fallback else {}
            ),
            "selected_room_by_robot": selected_rooms,
            "selection_source_by_robot": {
                robot_id: selection_source for robot_id in assignments
            },
            "rejected_room_ids": [],
            "fallback_robots": sorted(assignments) if fallback else [],
            "final_frontier_assignments": dict(assignments),
            "llm_output_status": output_status,
            "fallback_reason": fallback_reason,
            "validation_errors": list(validation_errors or []),
            "tool_status": dict(stage_statuses),
            "tool_trace": list(tool_trace or []),
            "probability_evidence_by_room": {
                room_id: record.get("evidence_ids", [])
                for room_id, record in probability_records.items()
            },
            "probability_reason_by_room": {
                room_id: record.get("reason", "")
                for room_id, record in probability_records.items()
            },
            "assignment_reason_by_robot": dict(assignment_reasons),
            "llm_probability_spread": round(float(probability_spread), 6),
            "llm_probability_non_degenerate": bool(
                len(probability_values) <= 1 or probability_spread > 1e-6
            ),
            "llm_effective": bool(not fallback and assignments),
            "frontier_score_details": frontier_details,
            "llm_call_count": int(llm_call_count),
            "room_diversity_required": room_diversity_required,
            "room_diversity_achieved": bool(
                len(selected_room_values) <= 1
                or len(set(selected_room_values)) == len(
                    selected_room_values
                )
            ),
            "recent_assignment_history": list(recent_history or []),
            "room_history_summary": dict(room_history_summary or {}),
        }

    def _finish_fallback(self, current_frontiers, pose_pred,
                         probability_records, stage_statuses,
                         validation_errors, fallback_reason, tool_trace,
                         tools_called, step=None, llm_call_count=0,
                         recent_history=None, room_history_summary=None):
        assignments = self._fallback_assignments(
            current_frontiers,
            pose_pred,
            probability_records,
        )
        source = (
            "llm_probability_fallback"
            if probability_records else "geometry_fallback"
        )
        selected_rooms = {
            robot_id: current_frontiers.frontiers[frontier_id].room_id
            for robot_id, frontier_id in assignments.items()
        }
        reasons = {
            robot_id: fallback_reason for robot_id in assignments
        }
        self._store_audit(
            current_frontiers,
            assignments,
            probability_records,
            selected_rooms,
            reasons,
            source,
            "invalid_after_repair",
            stage_statuses,
            validation_errors=validation_errors,
            fallback_reason=fallback_reason,
            tool_trace=tool_trace,
            llm_call_count=llm_call_count,
            recent_history=recent_history,
            room_history_summary=room_history_summary,
        )
        if step is not None:
            self._record_assignment_history(
                current_frontiers, assignments, step
            )
        return (
            assignments,
            list(tools_called) + ["current_frontier_fallback"],
            f"helicase_fallback({fallback_reason})",
        )

    def decide(self, kg: KnowledgeGraph, target_name: str,
               enriched_frontiers, pose_pred, step: int, max_steps: int,
               decision_history: list,
               current_frontiers: Optional[CurrentFrontierView] = None
               ) -> Tuple[Dict, List[str], str]:
        """Run two LLM stages and return a validated room-first assignment."""
        _ = decision_history  # Stable history is maintained internally.
        if current_frontiers is None:
            current_frontiers = CurrentFrontierView.from_enriched(
                enriched_frontiers
            )
        if not current_frontiers.frontiers:
            self.last_decision_audit = {
                "allowed_room_ids": [],
                "allowed_frontier_ids": [],
                "raw_room_probabilities": {},
                "accepted_room_probabilities": {},
                "raw_room_assignments": {},
                "accepted_llm_rooms": {},
                "selected_room_by_robot": {},
                "selection_source_by_robot": {},
                "rejected_room_ids": [],
                "fallback_robots": [],
                "final_frontier_assignments": {},
                "llm_output_status": "no_current_frontiers",
                "tool_status": {},
                "llm_call_count": 0,
            }
            return {}, ["no_current_frontiers"], "helicase_no_frontiers"

        kg_text = kg.to_text(current_frontier_view=current_frontiers)
        current_room_ids = sorted(current_frontiers.room_ids)
        frontier_options = self._frontier_options(
            current_frontiers,
            pose_pred,
            self.num_agents,
        )
        recent_history = self._prepare_recent_history(kg, step)
        room_history_summary = self._room_history_summary(
            current_room_ids, recent_history
        )
        validation_errors = []
        stage_statuses = {"query_room_objects": "executed"}
        tools_called = ["query_room_objects"]
        llm_call_count = 0

        query_results, allowed_evidence = self._query_room_objects(
            kg,
            current_frontiers,
            current_room_ids,
        )
        tool_trace = [{
            "tool": "query_room_objects",
            "status": "executed",
            "result": query_results,
        }]

        probability_call_identities = [{
            "name": "estimate_room_probability",
            "room_id": room_id,
            "target": target_name,
            "allowed_evidence_ids": sorted(allowed_evidence[room_id]),
        } for room_id in current_room_ids]
        probability_prompt = (
            "You are the central MindNav KG reasoning LLM. This is LLM CALL "
            "1/2 for the current planning decision. Read the complete "
            "serialized KG and deterministic room-object query results. For "
            "every current room, call estimate_room_probability exactly once. "
            "Estimate your own P(target|room) using observed room type, object "
            "identity and certainty, exploration evidence, and your parametric "
            "world knowledge. probability means target presence likelihood; "
            "confidence means evidence reliability. Independent room "
            "probabilities need not sum to one. Do not use a Python target "
            "prior; none is provided. Return pure JSON with only tool_calls. "
            "Each call arguments must contain exactly room_id, target, "
            "probability, confidence, evidence_ids, reason. Copy evidence_ids "
            "only from that room's allowed list. These citations are stable "
            "room/object node IDs; never cite relation labels or frontier "
            "IDs. You may use the complete KG for reasoning even when a fact "
            "is not cited as an evidence_id.\n\n"
            f"TARGET: {target_name}\n"
            f"STEP: {int(step)}/{int(max_steps)}\n"
            f"CURRENT_ROOM_IDS: {json.dumps(current_room_ids)}\n"
            "REQUIRED_CALL_IDENTITIES: Output exactly one complete call for "
            "each entry below, in this order. Preserve its name, room_id and "
            "target. Choose probability, confidence, evidence_ids and reason "
            "yourself; evidence_ids must be a subset of that entry's allowed "
            "list.\n"
            f"{json.dumps(probability_call_identities, ensure_ascii=False)}\n\n"
            f"SERIALIZED_KG:\n{kg_text}\n\n"
            f"QUERY_ROOM_OBJECTS_RESULTS:\n"
            f"{json.dumps(query_results, ensure_ascii=False)}"
        )
        probability_records, probability_status, errors, responses = (
            self._call_json_stage(
                "estimate_room_probability",
                probability_prompt,
                lambda payload: self._validate_probability_calls(
                    payload,
                    current_room_ids,
                    target_name,
                    allowed_evidence,
                ),
                max_tokens=1200,
            )
        )
        llm_call_count += len(responses)
        stage_statuses["estimate_room_probability"] = probability_status
        validation_errors.extend(errors)
        tool_trace.append({
            "tool": "estimate_room_probability",
            "status": probability_status,
            "raw_response": responses[-1][:6000],
        })
        if probability_records is None:
            return self._finish_fallback(
                current_frontiers,
                pose_pred,
                {},
                stage_statuses,
                validation_errors,
                "invalid_estimate_room_probability",
                tool_trace,
                tools_called,
                step=step,
                llm_call_count=llm_call_count,
                recent_history=recent_history,
                room_history_summary=room_history_summary,
            )

        tools_called.append("estimate_room_probability")
        probability_tool_result = [{
            "room_id": room_id,
            **probability_records[room_id],
        } for room_id in current_room_ids]
        tool_trace[-1]["result"] = probability_tool_result

        decision_packet, room_history_summary = (
            self._build_current_decision_packet(
                query_results,
                probability_records,
                frontier_options,
                pose_pred,
                current_room_ids,
                recent_history,
            )
        )
        distinct_rooms_required = (
            len(current_room_ids) >= self.num_agents
        )
        valid_room_frontier_pairs = decision_packet["constraints"][
            "valid_room_frontier_pairs"
        ]
        assignment_prompt = (
            "You are the central MindNav allocation LLM. This is LLM CALL "
            "2/2. Use only the current decision packet below; it is the "
            "task-relevant projection of the KG. Make the final joint "
            "assignment yourself—Python will not rerank a valid answer.\n\n"
            "Follow a strict room-first procedure: (A) compare current rooms "
            "using target_probability, confidence, objects, exploration state "
            "and recent progress; (B) choose one room per robot; (C) within "
            "each chosen room select one of that room's frontier IDs using "
            "robot distance and frontier area. Avoid repeatedly sending the "
            "same robot to a room when recent assignments produced no new "
            "object or room evidence and another current room is available. "
            "Repetition is allowed when new evidence appeared, the room has "
            "clearly stronger target likelihood, or no alternative exists.\n\n"
            f"ROOM_DIVERSITY_REQUIRED: {str(distinct_rooms_required).lower()}. "
            "When true, assigned room_id values MUST be distinct. When false, "
            "robots may share the only available room but must use distinct "
            "frontiers whenever enough frontier choices exist. Frontier IDs "
            "must always belong to the stated room. The authoritative legal "
            "pairs are listed in VALID_ROOM_FRONTIER_PAIRS below. Copy each "
            "selected pair intact; NEVER combine a room_id from one entry "
            "with a frontier_id from another entry.\n\n"
            f"VALID_ROOM_FRONTIER_PAIRS: "
            f"{json.dumps(valid_room_frontier_pairs)}\n\n"
            "Return pure JSON with only tool_calls and exactly one "
            "assign_frontiers call. Its arguments must be {assignments, "
            "diversity}; diversity must be true. assignments must map every "
            "robot ID to exactly {frontier_id, room_id, reason}. No extra "
            "fields and no explanation outside JSON. The robot ID is already "
            "the assignments map key; NEVER repeat robot_id inside its value. "
            "Copy this exact wrapper "
            "shape (replace only the argument values): "
            '{"tool_calls":[{"name":"assign_frontiers","arguments":'
            '{"assignments":{"robot_0":{"frontier_id":0,"room_id":'
            '"room_ID","reason":"reason"},"robot_1":{"frontier_id":1,'
            '"room_id":"room_ID","reason":"reason"}},"diversity":true}}]}'
            " Do not add a top-level assign_frontiers field.\n\n"
            f"TARGET: {target_name}\n"
            f"STEP: {int(step)}/{int(max_steps)}\n"
            f"CURRENT_DECISION_PACKET:\n"
            f"{json.dumps(decision_packet, ensure_ascii=False)}"
        )
        assignment_result, assignment_status, errors, responses = (
            self._call_json_stage(
                "assign_frontiers",
                assignment_prompt,
                lambda payload: self._validate_assignment_calls(
                    payload, current_frontiers
                ),
                max_tokens=750,
            )
        )
        llm_call_count += len(responses)
        stage_statuses["assign_frontiers"] = assignment_status
        validation_errors.extend(errors)
        tool_trace.append({
            "tool": "assign_frontiers",
            "status": assignment_status,
            "raw_response": responses[-1][:5000],
        })
        if assignment_result is None:
            return self._finish_fallback(
                current_frontiers,
                pose_pred,
                probability_records,
                stage_statuses,
                validation_errors,
                "invalid_assign_frontiers",
                tool_trace,
                tools_called,
                step=step,
                llm_call_count=llm_call_count,
                recent_history=recent_history,
                room_history_summary=room_history_summary,
            )

        assignments, assigned_rooms, assignment_reasons = assignment_result
        tools_called.append("assign_frontiers")
        tool_trace[-1]["result"] = {
            "assignments": assignments,
            "rooms": assigned_rooms,
            "diversity": True,
        }
        overall_status = (
            "repaired"
            if "repaired" in stage_statuses.values() else "valid"
        )
        self._store_audit(
            current_frontiers,
            assignments,
            probability_records,
            assigned_rooms,
            assignment_reasons,
            "llm_tool_assignment",
            overall_status,
            stage_statuses,
            validation_errors=validation_errors,
            tool_trace=tool_trace,
            llm_call_count=llm_call_count,
            recent_history=recent_history,
            room_history_summary=room_history_summary,
        )
        self._record_assignment_history(
            current_frontiers, assignments, step
        )

        probability_text = ",".join(
            f"{room_id}={probability_records[room_id]['probability']:.2f}"
            for room_id in sorted(
                probability_records,
                key=lambda room_id: (
                    -probability_records[room_id]["probability"], room_id
                ),
            )
        )
        assignment_text = ",".join(
            f"{robot_id}=f{assignments[robot_id]}"
            for robot_id in sorted(assignments)
        )
        return (
            assignments,
            tools_called,
            f"helicase_room_first({overall_status};P={probability_text};"
            f"{assignment_text})",
        )

    def deterministic_fallback(self, current_frontiers, pose_pred,
                               reason="brain_error"):
        """Transport-level current-only fallback after external retries."""
        assignments = self._fallback_assignments(
            current_frontiers,
            pose_pred,
        )
        selected_rooms = {
            robot_id: current_frontiers.frontiers[frontier_id].room_id
            for robot_id, frontier_id in assignments.items()
        }
        self._store_audit(
            current_frontiers,
            assignments,
            {},
            selected_rooms,
            {robot_id: reason for robot_id in assignments},
            "geometry_fallback",
            "transport_fallback",
            {},
            fallback_reason=reason,
        )
        return (
            assignments,
            ["current_frontier_fallback"],
            f"helicase_fallback({reason})",
        )

    def reflect(self, brain, target_name, success, steps, dtg):
        """Store a short post-episode lesson for later LLM decisions."""
        outcome = "SUCCESS" if success else "FAILED"
        lesson = brain.call(
            f"Episode: {outcome}. Target: {target_name}. Steps: {steps}. "
            f"Distance: {dtg:.1f}m. What one lesson should the KG decision "
            "brain remember?",
            max_tokens=60,
        )
        self.reflexion_memory.append(
            f"[{target_name},{outcome}] {lesson[:120]}"
        )
        if len(self.reflexion_memory) > 20:
            self.reflexion_memory = self.reflexion_memory[-20:]


__all__ = [
    "CurrentFrontierView",
    "HelicaseBrain",
    "KGEdge",
    "KGNode",
    "KGUpdater",
    "KnowledgeGraph",
]
