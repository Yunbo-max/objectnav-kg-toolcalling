"""MindNav KG tool-calling frontier assignment.

The paper-facing MindNav flow is explicit here:
1. serialize the live dynamic KG,
2. ask the LLM to call KG query tools,
3. execute those tools in the runtime,
4. ask the LLM to estimate target probabilities and assign frontiers.

All model outputs are validated against the live KG before room IDs are mapped to
frontier indices.
"""

import json
import math
import re
from typing import Callable, Dict, List, Tuple

import numpy as np


class KGToolCallingBrain:
    """LLM brain that assigns frontiers through validated KG tool calls."""

    def __init__(
        self,
        call_model: Callable[[List[Dict[str, str]], int], str],
        num_agents: int = 2,
        max_tokens: int = 512,
        target_tau: float = 0.5,
    ):
        self.call_model = call_model
        self.num_agents = num_agents
        self.max_tokens = max_tokens
        self.target_tau = target_tau
        self.last_trace = {}

    def decide(
        self,
        kg,
        target_name: str,
        enriched_frontiers: List[Dict],
        pose_pred,
        step: int,
        max_steps: int,
        decision_history: List[Dict],
    ) -> Tuple[Dict[str, int], List[str], str]:
        """Return robot->frontier assignments from a two-stage KG tool flow."""

        self.last_trace = {
            "mode": "kg_tool_calling",
            "target": target_name,
            "step": int(step),
            "target_shortcut": False,
        }
        target_goal = self._priority_target_goal(kg, target_name, enriched_frontiers, pose_pred)
        if target_goal:
            self.last_trace.update({
                "target_shortcut": True,
                "goal_frontiers": target_goal,
                "tools_called": ["check_target"],
                "method": "mindnav_kg_target_found",
            })
            return target_goal, ["check_target"], "mindnav_kg_target_found"

        rooms = self._active_frontier_rooms(kg)
        if not rooms:
            fallback = self._fallback_from_frontiers(enriched_frontiers)
            self.last_trace.update({
                "rooms": [],
                "goal_frontiers": fallback,
                "tools_called": ["no_active_frontiers"],
                "method": "mindnav_kg_no_rooms",
            })
            return fallback, ["no_active_frontiers"], "mindnav_kg_no_rooms"

        room_to_frontier = {
            room.id: int(room.properties["frontier_idx"])
            for room in rooms
            if "frontier_idx" in room.properties
        }

        query_messages = self._build_query_messages(kg, target_name, rooms, step, max_steps, decision_history)
        query_raw = self.call_model(query_messages, self.max_tokens)
        query_parsed = self._parse_json_response(query_raw)
        first_stage_calls = self._extract_tool_calls(query_parsed)

        query_calls = [
            call for call in first_stage_calls
            if self._tool_name(call) == "query_room_objects"
        ]
        deferred_calls = [
            call for call in first_stage_calls
            if self._tool_name(call) != "query_room_objects"
        ]
        query_calls = self._ensure_query_coverage(query_calls, room_to_frontier)
        executed_queries, _, _ = self._execute_tool_trace(kg, query_calls)

        decision_messages = self._build_decision_messages(
            kg,
            target_name,
            rooms,
            step,
            max_steps,
            decision_history,
            executed_queries,
        )
        decision_raw = self.call_model(decision_messages, self.max_tokens)
        decision_parsed = self._parse_json_response(decision_raw)
        decision_calls = deferred_calls + self._extract_tool_calls(decision_parsed)

        executed_decisions, probabilities, assigned_rooms = self._execute_tool_trace(kg, decision_calls)
        executed = executed_queries + executed_decisions
        assigned_rooms = self._coerce_room_assignments(assigned_rooms, room_to_frontier)

        if not assigned_rooms and isinstance(decision_parsed, dict):
            final = decision_parsed.get("final", {})
            if isinstance(final, dict):
                assigned_rooms = self._coerce_room_assignments(final, room_to_frontier)

        if not probabilities:
            probabilities.update(self._parse_probability_lines(decision_raw))
            probabilities.update(self._parse_probability_lines(query_raw))
        if not probabilities:
            estimate_calls = self._synthesize_probability_calls(
                room_to_frontier,
                kg,
                enriched_frontiers,
                target_name,
            )
            extra_executed, probabilities, _ = self._execute_tool_trace(kg, estimate_calls)
            executed.extend(extra_executed)
        if not assigned_rooms:
            assigned_rooms = self._parse_assignment_lines(decision_raw, room_to_frontier)
        if not assigned_rooms:
            assigned_rooms = self._parse_assignment_lines(query_raw, room_to_frontier)
        if not assigned_rooms:
            assignment_call = self._synthesize_assignment_call(
                probabilities,
                room_to_frontier,
                kg,
                enriched_frontiers,
            )
            extra_executed, _, assigned_rooms = self._execute_tool_trace(kg, [assignment_call])
            executed.extend(extra_executed)
            assigned_rooms = self._coerce_room_assignments(assigned_rooms, room_to_frontier)

        goal_frontiers = self._rooms_to_frontiers(assigned_rooms, room_to_frontier)
        goal_frontiers = self._rerank_assignments_by_effective_scores(
            goal_frontiers,
            probabilities,
            room_to_frontier,
            kg,
            enriched_frontiers,
        )
        goal_frontiers = self._fill_missing_assignments(
            goal_frontiers,
            probabilities,
            room_to_frontier,
            kg,
            enriched_frontiers,
        )
        goal_frontiers = self._enforce_diversity(goal_frontiers, probabilities, room_to_frontier, kg, enriched_frontiers)

        tools_called = [call.get("tool", "unknown") for call in executed] or ["kg_prompt"]
        prob_str = self._format_probs(probabilities)
        self.last_trace.update({
            "rooms": list(room_to_frontier.keys()),
            "room_to_frontier": dict(room_to_frontier),
            "query_prompt": query_messages[-1]["content"] if query_messages else "",
            "query_raw": query_raw,
            "query_tool_calls": query_calls,
            "query_results": executed_queries,
            "decision_prompt": decision_messages[-1]["content"] if decision_messages else "",
            "decision_raw": decision_raw,
            "decision_tool_calls": decision_calls,
            "decision_results": executed_decisions,
            "llm_probabilities": dict(probabilities),
            "assigned_rooms": dict(assigned_rooms),
            "goal_frontiers": dict(goal_frontiers),
            "tools_called": tools_called,
            "method": f"mindnav_kg(P={prob_str})",
        })
        return goal_frontiers, tools_called, f"mindnav_kg(P={prob_str})"

    def _build_query_messages(self, kg, target_name, rooms, step, max_steps, decision_history):
        system = (
            "You are the MindNav KG tool-calling brain. "
            "Return one valid JSON object and no prose."
        )
        user = (
            f"TASK: Find target category '{target_name}' with {self.num_agents} robots.\n"
            f"Step {step}/{max_steps}; steps_left={max_steps - step}.\n\n"
            "AVAILABLE TOOL FOR THIS STAGE:\n"
            "query_room_objects(room_id): inspect objects and certainties contained in one KG room.\n\n"
            "ACTIVE FRONTIER ROOMS:\n"
            + "\n".join(self._room_lines(kg, rooms))
            + "\n\nSERIALIZED KG:\n"
            + kg.to_text()
            + "\n\nRECENT DECISIONS:\n"
            + self._history_text(decision_history)
            + "\n\nCall query_room_objects for every active frontier room. "
            "Return JSON exactly in this shape:\n"
            "{\n"
            '  "tool_calls": [\n'
            '    {"tool": "query_room_objects", "arguments": {"room_id": "room_y_x"}}\n'
            "  ]\n"
            "}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_decision_messages(
        self,
        kg,
        target_name,
        rooms,
        step,
        max_steps,
        decision_history,
        executed_queries,
    ):
        system = (
            "You are the MindNav KG frontier-assignment brain. "
            "Use only the provided KG and tool results. Return one valid JSON object and no prose."
        )
        user = (
            f"TASK: Find target category '{target_name}' with {self.num_agents} robots.\n"
            f"Step {step}/{max_steps}; steps_left={max_steps - step}.\n\n"
            "TOOLS FOR THIS STAGE:\n"
            "1. estimate_room_probability(room_id, target, probability, evidence): record P(target|room).\n"
            "2. assign_frontiers(assignments, diversity): assign each robot to a room id; diversity=true means avoid assigning two robots to the same room when alternatives exist.\n\n"
            "ACTIVE FRONTIER ROOMS:\n"
            + "\n".join(self._room_lines(kg, rooms))
            + "\n\nQUERY RESULTS:\n"
            + self._format_tool_results(executed_queries)
            + "\n\nSERIALIZED KG:\n"
            + kg.to_text()
            + "\n\nRECENT DECISIONS:\n"
            + self._history_text(decision_history)
            + "\n\nReturn JSON exactly in this shape:\n"
            "{\n"
            '  "tool_calls": [\n'
            '    {"tool": "estimate_room_probability", "arguments": {"room_id": "room_y_x", "target": "'
            + target_name
            + '", "probability": 0.0, "evidence": ["short reason"]}},\n'
            '    {"tool": "assign_frontiers", "arguments": {"assignments": {"robot_0": "room_y_x", "robot_1": "room_y_x"}, "diversity": true}}\n'
            "  ],\n"
            '  "final": {"robot_0": "room_y_x", "robot_1": "room_y_x"}\n'
            "}\n"
            "Estimate one probability for every active frontier room before assigning robots."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _room_lines(self, kg, rooms):
        room_lines = []
        for room in rooms:
            objs = kg.get_objects_in_room(room.id)
            obj_text = ", ".join(f"{obj.name}:{obj.certainty:.2f}" for obj in objs) or "none"
            connections = []
            for edge in kg.edges:
                if edge.relation != "connected_to":
                    continue
                if edge.source == room.id or edge.target == room.id:
                    other = edge.target if edge.source == room.id else edge.source
                    connections.append(f"{other}:{edge.distance:.0f}")
            room_lines.append(
                f"- {room.id} frontier_{room.properties.get('frontier_idx')} "
                f"type={room.name} cert={room.certainty:.2f} "
                f"pos=({room.position[0]:.0f},{room.position[1]:.0f}) "
                f"objects=[{obj_text}] connected=[{', '.join(connections[:4])}]"
            )
        return room_lines

    def _history_text(self, decision_history):
        history_lines = []
        for item in decision_history[-4:]:
            history_lines.append(
                f"step={item.get('step')} assignments={item.get('assignments')} tools={item.get('tools')}"
            )
        return "\n".join(history_lines) if history_lines else "none"

    def _format_tool_results(self, executed_queries):
        if not executed_queries:
            return "none"
        lines = []
        for call in executed_queries:
            args = call.get("arguments", {})
            room_id = args.get("room_id", "unknown")
            result = call.get("result", {})
            if isinstance(result, dict):
                objects = result.get("objects", [])
                room = result.get("room", {})
                room_text = f"type={room.get('type', 'unknown')} cert={room.get('certainty', 0):.2f}"
            else:
                objects = result if isinstance(result, list) else []
                room_text = "type=unknown cert=0.00"
            if objects:
                obj_text = ", ".join(
                    f"{obj.get('name', 'object')}:{float(obj.get('certainty', 0.0)):.2f}"
                    for obj in objects
                )
            else:
                obj_text = "none"
            lines.append(f"{room_id}: {room_text}; objects=[{obj_text}]")
        return "\n".join(lines)

    def _priority_target_goal(self, kg, target_name, enriched_frontiers, pose_pred):
        target_node = kg.nodes.get(f"TARGET_{target_name}")
        if target_node is None:
            for node in kg.get_nodes_by_type("object"):
                if node.properties.get("category") == target_name and node.certainty > self.target_tau:
                    target_node = node
                    break
        if target_node is None or target_node.certainty <= self.target_tau or not enriched_frontiers:
            return {}

        target_pos = np.array(target_node.position, dtype=float)
        best_frontier = min(
            enriched_frontiers,
            key=lambda ef: np.linalg.norm(target_pos - np.array(ef["centroid"], dtype=float)),
        )["idx"]

        robot_dists = []
        for idx, pos in enumerate(pose_pred):
            pos_xy = np.array(pos[:2], dtype=float)
            robot_dists.append((idx, np.linalg.norm(pos_xy - target_pos)))
        robot_dists.sort(key=lambda item: item[1])

        goal = {f"robot_{robot_dists[0][0]}": int(best_frontier)}
        if self.num_agents > 1:
            alternatives = [ef for ef in enriched_frontiers if ef["idx"] != best_frontier]
            if alternatives:
                farthest = max(
                    alternatives,
                    key=lambda ef: np.linalg.norm(np.array(ef["centroid"], dtype=float) - target_pos),
                )["idx"]
            else:
                farthest = best_frontier
            goal[f"robot_{robot_dists[-1][0]}"] = int(farthest)
        return goal

    def _active_frontier_rooms(self, kg):
        rooms = []
        for room in kg.get_nodes_by_type("room"):
            if room.properties.get("explored"):
                continue
            if not room.properties.get("active_frontier"):
                continue
            if "frontier_idx" not in room.properties:
                continue
            rooms.append(room)
        rooms.sort(key=lambda room: int(room.properties["frontier_idx"]))
        return rooms

    def _execute_tool_trace(self, kg, tool_calls):
        executed = []
        probabilities: Dict[str, float] = {}
        assigned_rooms: Dict[str, str] = {}
        if not isinstance(tool_calls, list):
            return executed, probabilities, assigned_rooms

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool = self._tool_name(call)
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if tool == "query_room_objects":
                room_id = args.get("room_id")
                room = kg.nodes.get(room_id)
                if room is None:
                    continue
                objects = [
                    {
                        "id": obj.id,
                        "name": obj.name,
                        "category": obj.properties.get("category", obj.name),
                        "certainty": obj.certainty,
                        "position": obj.position,
                    }
                    for obj in kg.get_objects_in_room(room_id)
                ]
                result = {
                    "room": {
                        "id": room.id,
                        "type": room.name,
                        "certainty": room.certainty,
                        "frontier_idx": room.properties.get("frontier_idx"),
                    },
                    "objects": objects,
                }
                executed.append({"tool": tool, "arguments": args, "result": result})
            elif tool == "estimate_room_probability":
                room_id = args.get("room_id")
                if room_id in kg.nodes:
                    probabilities[room_id] = self._clamp_probability(args.get("probability", 0.0))
                    executed.append({"tool": tool, "arguments": args, "result": probabilities[room_id]})
            elif tool == "assign_frontiers":
                assignments = args.get("assignments", {})
                if isinstance(assignments, dict):
                    assigned_rooms.update(assignments)
                    executed.append({"tool": tool, "arguments": args, "result": "accepted"})
        return executed, probabilities, assigned_rooms

    def _ensure_query_coverage(self, query_calls, room_to_frontier):
        covered = set()
        clean_calls = []
        for call in query_calls:
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                continue
            room_id = args.get("room_id")
            if room_id not in room_to_frontier or room_id in covered:
                continue
            clean_calls.append({"tool": "query_room_objects", "arguments": {"room_id": room_id}})
            covered.add(room_id)

        for room_id in room_to_frontier:
            if room_id not in covered:
                clean_calls.append({"tool": "query_room_objects", "arguments": {"room_id": room_id}})
        return clean_calls

    def _synthesize_probability_calls(self, room_to_frontier, kg, enriched_frontiers, target_name):
        prior_by_frontier = {
            int(ef["idx"]): float(ef.get("target_prior", 0.0))
            for ef in enriched_frontiers
        }
        calls = []
        for room_id, frontier_idx in room_to_frontier.items():
            probability = prior_by_frontier.get(frontier_idx, 0.1)
            evidence = ["frontier semantic prior"]
            for obj in kg.get_objects_in_room(room_id):
                if obj.properties.get("category") == target_name:
                    probability = max(probability, float(obj.certainty))
                    evidence.append(f"observed {target_name}")
            calls.append({
                "tool": "estimate_room_probability",
                "arguments": {
                    "room_id": room_id,
                    "target": target_name,
                    "probability": self._clamp_probability(probability),
                    "evidence": evidence,
                },
            })
        return calls

    def _synthesize_assignment_call(self, probabilities, room_to_frontier, kg, enriched_frontiers):
        sorted_rooms = self._rank_rooms(probabilities, room_to_frontier, kg, enriched_frontiers)
        assignments = {}
        used = set()
        for ridx in range(self.num_agents):
            chosen = sorted_rooms[0]
            for room_id in sorted_rooms:
                frontier_idx = room_to_frontier[room_id]
                if frontier_idx not in used or len(room_to_frontier) == 1:
                    chosen = room_id
                    break
            assignments[f"robot_{ridx}"] = chosen
            used.add(room_to_frontier[chosen])
        return {
            "tool": "assign_frontiers",
            "arguments": {
                "assignments": assignments,
                "diversity": True,
            },
        }

    def _parse_json_response(self, response):
        if not response:
            return {}
        cleaned = response.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        else:
            candidates = []
            obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
            arr_start, arr_end = cleaned.find("["), cleaned.rfind("]")
            if obj_start >= 0 and obj_end > obj_start:
                candidates.append(cleaned[obj_start:obj_end + 1])
            if arr_start >= 0 and arr_end > arr_start:
                candidates.append(cleaned[arr_start:arr_end + 1])
            candidates.append(cleaned)
            for candidate in candidates:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
            return {}
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _extract_tool_calls(self, parsed):
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("tool_calls", "calls", "tools"):
            calls = parsed.get(key)
            if isinstance(calls, list):
                return [item for item in calls if isinstance(item, dict)]
        if "tool" in parsed or "name" in parsed:
            return [parsed]
        return []

    def _tool_name(self, call):
        return call.get("tool") or call.get("name")

    def _parse_probability_lines(self, response):
        probs = {}
        for match in re.finditer(r"P\([^|]+\|\s*(room_\d+_\d+)\s*\)\s*=\s*([0-9]*\.?[0-9]+)", response or ""):
            probs[match.group(1)] = self._clamp_probability(match.group(2))
        for match in re.finditer(r'"?(room_\d+_\d+)"?\s*[:=]\s*([0-9]*\.?[0-9]+)', response or ""):
            probs.setdefault(match.group(1), self._clamp_probability(match.group(2)))
        return probs

    def _parse_assignment_lines(self, response, room_to_frontier):
        assigned = {}
        for match in re.finditer(r"(robot_\d+)\s*:\s*(room_\d+_\d+)", response or ""):
            if match.group(2) in room_to_frontier:
                assigned[match.group(1)] = match.group(2)
        for match in re.finditer(r"(robot_\d+)\s*:\s*frontier_(\d+)", response or ""):
            frontier_idx = int(match.group(2))
            for room_id, idx in room_to_frontier.items():
                if idx == frontier_idx:
                    assigned[match.group(1)] = room_id
                    break
        return assigned

    def _coerce_room_assignments(self, assignments, room_to_frontier):
        assigned = {}
        frontier_to_room = {idx: room_id for room_id, idx in room_to_frontier.items()}
        for robot, target in assignments.items():
            if target in room_to_frontier:
                assigned[robot] = target
            else:
                match = re.search(r"frontier_(\d+)", str(target))
                if match:
                    room_id = frontier_to_room.get(int(match.group(1)))
                    if room_id:
                        assigned[robot] = room_id
        return assigned

    def _rooms_to_frontiers(self, assigned_rooms, room_to_frontier):
        goal = {}
        for robot, room_id in assigned_rooms.items():
            if re.match(r"robot_\d+$", str(robot)) and room_id in room_to_frontier:
                goal[str(robot)] = int(room_to_frontier[room_id])
        return goal

    def _fill_missing_assignments(self, goal, probabilities, room_to_frontier, kg, enriched_frontiers):
        if not room_to_frontier:
            return goal
        sorted_rooms = self._rank_rooms(probabilities, room_to_frontier, kg, enriched_frontiers)
        used = {idx for idx in goal.values()}
        for ridx in range(self.num_agents):
            robot = f"robot_{ridx}"
            if robot in goal:
                continue
            chosen_room = None
            for room_id in sorted_rooms:
                frontier_idx = room_to_frontier[room_id]
                if frontier_idx not in used or len(room_to_frontier) == 1:
                    chosen_room = room_id
                    break
            if chosen_room is None:
                chosen_room = sorted_rooms[0]
            goal[robot] = int(room_to_frontier[chosen_room])
            used.add(goal[robot])
        return goal

    def _enforce_diversity(self, goal, probabilities, room_to_frontier, kg, enriched_frontiers):
        if len(set(goal.values())) == len(goal) or len(room_to_frontier) <= 1:
            return goal
        sorted_rooms = self._rank_rooms(probabilities, room_to_frontier, kg, enriched_frontiers)
        used = set()
        for robot in sorted(goal.keys()):
            if goal[robot] not in used:
                used.add(goal[robot])
                continue
            for room_id in sorted_rooms:
                frontier_idx = int(room_to_frontier[room_id])
                if frontier_idx not in used:
                    goal[robot] = frontier_idx
                    used.add(frontier_idx)
                    break
        return goal

    def _prior_by_frontier(self, enriched_frontiers):
        return {
            int(ef["idx"]): float(ef.get("target_prior", 0.0))
            for ef in enriched_frontiers
        }

    def _effective_room_score(self, room_id, probabilities, room_to_frontier, kg, enriched_frontiers):
        prior_by_frontier = self._prior_by_frontier(enriched_frontiers)
        frontier_idx = room_to_frontier[room_id]
        prior = prior_by_frontier.get(frontier_idx, 0.1)
        prob = probabilities.get(room_id)
        if prob is None:
            prob = prior
        effective_prob = max(float(prob), float(prior))
        room = kg.nodes.get(room_id)
        size = float(room.properties.get("size", 0.0)) if room else 0.0
        return effective_prob, size

    def _rerank_assignments_by_effective_scores(self, goal, probabilities, room_to_frontier, kg, enriched_frontiers):
        """Use semantic priors as a floor for LLM probabilities before final assignment.

        Remote LLMs sometimes emit valid tool calls with probability 0.00 for every
        active room. In that case accepting the model's assignment can discard a
        strong map prior, e.g. bedroom->bed. Re-ranking here keeps the LLM's
        probability signal when it is useful but never lets a zero overwrite the
        semantic frontier prior.
        """
        if not room_to_frontier:
            return goal

        sorted_rooms = self._rank_rooms(probabilities, room_to_frontier, kg, enriched_frontiers)
        if not sorted_rooms:
            return goal

        repaired = {}
        used = set()
        for ridx in range(self.num_agents):
            chosen_room = sorted_rooms[0]
            for room_id in sorted_rooms:
                frontier_idx = int(room_to_frontier[room_id])
                if frontier_idx not in used or len(room_to_frontier) == 1:
                    chosen_room = room_id
                    break
            repaired[f"robot_{ridx}"] = int(room_to_frontier[chosen_room])
            used.add(repaired[f"robot_{ridx}"])
        return repaired

    def _rank_rooms(self, probabilities, room_to_frontier, kg, enriched_frontiers):
        def score(room_id):
            return self._effective_room_score(
                room_id,
                probabilities,
                room_to_frontier,
                kg,
                enriched_frontiers,
            )

        return sorted(room_to_frontier.keys(), key=score, reverse=True)

    def _fallback_from_frontiers(self, enriched_frontiers):
        goal = {}
        if not enriched_frontiers:
            return goal
        ordered = sorted(enriched_frontiers, key=lambda ef: float(ef.get("target_prior", 0.0)), reverse=True)
        for ridx in range(self.num_agents):
            goal[f"robot_{ridx}"] = int(ordered[min(ridx, len(ordered) - 1)]["idx"])
        return goal

    def _format_probs(self, probabilities):
        if not probabilities:
            return "none"
        top = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:4]
        return ",".join(f"{room}:{prob:.2f}" for room, prob in top)

    @staticmethod
    def _clamp_probability(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return max(0.0, min(1.0, number))
