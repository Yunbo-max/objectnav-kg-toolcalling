"""MindNav heuristic frontier assignment baseline.

This baseline intentionally does not call any LLM. It uses the same frontier
enrichment and optional live KG target detection as MindNav-KG, then assigns
robots to high-scoring diverse frontiers with deterministic rules.
"""

import math
from typing import Dict, List, Tuple

import numpy as np


class MindNavHeuristicBrain:
    def __init__(
        self,
        num_agents: int = 2,
        target_tau: float = 0.5,
        target_prior_weight: float = 4.0,
        room_confidence_weight: float = 0.25,
        area_weight: float = 0.15,
        distance_weight: float = 0.35,
        revisit_penalty: float = 0.45,
        diversity_weight: float = 0.15,
    ):
        self.num_agents = num_agents
        self.target_tau = target_tau
        self.target_prior_weight = target_prior_weight
        self.room_confidence_weight = room_confidence_weight
        self.area_weight = area_weight
        self.distance_weight = distance_weight
        self.revisit_penalty = revisit_penalty
        self.diversity_weight = diversity_weight

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
        if not enriched_frontiers:
            return {}, ["heuristic_no_frontiers"], "mindnav_heuristic(no_frontiers)"

        target_goal = self._priority_target_goal(kg, target_name, enriched_frontiers, pose_pred)
        if target_goal:
            return target_goal, ["heuristic_check_target"], "mindnav_heuristic_target_found"

        recent = self._recent_frontier_counts(decision_history)
        scores = self._score_frontiers(enriched_frontiers, pose_pred, recent)
        assignments = self._assign_diverse(enriched_frontiers, scores)
        score_str = self._format_scores(enriched_frontiers, scores)
        return assignments, ["heuristic_score_frontiers", "heuristic_assign_frontiers"], f"mindnav_heuristic({score_str})"

    def _priority_target_goal(self, kg, target_name, enriched_frontiers, pose_pred):
        target_node = kg.nodes.get(f"TARGET_{target_name}")
        if target_node is None:
            for node in kg.get_nodes_by_type("object"):
                if node.properties.get("category") == target_name and node.certainty > self.target_tau:
                    target_node = node
                    break
        if target_node is None or target_node.certainty <= self.target_tau:
            return {}

        target_pos = np.array(target_node.position, dtype=float)
        best_frontier = min(
            enriched_frontiers,
            key=lambda ef: np.linalg.norm(target_pos - np.array(ef["centroid"], dtype=float)),
        )["idx"]

        robot_dists = []
        for idx, pos in enumerate(pose_pred):
            robot_dists.append((idx, np.linalg.norm(np.array(pos[:2], dtype=float) - target_pos)))
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

    def _recent_frontier_counts(self, decision_history):
        counts = {}
        for item in decision_history[-4:]:
            assignments = item.get("assignments", {})
            if not isinstance(assignments, dict):
                continue
            for frontier_idx in assignments.values():
                try:
                    frontier_idx = int(frontier_idx)
                except (TypeError, ValueError):
                    continue
                counts[frontier_idx] = counts.get(frontier_idx, 0) + 1
        return counts

    def _score_frontiers(self, enriched_frontiers, pose_pred, recent):
        scores = {}
        for robot_idx in range(self.num_agents):
            robot_pos = np.array(pose_pred[min(robot_idx, len(pose_pred) - 1)][:2], dtype=float)
            robot_scores = {}
            for frontier in enriched_frontiers:
                idx = int(frontier["idx"])
                centroid = np.array(frontier["centroid"], dtype=float)
                target_prior = self._safe_float(frontier.get("target_prior", 0.0))
                room_conf = self._safe_float(frontier.get("room_confidence", 0.0))
                area = self._safe_float(frontier.get("area", 0.0))
                dist = float(np.linalg.norm(robot_pos - centroid))

                score = 0.0
                score += self.target_prior_weight * target_prior
                score += self.room_confidence_weight * min(room_conf, 1.0)
                score += self.area_weight * min(area / 80.0, 1.0)
                score -= self.distance_weight * min(dist / 500.0, 1.0)
                score -= self.revisit_penalty * recent.get(idx, 0)
                robot_scores[idx] = score
            scores[f"robot_{robot_idx}"] = robot_scores
        return scores

    def _assign_diverse(self, enriched_frontiers, scores):
        frontier_idxs = [int(frontier["idx"]) for frontier in enriched_frontiers]
        if not frontier_idxs:
            return {}
        if self.num_agents == 1 or len(frontier_idxs) == 1:
            best = max(frontier_idxs, key=lambda idx: scores["robot_0"].get(idx, -math.inf))
            return {f"robot_{idx}": int(best) for idx in range(self.num_agents)}

        centroid_by_idx = {
            int(frontier["idx"]): np.array(frontier["centroid"], dtype=float)
            for frontier in enriched_frontiers
        }
        best_pair = None
        best_score = -math.inf
        for first in frontier_idxs:
            for second in frontier_idxs:
                if first == second:
                    continue
                pair_dist = float(np.linalg.norm(centroid_by_idx[first] - centroid_by_idx[second]))
                pair_score = (
                    scores["robot_0"].get(first, -math.inf)
                    + scores["robot_1"].get(second, -math.inf)
                    + self.diversity_weight * min(pair_dist / 500.0, 1.0)
                )
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (first, second)

        assignments = {"robot_0": int(best_pair[0]), "robot_1": int(best_pair[1])}
        for robot_idx in range(2, self.num_agents):
            robot = f"robot_{robot_idx}"
            used = set(assignments.values())
            candidates = [idx for idx in frontier_idxs if idx not in used] or frontier_idxs
            assignments[robot] = int(max(candidates, key=lambda idx: scores[robot].get(idx, -math.inf)))
        return assignments

    def _format_scores(self, enriched_frontiers, scores):
        if not enriched_frontiers:
            return "none"
        combined = []
        for frontier in enriched_frontiers:
            idx = int(frontier["idx"])
            vals = [robot_scores.get(idx, 0.0) for robot_scores in scores.values()]
            combined.append((idx, max(vals)))
        combined.sort(key=lambda item: item[1], reverse=True)
        return "S=" + ",".join(f"frontier_{idx}:{score:.2f}" for idx, score in combined[:4])

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return value
