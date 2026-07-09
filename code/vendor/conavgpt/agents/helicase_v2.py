"""
Helicase v2: KG + Real Bayesian Belief — all reasoning in code, not LLM.

Fixes from v1:
1. No LLM reasoning over KG text — belief computed algorithmically
2. Real Bayesian update P(target|area) from detected objects
3. Frontier selection by belief × unexplored score, not LLM choice
4. Spatial diversity enforced in code
5. Works with 0.5B brain (brain only does VLM perception, not reasoning)
"""
import numpy as np
import math
from typing import Dict, List, Tuple


# ═══════════════════════════════════════
# BAYESIAN BELIEF MAP
# ═══════════════════════════════════════

# P(target | room_type) — hardcoded priors
TARGET_ROOM_PRIOR = {
    "chair": {"bathroom": 0.02, "bedroom": 0.10, "living_room": 0.50, "kitchen": 0.25, "hallway": 0.05, "unknown": 0.08},
    "bed": {"bathroom": 0.01, "bedroom": 0.85, "living_room": 0.02, "kitchen": 0.01, "hallway": 0.01, "unknown": 0.10},
    "plant": {"bathroom": 0.05, "bedroom": 0.10, "living_room": 0.35, "kitchen": 0.10, "hallway": 0.30, "unknown": 0.10},
    "toilet": {"bathroom": 0.85, "bedroom": 0.01, "living_room": 0.01, "kitchen": 0.01, "hallway": 0.02, "unknown": 0.10},
    "tv_monitor": {"bathroom": 0.01, "bedroom": 0.20, "living_room": 0.60, "kitchen": 0.05, "hallway": 0.02, "unknown": 0.12},
    "sofa": {"bathroom": 0.01, "bedroom": 0.05, "living_room": 0.75, "kitchen": 0.02, "hallway": 0.02, "unknown": 0.15},
}

# Object → room type mapping
OBJECT_ROOM = {
    "toilet": "bathroom", "sink": "bathroom", "bathtub": "bathroom",
    "shower": "bathroom", "towel": "bathroom",
    "bed": "bedroom", "chest_of_drawers": "bedroom",
    "sofa": "living_room", "tv_monitor": "living_room", "fireplace": "living_room",
    "table": "kitchen", "chair": "living_room", "plant": "living_room",
    "stairs": "hallway",
}

# Object co-occurrence: P(target | nearby_object)
OBJECT_COOCCURRENCE = {
    "toilet": {"sink": 0.85, "bathtub": 0.9, "towel": 0.7},
    "bed": {"chest_of_drawers": 0.85},
    "chair": {"table": 0.6, "sofa": 0.4, "tv_monitor": 0.3},
    "tv_monitor": {"sofa": 0.7, "chair": 0.4},
    "sofa": {"tv_monitor": 0.6, "chair": 0.4, "fireplace": 0.3},
    "plant": {"chair": 0.2, "sofa": 0.2, "stairs": 0.3},
}


class BayesianBeliefMap:
    """Real Bayesian belief over frontier areas. All computation in code."""

    def __init__(self, target_name):
        self.target = target_name
        self.beliefs = {}  # frontier_idx → P(target here)
        self.explored = {}  # frontier_idx → exploration level (0-1)
        self.room_types = {}  # frontier_idx → inferred room type
        self.visit_count = {}  # frontier_idx → how many times chosen
        self.total_updates = 0

    def reset(self):
        self.beliefs.clear()
        self.explored.clear()
        self.room_types.clear()
        self.visit_count.clear()
        self.total_updates = 0

    def update_from_frontiers(self, enriched_frontiers, pose_pred):
        """Update belief for all frontiers based on nearby objects."""
        target_priors = TARGET_ROOM_PRIOR.get(self.target, {})
        cooccurrence = OBJECT_COOCCURRENCE.get(self.target, {})

        for ef in enriched_frontiers:
            idx = ef['idx']
            nearby = ef.get('nearby_objects', [])

            # Initialize if new frontier
            if idx not in self.beliefs:
                self.beliefs[idx] = 0.1  # uniform prior
                self.explored[idx] = 0.0
                self.visit_count[idx] = 0

            # Infer room type from objects
            room_scores = {}
            for obj in nearby:
                room = OBJECT_ROOM.get(obj)
                if room:
                    room_scores[room] = room_scores.get(room, 0) + 1

            if room_scores:
                best_room = max(room_scores, key=room_scores.get)
                self.room_types[idx] = best_room
            else:
                best_room = self.room_types.get(idx, "unknown")

            # Bayesian update: P(target|frontier) = P(target|room) * P(room|objects)
            room_prior = target_priors.get(best_room, 0.08)
            likelihood = room_prior

            # Boost from co-occurring objects
            for obj in nearby:
                if obj in cooccurrence:
                    likelihood = max(likelihood, cooccurrence[obj])

            # Target object directly nearby → very high
            if self.target in nearby:
                likelihood = 0.95

            # Update belief (soft update, don't replace)
            alpha = 0.3  # learning rate
            self.beliefs[idx] = (1 - alpha) * self.beliefs[idx] + alpha * likelihood

        # Decay belief for explored areas where target wasn't found
        for idx in self.explored:
            if self.explored[idx] > 0.5:
                self.beliefs[idx] *= 0.85  # reduce belief in explored areas

        # Normalize
        total = sum(self.beliefs.values())
        if total > 0:
            for idx in self.beliefs:
                self.beliefs[idx] /= total

        self.total_updates += 1

    def mark_explored(self, frontier_idx, robot_positions, frontier_centroids):
        """Mark frontier as explored if robot is nearby."""
        if frontier_idx not in self.explored:
            return

        for pos in robot_positions:
            if frontier_idx < len(frontier_centroids):
                centroid = frontier_centroids[frontier_idx]
                dist = np.sqrt((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)
                if dist < 40:
                    self.explored[frontier_idx] = min(1.0, self.explored[frontier_idx] + 0.2)

    def select_frontiers(self, enriched_frontiers, num_robots=2):
        """Select best frontiers for each robot — algorithmic, no LLM.

        Score = belief × (1 - explored) × (1 / (1 + visit_count))
        Robot 1 gets best score. Robot 2 gets best score that is FAR from robot 1.
        """
        if not enriched_frontiers:
            return {}

        # Score each frontier
        scores = []
        for ef in enriched_frontiers:
            idx = ef['idx']
            belief = self.beliefs.get(idx, 0.1)
            explored = self.explored.get(idx, 0.0)
            visits = self.visit_count.get(idx, 0)

            score = belief * (1.0 - explored) * (1.0 / (1.0 + visits))
            scores.append((idx, score, ef['centroid']))

        scores.sort(key=lambda x: -x[1])

        if not scores:
            return {}

        # Robot 0: best score
        r0_idx = scores[0][0]
        r0_centroid = scores[0][2]
        self.visit_count[r0_idx] = self.visit_count.get(r0_idx, 0) + 1

        # Robot 1: best score that is FAR from robot 0
        r1_idx = scores[min(1, len(scores)-1)][0]  # default: second best
        best_dist = 0
        for idx, score, centroid in scores:
            if idx == r0_idx:
                continue
            dist = np.sqrt((centroid[0] - r0_centroid[0])**2 +
                          (centroid[1] - r0_centroid[1])**2)
            # Combine score and distance
            combined = score * 0.5 + (dist / 500) * 0.5  # balance quality and diversity
            if combined > best_dist:
                best_dist = combined
                r1_idx = idx

        self.visit_count[r1_idx] = self.visit_count.get(r1_idx, 0) + 1

        result = {"robot_0": r0_idx, "robot_1": r1_idx}
        return result

    def get_entropy(self):
        """Shannon entropy of belief distribution."""
        values = list(self.beliefs.values())
        if not values:
            return 0
        total = sum(values)
        if total == 0:
            return 0
        entropy = 0
        for v in values:
            p = v / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_status(self):
        """Short status string for logging."""
        if not self.beliefs:
            return "no beliefs"
        best_idx = max(self.beliefs, key=self.beliefs.get)
        best_val = self.beliefs[best_idx]
        room = self.room_types.get(best_idx, "?")
        entropy = self.get_entropy()
        n_explored = sum(1 for v in self.explored.values() if v > 0.5)
        return f"best=f{best_idx}({room},{best_val:.2f}) entropy={entropy:.1f} explored={n_explored}"


# ═══════════════════════════════════════
# HELICASE v2 BRAIN
# ═══════════════════════════════════════

class HelicaseV2Brain:
    """Bayesian belief (code) + LLM brain (simple decisions).

    Code does:  Bayesian belief update, score computation, room inference
    Brain does: Read pre-computed scores → pick frontiers (simple, even 0.5B can do)
    """

    def __init__(self, brain=None, num_agents=2):
        self.brain = brain  # LLM brain (can be 0.5B, MiniMax, etc.)
        self.num_agents = num_agents
        self.belief_map = None
        self.reflexion_memory = []

    def reset(self, target_name):
        self.belief_map = BayesianBeliefMap(target_name)

    def decide(self, target_name, enriched_frontiers, pose_pred,
               full_map_pred, target_point_map, semantic_categories=None):
        """Bayesian belief + brain decision.

        1. Code computes belief scores (hard reasoning)
        2. Brain reads scores and picks frontiers (simple decision)

        Returns: (goal_frontiers dict, tools_list, method_name)
        """
        if self.belief_map is None:
            self.reset(target_name)

        # Check if target found on map
        if full_map_pred is not None:
            if semantic_categories is None:
                from constants import HM3D_SEMANTIC_CATEGORIES
                semantic_categories = HM3D_SEMANTIC_CATEGORIES
            import torch
            sem = full_map_pred[4:]
            for i, cat in enumerate(semantic_categories):
                if i >= sem.shape[0]:
                    continue
                if cat == target_name:
                    count = (sem[i] > 0.1).sum().item()
                    if count > 5:
                        ys, xs = torch.where(sem[i] > 0.1)
                        target_pos = (int(ys.float().mean()), int(xs.float().mean()))
                        best_f = 0
                        min_dist = float('inf')
                        for ef in enriched_frontiers:
                            d = np.sqrt((target_pos[0] - ef['centroid'][0])**2 +
                                       (target_pos[1] - ef['centroid'][1])**2)
                            if d < min_dist:
                                min_dist = d
                                best_f = ef['idx']
                        return (
                            {f"robot_{i}": best_f for i in range(self.num_agents)},
                            ["check_target"], "helicase_v2_found"
                        )

        if not enriched_frontiers or not target_point_map:
            return {}, ["no_frontiers"], "helicase_v2_no_frontier"

        # ── CODE: Bayesian belief update (hard reasoning) ──
        self.belief_map.update_from_frontiers(enriched_frontiers, pose_pred)

        centroids = [ef['centroid'] for ef in enriched_frontiers]
        for ef in enriched_frontiers:
            self.belief_map.mark_explored(ef['idx'], pose_pred, centroids)

        # ── CODE: Compute scores for each frontier ──
        scored_frontiers = []
        for ef in enriched_frontiers:
            idx = ef['idx']
            belief = self.belief_map.beliefs.get(idx, 0.1)
            explored = self.belief_map.explored.get(idx, 0.0)
            visits = self.belief_map.visit_count.get(idx, 0)
            room = self.belief_map.room_types.get(idx, "unknown")
            score = belief * (1.0 - explored) * (1.0 / (1.0 + visits))
            scored_frontiers.append({
                "idx": idx, "score": score, "belief": belief,
                "room": room, "explored": explored,
                "centroid": ef['centroid']
            })
        scored_frontiers.sort(key=lambda x: -x['score'])

        # ── BRAIN: Read scores and pick frontiers ──
        # Format as simple numbered list — even 0.5B can read this
        frontier_lines = []
        for sf in scored_frontiers[:6]:
            frontier_lines.append(
                f"frontier_{sf['idx']}: score={sf['score']:.2f}, "
                f"room={sf['room']}, belief={sf['belief']:.2f}, "
                f"explored={sf['explored']:.0%}"
            )

        if self.brain:
            prompt = (
                f"Find {target_name}. Frontiers ranked by score:\n"
                + "\n".join(frontier_lines) + "\n\n"
                f"Pick 2 DIFFERENT frontiers. Best score for robot_0, "
                f"a distant one for robot_1.\n"
                f"robot_0: frontier_N\nrobot_1: frontier_M"
            )
            response = self.brain.call(prompt, max_tokens=50)

            # Parse brain output
            import re
            goal_frontiers = {}
            for match in re.finditer(r'robot_(\d+)\s*:\s*frontier_(\d+)', response):
                goal_frontiers[f"robot_{match.group(1)}"] = int(match.group(2))

            # If brain parsed robot_0 but not robot_1, assign robot_1
            if "robot_0" in goal_frontiers and "robot_1" not in goal_frontiers:
                r0 = goal_frontiers["robot_0"]
                r0_pos = np.array(scored_frontiers[0]['centroid'])
                best_dist = 0
                best_idx = scored_frontiers[min(1, len(scored_frontiers)-1)]['idx']
                for sf in scored_frontiers:
                    if sf['idx'] == r0:
                        continue
                    d = np.linalg.norm(r0_pos - np.array(sf['centroid']))
                    if d > best_dist:
                        best_dist = d
                        best_idx = sf['idx']
                goal_frontiers["robot_1"] = best_idx

            # If brain completely failed, use algorithmic fallback
            if "robot_0" not in goal_frontiers:
                goal_frontiers = self.belief_map.select_frontiers(
                    enriched_frontiers, self.num_agents)
        else:
            # No brain — pure algorithmic
            goal_frontiers = self.belief_map.select_frontiers(
                enriched_frontiers, self.num_agents)

        # Update visit counts
        for key, idx in goal_frontiers.items():
            self.belief_map.visit_count[idx] = self.belief_map.visit_count.get(idx, 0) + 1

        tools = ["belief_update", "belief_score", "brain_select"]
        status = self.belief_map.get_status()
        return goal_frontiers, tools, f"helicase_v2({status})"

    def reflect(self, target_name, success, steps, dtg):
        """Store lesson for cross-episode learning."""
        if self.belief_map:
            lesson = f"[{target_name},{'OK' if success else 'FAIL'}] "
            if success:
                best_room = "unknown"
                if self.belief_map.room_types:
                    best_idx = max(self.belief_map.beliefs, key=self.belief_map.beliefs.get)
                    best_room = self.belief_map.room_types.get(best_idx, "unknown")
                lesson += f"found in {best_room}"
            else:
                explored_rooms = [self.belief_map.room_types.get(idx, "?")
                                  for idx, v in self.belief_map.explored.items() if v > 0.5]
                lesson += f"not in {explored_rooms}"
            self.reflexion_memory.append(lesson)
