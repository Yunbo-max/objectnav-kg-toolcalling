"""Read-only reproduction of MCoCoNav's custom SR/SPL metrics.

This evaluator observes an existing navigation trajectory.  It never changes
agent state, actions, RNG state, or Habitat measurements.
"""

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


_CATEGORY_ALIASES = {
    "couch": "sofa",
    "potted_plant": "plant",
    "television": "tv_monitor",
    "tv": "tv_monitor",
    "tv_screen": "tv_monitor",
}


def normalize_category_name(name: str) -> str:
    """Normalize Habitat/HM3D category spelling to MindNav target names."""
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(name).strip().lower()
    ).strip("_")
    return _CATEGORY_ALIASES.get(normalized, normalized)


def build_instance_category_map(
    semantic_objects: Iterable[object],
    category_mapping: Optional[Mapping[str, str]] = None,
) -> Dict[int, str]:
    """Map Habitat semantic instance IDs to normalized category names."""
    remap = category_mapping or {}
    result: Dict[int, str] = {}
    for list_index, semantic_object in enumerate(semantic_objects):
        if (
            semantic_object is None
            or getattr(semantic_object, "category", None) is None
        ):
            continue
        raw_name = semantic_object.category.name()
        mapped_name = remap.get(raw_name, raw_name)
        category_name = normalize_category_name(mapped_name)

        object_id = getattr(semantic_object, "id", "")
        match = re.search(r"(-?\d+)$", str(object_id))
        instance_id = int(match.group(1)) if match else list_index
        result[instance_id] = category_name
        # Habitat usually indexes semantic objects by instance ID.  Keeping the
        # list index as a fallback also supports lightweight test simulators.
        result.setdefault(list_index, category_name)
    return result


@dataclass
class _AgentMetricState:
    initial_world_xy: Tuple[float, float]
    previous_world_xy: Tuple[float, float]
    start_grid: Optional[Tuple[int, int]] = None
    end_grid: Optional[Tuple[int, int]] = None
    path_length: float = 1e-5
    predicted_find_goal: bool = False
    gt_find_goal: bool = False
    gt_observation_available: bool = False


class MCoCoNavMetricTracker:
    """Track MCoCoNav-compatible metrics for one multi-agent episode."""

    def __init__(self, num_agents: int, map_resolution_cm: float):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if map_resolution_cm <= 0:
            raise ValueError("map_resolution_cm must be positive")
        self.num_agents = int(num_agents)
        self.map_resolution_cm = float(map_resolution_cm)
        self.goal_name = ""
        self.instance_categories: Dict[int, str] = {}
        self._states = []

    def reset(
        self,
        goal_name: str,
        initial_world_poses: Sequence[Sequence[float]],
        instance_categories: Mapping[int, str],
    ) -> None:
        if len(initial_world_poses) != self.num_agents:
            raise ValueError("one initial pose is required for every agent")
        self.goal_name = normalize_category_name(goal_name)
        self.instance_categories = {
            int(instance_id): normalize_category_name(category)
            for instance_id, category in instance_categories.items()
        }
        self._states = []
        for pose in initial_world_poses:
            world_xy = (float(pose[0]), float(pose[1]))
            self._states.append(
                _AgentMetricState(
                    initial_world_xy=world_xy,
                    previous_world_xy=world_xy,
                )
            )

    def _to_local_grid(
        self,
        world_xy: Tuple[float, float],
        planner_pose_inputs: Sequence[float],
        map_shape: Sequence[int],
    ) -> Tuple[int, int]:
        gx1 = int(planner_pose_inputs[3])
        gy1 = int(planner_pose_inputs[5])
        scale = 100.0 / self.map_resolution_cm
        row = int(world_xy[1] * scale - gx1)
        col = int(world_xy[0] * scale - gy1)
        row = min(max(row, 0), int(map_shape[0]) - 1)
        col = min(max(col, 0), int(map_shape[1]) - 1)
        return row, col

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    def observe_agent(
        self,
        agent_id: int,
        planner_pose_inputs: Sequence[float],
        map_shape: Sequence[int],
        predicted_find_goal: bool,
        semantic_observation=None,
    ) -> None:
        """Observe one agent after mapping/acting and before ``env.step``."""
        if not self._states:
            raise RuntimeError("reset must be called before observing an episode")
        state = self._states[int(agent_id)]
        current_world_xy = (
            float(planner_pose_inputs[0]),
            float(planner_pose_inputs[1]),
        )
        previous_grid = self._to_local_grid(
            state.previous_world_xy, planner_pose_inputs, map_shape
        )
        current_grid = self._to_local_grid(
            current_world_xy, planner_pose_inputs, map_shape
        )
        if state.start_grid is None:
            # This matches MCoCoNav's first-step ``Start_Location=last_start``.
            state.start_grid = previous_grid
        state.path_length += self._distance(previous_grid, current_grid)
        state.end_grid = current_grid
        state.previous_world_xy = current_world_xy
        state.predicted_find_goal |= bool(predicted_find_goal)

        if semantic_observation is not None:
            state.gt_observation_available = True
            visible_instance_ids = np.unique(np.asarray(semantic_observation))
            for instance_id in visible_instance_ids:
                category = self.instance_categories.get(int(instance_id))
                if category == self.goal_name:
                    state.gt_find_goal = True
                    break

    def result(self) -> Dict[str, object]:
        """Return episode metrics using MCoCoNav's original aggregation."""
        if not self._states:
            raise RuntimeError("reset must be called before requesting results")

        agent_metrics = {}
        team_spl = 0.0
        for agent_id, state in enumerate(self._states):
            success = bool(state.predicted_find_goal and state.gt_find_goal)
            if state.start_grid is None or state.end_grid is None:
                straight_line_distance = 0.0
            else:
                straight_line_distance = self._distance(
                    state.start_grid, state.end_grid
                )
            if state.path_length <= 1e-3:
                spl = float(success)
            else:
                spl = min(
                    float(success) * straight_line_distance / state.path_length,
                    1.0,
                )
            team_spl = max(team_spl, spl)
            agent_metrics[f"robot_{agent_id}"] = {
                "predicted_find_goal": bool(state.predicted_find_goal),
                "gt_find_goal": bool(state.gt_find_goal),
                "gt_observation_available": bool(
                    state.gt_observation_available
                ),
                "success": float(success),
                "spl": float(spl),
                "path_length_cells": float(state.path_length),
                "start_end_distance_cells": float(straight_line_distance),
            }

        # Preserve MCoCoNav's order-dependent team SR: the first robot with a
        # predicted detection decides whether the episode is counted as a true
        # success, even if a later robot has a GT-confirmed detection.
        ordered_success = 0.0
        navigation_success = 0.0
        deciding_robot = None
        for agent_id, state in enumerate(self._states):
            if state.predicted_find_goal:
                navigation_success = 1.0
                ordered_success = float(state.gt_find_goal)
                deciding_robot = f"robot_{agent_id}"
                break

        any_agent_success = float(
            any(
                state.predicted_find_goal and state.gt_find_goal
                for state in self._states
            )
        )
        return {
            "success": ordered_success,
            "navigation_success": navigation_success,
            "spl": float(team_spl),
            "any_agent_success": any_agent_success,
            "deciding_robot": deciding_robot,
            "agent_metrics": agent_metrics,
        }
