"""Isolated runtime helpers for the 1/3-robot MindNav ablation.

The production two-robot path deliberately does not call these helpers.  They
exist so the robot-count ablation can vary Habitat agent cardinality without
rewriting the already-audited two-agent statements in ``exp_main_brain.py``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ABLATION_AGENT_COUNTS = (1, 3)


def validate_ablation_agent_count(num_agents: int) -> int:
    num_agents = int(num_agents)
    if num_agents not in ABLATION_AGENT_COUNTS:
        raise ValueError(
            "robot-count ablation only accepts num_agents=1 or 3; "
            f"got {num_agents}"
        )
    return num_agents


def _clone_config(config: Any) -> Any:
    clone = getattr(config, "clone", None)
    return clone() if callable(clone) else deepcopy(config)


def configure_habitat_agents(config_env: Any, num_agents: int) -> None:
    """Configure only the isolated 1/3-agent Habitat path in-place."""
    num_agents = validate_ablation_agent_count(num_agents)
    simulator = config_env.SIMULATOR
    simulator.NUM_AGENTS = num_agents
    simulator.AGENTS = [f"AGENT_{index}" for index in range(num_agents)]
    template = simulator.AGENT_0
    for index in range(1, num_agents):
        name = f"AGENT_{index}"
        if not hasattr(simulator, name):
            setattr(simulator, name, _clone_config(template))


def initialize_actions(num_agents: int) -> list[int]:
    return [0] * validate_ablation_agent_count(num_agents)


def fuse_agent_maps(full_maps: Sequence[torch.Tensor]) -> torch.Tensor:
    """Fuse every active robot map; used only by the ablation branch."""
    if len(full_maps) not in ABLATION_AGENT_COUNTS:
        raise ValueError(
            "ablation map fusion expects 1 or 3 maps; "
            f"got {len(full_maps)}"
        )
    return torch.stack(list(full_maps), dim=0).max(dim=0).values


def build_history_record(
    step: int,
    goal_frontiers: Mapping[str, int],
    object_names: Iterable[str],
    num_agents: int,
) -> dict[str, Any]:
    num_agents = validate_ablation_agent_count(num_agents)
    assignments = {
        f"robot_{index}": int(goal_frontiers[f"robot_{index}"])
        for index in range(num_agents)
    }
    return {
        "step": int(step),
        "assignments": assignments,
        "objects_found": ",".join(sorted(object_names)),
    }


def build_probe_fields(
    goal_frontiers: Mapping[str, int],
    num_agents: int,
) -> dict[str, Any]:
    num_agents = validate_ablation_agent_count(num_agents)
    chosen = [
        int(goal_frontiers[f"robot_{index}"])
        for index in range(num_agents)
    ]
    return {
        "chosen_frontiers": chosen,
        "duplicate_frontier_count": len(chosen) - len(set(chosen)),
        "same_frontier": len(chosen) > 1 and len(set(chosen)) == 1,
    }


def capture_initial_agent_poses(env: Any, num_agents: int) -> list[dict[str, Any]]:
    num_agents = validate_ablation_agent_count(num_agents)
    poses = []
    for index in range(num_agents):
        state = env.sim.get_agent_state(agent_id=index)
        rotation = state.rotation
        if hasattr(rotation, "real") and hasattr(rotation, "imag"):
            rotation_wxyz = [
                float(rotation.real),
                *[float(value) for value in np.asarray(rotation.imag).ravel()],
            ]
        else:
            rotation_wxyz = [float(value) for value in rotation]
        poses.append({
            "robot_id": f"robot_{index}",
            "position": [float(value) for value in state.position],
            "rotation_wxyz": rotation_wxyz,
        })
    return poses

