"""Prompt-only MindNav extension for the 1/3-robot ablation.

The audited two-robot :class:`brain.HelicaseBrain` remains untouched.  Its
validation, fallback, KG, and history logic are already cardinality-aware; the
only active two-agent assumption is in the assignment-stage prompt.  This
subclass rewrites that prompt immediately before transport for N=1 or N=3.
"""
from __future__ import annotations

import json

from brain import HelicaseBrain
from robot_count_runtime import validate_ablation_agent_count


_OLD_POLICY = (
    "(A) If at least two distinct grounded promising regions exist, "
    "select the best distinct grounded regions.\n"
    "(B) If exactly one grounded promising region exists, assign one "
    "suitable robot to exploit it and send the other robot to the best "
    "spatially complementary fresh region.\n"
    "(C) If no grounded promising region exists, enter COVERAGE MODE: "
    "treat evidence-equivalent low-confidence unknown regions as "
    "semantically tied and choose by new visual exposure, spatial "
    "complementarity, and travel cost.\n\n"
)

_OLD_WRAPPER = (
    '{"tool_calls":[{"name":"assign_frontiers","arguments":'
    '{"assignments":{"robot_0":{"frontier_id":0,"room_id":'
    '"room_ID","reason":"reason"},"robot_1":{"frontier_id":1,'
    '"room_id":"room_ID","reason":"reason"}},"diversity":true}}]}'
)


def _replace_once(prompt: str, old: str, new: str, label: str) -> str:
    count = prompt.count(old)
    if count != 1:
        raise RuntimeError(
            f"robot-count prompt marker {label!r} expected once, found {count}"
        )
    return prompt.replace(old, new, 1)


def _example_wrapper(num_agents: int) -> str:
    assignments = {
        f"robot_{index}": {
            "frontier_id": index,
            "room_id": "room_ID",
            "reason": "reason",
        }
        for index in range(num_agents)
    }
    return json.dumps({
        "tool_calls": [{
            "name": "assign_frontiers",
            "arguments": {
                "assignments": assignments,
                "diversity": True,
            },
        }],
    }, separators=(",", ":"))


def rewrite_assignment_prompt(prompt: str, num_agents: int) -> str:
    """Replace only the cardinality-specific assignment instructions."""
    num_agents = validate_ablation_agent_count(num_agents)
    if num_agents == 1:
        policy = (
            "(A) If one or more GROUNDED_PROMISING regions exist, send the "
            "single robot to the best grounded legal room/frontier pair.\n"
            "(B) If none exists, enter COVERAGE MODE and choose the single "
            "legal frontier with the best new visual exposure and travel "
            "cost. Cross-robot diversity is not applicable.\n\n"
        )
        history_rule = (
            "Prefer a frontier that exposes a spatially new area; distinct "
            "room IDs alone do not guarantee a new view."
        )
    else:
        policy = (
            "(A) Rank all GROUNDED_PROMISING regions and assign up to three "
            "robots to the best distinct legal room/frontier pairs.\n"
            "(B) If fewer grounded promising regions than robots exist, use "
            "one suitable robot per grounded region and send every remaining "
            "robot to the best spatially complementary fresh legal pair.\n"
            "(C) If none exists, enter COVERAGE MODE: treat evidence-equivalent "
            "low-confidence unknown regions as semantically tied and allocate "
            "all three robots by new visual exposure, spatial complementarity, "
            "and travel cost.\n\n"
        )
        history_rule = (
            "Prefer three frontiers that expose spatially different areas; "
            "distinct room IDs alone do not guarantee non-overlapping views."
        )

    prompt = _replace_once(prompt, _OLD_POLICY, policy, "allocation_policy")
    prompt = _replace_once(prompt, _OLD_WRAPPER, _example_wrapper(num_agents),
                           "json_wrapper")
    old_history_rule = (
        "Prefer two frontiers that expose spatially different areas; distinct "
        "room IDs alone do not guarantee non-overlapping views."
    )
    if old_history_rule in prompt:
        prompt = _replace_once(
            prompt, old_history_rule, history_rule, "history_diversity"
        )
    return prompt


class RobotCountAblationBrain(HelicaseBrain):
    """HelicaseBrain with an isolated N=1/N=3 assignment prompt."""

    def __init__(self, *args, num_agents: int, **kwargs):
        validate_ablation_agent_count(num_agents)
        super().__init__(*args, num_agents=num_agents, **kwargs)

    def _call_json_stage(self, stage_name, prompt, validator, max_tokens):
        if stage_name == "assign_frontiers":
            prompt = rewrite_assignment_prompt(prompt, self.num_agents)
        return super()._call_json_stage(
            stage_name, prompt, validator, max_tokens
        )

