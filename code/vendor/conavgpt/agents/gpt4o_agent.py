"""
GPT-4o Navigation Agent — Uses OpenAI function calling API.

GPT-4o sees the RGB image + tool definitions → decides which tools to call →
executes tools locally → sends results back → GPT-4o reasons → picks action.

This is the TEACHER agent. Its trajectories (with tool calls) are distilled
into a 0.5B student model.

Design:
  1. Each navigation step = one GPT-4o conversation
  2. GPT-4o can call 0-5 tools per step (it decides)
  3. Tool results are sent back, GPT-4o may call more tools
  4. Final response must contain an ACTION
  5. Everything is logged for distillation

Usage:
    agent = GPT4oNavigationAgent(api_key="sk-...", toolkit=toolkit)
    action, trace = agent.step(rgb_image, target="toilet", step_info={...})
"""

import os
import json
import base64
import io
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
from dataclasses import dataclass, field, asdict


@dataclass
class ToolCallRecord:
    """Record of one tool call for distillation."""
    tool_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class StepTrace:
    """Complete trace of one navigation step — training data for student."""
    step: int
    target: str
    rgb_path: str
    system_prompt: str
    tool_calls: List[ToolCallRecord]
    reasoning: str          # GPT-4o's thinking
    action: str             # final action: forward/left/right/stop
    num_tool_calls: int
    total_tokens: int


class GPT4oNavigationAgent:
    """GPT-4o teacher agent with dynamic tool calling."""

    def __init__(self, api_key: str, toolkit, max_tool_rounds: int = 3):
        """
        Args:
            api_key: OpenAI API key
            toolkit: CompleteToolkit instance (from complete_tools.py)
            max_tool_rounds: max rounds of tool calling per step
        """
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.toolkit = toolkit
        self.max_tool_rounds = max_tool_rounds

        # State
        self.step_count = 0
        self.target = ""
        self.observations_summary = ""
        self.room_history = []

    def reset(self, target: str):
        """Reset for new episode."""
        self.step_count = 0
        self.target = target
        self.observations_summary = ""
        self.room_history = []

    def step(self, rgb: np.ndarray, target: str, step_info: Dict = None) -> Tuple[str, StepTrace]:
        """
        One navigation step with dynamic tool calling.

        Args:
            rgb: current RGB observation (H, W, 3)
            target: target object name
            step_info: optional dict with extra info (step number, explored %, etc.)

        Returns:
            (action, trace) — action string + full trace for distillation
        """
        if step_info is None:
            step_info = {}

        self.target = target
        step_num = step_info.get("step", self.step_count)

        # Build system prompt
        system_prompt = self._build_system_prompt(target, step_info)

        # Build initial message with image
        user_message = self._build_user_message(rgb, target, step_info)

        # Conversation loop with tool calling
        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        tool_calls_log = []
        action = None
        reasoning = ""

        for round_idx in range(self.max_tool_rounds + 1):
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self._get_tool_definitions(),
                tool_choice="auto",  # GPT-4o decides whether to call tools
                max_tokens=500,
                temperature=0,
            )

            choice = response.choices[0]
            total_tokens = response.usage.total_tokens if response.usage else 0

            # Check if GPT-4o wants to call tools
            if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
                # Execute each tool call
                tool_results = []
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                    # Map function name to tool ID and execute
                    result = self._execute_tool(fn_name, rgb, step_info, fn_args)

                    tool_calls_log.append(ToolCallRecord(
                        tool_id=self._name_to_id(fn_name),
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=result,
                    ))

                    tool_results.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "content": result,
                    })

                # Add assistant message with tool calls + tool results
                messages.append(choice.message)
                messages.extend(tool_results)
                continue

            # No more tool calls — GPT-4o gave final response
            reasoning = choice.message.content or ""

            # Extract action from response
            action = self._extract_action(reasoning)
            break

        # Default action if nothing extracted
        if action is None:
            action = "forward"

        # Update state
        self.step_count += 1

        # Build trace for distillation
        trace = StepTrace(
            step=step_num,
            target=target,
            rgb_path="",  # filled by caller
            system_prompt=system_prompt,
            tool_calls=tool_calls_log,
            reasoning=reasoning,
            action=action,
            num_tool_calls=len(tool_calls_log),
            total_tokens=total_tokens,
        )

        return action, trace

    # ════════════════════════════════════════════════════
    # Prompt Building
    # ════════════════════════════════════════════════════

    def _build_system_prompt(self, target: str, step_info: Dict) -> str:
        return f"""You are a navigation robot searching for a {target} in an indoor house.

You can see through an RGB camera and have access to tools for perception, localization, planning, and execution.

RULES:
1. Call tools to gather information BEFORE deciding your action.
2. Don't call all tools every step — only call what you need for the current situation.
3. When you enter a new room: call classify_room_type and detect_doors.
4. When you see a door/opening: call check_opening or detect_doors.
5. When you think the target might be visible: call identify_target.
6. Always end with an ACTION: forward, left, right, or stop.
7. Only say "stop" if you are CERTAIN the {target} is clearly visible and within 1 meter.

NAVIGATION STRATEGY:
- Think about what room typically contains a {target}
- Use room classification and door detection to find that room type
- Check floor type (tile=bathroom, carpet=bedroom) as a clue
- If stuck, try a different direction
- Adapt strategy based on exploration progress

Step: {step_info.get('step', self.step_count)}
Max steps: {step_info.get('max_steps', 500)}
Explored: {step_info.get('explored_pct', 'unknown')}%
Rooms visited: {', '.join(self.room_history) if self.room_history else 'none yet'}"""

    def _build_user_message(self, rgb: np.ndarray, target: str, step_info: Dict) -> Dict:
        """Build user message with RGB image."""
        # Encode image
        image = Image.fromarray(rgb).resize((512, 384))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        text = f"Current observation. I'm looking for a {target}. What tools should I call? Then decide an action."

        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": text},
            ]
        }

    # ════════════════════════════════════════════════════
    # Tool Definitions (OpenAI function calling format)
    # ════════════════════════════════════════════════════

    def _get_tool_definitions(self) -> List[Dict]:
        """All 38 tools in OpenAI function calling format."""
        tools = [
            # === PERCEPTION ===
            self._tool_def("observe_scene", "Describe the current scene: room type, objects, doors, floor, layout", {}),
            self._tool_def("identify_target", "Check if target object is visible in current view",
                          {"target_object": {"type": "string", "description": "object to look for"}}),
            self._tool_def("detect_objects", "Detect all objects in current view with positions", {}),
            self._tool_def("detect_doors", "Detect doors, doorways, and openings", {}),
            self._tool_def("classify_room_type", "Classify what type of room this is", {}),
            self._tool_def("estimate_depth", "Estimate depth/distance in a direction",
                          {"region": {"type": "string", "enum": ["center", "left", "right"], "description": "which part of image"}}),
            self._tool_def("estimate_room_size", "Estimate if this is a big room, small room, or corridor", {}),
            self._tool_def("detect_floor_type", "Detect floor material: tile/carpet/wood", {}),
            self._tool_def("segment_scene", "Unified detection + captioning of the scene", {}),
            self._tool_def("look_direction", "Focus on a specific direction and describe it",
                          {"direction": {"type": "string", "enum": ["left", "right", "ahead"], "description": "which direction"}}),
            self._tool_def("check_opening", "Check if there's a door/opening and describe what's through it", {}),
            self._tool_def("estimate_metric_depth", "Estimate absolute depth in meters", {}),
            self._tool_def("get_spatial_relations", "Describe spatial layout of objects", {}),
            self._tool_def("read_semantic_map", "Read already-detected objects from the built map", {}),

            # === LOCALIZATION ===
            self._tool_def("get_position_and_heading", "Get current position and compass direction", {}),
            self._tool_def("get_exploration_progress", "How much of the scene has been explored", {}),
            self._tool_def("get_room_size_here", "How big is the current room (from pathfinder)", {}),
            self._tool_def("get_visited_rooms", "List of rooms visited so far", {}),
            self._tool_def("check_if_stuck", "Check if the agent is stuck (not moving)", {}),
            self._tool_def("check_room_revisit", "Check if we've been in this room before", {}),
            self._tool_def("get_distance_walked", "Total distance traveled so far", {}),
            self._tool_def("get_scene_objects", "All object categories in this scene", {}),

            # === PLANNING ===
            self._tool_def("select_frontier", "Pick the best frontier to explore",
                          {"scene_description": {"type": "string", "description": "what you see"},
                           "frontiers": {"type": "string", "description": "available frontiers"}}),
            self._tool_def("get_frontiers_from_map", "Extract frontier boundaries from the map", {}),
            self._tool_def("should_stop", "Decide if target is found — should I stop?",
                          {"observation": {"type": "string", "description": "what you see"},
                           "detection_result": {"type": "string", "description": "target detection result"}}),
            self._tool_def("reason_where_to_search", "Reason about where the target is likely located",
                          {"observations_so_far": {"type": "string", "description": "summary of observations"}}),
            self._tool_def("predict_adjacent_room", "Predict what room is through the nearest opening",
                          {"current_room": {"type": "string", "description": "current room type"},
                           "visible_cues": {"type": "string", "description": "what you see near the opening"}}),
            self._tool_def("should_change_strategy", "Should I change exploration strategy?", {}),
            self._tool_def("rank_frontiers_by_room", "Re-rank frontiers by room type match",
                          {"frontiers": {"type": "string", "description": "frontier list"},
                           "room_info": {"type": "string", "description": "room type observations"}}),
            self._tool_def("suggest_backtrack", "Suggest where to go back when stuck",
                          {"visited_rooms": {"type": "string", "description": "rooms visited"},
                           "unexplored": {"type": "string", "description": "unexplored areas"}}),

            # === EXECUTION ===
            self._tool_def("get_path_to_point", "Get direction to reach a specific point",
                          {"goal": {"type": "string", "description": "goal point as x,y,z"}}),
            self._tool_def("can_move_forward", "Check if forward path is clear", {}),
            self._tool_def("get_distance_to_goal", "Get walking distance to a point",
                          {"goal": {"type": "string", "description": "goal point as x,y,z"}}),
            self._tool_def("can_move_direction", "Check if a direction is clear",
                          {"direction": {"type": "string", "enum": ["forward", "left", "right", "back"]}}),
            self._tool_def("get_closest_wall_distance", "Distance to nearest wall", {}),
            self._tool_def("plan_full_path", "Get full path with waypoints to a goal",
                          {"goal": {"type": "string", "description": "goal point as x,y,z"}}),
            self._tool_def("get_random_target", "Get a random navigable point (when stuck)", {}),
            self._tool_def("estimate_steps_to_goal", "Estimate how many steps to reach a point",
                          {"goal": {"type": "string", "description": "goal point as x,y,z"}}),
        ]
        return tools

    def _tool_def(self, name: str, description: str, params: Dict) -> Dict:
        """Create OpenAI function calling tool definition."""
        properties = {}
        required = []
        for pname, pspec in params.items():
            properties[pname] = pspec
            required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ════════════════════════════════════════════════════
    # Tool Execution
    # ════════════════════════════════════════════════════

    def _execute_tool(self, name: str, rgb: np.ndarray, step_info: Dict, args: Dict) -> str:
        """Execute a tool by name, route to toolkit."""
        # Map function names to tool IDs
        name_to_id = {
            # Perception
            "observe_scene": "P1", "identify_target": "P2", "detect_objects": "P3",
            "detect_doors": "P4", "classify_room_type": "P5", "estimate_depth": "P6",
            "estimate_room_size": "P7", "detect_floor_type": "P8", "segment_scene": "P9",
            "look_direction": "P10", "check_opening": "P11", "estimate_metric_depth": "P12",
            "get_spatial_relations": "P13", "read_semantic_map": "P14",
            # Localization
            "get_position_and_heading": "L1", "get_exploration_progress": "L2",
            "get_room_size_here": "L3", "get_visited_rooms": "L4", "check_if_stuck": "L5",
            "check_room_revisit": "L6", "get_distance_walked": "L7", "get_scene_objects": "L8",
            # Planning
            "select_frontier": "G1", "get_frontiers_from_map": "G2", "should_stop": "G3",
            "reason_where_to_search": "G4", "predict_adjacent_room": "G5",
            "should_change_strategy": "G6", "rank_frontiers_by_room": "G7", "suggest_backtrack": "G8",
            # Execution
            "get_path_to_point": "E1", "can_move_forward": "E2", "get_distance_to_goal": "E3",
            "can_move_direction": "E4", "get_closest_wall_distance": "E5", "plan_full_path": "E6",
            "get_random_target": "E7", "estimate_steps_to_goal": "E8",
        }

        tool_id = name_to_id.get(name, "")
        if not tool_id:
            return f"Unknown tool: {name}"

        # Merge args with step_info for context
        kwargs = {**args}
        kwargs["target"] = self.target

        # Pass step_info fields that tools need
        if "explored_map" in step_info:
            kwargs["explored_map"] = step_info["explored_map"]
        if "occ_map" in step_info:
            kwargs["occ_map"] = step_info["occ_map"]
        if "agent_pos" in step_info:
            kwargs["agent_pos"] = step_info["agent_pos"]
        if "semantic_map" in step_info:
            kwargs["semantic_map"] = step_info["semantic_map"]
        if "categories" in step_info:
            kwargs["categories"] = step_info["categories"]
        if "image_history" in step_info:
            kwargs["image_history"] = step_info["image_history"]

        # Planning tools need scene/frontier info from args
        if name == "select_frontier":
            kwargs["scene"] = args.get("scene_description", "")
            kwargs["frontiers"] = args.get("frontiers", "")
        elif name == "should_stop":
            kwargs["observation"] = args.get("observation", "")
            kwargs["detection"] = args.get("detection_result", "")
        elif name == "reason_where_to_search":
            kwargs["observations"] = args.get("observations_so_far", "")
        elif name == "predict_adjacent_room":
            kwargs["current_room"] = args.get("current_room", "")
            kwargs["cues"] = args.get("visible_cues", "")
        elif name in ("get_path_to_point", "get_distance_to_goal", "plan_full_path", "estimate_steps_to_goal"):
            goal_str = args.get("goal", "0,0,0")
            try:
                kwargs["goal"] = [float(x.strip()) for x in goal_str.split(",")]
            except:
                kwargs["goal"] = [0, 0, 0]
        elif name == "can_move_direction":
            kwargs["direction"] = args.get("direction", "forward")
        elif name == "estimate_depth":
            kwargs["region"] = args.get("region", "center")
        elif name == "look_direction":
            kwargs["direction"] = args.get("direction", "ahead")
        elif name == "rank_frontiers_by_room":
            kwargs["frontiers"] = args.get("frontiers", "")
            kwargs["room_info"] = args.get("room_info", "")
        elif name == "suggest_backtrack":
            kwargs["visited"] = args.get("visited_rooms", "")
            kwargs["unexplored"] = args.get("unexplored", "")

        # Execute via toolkit
        result = self.toolkit.execute(tool_id, rgb=rgb, **kwargs)
        return result.output

    def _name_to_id(self, name: str) -> str:
        """Map function name to tool ID."""
        mapping = {
            "observe_scene": "P1", "identify_target": "P2", "detect_objects": "P3",
            "detect_doors": "P4", "classify_room_type": "P5", "estimate_depth": "P6",
            "estimate_room_size": "P7", "detect_floor_type": "P8", "segment_scene": "P9",
            "look_direction": "P10", "check_opening": "P11", "estimate_metric_depth": "P12",
            "get_spatial_relations": "P13", "read_semantic_map": "P14",
            "get_position_and_heading": "L1", "get_exploration_progress": "L2",
            "get_room_size_here": "L3", "get_visited_rooms": "L4", "check_if_stuck": "L5",
            "check_room_revisit": "L6", "get_distance_walked": "L7", "get_scene_objects": "L8",
            "select_frontier": "G1", "get_frontiers_from_map": "G2", "should_stop": "G3",
            "reason_where_to_search": "G4", "predict_adjacent_room": "G5",
            "should_change_strategy": "G6", "rank_frontiers_by_room": "G7", "suggest_backtrack": "G8",
            "get_path_to_point": "E1", "can_move_forward": "E2", "get_distance_to_goal": "E3",
            "can_move_direction": "E4", "get_closest_wall_distance": "E5", "plan_full_path": "E6",
            "get_random_target": "E7", "estimate_steps_to_goal": "E8",
        }
        return mapping.get(name, name)

    def _extract_action(self, text: str) -> Optional[str]:
        """Extract navigation action from GPT-4o response."""
        text_lower = text.lower()

        # Check for explicit action keywords
        if "stop" in text_lower and ("found" in text_lower or "visible" in text_lower or "see" in text_lower):
            return "stop"

        action_patterns = [
            (r'\bforward\b', "forward"),
            (r'\bturn\s*left\b|\bleft\b', "left"),
            (r'\bturn\s*right\b|\bright\b', "right"),
            (r'\bstop\b', "stop"),
            (r'\bmove\s*forward\b', "forward"),
        ]

        # Look for ACTION: prefix first
        action_match = re.search(r'ACTION:\s*(forward|left|right|stop)', text_lower)
        if action_match:
            return action_match.group(1)

        # Then look for action words
        for pattern, action in action_patterns:
            if re.search(pattern, text_lower):
                return action

        return None


# ════════════════════════════════════════════════════════════
# Example usage
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("GPT-4o Navigation Agent")
    print("="*50)
    print()
    print("Usage:")
    print("  from agents.gpt4o_agent import GPT4oNavigationAgent")
    print("  from tools.complete_tools import CompleteToolkit")
    print()
    print("  toolkit = CompleteToolkit(sim=env._sim, vlm_device='cuda:0', llm_device='cuda:1')")
    print("  agent = GPT4oNavigationAgent(api_key='sk-...', toolkit=toolkit)")
    print("  agent.reset(target='toilet')")
    print()
    print("  action, trace = agent.step(rgb, target='toilet', step_info={...})")
    print("  # trace contains full tool calling history for distillation")
    print()
    print("Flow per step:")
    print("  1. Send image + 38 tool definitions to GPT-4o")
    print("  2. GPT-4o decides: call classify_room_type + detect_doors")
    print("  3. Execute tools locally (GPU/CPU)")
    print("  4. Send results back to GPT-4o")
    print("  5. GPT-4o reasons: 'tile floor + small door = bathroom'")
    print("  6. GPT-4o decides: ACTION: turn_left")
    print("  7. Log everything for distillation")
