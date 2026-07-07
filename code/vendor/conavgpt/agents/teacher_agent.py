"""
Teacher agent: Large VLM + dynamic tool calling for navigation.
Generates trajectories that will be distilled into the student model.

Architecture:
  1. Tools are called selectively based on step context (dynamic scheduling)
  2. VLM receives RGB image + tool results, outputs scene analysis
  3. Action is parsed from natural language VLM output
  4. Full trace (reasoning, tool calls, action) is logged for distillation

The teacher receives ONLY RGB images -- no GT depth or semantics.
"""
import gc
import json
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from tools.navigation_tools import ToolResult


class TeacherAgent:
    """VLM-based teacher agent with dynamic tool calling."""

    def __init__(
        self,
        model_name: str,
        toolkit,
        max_tool_calls: int = 3,
        max_new_tokens: int = 256,
        device: str = "cuda:0",
    ):
        self.toolkit = toolkit
        self.max_tool_calls = max_tool_calls
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.device = device
        self.use_api = False
        self._load_model(model_name)

        self.category_map = {
            0: "chair", 1: "bed", 2: "plant",
            3: "toilet", 4: "tv_monitor", 5: "sofa",
        }

        # Per-episode tool-scheduling state
        self._last_detect_step = -999
        self._last_depth_step = -999
        self._queried_location = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str):
        if "gpt" in model_name.lower():
            import openai
            self.client = openai.OpenAI()
            self.use_api = True
        else:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map=self.device,
            )
            self.model.eval()
            self.use_api = False

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_nav_prompt(
        self,
        target_object: str,
        step_info: dict,
        tool_results: Optional[List[Dict]] = None,
    ) -> str:
        """Build navigation prompt.  Avoids structured output keywords
        (ACTION:, TOOL:, etc.) that trigger Qwen2.5's native function
        calling mode.  Instead asks for natural language that we parse."""
        parts = [
            f"I am a robot navigating indoors. My goal is to find a {target_object}.",
            f"This is step {step_info.get('step', 0)}. "
            f"My previous move was: {step_info.get('prev_action', 'none')}.",
        ]

        if tool_results:
            parts.append("")
            parts.append("Here is what I already know:")
            for tr in tool_results:
                # Present tool results as natural observations
                parts.append(f"- {tr['result']}")

        parts.extend([
            "",
            "Look at this image from my camera and help me navigate.",
            f"Tell me what you see and whether {target_object} is visible.",
            "Then tell me: should I go forward, turn left, turn right, or stop?",
        ])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        rgb_image: np.ndarray,
        target_object: str,
        step_info: dict,
    ) -> Tuple[str, Dict]:
        """One navigation step with dynamic tool calling."""
        step_num = step_info.get("step", 0)

        # Phase 1: Run tools dynamically
        tool_calls_log = self._run_tools_dynamically(
            rgb_image, target_object, step_info
        )

        # Phase 2: Check if target detected -- short-circuit to stop
        if self._target_detected(tool_calls_log, target_object):
            reasoning = f"[Target '{target_object}' detected by object detection. Stopping.]"
            return "stop", self._make_entry(
                target_object, step_info, reasoning, tool_calls_log, "stop"
            )

        # Phase 3: VLM inference
        prompt = self._build_nav_prompt(
            target_object, step_info, tool_calls_log or None
        )
        vlm_output = self._generate(prompt, rgb_image)
        reasoning = vlm_output

        # Phase 4: Parse action from natural language
        action = self._parse_action(vlm_output)

        return action, self._make_entry(
            target_object, step_info, reasoning, tool_calls_log, action
        )

    # ------------------------------------------------------------------
    # Dynamic tool scheduling
    # ------------------------------------------------------------------

    def _run_tools_dynamically(
        self,
        rgb_image: np.ndarray,
        target_object: str,
        step_info: dict,
    ) -> List[Dict]:
        """Decide which tools to call. Averages ~1.5 calls/step."""
        step_num = step_info.get("step", 0)
        tool_calls: List[Dict] = []
        detected_objects = []

        # Object detection: every 3 steps or first step
        if step_num - self._last_detect_step >= 3 or step_num == 0:
            result = self.toolkit.detect_objects(rgb_image)
            tool_calls.append({
                "tool": "detect_objects", "args": "",
                "result": result.output,
            })
            self._last_detect_step = step_num
            for item in result.output.split(","):
                name = item.strip().split("(")[0].strip()
                if name and name != "no objects detected":
                    detected_objects.append(name)

        # Location query: once per episode
        if not self._queried_location:
            result = self.toolkit.query_object_location(target_object)
            tool_calls.append({
                "tool": "query_object_location", "args": target_object,
                "result": result.output,
            })
            self._queried_location = True

        # Depth: every 5 steps
        if step_num - self._last_depth_step >= 5 or step_num == 0:
            result = self.toolkit.estimate_depth(rgb_image, "center")
            tool_calls.append({
                "tool": "estimate_depth", "args": "center",
                "result": result.output,
            })
            self._last_depth_step = step_num

        # Room classification: every 10 steps
        if step_num % 10 == 0 and detected_objects:
            result = self.toolkit.classify_room(detected_objects)
            tool_calls.append({
                "tool": "classify_room",
                "args": ",".join(detected_objects),
                "result": result.output,
            })

        return tool_calls

    def _target_detected(self, tool_calls: List[Dict], target: str) -> bool:
        """Check if detection found target with >= 30% confidence."""
        aliases = {
            "tv_monitor": ["tv", "monitor", "television"],
            "sofa": ["couch", "sofa"],
            "plant": ["plant"], "bed": ["bed"],
            "chair": ["chair"], "toilet": ["toilet"],
        }
        search_terms = [target.lower()]
        search_terms.extend(aliases.get(target.lower(), []))

        for tc in tool_calls:
            if tc["tool"] == "detect_objects":
                for term in search_terms:
                    match = re.search(
                        rf"{re.escape(term)}\((\d+)%\)", tc["result"].lower()
                    )
                    if match and int(match.group(1)) >= 30:
                        return True
        return False

    def reset_episode(self):
        """Reset per-episode state."""
        self._last_detect_step = -999
        self._last_depth_step = -999
        self._queried_location = False

    # ------------------------------------------------------------------
    # Action parsing from natural language
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(text: str) -> str:
        """Parse navigation action from VLM natural language output."""
        text_lower = text.lower()

        # 1. Explicit "ACTION: X" format (in case model follows it)
        m = re.search(r"action:\s*(forward|left|right|stop)", text_lower)
        if m:
            return m.group(1)

        # 2. "go forward", "move forward", "continue forward"
        if re.search(r"\b(go|move|continue|walk|proceed)\s+(forward|ahead|straight)", text_lower):
            return "forward"

        # 3. "turn left"
        if re.search(r"\bturn\s+left\b", text_lower):
            return "left"

        # 4. "turn right"
        if re.search(r"\bturn\s+right\b", text_lower):
            return "right"

        # 5. "stop" in context of finding target
        if re.search(r"\bstop\b", text_lower) and re.search(
            r"(found|see|visible|spotted|target|close|nearby)", text_lower
        ):
            return "stop"

        # 6. Last resort: check for directional keywords
        fwd_count = len(re.findall(r"\b(forward|ahead|straight|continue)\b", text_lower))
        left_count = len(re.findall(r"\bleft\b", text_lower))
        right_count = len(re.findall(r"\bright\b", text_lower))

        if fwd_count > 0 and fwd_count >= left_count and fwd_count >= right_count:
            return "forward"
        if left_count > right_count:
            return "left"
        if right_count > left_count:
            return "right"

        # Default: forward (explore)
        return "forward"

    # ------------------------------------------------------------------
    # VLM generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, rgb_image: np.ndarray) -> str:
        if self.use_api:
            return self._generate_api(prompt, rgb_image)
        return self._generate_local(prompt, rgb_image)

    def _generate_api(self, prompt: str, rgb_image: np.ndarray) -> str:
        import base64, io
        from PIL import Image
        img = Image.fromarray(rgb_image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content

    def _generate_local(self, prompt: str, rgb_image: np.ndarray) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        pil_img = Image.fromarray(rgb_image)
        max_side = 320
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.BILINEAR
            )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_entry(target, step_info, reasoning, tool_calls, action) -> Dict:
        return {
            "target": target,
            "step": step_info.get("step", 0),
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "action": action,
            "num_tool_calls": len(tool_calls),
        }

    def resolve_target_name(self, objectgoal_id: int) -> str:
        return self.category_map.get(objectgoal_id, f"object_{objectgoal_id}")
