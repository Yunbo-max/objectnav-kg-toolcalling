"""LLM API configuration and OpenAI-compatible MindNav adapter."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class BrainAPIConfig:
    """Resolved configuration for the MindNav reasoning backend."""

    backend: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    extra_body: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, str]] = None
    supports_seed: bool = True
    uses_max_completion_tokens: bool = False
    loads_local_model: bool = False


def load_brain_api_config(repo_root: str) -> BrainAPIConfig:
    """Load ``.env`` and resolve the selected MindNav LLM backend."""
    load_dotenv(Path(repo_root) / ".env", override=False)
    backend = os.environ.get("BRAIN_BACKEND", "deepseek").strip().lower()

    if backend == "deepseek":
        api_key = os.environ.get(
            "DEEPSEEK_API_KEY", os.environ.get("BRAIN_API_KEY", "")
        ).strip()
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for BRAIN_BACKEND=deepseek; "
                "set it in the repository .env file"
            )

        thinking = os.environ.get("DEEPSEEK_THINKING", "disabled").strip().lower()
        if thinking not in {"enabled", "disabled"}:
            raise ValueError(
                "DEEPSEEK_THINKING must be either 'enabled' or 'disabled'"
            )
        return BrainAPIConfig(
            backend=backend,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL",
                os.environ.get("BRAIN_BASE_URL", "https://api.deepseek.com"),
            ).rstrip("/"),
            api_key=api_key,
            model=os.environ.get(
                "DEEPSEEK_MODEL",
                os.environ.get("BRAIN_MODEL", "deepseek-v4-flash"),
            ),
            extra_body={"thinking": {"type": thinking}},
            # DeepSeek JSON Output guarantees syntactically valid JSON.  The
            # stage-specific validators in ``brain.py`` still enforce the
            # exact tool schema and all navigation constraints.
            response_format={"type": "json_object"},
            # DeepSeek's Chat Completions schema does not document ``seed``.
            supports_seed=False,
        )

    if backend == "qqqapi":
        api_key = os.environ.get(
            "QQQAPI_API_KEY", os.environ.get("BRAIN_API_KEY", "")
        ).strip()
        if not api_key:
            raise RuntimeError(
                "QQQAPI_API_KEY is required for BRAIN_BACKEND=qqqapi; "
                "set it in the repository .env file"
            )

        reasoning_effort = os.environ.get(
            "QQQAPI_REASONING_EFFORT", "none"
        ).strip().lower()
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError(
                "QQQAPI_REASONING_EFFORT must be one of: "
                "none, low, medium, high, xhigh"
            )
        return BrainAPIConfig(
            backend=backend,
            base_url=os.environ.get(
                "QQQAPI_BASE_URL", "https://qqqapi.com/v1"
            ).rstrip("/"),
            api_key=api_key,
            model=os.environ.get("QQQAPI_MODEL", "gpt-5.4"),
            extra_body={"reasoning_effort": reasoning_effort},
            response_format={"type": "json_object"},
            supports_seed=False,
            uses_max_completion_tokens=True,
        )

    if backend == "vllm":
        return BrainAPIConfig(
            backend=backend,
            base_url=os.environ.get(
                "BRAIN_BASE_URL", "http://127.0.0.1:8000/v1"
            ).rstrip("/"),
            api_key=os.environ.get("BRAIN_API_KEY", "local-vllm"),
            model=os.environ.get("BRAIN_MODEL", "qwen2.5-3b"),
        )

    if backend == "siliconflow":
        return BrainAPIConfig(
            backend=backend,
            base_url=os.environ.get(
                "BRAIN_BASE_URL", "https://api.siliconflow.cn/v1"
            ).rstrip("/"),
            api_key=os.environ.get(
                "BRAIN_API_KEY", os.environ.get("SILICONFLOW_API_KEY", "")
            ),
            model=os.environ.get("BRAIN_MODEL", "Pro/MiniMaxAI/MiniMax-M2.5"),
            loads_local_model=True,
        )

    raise ValueError(
        "BRAIN_BACKEND must be one of: deepseek, qqqapi, siliconflow, vllm"
    )


class OpenAICompatibleBrainAdapter:
    """Expose the ``brain.call`` interface expected by HelicaseBrain."""

    def __init__(
        self,
        config: BrainAPIConfig,
        usage_sink,
        seed: Optional[int] = None,
    ):
        import openai

        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.model
        self.extra_body = config.extra_body
        self.response_format = config.response_format
        self.usage_sink = usage_sink
        self.seed = seed if config.supports_seed else None
        self.last_prompt = ""
        self.last_response = ""
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 0,
        }

    @staticmethod
    def _usage_value(obj, name: str, default=0):
        """Read an OpenAI SDK usage field from objects or dictionaries."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @classmethod
    def _usage_int(cls, obj, name: str) -> int:
        value = cls._usage_value(obj, name, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    def usage_snapshot(self) -> Dict[str, int]:
        """Return cumulative provider-reported usage and real request count."""
        return dict(self._usage_totals)

    def _build_request(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the MindNav Helicase KG reasoning brain. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        if self.config.uses_max_completion_tokens:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens
        if self.seed is not None:
            request["seed"] = int(self.seed)
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        if self.response_format is not None:
            request["response_format"] = self.response_format
        return request

    def call(self, prompt, max_tokens=300):
        self.last_prompt = prompt
        # Count the actual transport attempt even when the provider raises.
        self._usage_totals["api_calls"] += 1
        try:
            response = self.client.chat.completions.create(
                **self._build_request(prompt, max_tokens)
            )
        except Exception:
            if self.usage_sink is not None:
                self.usage_sink.append({
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "api_calls": 1,
                    "request_failed": True,
                })
            raise
        content = response.choices[0].message.content.strip()
        self.last_response = content
        usage = getattr(response, "usage", None)
        prompt_details = self._usage_value(usage, "prompt_tokens_details", None)
        completion_details = self._usage_value(
            usage, "completion_tokens_details", None
        )
        record = {
            "input_tokens": self._usage_int(usage, "prompt_tokens"),
            "output_tokens": self._usage_int(usage, "completion_tokens"),
            "cached_input_tokens": self._usage_int(
                prompt_details, "cached_tokens"
            ),
            "reasoning_tokens": self._usage_int(
                completion_details, "reasoning_tokens"
            ),
            "api_calls": 1,
            "request_failed": False,
        }
        for key in (
            "input_tokens", "output_tokens", "cached_input_tokens",
            "reasoning_tokens",
        ):
            self._usage_totals[key] += record[key]
        if self.usage_sink is not None:
            self.usage_sink.append(record)
        print(f"MindNav core brain ({self.model}) response:")
        print(content)
        return content


class OpenAICompatibleTextAdapter:
    """OpenAI-compatible chat adapter for plain-text navigation brains.

    Co-NavGPT expects ``robot_i: frontier_j`` lines rather than MindNav's JSON
    tool calls, so it must not inherit the JSON response-format constraint.
    """

    def __init__(
        self,
        config: BrainAPIConfig,
        usage_sink=None,
        seed: Optional[int] = None,
    ):
        import openai

        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.model
        self.usage_sink = usage_sink
        self.seed = seed if config.supports_seed else None
        self.last_response = ""
        self._usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls": 0,
        }

    def usage_snapshot(self) -> Dict[str, int]:
        return dict(self._usage_totals)

    def call(self, messages, max_tokens: int = 256) -> str:
        self._usage_totals["api_calls"] += 1
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if self.config.uses_max_completion_tokens:
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens
        if self.seed is not None:
            request["seed"] = int(self.seed)
        if self.config.extra_body is not None:
            request["extra_body"] = self.config.extra_body

        try:
            response = self.client.chat.completions.create(**request)
        except Exception:
            if self.usage_sink is not None:
                self.usage_sink.append({
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "api_calls": 1,
                    "request_failed": True,
                })
            raise

        content = (response.choices[0].message.content or "").strip()
        self.last_response = content
        usage = getattr(response, "usage", None)
        prompt_details = OpenAICompatibleBrainAdapter._usage_value(
            usage, "prompt_tokens_details", None
        )
        completion_details = OpenAICompatibleBrainAdapter._usage_value(
            usage, "completion_tokens_details", None
        )
        record = {
            "input_tokens": OpenAICompatibleBrainAdapter._usage_int(
                usage, "prompt_tokens"
            ),
            "output_tokens": OpenAICompatibleBrainAdapter._usage_int(
                usage, "completion_tokens"
            ),
            "cached_input_tokens": OpenAICompatibleBrainAdapter._usage_int(
                prompt_details, "cached_tokens"
            ),
            "reasoning_tokens": OpenAICompatibleBrainAdapter._usage_int(
                completion_details, "reasoning_tokens"
            ),
            "api_calls": 1,
            "request_failed": False,
        }
        for key in (
            "input_tokens", "output_tokens", "cached_input_tokens",
            "reasoning_tokens",
        ):
            self._usage_totals[key] += record[key]
        if self.usage_sink is not None:
            self.usage_sink.append(record)
        return content
