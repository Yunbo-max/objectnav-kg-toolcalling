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
        "BRAIN_BACKEND must be one of: deepseek, siliconflow, vllm"
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
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self.seed is not None:
            request["seed"] = int(self.seed)
        if self.extra_body is not None:
            request["extra_body"] = self.extra_body
        if self.response_format is not None:
            request["response_format"] = self.response_format
        return request

    def call(self, prompt, max_tokens=300):
        self.last_prompt = prompt
        response = self.client.chat.completions.create(
            **self._build_request(prompt, max_tokens)
        )
        content = response.choices[0].message.content.strip()
        self.last_response = content
        self.usage_sink.append(0)
        print(f"MindNav core brain ({self.model}) response:")
        print(content)
        return content
