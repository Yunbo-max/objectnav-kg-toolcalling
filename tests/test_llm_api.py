import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from llm_api import OpenAICompatibleBrainAdapter, load_brain_api_config


class DeepSeekConfigTests(unittest.TestCase):
    def test_loads_deepseek_config_from_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "BRAIN_BACKEND=deepseek\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
                "DEEPSEEK_MODEL=deepseek-v4-flash\n"
                "DEEPSEEK_THINKING=disabled\n"
                "DEEPSEEK_API_KEY=test-key\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                config = load_brain_api_config(directory)

        self.assertEqual(config.backend, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(
            config.extra_body,
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            config.response_format,
            {"type": "json_object"},
        )
        self.assertFalse(config.supports_seed)

    def test_adapter_sends_non_thinking_request_without_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".env").write_text(
                "BRAIN_BACKEND=deepseek\n"
                "DEEPSEEK_API_KEY=test-key\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                config = load_brain_api_config(directory)

        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="robot_0: frontier_0"))]
        )
        with patch("openai.OpenAI", return_value=client):
            adapter = OpenAICompatibleBrainAdapter(config, [], seed=123)
            adapter.call("assign", max_tokens=64)

        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            request["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("seed", request)


if __name__ == "__main__":
    unittest.main()
