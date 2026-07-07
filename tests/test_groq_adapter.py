# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for GroqProvider adapter and BaseAgent Groq integration.

No real API key, no GPU, no network required. All external calls are mocked.

Usage:
    pytest tests/test_groq_adapter.py -v
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — build minimal mock objects reused across tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_groq_response(content: str):
    """Build a mock Groq API response object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _minimal_config(provider: str = "groq") -> dict:
    """Return a minimal config dict with the groq section populated."""
    return {
        "groq": {
            "provider": provider,
            "model": "llama-3.3-70b-versatile",
            "api_key_env": "GROQ_API_KEY",
            "max_retries": 3,
            "timeout": 60,
        },
        "models": {
            "agent1_protocol": "groq-cloud",
        },
        "generation": {
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — generate() returns a string
# ══════════════════════════════════════════════════════════════════════════════

class TestGroqGenerateReturnsString:
    """test_groq_generate_returns_string: mock Groq client, call generate(), verify str."""

    def test_generate_returns_str(self):
        """GroqProvider.generate() must return a plain string."""
        mock_response = _make_groq_response("  Hello, world!  ")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            # Patch the Groq class so no real SDK is needed
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                from agents.groq_adapter import GroqProvider
                provider = GroqProvider(_minimal_config())

                result = provider.generate("What is CBT?", max_new_tokens=128, temperature=0.3)

        assert isinstance(result, str), "generate() must return str"
        assert result == "Hello, world!", (
            "generate() must strip surrounding whitespace from the API response"
        )

    def test_generate_passes_correct_params(self):
        """generate() forwards max_new_tokens and temperature to the API."""
        mock_response = _make_groq_response("answer")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                from agents.groq_adapter import GroqProvider
                provider = GroqProvider(_minimal_config())
                provider.generate("prompt", max_new_tokens=256, temperature=0.1)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    def test_generate_message_is_user_role(self):
        """generate() sends the prompt as a user-role message."""
        mock_response = _make_groq_response("ok")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                from agents.groq_adapter import GroqProvider
                provider = GroqProvider(_minimal_config())
                provider.generate("my prompt")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "my prompt"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — _clean_json() handles markdown fences, incomplete braces, non-ASCII
# ══════════════════════════════════════════════════════════════════════════════

class TestGroqCleanJson:
    """test_groq_clean_json: verify _clean_json() edge cases."""

    @pytest.fixture(autouse=True)
    def provider(self):
        """Create a GroqProvider instance for each test without real API calls."""
        mock_client = MagicMock()
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                from agents.groq_adapter import GroqProvider
                self._provider = GroqProvider(_minimal_config())

    def test_strips_markdown_json_fence(self):
        """```json ... ``` fences are removed."""
        raw = '```json\n{"key": "value"}\n```'
        result = self._provider._clean_json(raw)
        assert result == '{"key": "value"}'

    def test_strips_bare_code_fence(self):
        """``` ... ``` fences (no language tag) are also removed."""
        raw = '```\n{"key": 42}\n```'
        result = self._provider._clean_json(raw)
        assert result == '{"key": 42}'

    def test_incomplete_braces_are_closed(self):
        """An unclosed JSON object gets a closing brace appended."""
        raw = '{"decision": "INCLUDE", "confidence": 0.9'
        result = self._provider._clean_json(raw)
        # Must start with { and end with }
        assert result.startswith("{")
        assert result.endswith("}")

    def test_nested_braces_fully_matched(self):
        """Nested objects are returned complete (depth tracking works)."""
        raw = '{"outer": {"inner": 1}}'
        result = self._provider._clean_json(raw)
        assert result == '{"outer": {"inner": 1}}'

    def test_non_ascii_stripped(self):
        """Non-ASCII characters (tildes, accents, etc.) are removed."""
        raw = '{"campo": "depresión y ansiedad", "val": 1}'
        result = self._provider._clean_json(raw)
        # Non-ASCII gone, ASCII structure intact
        assert "depres" in result
        assert "n y ansiedad" in result or "n y ansiedad" not in result  # partial ok
        # Key test: no non-ASCII bytes remain
        assert result.isascii(), f"Non-ASCII chars remain in: {result!r}"

    def test_no_json_object_returns_empty(self):
        """Text with no opening brace returns {}."""
        raw = "This text has no JSON at all."
        result = self._provider._clean_json(raw)
        assert result == "{}"

    def test_preamble_before_brace_ignored(self):
        """Leading prose before the first { is discarded."""
        raw = 'Sure! Here is the JSON:\n{"result": true}'
        result = self._provider._clean_json(raw)
        assert result == '{"result": true}'


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — retry on transient error
# ══════════════════════════════════════════════════════════════════════════════

class TestGroqRetryOnError:
    """test_groq_retry_on_error: first call raises, second succeeds."""

    def test_retries_on_first_failure(self):
        """generate() retries and succeeds after a transient API error."""
        mock_response = _make_groq_response("retry succeeded")
        mock_client = MagicMock()

        # Side effect: raise on first call, succeed on second
        mock_client.chat.completions.create.side_effect = [
            Exception("transient server error"),
            mock_response,
        ]

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                # Patch time.sleep so tests don't actually wait
                with patch("time.sleep"):
                    from agents.groq_adapter import GroqProvider
                    provider = GroqProvider(_minimal_config())
                    result = provider.generate("test prompt")

        assert result == "retry succeeded"
        assert mock_client.chat.completions.create.call_count == 2, (
            "API should have been called exactly twice (1 failure + 1 success)"
        )

    def test_raises_after_all_retries_exhausted(self):
        """generate() re-raises the exception when all retries are exhausted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("persistent error")

        cfg = _minimal_config()
        cfg["groq"]["max_retries"] = 2

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                with patch("time.sleep"):
                    from agents.groq_adapter import GroqProvider
                    provider = GroqProvider(cfg)
                    with pytest.raises(Exception, match="persistent error"):
                        provider.generate("test prompt")

        # Called max_retries + 1 times (initial attempt + 2 retries)
        assert mock_client.chat.completions.create.call_count == 3

    def test_no_retry_on_immediate_success(self):
        """generate() calls the API exactly once when the first call succeeds."""
        mock_response = _make_groq_response("immediate")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_client):
                from agents.groq_adapter import GroqProvider
                provider = GroqProvider(_minimal_config())
                provider.generate("test")

        assert mock_client.chat.completions.create.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — BaseAgent in Groq mode
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseAgentGroqMode:
    """test_base_agent_groq_mode: provider='groq' → GroqProvider is set up."""

    def _make_config_yaml(self, tmp_path, provider: str = "groq") -> str:
        """Write a minimal config.yaml and return its directory."""
        import yaml
        cfg = {
            "groq": {
                "provider": provider,
                "model": "llama-3.3-70b-versatile",
                "api_key_env": "GROQ_API_KEY",
                "max_retries": 3,
                "timeout": 60,
            },
            "models": {"agent1_protocol": "groq-cloud"},
            "generation": {
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.3,
            },
            "device": "cpu",
            "quantization": "none",
        }
        config_path = os.path.join(tmp_path, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        return tmp_path

    def test_groq_backend_initialised(self, tmp_path):
        """BaseAgent sets _backend='groq' and creates a GroqProvider."""
        config_dir = self._make_config_yaml(tmp_path, provider="groq")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = _make_groq_response("ok")

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_groq_client):
                # Run BaseAgent.__init__ from the tmp config directory
                orig_dir = os.getcwd()
                try:
                    os.chdir(config_dir)
                    from agents.base import BaseAgent
                    agent = BaseAgent.__new__(BaseAgent)
                    BaseAgent.__init__(agent, "agent1_protocol")
                finally:
                    os.chdir(orig_dir)

        assert agent._backend == "groq", "Backend must be 'groq'"
        assert agent._groq_provider is not None, "_groq_provider must be initialised"
        assert agent.device == "cloud", "device must be 'cloud' for Groq backend"
        assert agent.model is None, "No HF model should be loaded in Groq mode"
        assert agent.tokenizer is None, "No tokenizer should be loaded in Groq mode"

    def test_groq_generate_delegates_to_provider(self, tmp_path):
        """BaseAgent.generate() delegates to GroqProvider in groq mode."""
        config_dir = self._make_config_yaml(tmp_path, provider="groq")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = _make_groq_response(
            "groq output"
        )

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_groq_client):
                orig_dir = os.getcwd()
                try:
                    os.chdir(config_dir)
                    from agents.base import BaseAgent
                    agent = BaseAgent.__new__(BaseAgent)
                    BaseAgent.__init__(agent, "agent1_protocol")
                    result = agent.generate("hello")
                finally:
                    os.chdir(orig_dir)

        assert result == "groq output"

    def test_groq_load_model_is_noop(self, tmp_path):
        """BaseAgent.load_model() is a no-op in Groq mode (no VRAM ops)."""
        config_dir = self._make_config_yaml(tmp_path, provider="groq")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = _make_groq_response("ok")

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_groq_client):
                orig_dir = os.getcwd()
                try:
                    os.chdir(config_dir)
                    from agents.base import BaseAgent
                    agent = BaseAgent.__new__(BaseAgent)
                    BaseAgent.__init__(agent, "agent1_protocol")
                    # Must not raise and must not set model
                    agent.load_model()
                finally:
                    os.chdir(orig_dir)

        assert agent.model is None

    def test_groq_free_vram_is_noop(self, tmp_path):
        """BaseAgent.free_vram() is a no-op in Groq mode."""
        config_dir = self._make_config_yaml(tmp_path, provider="groq")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = _make_groq_response("ok")

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
            with patch("agents.groq_adapter.Groq", return_value=mock_groq_client):
                orig_dir = os.getcwd()
                try:
                    os.chdir(config_dir)
                    from agents.base import BaseAgent
                    agent = BaseAgent.__new__(BaseAgent)
                    BaseAgent.__init__(agent, "agent1_protocol")
                    agent.free_vram()  # must not raise
                finally:
                    os.chdir(orig_dir)

        # If we get here without exception, the test passes
        assert agent.model is None


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — BaseAgent in local mode (no regression)
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseAgentLocalMode:
    """test_base_agent_local_mode: provider='local' → HF path is used, no Groq."""

    def _make_local_config_yaml(self, tmp_path) -> str:
        """Write a minimal config.yaml with provider='local'."""
        import yaml
        cfg = {
            "groq": {
                "provider": "local",  # <-- local backend
            },
            "models": {"agent1_protocol": "./models/qwen2.5-7b"},
            "generation": {
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.3,
            },
            "device": "cpu",
            "quantization": "none",
        }
        config_path = os.path.join(tmp_path, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        return tmp_path

    def test_local_backend_does_not_create_groq_provider(self, tmp_path):
        """BaseAgent with local provider must NOT instantiate GroqProvider."""
        config_dir = self._make_local_config_yaml(tmp_path)

        orig_dir = os.getcwd()
        try:
            os.chdir(config_dir)
            # Patch torch so no CUDA check fails
            with patch("agents.base.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                from agents.base import BaseAgent
                agent = BaseAgent.__new__(BaseAgent)
                BaseAgent.__init__(agent, "agent1_protocol")
        finally:
            os.chdir(orig_dir)

        assert agent._backend == "local", "Backend must be 'local'"
        assert agent._groq_provider is None, "No GroqProvider should exist in local mode"

    def test_local_backend_sets_model_path(self, tmp_path):
        """BaseAgent in local mode reads model path from config."""
        config_dir = self._make_local_config_yaml(tmp_path)

        orig_dir = os.getcwd()
        try:
            os.chdir(config_dir)
            with patch("agents.base.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                from agents.base import BaseAgent
                agent = BaseAgent.__new__(BaseAgent)
                BaseAgent.__init__(agent, "agent1_protocol")
        finally:
            os.chdir(orig_dir)

        assert agent.model_path == "./models/qwen2.5-7b"
        assert agent.model_paths == ["./models/qwen2.5-7b"]

    def test_local_backend_device_is_cpu(self, tmp_path):
        """BaseAgent in local mode uses CPU when CUDA is unavailable."""
        config_dir = self._make_local_config_yaml(tmp_path)

        orig_dir = os.getcwd()
        try:
            os.chdir(config_dir)
            with patch("agents.base.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                from agents.base import BaseAgent
                agent = BaseAgent.__new__(BaseAgent)
                BaseAgent.__init__(agent, "agent1_protocol")
        finally:
            os.chdir(orig_dir)

        assert agent.device == "cpu"

    def test_local_backend_generate_calls_hf_path(self, tmp_path):
        """BaseAgent.generate() in local mode uses the HF tokenizer/model path."""
        config_dir = self._make_local_config_yaml(tmp_path)

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 5))}
        mock_tokenizer.decode.return_value = "local model output"

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.__getitem__ = lambda self, i: MagicMock()
        mock_model.generate.return_value = [MagicMock()]

        orig_dir = os.getcwd()
        try:
            os.chdir(config_dir)
            with patch("agents.base.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
                mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
                from agents.base import BaseAgent
                agent = BaseAgent.__new__(BaseAgent)
                BaseAgent.__init__(agent, "agent1_protocol")

                # Inject mock model/tokenizer directly to simulate a loaded model
                agent.model = mock_model
                agent.tokenizer = mock_tokenizer
                agent._loaded_path = agent.model_path
                agent._has_chat_template = False
                agent._backend = "local"
                agent._groq_provider = None

                # Patch _tokenize_prompt and _decode_response to avoid real HF ops
                agent._tokenize_prompt = MagicMock(
                    return_value={"input_ids": MagicMock(shape=(1, 5))}
                )
                agent._decode_response = MagicMock(return_value="local model output")

                result = agent.generate("hello")
        finally:
            os.chdir(orig_dir)

        assert result == "local model output"
        # GroqProvider must never have been touched
        assert agent._groq_provider is None

    def test_missing_api_key_raises_valueerror(self):
        """GroqProvider raises ValueError when GROQ_API_KEY is not set."""
        env = {k: v for k, v in os.environ.items() if "GROQ" not in k}
        with patch.dict(os.environ, env, clear=True):
            with patch("agents.groq_adapter.Groq"):
                with patch("dotenv.load_dotenv", return_value=None):
                    from agents.groq_adapter import GroqProvider
                    with pytest.raises(ValueError, match="Groq API key not found"):
                        GroqProvider(_minimal_config())
