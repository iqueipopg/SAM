"""Groq cloud LLM adapter for SAM pipeline.

Provides a GroqProvider class that mirrors BaseAgent's generate() interface,
delegating inference to Groq's cloud API instead of a local HuggingFace model.

Supports per-agent model selection (e.g., 8B for simple tasks, 70B for critical)
and response_format: json_object for guaranteed valid JSON output.

Usage:
    provider = GroqProvider(config, agent_name="agent3_screening")
    response = provider.generate(prompt, max_tokens=512, temperature=0.3)
"""

import os
import re
import time
import logging

from agents.retry import retry_with_backoff

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)

# Rate limit tracking (shared across all GroqProvider instances)
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 2.1  # 30 RPM = 1 req per 2s, add margin


class GroqProvider:
    """Cloud LLM provider using the Groq Python SDK.

    Mirrors BaseAgent's generate() interface so agents can transparently
    switch between local HF models and Groq cloud inference.

    Input: config dict with groq section, optional agent_name for per-agent model.
    Output: generated text string from Groq API.
    GPU: no
    """

    def __init__(self, config: dict, agent_name: str = None):
        groq_cfg = config.get("groq", {})
        self.default_model = groq_cfg.get("model", "llama-3.3-70b-versatile")
        self.max_retries = groq_cfg.get("max_retries", 3)
        self.timeout = groq_cfg.get("timeout", 60)

        # Per-agent model selection from groq.agent_models
        agent_models = groq_cfg.get("agent_models", {})
        self.model = agent_models.get(agent_name, self.default_model) if agent_name else self.default_model

        # Load API key from environment variable
        api_key_env = groq_cfg.get("api_key_env", "GROQ_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get(api_key_env)
            except ImportError:
                pass

        if not api_key:
            raise ValueError(
                f"Groq API key not found. Set the {api_key_env} environment "
                f"variable or add it to your .env file."
            )

        if Groq is None:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )

        self.client = Groq(api_key=api_key)
        logger.info(
            "GroqProvider initialized: agent=%s, model=%s, timeout=%ds",
            agent_name or "default", self.model, self.timeout,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7, json_mode: bool = False) -> str:
        """Generate text using Groq cloud API.

        Args:
            prompt: The user prompt to send.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            json_mode: If True, use response_format: json_object for valid JSON.

        Returns:
            Generated text string.
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            timeout=self.timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _call():
            global _last_request_time
            # Rate limiting: ensure minimum interval between requests
            elapsed = time.time() - _last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            _last_request_time = time.time()

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            base_delay=2.0,
            max_delay=60.0,
            backoff_factor=2.0,
        )

    def _clean_json(self, text: str) -> str:
        """Clean LLM output to extract valid JSON object.

        Mirrors the _clean_json() pattern used across SAM agents:
        strips markdown fences, removes non-ASCII, finds matched braces.
        """
        cleaned = text.replace("```json", "").replace("```", "").strip()
        cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)
        start = cleaned.find("{")
        if start < 0:
            return "{}"
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start:i + 1]
        return cleaned[start:] + "}"
