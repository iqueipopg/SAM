"""Groq cloud LLM adapter for SAM pipeline.

Provides a GroqProvider class that mirrors BaseAgent's generate() interface,
delegating inference to Groq's cloud API instead of a local HuggingFace model.

Usage:
    provider = GroqProvider(config)
    response = provider.generate(prompt, max_tokens=512, temperature=0.3)
"""

import os
import re
import logging

from agents.retry import retry_with_backoff

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)


class GroqProvider:
    """Cloud LLM provider using the Groq Python SDK.

    Mirrors BaseAgent's generate() interface so agents can transparently
    switch between local HF models and Groq cloud inference.
    """

    def __init__(self, config: dict):
        """Initialize Groq provider from config.yaml groq section.

        Args:
            config: Full config dict (expects config['groq'] sub-dict).
        """
        groq_cfg = config.get("groq", {})
        self.model = groq_cfg.get("model", "llama-3.3-70b-versatile")
        self.max_retries = groq_cfg.get("max_retries", 3)
        self.timeout = groq_cfg.get("timeout", 60)

        # Load API key from environment variable
        api_key_env = groq_cfg.get("api_key_env", "GROQ_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            # Try loading from .env file via python-dotenv
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

        # Initialize Groq client
        if Groq is None:
            raise ImportError(
                "groq package not installed. Install with: pip install groq"
            )

        self.client = Groq(api_key=api_key)
        logger.info(
            "GroqProvider initialized: model=%s, timeout=%ds",
            self.model, self.timeout,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        """Generate text using Groq cloud API.

        Args:
            prompt: The user prompt to send.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        messages = [{"role": "user", "content": prompt}]

        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=self.timeout,
            )
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
        # Unmatched braces — close and return
        return cleaned[start:] + "}"
