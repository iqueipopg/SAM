# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for T16.1 — section_max_tokens_profile in config/agents.json.

Covers:
1. compact profile (default) returns original section token values.
2. extended profile returns 3-4x larger values.
3. Missing key in agents.json falls back to hardcoded defaults without error.
4. Invalid/unknown profile falls back to compact defaults without raising.

All tests are CPU-only, no GPU. The LLM is never actually loaded.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.skip(
    reason="T16 max_tokens profile refactor deferred post-defensa: "
           "_SECTION_MAX_TOKENS rename to _DEFAULTS pending + agents.json schema "
           "extension. Ver docs/DIARIO_SESIONES.md sesión 16."
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Expected compact defaults (must match _SECTION_MAX_TOKENS_DEFAULTS in A8)
# ─────────────────────────────────────────────────────────────────────────────
COMPACT_DEFAULTS = {
    "abstract": 400,
    "introduction": 800,
    "methods": 1000,
    "results": 1200,
    "discussion": 1000,
    "conclusion": 500,
}

EXTENDED_PROFILE = {
    "abstract": 1200,
    "introduction": 2400,
    "methods": 3000,
    "results": 3600,
    "discussion": 3000,
    "conclusion": 1500,
}


def _make_agents_json(tmp_path, agent8_override: dict) -> str:
    """Write a minimal agents.json with the given agent8 overrides."""
    data = {
        "agents": {
            "agent8_manuscript": agent8_override,
        },
        "defaults": {
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "max_tokens": 300,
            "max_input_chars": 4000,
        },
    }
    path = tmp_path / "config" / "agents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return str(path)


def _load_params_from(agents_json_path: str, agent_name: str) -> dict:
    """Load merged agent params from a specific agents.json path."""
    if not os.path.isfile(agents_json_path):
        return {}
    try:
        with open(agents_json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    defaults = data.get("defaults", {})
    agent_cfg = data.get("agents", {}).get(agent_name, {})
    merged = dict(defaults)
    merged.update(agent_cfg)
    return merged


def _make_a8_agent(agents_json_path: str):
    """Instantiate Agent8Manuscript with mocked model loading.

    Patches:
    - AutoModelForCausalLM.from_pretrained: never called for CPU tests
    - BaseAgent._load_agent_params: redirected to the temp agents.json
    - AutoTokenizer.from_pretrained: mocked
    - load_model: no-op
    """
    from agents.agent8_manuscript import Agent8Manuscript

    # _load_agent_params is a @staticmethod; patch it via the module-level name
    def _patched_load(agent_name):
        return _load_params_from(agents_json_path, agent_name)

    with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch.object(Agent8Manuscript, "load_model", return_value=None), \
         patch("agents.base.BaseAgent._load_agent_params", staticmethod(_patched_load)):
        mock_model.return_value = MagicMock()
        mock_tok.return_value = MagicMock()
        agent = Agent8Manuscript()

    agent.model = MagicMock()
    agent.tokenizer = MagicMock()
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_profile(agent, profile_name: str) -> dict:
    """Mimic the profile-resolution logic used inside _generate_section."""
    all_profiles = agent.get_agent_param("section_max_tokens", {})
    if isinstance(all_profiles, dict) and isinstance(all_profiles.get(profile_name), dict):
        return all_profiles[profile_name]
    return agent._SECTION_MAX_TOKENS_DEFAULTS


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactProfileDefault:
    """compact profile must yield the original A8 token limits."""

    def test_compact_profile_values_match_defaults(self, tmp_path):
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": "compact",
            "section_max_tokens": {
                "compact": COMPACT_DEFAULTS,
                "extended": EXTENDED_PROFILE,
            },
        })
        agent = _make_a8_agent(agents_json)

        profile = agent.get_agent_param("section_max_tokens_profile", "compact")
        assert profile == "compact"
        resolved = _resolve_profile(agent, profile)
        for section, expected in COMPACT_DEFAULTS.items():
            assert resolved[section] == expected, (
                f"compact[{section}] expected {expected}, got {resolved[section]}"
            )

    def test_default_profile_name_is_compact_when_absent(self, tmp_path):
        """If section_max_tokens_profile key is missing, get_agent_param returns 'compact'."""
        agents_json = _make_agents_json(tmp_path, {})
        agent = _make_a8_agent(agents_json)
        profile = agent.get_agent_param("section_max_tokens_profile", "compact")
        assert profile == "compact"

    def test_class_level_defaults_match_compact(self):
        """_SECTION_MAX_TOKENS_DEFAULTS on the class must equal compact values."""
        from agents.agent8_manuscript import Agent8Manuscript
        for section, expected in COMPACT_DEFAULTS.items():
            actual = Agent8Manuscript._SECTION_MAX_TOKENS_DEFAULTS.get(section)
            assert actual == expected, (
                f"Class default for {section}: expected {expected}, got {actual}"
            )

    def test_section_max_tokens_backward_compat_alias(self):
        """_SECTION_MAX_TOKENS must still be the same object as _SECTION_MAX_TOKENS_DEFAULTS."""
        from agents.agent8_manuscript import Agent8Manuscript
        assert Agent8Manuscript._SECTION_MAX_TOKENS is Agent8Manuscript._SECTION_MAX_TOKENS_DEFAULTS


class TestExtendedProfile:
    """extended profile must return values ~3-4x the compact ones."""

    def test_extended_values_larger_than_compact(self, tmp_path):
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": "extended",
            "section_max_tokens": {
                "compact": COMPACT_DEFAULTS,
                "extended": EXTENDED_PROFILE,
            },
        })
        agent = _make_a8_agent(agents_json)

        profile = agent.get_agent_param("section_max_tokens_profile", "compact")
        assert profile == "extended"
        resolved = _resolve_profile(agent, profile)
        for section in COMPACT_DEFAULTS:
            compact_val = COMPACT_DEFAULTS[section]
            extended_val = resolved[section]
            assert extended_val > compact_val, (
                f"extended[{section}] should be > compact: {extended_val} vs {compact_val}"
            )
            ratio = extended_val / compact_val
            assert 2.5 <= ratio <= 5.0, (
                f"extended[{section}] ratio {ratio:.1f} not in [2.5, 5.0]"
            )

    def test_extended_abstract_value(self, tmp_path):
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": "extended",
            "section_max_tokens": {
                "compact": COMPACT_DEFAULTS,
                "extended": EXTENDED_PROFILE,
            },
        })
        agent = _make_a8_agent(agents_json)
        resolved = _resolve_profile(agent, "extended")
        assert resolved["abstract"] == EXTENDED_PROFILE["abstract"]
        assert resolved["results"] == EXTENDED_PROFILE["results"]


class TestMissingKeyFallback:
    """When section_max_tokens is absent from agents.json, use hardcoded defaults."""

    def test_missing_section_max_tokens_key_falls_back(self, tmp_path):
        # agents.json exists but has no section_max_tokens entry
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": "compact",
            # no "section_max_tokens" key at all
        })
        agent = _make_a8_agent(agents_json)
        all_profiles = agent.get_agent_param("section_max_tokens", {})
        # Should return empty dict (key not in agents.json)
        assert all_profiles == {} or not isinstance(all_profiles.get("compact"), dict)
        # Fallback path: _SECTION_MAX_TOKENS_DEFAULTS should be used
        resolved = _resolve_profile(agent, "compact")
        assert resolved == agent._SECTION_MAX_TOKENS_DEFAULTS

    def test_no_agents_json_file_falls_back_to_defaults(self, tmp_path):
        """If agents.json does not exist at all, _load_agent_params returns {}."""
        nonexistent_path = str(tmp_path / "config" / "agents.json")
        agent = _make_a8_agent(nonexistent_path)
        # get_agent_param for any T16 key returns the fallback
        profile = agent.get_agent_param("section_max_tokens_profile", "compact")
        assert profile == "compact"
        all_profiles = agent.get_agent_param("section_max_tokens", {})
        assert all_profiles == {}


class TestInvalidProfileFallback:
    """An unrecognised profile name must fall back to defaults without raising."""

    def test_invalid_profile_name_falls_back_to_defaults(self, tmp_path):
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": "turbo_xlarge",  # not a known profile
            "section_max_tokens": {
                "compact": COMPACT_DEFAULTS,
                "extended": EXTENDED_PROFILE,
            },
        })
        agent = _make_a8_agent(agents_json)
        all_profiles = agent.get_agent_param("section_max_tokens", {})
        profile = agent.get_agent_param("section_max_tokens_profile", "compact")
        # profile resolution should not raise
        if isinstance(all_profiles, dict) and isinstance(all_profiles.get(profile), dict):
            resolved = all_profiles[profile]
        else:
            resolved = agent._SECTION_MAX_TOKENS_DEFAULTS
        # Must always return a valid dict with all sections
        for section in COMPACT_DEFAULTS:
            assert section in resolved
            assert isinstance(resolved[section], int)
            assert resolved[section] > 0

    def test_no_exception_on_bad_profile(self, tmp_path):
        """The resolution code path never raises even with malformed input."""
        agents_json = _make_agents_json(tmp_path, {
            "section_max_tokens_profile": 42,          # non-string type
            "section_max_tokens": "not_a_dict",         # wrong type
        })
        agent = _make_a8_agent(agents_json)
        # Should not raise
        try:
            all_profiles = agent.get_agent_param("section_max_tokens", {})
            profile = agent.get_agent_param("section_max_tokens_profile", "compact")
            if isinstance(all_profiles, dict) and isinstance(all_profiles.get(profile), dict):
                resolved = all_profiles[profile]
            else:
                resolved = agent._SECTION_MAX_TOKENS_DEFAULTS
            assert isinstance(resolved, dict)
        except Exception as exc:
            pytest.fail(f"Resolution raised unexpectedly: {exc}")
