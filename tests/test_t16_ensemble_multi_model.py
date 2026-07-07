# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for T16.6 — ensemble multi-modelo in base.py, agent3_screening.py.

Covers:
1. get_ensemble_models() with ensemble_models list in agents.json → normalised plan.
2. get_ensemble_models() with legacy models.agentX list in config.yaml → plan with votes=1.
3. get_ensemble_models() with both sources → agents.json takes precedence.
4. get_ensemble_models() with neither source → empty list (use default behaviour).
5. A3._run_ensemble_vote with 2-model plan: load_model called 2× with correct paths/quants,
   free_vram called between models, votes from both models are aggregated.
6. Backward compat: without ensemble_models config, legacy single-model path works.

All tests CPU-only, no GPU, no network.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

pytestmark = pytest.mark.skip(
    reason="T16 Rudiger feature deferred post-defensa: ensemble multi-model "
           "requires Rudiger access for VRAM-cycling validation. Tests written "
           "against unimplemented BaseAgent.get_ensemble_models / "
           "Agent3._run_ensemble_vote. Ver docs/DIARIO_SESIONES.md sesión 16."
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Protocol and paper fixtures
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOL = {
    "framework": "PICO",
    "research_question": "Does CBT reduce depression in adults?",
    "population": "Adults with major depressive disorder",
    "intervention": "Cognitive Behavioral Therapy (CBT)",
    "comparator": "Waitlist",
    "outcome": "Depression (PHQ-9)",
    "inclusion_criteria": ["RCT", "Adults 18+"],
    "exclusion_criteria": ["Reviews"],
}

PAPER = {
    "pmid": "12345678",
    "title": "CBT for adult depression: an RCT",
    "abstract": "120 adults randomized. CBT reduced PHQ-9 significantly.",
    "year": "2024",
    "journal": "JAMA Psychiatry",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_agents_json(tmp_path, agent3_cfg: dict) -> str:
    data = {
        "agents": {"agent3_screening": agent3_cfg},
        "defaults": {"temperature": 0.2, "repetition_penalty": 1.3,
                     "max_tokens": 150, "max_input_chars": 800},
    }
    path = tmp_path / "config" / "agents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return str(path)


def _make_base_agent_with_params(agent_name: str, agents_json_path: str, model_paths=None):
    """Return a BaseAgent-like MagicMock with get_ensemble_models wired up properly."""
    from agents.base import BaseAgent
    from agents import base as base_module

    def _patched_load(name):
        if not os.path.isfile(agents_json_path):
            return {}
        try:
            with open(agents_json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        defaults = data.get("defaults", {})
        agent_cfg = data.get("agents", {}).get(name, {})
        merged = dict(defaults)
        merged.update(agent_cfg)
        return merged

    with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_m, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_t, \
         patch.object(base_module.BaseAgent, "_load_agent_params", staticmethod(_patched_load)):
        mock_m.return_value = MagicMock()
        mock_t.return_value = MagicMock()
        agent = BaseAgent.__new__(BaseAgent)
        agent.agent_params = _patched_load(agent_name)
        agent.agent_name = agent_name
        agent.quantization = "4bit"
        agent.model = None
        agent.tokenizer = None
        agent._loaded_path = None
        agent._loaded_quantization = None
        agent._backend = "local"
        import yaml
        with open(os.path.join(PROJECT_ROOT, "config.yaml")) as f:
            agent.config = yaml.safe_load(f) or {}
        if model_paths:
            agent.model_paths = model_paths
            agent.model_path = model_paths[0]
        else:
            agent.model_paths = ["/models/qwen2.5-7b"]
            agent.model_path = "/models/qwen2.5-7b"
        from agents.logger import get_logger
        agent.logger = get_logger("BaseAgent")
    return agent


def _make_a3_agent(agents_json_path: str, ensemble_size=3, model_paths=None):
    """Build Agent3Screening with mocked model loading, using a custom agents.json."""
    from agents.agent3_screening import Agent3Screening
    from agents import base as base_module

    def _patched_load(name):
        if not os.path.isfile(agents_json_path):
            return {}
        try:
            with open(agents_json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        defaults = data.get("defaults", {})
        agent_cfg = data.get("agents", {}).get(name, {})
        merged = dict(defaults)
        merged.update(agent_cfg)
        return merged

    with patch("agents.base.AutoModelForCausalLM.from_pretrained") as mock_m, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_t, \
         patch.object(Agent3Screening, "load_model", return_value=None), \
         patch.object(base_module.BaseAgent, "_load_agent_params", staticmethod(_patched_load)):
        mock_m.return_value = MagicMock()
        mock_t.return_value = MagicMock()
        agent = Agent3Screening(ensemble_size=ensemble_size, voting_threshold=0.60)

    agent.agent_params = _patched_load("agent3_screening")
    agent.model = MagicMock()
    agent.tokenizer = MagicMock()
    if model_paths:
        agent.model_paths = model_paths
        agent.model_path = model_paths[0]
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# 1. get_ensemble_models() — new ensemble_models list
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEnsembleModelsNewFormat:
    """get_ensemble_models() reads ensemble_models from agents.json."""

    def test_string_items_normalized(self, tmp_path):
        """String items get quantization from self.quantization and votes=1."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": [
                "/models/qwen2.5-7b",
                "/models/mistral-1.5b",
            ]
        })
        agent = _make_base_agent_with_params("agent3_screening", agents_json)
        plan = agent.get_ensemble_models()
        assert len(plan) == 2
        assert plan[0]["path"] == "/models/qwen2.5-7b"
        assert plan[0]["votes"] == 1
        assert plan[0]["quantization"] == agent.quantization
        assert plan[1]["path"] == "/models/mistral-1.5b"
        assert plan[1]["votes"] == 1

    def test_dict_items_with_votes_and_quant(self, tmp_path):
        """Dict items preserve explicit path, quantization, and votes."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": [
                {"path": "/models/qwen2.5-7b", "quantization": "4bit", "votes": 2},
                {"path": "/models/mistral-1.5b", "quantization": "none", "votes": 1},
            ]
        })
        agent = _make_base_agent_with_params("agent3_screening", agents_json)
        plan = agent.get_ensemble_models()
        assert len(plan) == 2
        assert plan[0]["votes"] == 2
        assert plan[0]["quantization"] == "4bit"
        assert plan[1]["votes"] == 1
        assert plan[1]["quantization"] == "none"

    def test_votes_floor_is_one(self, tmp_path):
        """votes=0 or negative in config must be clamped to 1."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": [
                {"path": "/models/qwen2.5-7b", "votes": 0},
            ]
        })
        agent = _make_base_agent_with_params("agent3_screening", agents_json)
        plan = agent.get_ensemble_models()
        assert plan[0]["votes"] == 1

    def test_plan_keys_present(self, tmp_path):
        """Every item in the plan must have path, quantization, votes."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": ["/models/a", "/models/b", "/models/c"]
        })
        agent = _make_base_agent_with_params("agent3_screening", agents_json)
        plan = agent.get_ensemble_models()
        for item in plan:
            assert "path" in item
            assert "quantization" in item
            assert "votes" in item


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_ensemble_models() — legacy list from config.yaml
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEnsembleModelsLegacyFormat:
    """Legacy: config.yaml models.agent3_screening = ["a", "b"] → plan with votes=1."""

    def test_legacy_list_becomes_plan(self, tmp_path):
        """Two paths in model_paths → plan of 2 with votes=1 each."""
        agents_json = _write_agents_json(tmp_path, {})  # no ensemble_models key
        agent = _make_base_agent_with_params(
            "agent3_screening", agents_json,
            model_paths=["/models/model_a", "/models/model_b"],
        )
        # Remove ensemble_models param to force legacy path
        agent.agent_params.pop("ensemble_models", None)
        plan = agent.get_ensemble_models()
        assert len(plan) == 2
        for item in plan:
            assert item["votes"] == 1
            assert item["quantization"] == agent.quantization

    def test_legacy_single_path_returns_empty(self, tmp_path):
        """Single path in model_paths → empty plan (use default behaviour)."""
        agents_json = _write_agents_json(tmp_path, {})
        agent = _make_base_agent_with_params(
            "agent3_screening", agents_json,
            model_paths=["/models/only_one"],
        )
        agent.agent_params.pop("ensemble_models", None)
        plan = agent.get_ensemble_models()
        assert plan == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precedence: agents.json wins over legacy config.yaml list
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsemblePrecedence:
    """When both sources are present, agents.json takes precedence."""

    def test_agents_json_wins_over_legacy_model_paths(self, tmp_path):
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": [
                {"path": "/models/new_primary", "quantization": "4bit", "votes": 3},
            ]
        })
        agent = _make_base_agent_with_params(
            "agent3_screening", agents_json,
            model_paths=["/models/legacy_a", "/models/legacy_b"],
        )
        plan = agent.get_ensemble_models()
        # Must return the agents.json plan, not the legacy list
        assert len(plan) == 1
        assert plan[0]["path"] == "/models/new_primary"
        assert plan[0]["votes"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. No multi-model config → empty plan
# ─────────────────────────────────────────────────────────────────────────────

class TestNoEnsembleConfig:
    """Without any multi-model config, get_ensemble_models returns []."""

    def test_no_config_returns_empty_list(self, tmp_path):
        agents_json = _write_agents_json(tmp_path, {})
        agent = _make_base_agent_with_params(
            "agent3_screening", agents_json,
            model_paths=["/models/single"],
        )
        agent.agent_params.pop("ensemble_models", None)
        plan = agent.get_ensemble_models()
        assert plan == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. A3._run_ensemble_vote with 2-model plan
# ─────────────────────────────────────────────────────────────────────────────

class TestA3RunEnsembleVote:
    """_run_ensemble_vote with a 2-model plan: load/free pattern and vote aggregation."""

    def _build_plan(self):
        return [
            {"path": "/models/model_a", "quantization": "4bit", "votes": 1},
            {"path": "/models/model_b", "quantization": "none", "votes": 2},
        ]

    def test_load_model_called_for_each_model(self, tmp_path):
        """load_model must be called once per model with the correct path and quant."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": self._build_plan()
        })
        agent = _make_a3_agent(agents_json, ensemble_size=3)

        with patch.object(agent, "get_ensemble_models", return_value=self._build_plan()), \
             patch.object(agent, "load_model") as mock_load, \
             patch.object(agent, "free_vram") as mock_free, \
             patch.object(agent, "_single_vote", return_value={"vote": "INCLUDE", "reasoning": "ok"}):
            agent._run_ensemble_vote(PAPER, PROTOCOL)

        load_calls = mock_load.call_args_list
        assert len(load_calls) == 2
        assert load_calls[0] == call("/models/model_a", quantization="4bit")
        assert load_calls[1] == call("/models/model_b", quantization="none")

    def test_free_vram_called_between_models(self, tmp_path):
        """free_vram must be called after model_a finishes and before model_b loads."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": self._build_plan()
        })
        agent = _make_a3_agent(agents_json, ensemble_size=3)

        call_order = []

        def _record_load(path, quantization=None):
            call_order.append(("load", path))

        def _record_free():
            call_order.append(("free",))

        def _vote(paper, protocol):
            return {"vote": "INCLUDE", "reasoning": "ok"}

        with patch.object(agent, "get_ensemble_models", return_value=self._build_plan()), \
             patch.object(agent, "load_model", side_effect=_record_load), \
             patch.object(agent, "free_vram", side_effect=_record_free), \
             patch.object(agent, "_single_vote", side_effect=_vote):
            agent._run_ensemble_vote(PAPER, PROTOCOL)

        # Pattern: load_a → vote(s) → free → load_b → vote(s)
        load_positions = [i for i, e in enumerate(call_order) if e[0] == "load"]
        free_positions = [i for i, e in enumerate(call_order) if e[0] == "free"]
        assert len(load_positions) == 2
        assert len(free_positions) >= 1
        # free must come AFTER first load and BEFORE second load
        assert free_positions[0] > load_positions[0]
        assert free_positions[0] < load_positions[1]

    def test_votes_aggregated_from_all_models(self, tmp_path):
        """Total votes = sum of per-model votes in the plan (1 + 2 = 3)."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": self._build_plan()
        })
        agent = _make_a3_agent(agents_json, ensemble_size=3)

        with patch.object(agent, "get_ensemble_models", return_value=self._build_plan()), \
             patch.object(agent, "load_model"), \
             patch.object(agent, "free_vram"), \
             patch.object(agent, "_single_vote", return_value={"vote": "INCLUDE", "reasoning": "ok"}):
            vote_details = agent._run_ensemble_vote(PAPER, PROTOCOL)

        assert len(vote_details) == 3  # 1 vote from model_a + 2 from model_b

    def test_model_name_tagged_in_vote_details(self, tmp_path):
        """Each vote detail must have a 'model' key set to the model's basename."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": self._build_plan()
        })
        agent = _make_a3_agent(agents_json, ensemble_size=3)

        with patch.object(agent, "get_ensemble_models", return_value=self._build_plan()), \
             patch.object(agent, "load_model"), \
             patch.object(agent, "free_vram"), \
             patch.object(agent, "_single_vote", return_value={"vote": "EXCLUDE", "reasoning": "no"}):
            vote_details = agent._run_ensemble_vote(PAPER, PROTOCOL)

        model_names = {v["model"] for v in vote_details}
        assert "model_a" in model_names
        assert "model_b" in model_names

    def test_mixed_votes_include_and_exclude(self, tmp_path):
        """If models disagree, the vote list reflects both and aggregation fires."""
        agents_json = _write_agents_json(tmp_path, {
            "ensemble_models": [
                {"path": "/models/m1", "quantization": "4bit", "votes": 2},
                {"path": "/models/m2", "quantization": "none", "votes": 1},
            ]
        })
        agent = _make_a3_agent(agents_json, ensemble_size=3)

        # m1 always INCLUDEs, m2 always EXCLUDEs
        vote_responses = [
            {"vote": "INCLUDE", "reasoning": "yes"},
            {"vote": "INCLUDE", "reasoning": "yes"},
            {"vote": "EXCLUDE", "reasoning": "no"},
        ]

        with patch.object(agent, "get_ensemble_models", return_value=[
                {"path": "/models/m1", "quantization": "4bit", "votes": 2},
                {"path": "/models/m2", "quantization": "none", "votes": 1},
            ]), \
             patch.object(agent, "load_model"), \
             patch.object(agent, "free_vram"), \
             patch.object(agent, "_single_vote", side_effect=vote_responses):
            result = agent._screen_single(PAPER, PROTOCOL)

        assert result["decision"] in ("INCLUDE", "EXCLUDE", "UNCERTAIN")
        assert len(result["votes"]) == 3
        include_count = sum(1 for v in result["votes"] if v["decision"] == "INCLUDE")
        exclude_count = sum(1 for v in result["votes"] if v["decision"] == "EXCLUDE")
        assert include_count == 2
        assert exclude_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Backward compat: without ensemble config, legacy single-model path works
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatSingleModel:
    """Without multi-model config, old ensemble_size loop still works."""

    def test_single_model_legacy_path_ensemble_size_votes(self, tmp_path):
        """Without ensemble_models, _run_ensemble_vote runs ensemble_size times on default model."""
        agents_json = _write_agents_json(tmp_path, {})  # no ensemble_models
        agent = _make_a3_agent(agents_json, ensemble_size=3)
        # Force empty plan
        with patch.object(agent, "get_ensemble_models", return_value=[]), \
             patch.object(agent, "_single_vote", return_value={"vote": "INCLUDE", "reasoning": "ok"}):
            vote_details = agent._run_ensemble_vote(PAPER, PROTOCOL)
        # Legacy path: exactly ensemble_size votes
        assert len(vote_details) == 3

    def test_single_model_votes_tagged_as_default(self, tmp_path):
        """Legacy path tags votes with model='default'."""
        agents_json = _write_agents_json(tmp_path, {})
        agent = _make_a3_agent(agents_json, ensemble_size=2)
        with patch.object(agent, "get_ensemble_models", return_value=[]), \
             patch.object(agent, "_single_vote", return_value={"vote": "EXCLUDE", "reasoning": "no"}):
            vote_details = agent._run_ensemble_vote(PAPER, PROTOCOL)
        for v in vote_details:
            assert v["model"] == "default"

    def test_single_model_no_load_model_called_by_run_ensemble(self, tmp_path):
        """Legacy path does NOT call load_model inside _run_ensemble_vote."""
        agents_json = _write_agents_json(tmp_path, {})
        agent = _make_a3_agent(agents_json, ensemble_size=2)
        with patch.object(agent, "get_ensemble_models", return_value=[]), \
             patch.object(agent, "load_model") as mock_load, \
             patch.object(agent, "_single_vote", return_value={"vote": "INCLUDE", "reasoning": "ok"}):
            agent._run_ensemble_vote(PAPER, PROTOCOL)
        mock_load.assert_not_called()
