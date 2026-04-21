"""Tests for T16.9 — --config flag in batch_run.py and SAM_CONFIG env var in api.py.

Covers:
1. _load_config with a non-existent path returns {} (graceful).
2. _load_config with a valid temp YAML returns the correct dict.
3. api.SAM_CONFIG_PATH resolves correctly when SAM_CONFIG env var is set.
4. api.SAM_CONFIG_PATH defaults to project_root/config.yaml when env var absent.
5. Relative SAM_CONFIG path is resolved against PROJECT_ROOT.

All tests are CPU-only. No LLM, no GPU, no network.
"""

import importlib
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# batch_run._load_config
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchRunLoadConfig:
    """Tests for batch_run._load_config()."""

    def test_nonexistent_path_returns_empty_dict(self, tmp_path):
        """Path that does not exist → {} without raising."""
        import batch_run
        missing = str(tmp_path / "does_not_exist.yaml")
        result = batch_run._load_config(missing)
        assert result == {}

    def test_valid_yaml_is_loaded_correctly(self, tmp_path):
        """A valid YAML file at the given path is parsed and returned."""
        import batch_run
        cfg = {
            "search": {"max_results_per_database": 42},
            "screening": {"ensemble_size": 5, "voting_threshold": 0.75},
            "custom_key": "rudiger",
        }
        path = tmp_path / "my_config.yaml"
        path.write_text(yaml.dump(cfg))
        result = batch_run._load_config(str(path))
        assert result["search"]["max_results_per_database"] == 42
        assert result["screening"]["ensemble_size"] == 5
        assert result["custom_key"] == "rudiger"

    def test_default_path_loads_real_config(self):
        """Calling _load_config(None) loads the real config.yaml at project root."""
        import batch_run
        result = batch_run._load_config(None)
        # Real config.yaml has at minimum a 'models' or 'search' key
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """An empty YAML file returns {} (yaml.safe_load returns None → converted)."""
        import batch_run
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = batch_run._load_config(str(path))
        assert result == {}

    def test_spanish_characters_in_yaml(self, tmp_path):
        """YAML with Spanish/accented characters is loaded without mangling."""
        import batch_run
        cfg = {
            "descripcion": "Revisión sistemática automatizada",
            "autor": "García-López",
        }
        path = tmp_path / "es_config.yaml"
        path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
        result = batch_run._load_config(str(path))
        assert "García" in result["autor"]
        assert "ó" in result["descripcion"]


# ─────────────────────────────────────────────────────────────────────────────
# api.SAM_CONFIG_PATH and api.load_config()
# ─────────────────────────────────────────────────────────────────────────────

class TestApiConfigPath:
    """Tests for SAM_CONFIG env var resolution in api.py."""

    def _reload_api(self, monkeypatch, env_vars: dict):
        """Set env vars then reimport api so module-level code re-executes."""
        # Clear keys we control
        for k in ["SAM_CONFIG", "SAM_SKIP_ENV_VALIDATION"]:
            monkeypatch.delenv(k, raising=False)
        # Always skip startup validation in tests
        monkeypatch.setenv("SAM_SKIP_ENV_VALIDATION", "1")
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)
        # Remove cached module so re-import re-runs module-level code
        if "api" in sys.modules:
            del sys.modules["api"]
        import api as api_module
        return api_module

    def test_default_path_is_project_root_config_yaml(self, monkeypatch):
        """Without SAM_CONFIG, SAM_CONFIG_PATH points to <project_root>/config.yaml."""
        api = self._reload_api(monkeypatch, {})
        expected = os.path.join(PROJECT_ROOT, "config.yaml")
        assert str(api.SAM_CONFIG_PATH) == expected

    def test_absolute_sam_config_env_var(self, monkeypatch, tmp_path):
        """SAM_CONFIG set to an absolute path → SAM_CONFIG_PATH equals that path."""
        custom_cfg = tmp_path / "custom.yaml"
        custom_cfg.write_text("search:\n  max_results_per_database: 77\n")
        api = self._reload_api(monkeypatch, {"SAM_CONFIG": str(custom_cfg)})
        assert str(api.SAM_CONFIG_PATH) == str(custom_cfg)

    def test_relative_sam_config_resolved_against_project_root(self, monkeypatch):
        """A relative SAM_CONFIG path is resolved against PROJECT_ROOT."""
        api = self._reload_api(monkeypatch, {"SAM_CONFIG": "config.yaml"})
        expected = os.path.join(PROJECT_ROOT, "config.yaml")
        assert str(api.SAM_CONFIG_PATH) == expected

    def test_load_config_reads_sam_config_path(self, monkeypatch, tmp_path):
        """api.load_config() loads from SAM_CONFIG_PATH (honours the env var)."""
        custom_cfg = tmp_path / "rudiger.yaml"
        custom_cfg.write_text(
            yaml.dump({"search": {"max_results_per_database": 99}}),
        )
        api = self._reload_api(monkeypatch, {"SAM_CONFIG": str(custom_cfg)})
        result = api.load_config()
        assert result["search"]["max_results_per_database"] == 99

    def test_load_config_default_reads_existing_config_yaml(self, monkeypatch):
        """api.load_config() without env var reads the real config.yaml."""
        api = self._reload_api(monkeypatch, {})
        result = api.load_config()
        assert isinstance(result, dict)
        assert len(result) > 0
