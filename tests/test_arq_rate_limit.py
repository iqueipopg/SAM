"""Tests for ARQ7 — single-concurrent-pipeline rate limit in api.py.

Covers:
- GET /api/status inicial → running=False.
- POST /api/run cuando pipeline activo → 429 con active_run_id.
- GET /api/status durante ejecución → active=True con run_id.
- Tras completarse el pipeline → running=False.
- El 429 incluye active_run_id en el body.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api as api_module


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Restaura el estado del pipeline antes y después de cada test."""
    original = dict(api_module._pipeline_state)
    api_module._pipeline_state.update({
        "running": False,
        "stop_requested": False,
        "kill_requested": False,
        "cancel_requested": False,
        "current_run_dir": None,
        "active_run_id": None,
    })
    yield
    api_module._pipeline_state.clear()
    api_module._pipeline_state.update(original)


class TestRateLimit:
    """ARQ7: single-concurrent-pipeline enforcement."""

    def test_status_initial_not_running(self):
        """GET /api/status devuelve running=False cuando no hay pipeline activo."""
        client = TestClient(api_module.app)
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["running"] is False

    def test_second_run_returns_429(self, tmp_path, monkeypatch):
        """Segunda petición a /api/run devuelve HTTP 429 mientras hay pipeline activo."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        api_module._pipeline_state["running"] = True
        api_module._pipeline_state["active_run_id"] = "run_first"

        client = TestClient(api_module.app)
        r = client.post("/api/run", json={"query": "segunda petición"})
        assert r.status_code == 429

    def test_429_body_contains_active_run_id(self, tmp_path, monkeypatch):
        """El body del 429 contiene active_run_id del pipeline en vuelo."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        api_module._pipeline_state["running"] = True
        api_module._pipeline_state["active_run_id"] = "run_inflight_xyz"

        client = TestClient(api_module.app)
        r = client.post("/api/run", json={"query": "concurrent"})
        assert r.status_code == 429
        body_text = json.dumps(r.json())
        assert "run_inflight_xyz" in body_text

    def test_status_shows_active_run_id_during_execution(self, monkeypatch):
        """GET /api/status muestra active_run_id cuando hay pipeline corriendo."""
        api_module._pipeline_state["running"] = True
        api_module._pipeline_state["active_run_id"] = "run_test_abc"
        api_module._pipeline_state["current_run_dir"] = "/tmp/run_test_abc"

        client = TestClient(api_module.app)
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["running"] is True
        assert data.get("active_run_id") == "run_test_abc"

    def test_status_after_pipeline_completes_not_running(self, tmp_path, monkeypatch):
        """Tras completarse el pipeline, /api/status vuelve a running=False."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        def fast_runner(req):
            api_module._pipeline_state["running"] = True
            api_module._pipeline_state["active_run_id"] = "run_done"
            try:
                yield api_module._sse_event("start", {"run_dir": str(tmp_path)})
                yield api_module._sse_event("pipeline_done", {"run_dir": str(tmp_path)})
            finally:
                api_module._release_pipeline_slot()

        monkeypatch.setattr(api_module, "run_pipeline_generator", fast_runner)

        client = TestClient(api_module.app)
        with client.stream("POST", "/api/run", json={"query": "test-done"}) as r:
            assert r.status_code == 200
            # consume the stream
            list(r.iter_text())

        # After stream finished the slot must be released
        assert api_module._pipeline_state["running"] is False
        assert api_module._pipeline_state["active_run_id"] is None

        # Subsequent GET /api/status must confirm not running
        r2 = client.get("/api/status")
        assert r2.json()["running"] is False

    def test_429_error_field_in_body(self, tmp_path, monkeypatch):
        """El body del 429 incluye campo 'error' descriptivo."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        api_module._pipeline_state["running"] = True
        api_module._pipeline_state["active_run_id"] = "run_active"

        client = TestClient(api_module.app)
        r = client.post("/api/run", json={"query": "extra"})
        assert r.status_code == 429
        body = r.json()
        # There should be some indication of "running" or "active" in the response
        body_str = json.dumps(body).lower()
        assert "running" in body_str or "active" in body_str or "pipeline" in body_str

    def test_first_run_allowed_when_not_running(self, tmp_path, monkeypatch):
        """Primera petición a /api/run se acepta cuando no hay pipeline activo."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        def instant_runner(req):
            yield api_module._sse_event("pipeline_done", {"run_dir": str(tmp_path)})
            api_module._release_pipeline_slot()

        monkeypatch.setattr(api_module, "run_pipeline_generator", instant_runner)

        client = TestClient(api_module.app)
        r = client.post("/api/run", json={"query": "primera"})
        assert r.status_code == 200

    def test_release_pipeline_slot_resets_both_flags(self):
        """_release_pipeline_slot limpia running y active_run_id."""
        api_module._pipeline_state["running"] = True
        api_module._pipeline_state["active_run_id"] = "run_clear"
        api_module._release_pipeline_slot()
        assert api_module._pipeline_state["running"] is False
        assert api_module._pipeline_state["active_run_id"] is None
