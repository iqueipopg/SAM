"""Tests for ARQ1 — resume from checkpoint in api.py.

Covers:
- discover_checkpoints con run_id inexistente → {}.
- discover_checkpoints con sólo checkpoint_a1_protocol.json → devuelve a1_protocol.
- discover_checkpoints con A1, A2, A3 → devuelve los tres.
- discover_checkpoints con gap A1+A5 → devuelve ambos (carga los disponibles).
- Resume vía /api/run con resume_run_id válido → emite SSE 'resumed'.
- Resume con run_id inexistente → HTTP 404.
- Resume con run_id que ya completó A8 → HTTP 400 "ya está completo".
- Resume sin checkpoints en el directorio → HTTP 400 "no hay nada que reanudar".
- GET /api/runs/{run_id}/checkpoints con run existente → lista agentes.
- GET /api/runs/nonexistent/checkpoints → HTTP 404.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


def _write_checkpoint(run_dir: Path, agent_name: str, payload: dict):
    """Helper: write a checkpoint file for a given agent."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"checkpoint_{agent_name}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


class TestDiscoverCheckpoints:
    """discover_checkpoints helper function tests."""

    def test_missing_directory_returns_empty_dict(self, tmp_path):
        """Directorio inexistente → {} sin excepción."""
        missing = str(tmp_path / "run_does_not_exist")
        result = api_module.discover_checkpoints(missing)
        assert result == {}

    def test_single_checkpoint_a1_returned(self, tmp_path):
        """Sólo checkpoint_a1_protocol.json → devuelve clave a1_protocol."""
        run_dir = tmp_path / "run_only_a1"
        _write_checkpoint(run_dir, "a1_protocol", {"protocol": {"framework": "PICO"}})
        result = api_module.discover_checkpoints(str(run_dir))
        assert "a1_protocol" in result
        assert result["a1_protocol"]["protocol"]["framework"] == "PICO"

    def test_three_checkpoints_returned(self, tmp_path):
        """A1, A2, A3 presentes → los tres devueltos."""
        run_dir = tmp_path / "run_three"
        _write_checkpoint(run_dir, "a1_protocol", {"step": 1})
        _write_checkpoint(run_dir, "a2_search", {"step": 2})
        _write_checkpoint(run_dir, "a3_screening", {"step": 3})
        result = api_module.discover_checkpoints(str(run_dir))
        assert set(result.keys()) == {"a1_protocol", "a2_search", "a3_screening"}

    def test_gap_in_checkpoints_returns_available(self, tmp_path):
        """A1 + A5 (gap en A2-A4) → devuelve ambos disponibles."""
        run_dir = tmp_path / "run_gap"
        _write_checkpoint(run_dir, "a1_protocol", {"step": 1})
        _write_checkpoint(run_dir, "a5_extraction", {"step": 5})
        result = api_module.discover_checkpoints(str(run_dir))
        assert "a1_protocol" in result
        assert "a5_extraction" in result
        assert "a2_search" not in result

    def test_corrupt_checkpoint_skipped(self, tmp_path):
        """Checkpoint corrupto es ignorado silenciosamente."""
        run_dir = tmp_path / "run_corrupt"
        run_dir.mkdir()
        _write_checkpoint(run_dir, "a1_protocol", {"ok": True})
        # Write corrupt file for a2
        (run_dir / "checkpoint_a2_search.json").write_text("{ invalid json }")
        result = api_module.discover_checkpoints(str(run_dir))
        assert "a1_protocol" in result
        assert "a2_search" not in result  # corrupt → silently skipped

    def test_all_checkpoints_returned(self, tmp_path):
        """Todos los checkpoints A1-A8 presentes → devuelve los 8."""
        run_dir = tmp_path / "run_all"
        for agent in api_module._CHECKPOINT_AGENT_ORDER:
            _write_checkpoint(run_dir, agent, {"agent": agent})
        result = api_module.discover_checkpoints(str(run_dir))
        assert len(result) == len(api_module._CHECKPOINT_AGENT_ORDER)


class TestResumeViaApi:
    """HTTP-level resume tests via /api/run."""

    def test_resume_nonexistent_run_id_returns_404(self, tmp_path, monkeypatch):
        """resume_run_id que no existe → HTTP 404."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        client = TestClient(api_module.app)
        r = client.post(
            "/api/run",
            json={"query": "test", "resume_run_id": "run_nonexistent"},
        )
        assert r.status_code == 404

    def test_resume_404_detail_mentions_run_id(self, tmp_path, monkeypatch):
        """El detalle del 404 menciona el run_id solicitado."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        client = TestClient(api_module.app)
        r = client.post(
            "/api/run",
            json={"query": "test", "resume_run_id": "run_missing_123"},
        )
        assert r.status_code == 404
        detail = r.json().get("detail", "")
        assert "run_missing_123" in detail

    def test_resume_empty_run_dir_returns_400(self, tmp_path, monkeypatch):
        """Directorio existe pero sin checkpoints → HTTP 400 no hay nada que reanudar."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        empty_run = tmp_path / "run_empty"
        empty_run.mkdir()
        client = TestClient(api_module.app)
        r = client.post(
            "/api/run",
            json={"query": "test", "resume_run_id": "run_empty"},
        )
        assert r.status_code == 400

    def test_resume_completed_pipeline_returns_400(self, tmp_path, monkeypatch):
        """run_id que ya tiene A8 checkpoint → HTTP 400 ya está completo."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "run_complete"
        # Write all checkpoints including the terminal a8_manuscript
        for agent in api_module._CHECKPOINT_AGENT_ORDER:
            _write_checkpoint(run_dir, agent, {"agent": agent})
        client = TestClient(api_module.app)
        r = client.post(
            "/api/run",
            json={"query": "test", "resume_run_id": "run_complete"},
        )
        assert r.status_code == 400
        detail = r.json().get("detail", "")
        # Message should indicate it's already complete (in Spanish per rule #12)
        assert "completo" in detail.lower() or "complete" in detail.lower() or "a8" in detail.lower()

    def test_resume_valid_run_id_emits_resumed_event(self, tmp_path, monkeypatch):
        """resume_run_id válido (incompleto) → stream contiene evento 'resumed'."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        run_id = "run_partial_abc"
        run_dir = tmp_path / run_id
        _write_checkpoint(run_dir, "a1_protocol", {
            "protocol": {"framework": "PICO", "population": "adults"},
            "pubmed_query": "(foo AND bar)",
        })

        def fake_runner(req):
            resume_state = api_module.discover_checkpoints(
                str(tmp_path / req.resume_run_id)
            )
            yield api_module._sse_event("start", {"run_dir": str(run_dir)})
            if resume_state:
                completed = list(resume_state.keys())
                yield api_module._sse_event("resumed", {
                    "run_dir": str(run_dir),
                    "from_agent": "A2",
                    "completed_agents": completed,
                    "resumed_from": completed[-1],
                })
            yield api_module._sse_event("pipeline_done", {"run_dir": str(run_dir)})
            api_module._release_pipeline_slot()

        monkeypatch.setattr(api_module, "run_pipeline_generator", fake_runner)

        client = TestClient(api_module.app)
        with client.stream("POST", "/api/run", json={
            "query": "resume test", "resume_run_id": run_id
        }) as r:
            assert r.status_code == 200
            text = "".join(chunk for chunk in r.iter_text())

        assert "event: resumed" in text

    def test_resume_includes_completed_agents_in_event(self, tmp_path, monkeypatch):
        """El evento 'resumed' incluye lista de agentes ya completados."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        run_id = "run_with_three"
        run_dir = tmp_path / run_id
        _write_checkpoint(run_dir, "a1_protocol", {
            "protocol": {"framework": "PICO"},
            "pubmed_query": "foo",
        })
        _write_checkpoint(run_dir, "a2_search", {"papers": [], "stats": {}})
        _write_checkpoint(run_dir, "a3_screening", {"include": [], "stats": {}})

        def fake_runner(req):
            resume_state = api_module.discover_checkpoints(
                str(tmp_path / req.resume_run_id)
            )
            completed = list(resume_state.keys())
            yield api_module._sse_event("resumed", {
                "completed_agents": completed,
                "from_agent": "A4",
            })
            yield api_module._sse_event("pipeline_done", {"run_dir": str(run_dir)})
            api_module._release_pipeline_slot()

        monkeypatch.setattr(api_module, "run_pipeline_generator", fake_runner)

        client = TestClient(api_module.app)
        with client.stream("POST", "/api/run", json={
            "query": "three agents", "resume_run_id": run_id
        }) as r:
            text = "".join(chunk for chunk in r.iter_text())

        assert "a1_protocol" in text
        assert "a2_search" in text


class TestCheckpointsEndpoint:
    """GET /api/runs/{run_id}/checkpoints endpoint tests."""

    def test_checkpoints_nonexistent_run_returns_404(self, tmp_path, monkeypatch):
        """run_id inexistente → HTTP 404."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_does_not_exist/checkpoints")
        assert r.status_code == 404

    def test_checkpoints_existing_run_returns_200(self, tmp_path, monkeypatch):
        """run_id existente → HTTP 200."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "run_has_ckpts"
        _write_checkpoint(run_dir, "a1_protocol", {"ok": True})

        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_has_ckpts/checkpoints")
        assert r.status_code == 200

    def test_checkpoints_lists_completed_agents(self, tmp_path, monkeypatch):
        """El endpoint lista los agentes con checkpoint disponible."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "run_ckpt_list"
        _write_checkpoint(run_dir, "a1_protocol", {"ok": True})
        _write_checkpoint(run_dir, "a2_search", {"ok": True})

        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_ckpt_list/checkpoints")
        data = r.json()
        completed = data.get("completed_agents", [])
        assert "a1_protocol" in completed
        assert "a2_search" in completed

    def test_checkpoints_next_agent_correct(self, tmp_path, monkeypatch):
        """next_agent apunta al primer agente aún sin checkpoint."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "run_next"
        _write_checkpoint(run_dir, "a1_protocol", {"ok": True})

        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_next/checkpoints")
        data = r.json()
        # After A1, next should be A2
        assert data.get("next_agent") == "A2"

    def test_checkpoints_resumable_true_when_incomplete(self, tmp_path, monkeypatch):
        """resumable=True cuando hay agentes pendientes."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        run_dir = tmp_path / "run_resumable"
        _write_checkpoint(run_dir, "a1_protocol", {"ok": True})

        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_resumable/checkpoints")
        assert r.json().get("resumable") is True

    def test_checkpoints_empty_run_dir_returns_no_completed(self, tmp_path, monkeypatch):
        """Directorio sin checkpoints → completed_agents vacío."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
        empty_run = tmp_path / "run_empty_ckpt"
        empty_run.mkdir()

        client = TestClient(api_module.app)
        r = client.get("/api/runs/run_empty_ckpt/checkpoints")
        assert r.status_code == 200
        assert r.json()["completed_agents"] == []
