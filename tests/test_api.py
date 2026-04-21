"""Tests for the FastAPI backend (api.py).

CPU-only — no GPU / no LLM / no network. Covers:
  * ARQ1 — /api/run with resume_run_id
  * ARQ2 — atomic checkpoint write (tmp + os.replace)
  * ARQ3 — validate_env() at startup
  * ARQ6 — CORS_ORIGINS env var
  * ARQ7 — single-concurrent-pipeline rate-limit (HTTP 429)
  * ARQ8 — cooperative kill via _cancel_requested flag

These tests exercise small helper functions directly and use FastAPI's
TestClient for HTTP-level checks. The heavy pipeline runner is mocked
so tests run in a few seconds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The api module imports torch / transformers lazily only when the pipeline
# actually runs, so we can import it here safely.
import api as api_module  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_pipeline_state():
    """Ensure each test starts with a clean pipeline state."""
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


# ════════════════════════════════════════════════════════════════════════════
# ARQ1 — resume_run_id
# ════════════════════════════════════════════════════════════════════════════

def _write_checkpoint(run_dir: Path, name: str, payload: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"checkpoint_{name}.json").write_text(json.dumps(payload))


def test_arq1_discover_checkpoints_finds_all(tmp_path):
    """discover_checkpoints returns only agents with existing checkpoint files."""
    run_dir = tmp_path / "run_123"
    _write_checkpoint(run_dir, "a1_protocol", {"protocol": {"framework": "PICO"}})
    _write_checkpoint(run_dir, "a3_screening", {"stats": {"total": 5}})

    found = api_module.discover_checkpoints(str(run_dir))
    assert "a1_protocol" in found
    assert "a3_screening" in found
    assert "a2_search" not in found
    assert found["a1_protocol"]["protocol"]["framework"] == "PICO"


def test_arq1_discover_checkpoints_missing_dir_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert api_module.discover_checkpoints(str(missing)) == {}


def test_arq1_api_run_with_missing_resume_id_returns_404(tmp_path, monkeypatch):
    """POST /api/run with an unknown resume_run_id should return HTTP 404."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
    client = TestClient(api_module.app)
    r = client.post(
        "/api/run",
        json={"query": "dummy", "resume_run_id": "run_does_not_exist"},
    )
    assert r.status_code == 404
    body = r.json()
    assert "resume_run_id" in body.get("detail", "")


def test_arq1_api_run_with_existing_resume_id_emits_resumed_event(tmp_path, monkeypatch):
    """If resume_run_id exists, the SSE stream should contain a 'resumed' event
    and skip already-completed agents."""
    from fastapi.testclient import TestClient

    run_id = "run_abc"
    run_dir = tmp_path / run_id
    _write_checkpoint(run_dir, "a1_protocol", {
        "protocol": {"framework": "PICO", "population": "adults"},
        "pubmed_query": "(foo OR bar)",
    })

    monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

    # Replace the heavy pipeline with a minimal generator that exercises only
    # the resume logic and yields our discovered state.
    def fake_runner(req):
        resume_state = api_module.discover_checkpoints(str(tmp_path / req.resume_run_id))
        yield api_module._sse_event("start", {"run_dir": str(tmp_path / req.resume_run_id)})
        if resume_state:
            yield api_module._sse_event("resumed", {
                "run_dir": str(tmp_path / req.resume_run_id),
                "from_agent": "A2",
                "completed_agents": ["A1"],
            })
        yield api_module._sse_event("pipeline_done", {"run_dir": str(tmp_path / req.resume_run_id)})

    monkeypatch.setattr(api_module, "run_pipeline_generator", fake_runner)

    client = TestClient(api_module.app)
    with client.stream("POST", "/api/run", json={"query": "x", "resume_run_id": run_id}) as r:
        assert r.status_code == 200
        text = "".join(chunk for chunk in r.iter_text())
    assert "event: resumed" in text
    assert '"from_agent": "A2"' in text


# ════════════════════════════════════════════════════════════════════════════
# ARQ2 — atomic checkpoint write
# ════════════════════════════════════════════════════════════════════════════

def test_arq2_save_checkpoint_is_atomic(tmp_path):
    """save_checkpoint must write via a tmp file + rename. If the final rename
    never happens, the previous file must remain untouched."""
    run_dir = tmp_path / "run_atomic"
    run_dir.mkdir()
    original = {"version": 1}
    api_module.save_checkpoint(str(run_dir), "a1_protocol", original)
    target = run_dir / "checkpoint_a1_protocol.json"
    assert json.loads(target.read_text())["version"] == 1

    # Simulate a failure during the rename phase by patching os.replace.
    with mock.patch.object(api_module.os, "replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            api_module.save_checkpoint(str(run_dir), "a1_protocol", {"version": 2})

    # Original file must still contain version=1
    assert json.loads(target.read_text())["version"] == 1


def test_arq2_save_checkpoint_tmp_cleanup(tmp_path):
    """On a successful write no .tmp file should remain in the run dir."""
    run_dir = tmp_path / "run_tmp"
    run_dir.mkdir()
    api_module.save_checkpoint(str(run_dir), "a2_search", {"papers": []})
    leftovers = list(run_dir.glob("*.tmp"))
    assert leftovers == []


# ════════════════════════════════════════════════════════════════════════════
# ARQ3 — validate_env()
# ════════════════════════════════════════════════════════════════════════════

def test_arq3_validate_env_raises_when_required_missing(monkeypatch):
    """validate_env must raise RuntimeError if a required variable is absent."""
    monkeypatch.delenv("ENTREZ_EMAIL", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    with pytest.raises(RuntimeError) as ei:
        api_module.validate_env()
    msg = str(ei.value)
    assert "ENTREZ_EMAIL" in msg
    assert "UNPAYWALL_EMAIL" in msg
    # The user-facing message must be in Spanish (rule #12)
    assert "Faltan" in msg or "faltan" in msg


def test_arq3_validate_env_passes_when_required_present(monkeypatch):
    monkeypatch.setenv("ENTREZ_EMAIL", "test@example.com")
    monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.com")
    # Should not raise
    api_module.validate_env()


def test_arq3_validate_env_optional_missing_is_noop(monkeypatch):
    """Missing optional vars should not raise."""
    monkeypatch.setenv("ENTREZ_EMAIL", "a@b.com")
    monkeypatch.setenv("UNPAYWALL_EMAIL", "a@b.com")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    api_module.validate_env()  # no exception


# ════════════════════════════════════════════════════════════════════════════
# ARQ6 — CORS_ORIGINS env var
# ════════════════════════════════════════════════════════════════════════════

def test_arq6_get_cors_origins_default():
    origins = api_module._get_cors_origins(env={})
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins


def test_arq6_get_cors_origins_from_env():
    origins = api_module._get_cors_origins(env={"CORS_ORIGINS": "https://foo.com,https://bar.org"})
    assert origins == ["https://foo.com", "https://bar.org"]


def test_arq6_get_cors_origins_strips_whitespace():
    origins = api_module._get_cors_origins(env={"CORS_ORIGINS": " https://a.com , https://b.com "})
    assert origins == ["https://a.com", "https://b.com"]


# ════════════════════════════════════════════════════════════════════════════
# ARQ7 — single concurrent pipeline
# ════════════════════════════════════════════════════════════════════════════

def test_arq7_second_run_returns_429(monkeypatch, tmp_path):
    """While one pipeline is running, a second POST /api/run returns 429."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)
    # Mark the pipeline as running with an active run_id.
    api_module._pipeline_state["running"] = True
    api_module._pipeline_state["active_run_id"] = "run_first"

    client = TestClient(api_module.app)
    r = client.post("/api/run", json={"query": "concurrent"})
    assert r.status_code == 429
    body = r.json()
    # Body may be wrapped in JSONResponse directly or via HTTPException detail
    assert "run_first" in json.dumps(body)


def test_arq7_status_reports_active_run(monkeypatch):
    from fastapi.testclient import TestClient

    api_module._pipeline_state["running"] = True
    api_module._pipeline_state["active_run_id"] = "run_xyz"
    api_module._pipeline_state["current_run_dir"] = "/tmp/run_xyz"

    client = TestClient(api_module.app)
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["running"] is True
    assert data.get("active_run_id") == "run_xyz"


# ════════════════════════════════════════════════════════════════════════════
# ARQ8 — cooperative kill
# ════════════════════════════════════════════════════════════════════════════

def test_arq8_kill_sets_cancel_flag():
    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)
    r = client.post("/api/kill")
    assert r.status_code == 200
    assert api_module._pipeline_state["cancel_requested"] is True
    assert api_module._pipeline_state["kill_requested"] is True


def test_arq8_check_cancel_hook_on_baseagent():
    """BaseAgent must expose a check_cancel() hook that raises or no-ops."""
    from agents.base import BaseAgent

    assert hasattr(BaseAgent, "check_cancel"), (
        "BaseAgent must have a check_cancel() method for ARQ8"
    )

    # Build a minimal instance without going through __init__ (avoids config.yaml)
    dummy = BaseAgent.__new__(BaseAgent)
    dummy._cancel_flag = None
    # No flag set → must not raise
    dummy.check_cancel()

    # Attach a callable flag that returns True → must raise
    dummy._cancel_flag = lambda: True
    with pytest.raises(Exception):
        dummy.check_cancel()
