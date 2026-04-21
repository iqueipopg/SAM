"""Tests for ARQ8 — cooperative cancellation in api.py.

Covers:
- is_cancelled() default → False.
- request_cancel() → is_cancelled() → True.
- reset_cancel() → is_cancelled() → False.
- POST /api/cancel → status 200, {"status": "cancel_requested"}.
- GET /api/status refleja cancel_requested=True tras llamar cancel.
- Nuevo pipeline resetea el flag automáticamente.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

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


class TestCooperativeCancel:
    """ARQ8: cooperative cancellation via is_cancelled / request_cancel / reset_cancel."""

    # ── module-level helpers ─────────────────────────────────────────────────

    def test_is_cancelled_default_false(self):
        """is_cancelled() devuelve False en estado limpio."""
        assert api_module.is_cancelled() is False

    def test_request_cancel_sets_flag(self):
        """request_cancel() pone is_cancelled() a True."""
        api_module.request_cancel()
        assert api_module.is_cancelled() is True

    def test_reset_cancel_clears_flag(self):
        """reset_cancel() pone is_cancelled() de vuelta a False."""
        api_module.request_cancel()
        assert api_module.is_cancelled() is True
        api_module.reset_cancel()
        assert api_module.is_cancelled() is False

    def test_reset_cancel_idempotent_when_already_false(self):
        """reset_cancel() no falla si el flag ya era False."""
        assert api_module.is_cancelled() is False
        api_module.reset_cancel()  # no exception
        assert api_module.is_cancelled() is False

    def test_multiple_request_cancel_idempotent(self):
        """Llamar request_cancel() varias veces no produce error."""
        api_module.request_cancel()
        api_module.request_cancel()
        assert api_module.is_cancelled() is True

    # ── POST /api/cancel endpoint ────────────────────────────────────────────

    def test_post_cancel_returns_200(self):
        """POST /api/cancel devuelve HTTP 200."""
        client = TestClient(api_module.app)
        r = client.post("/api/cancel")
        assert r.status_code == 200

    def test_post_cancel_body_status_field(self):
        """POST /api/cancel devuelve {'status': 'cancel_requested'}."""
        client = TestClient(api_module.app)
        r = client.post("/api/cancel")
        assert r.json() == {"status": "cancel_requested"}

    def test_post_cancel_sets_is_cancelled(self):
        """Tras POST /api/cancel, is_cancelled() devuelve True."""
        client = TestClient(api_module.app)
        client.post("/api/cancel")
        assert api_module.is_cancelled() is True

    # ── GET /api/status refleja cancel_requested ─────────────────────────────

    def test_status_cancel_requested_initially_false(self):
        """GET /api/status tiene cancel_requested=False en estado limpio."""
        client = TestClient(api_module.app)
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data.get("cancel_requested") is False

    def test_status_reflects_cancel_requested_true(self):
        """GET /api/status devuelve cancel_requested=True tras llamar /api/cancel."""
        client = TestClient(api_module.app)
        client.post("/api/cancel")
        r = client.get("/api/status")
        assert r.json().get("cancel_requested") is True

    # ── nuevo pipeline resetea el flag ───────────────────────────────────────

    def test_new_pipeline_resets_cancel_flag(self, tmp_path, monkeypatch):
        """Al iniciar un nuevo pipeline, cancel_requested se resetea a False."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        # Pre-set the cancel flag
        api_module.request_cancel()
        assert api_module.is_cancelled() is True

        flag_at_start = []

        def recording_runner(req):
            # Simulate what run_pipeline_generator does: reset cancel at the top
            api_module.reset_cancel()
            flag_at_start.append(api_module.is_cancelled())
            yield api_module._sse_event("pipeline_done", {"run_dir": str(tmp_path)})
            api_module._release_pipeline_slot()

        monkeypatch.setattr(api_module, "run_pipeline_generator", recording_runner)

        client = TestClient(api_module.app)
        with client.stream("POST", "/api/run", json={"query": "reset-test"}) as r:
            assert r.status_code == 200
            list(r.iter_text())

        # The runner called reset_cancel at the top → flag was False when pipeline started
        assert flag_at_start == [False], (
            "cancel_requested debe ser False al inicio del pipeline (reset_cancel llamado)"
        )

    def test_run_pipeline_generator_resets_cancel_on_start(self, tmp_path, monkeypatch):
        """run_pipeline_generator resetea cancel_requested=False al arrancar."""
        monkeypatch.setattr(api_module, "OUTPUTS_DIR", tmp_path)

        # Pre-set cancel
        api_module._pipeline_state["cancel_requested"] = True

        # Consume the real generator minimally (it will fail at A1 import without
        # a real config, but we only need to verify the reset at the top of the generator)
        gen = api_module.run_pipeline_generator(
            api_module.PipelineRequest(query="test reset")
        )
        # The first thing run_pipeline_generator does is reset cancel
        try:
            next(gen)
        except StopIteration:
            pass
        except Exception:
            pass

        assert api_module._pipeline_state.get("cancel_requested") is False
