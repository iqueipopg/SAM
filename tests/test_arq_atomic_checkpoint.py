# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ARQ2 — atomic checkpoint write in api.py.

Covers:
- Normal write: file created, no .tmp residual.
- Write with exception during json.dump: .tmp cleaned, target not corrupted.
- Write in non-existent directory: directory is created automatically.
- Overwrite of existing checkpoint: new content replaces atomically.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api as api_module


class TestAtomicCheckpointWrite:
    """ARQ2: atomic checkpoint write via tmp + os.replace."""

    def test_normal_write_creates_file(self, tmp_path):
        """save_checkpoint writes the expected JSON file."""
        run_dir = str(tmp_path / "run_normal")
        api_module.save_checkpoint(run_dir, "a1_protocol", {"key": "value"})
        target = tmp_path / "run_normal" / "checkpoint_a1_protocol.json"
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["key"] == "value"

    def test_normal_write_no_tmp_residual(self, tmp_path):
        """After a successful write no .tmp files remain in the directory."""
        run_dir = str(tmp_path / "run_tmp")
        os.makedirs(run_dir)
        api_module.save_checkpoint(run_dir, "a2_search", {"papers": []})
        leftovers = list((tmp_path / "run_tmp").glob("*.tmp"))
        assert leftovers == [], "No debe quedar ningún fichero .tmp tras escritura exitosa"

    def test_failed_write_cleans_tmp_and_preserves_original(self, tmp_path):
        """If json.dump raises, the .tmp is cleaned up and original is untouched."""
        run_dir = tmp_path / "run_atomic"
        run_dir.mkdir()
        # Write original checkpoint
        api_module.save_checkpoint(str(run_dir), "a1_protocol", {"version": 1})
        target = run_dir / "checkpoint_a1_protocol.json"
        assert json.loads(target.read_text())["version"] == 1

        # Simulate failure during json.dump
        with mock.patch("json.dump", side_effect=IOError("disco lleno")):
            with pytest.raises(IOError):
                api_module.save_checkpoint(str(run_dir), "a1_protocol", {"version": 2})

        # Original must remain intact
        assert json.loads(target.read_text())["version"] == 1
        # .tmp must be cleaned up
        assert not list(run_dir.glob("*.tmp")), ".tmp no debe quedar tras fallo"

    def test_failed_rename_preserves_original(self, tmp_path):
        """If os.replace raises, the previous checkpoint is not corrupted."""
        run_dir = tmp_path / "run_rename_fail"
        run_dir.mkdir()
        api_module.save_checkpoint(str(run_dir), "a3_screening", {"original": True})
        target = run_dir / "checkpoint_a3_screening.json"

        with mock.patch.object(api_module.os, "replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                api_module.save_checkpoint(str(run_dir), "a3_screening", {"original": False})

        # Original must still be there
        assert json.loads(target.read_text())["original"] is True

    def test_write_creates_directory_if_missing(self, tmp_path):
        """save_checkpoint creates run_dir if it does not exist yet."""
        run_dir = str(tmp_path / "new_dir" / "run_X")
        api_module.save_checkpoint(run_dir, "a1_protocol", {"created": True})
        target_path = (tmp_path / "new_dir" / "run_X" / "checkpoint_a1_protocol.json")
        assert target_path.exists()

    def test_overwrite_replaces_content_atomically(self, tmp_path):
        """Calling save_checkpoint twice updates the content atomically."""
        run_dir = str(tmp_path / "run_overwrite")
        api_module.save_checkpoint(run_dir, "a5_extraction", {"iteration": 1})
        api_module.save_checkpoint(run_dir, "a5_extraction", {"iteration": 2})
        target = tmp_path / "run_overwrite" / "checkpoint_a5_extraction.json"
        data = json.loads(target.read_text())
        assert data["iteration"] == 2, "El segundo checkpoint debe sobrescribir el primero"

    def test_checkpoint_content_preserved_unicode(self, tmp_path):
        """Unicode / Spanish characters are preserved (ensure_ascii=False)."""
        run_dir = str(tmp_path / "run_unicode")
        payload = {
            "titulo": "Revisión sistemática automatizada",
            "autor": "García-López",
            "nota": "Eficacia del tratamiento cognitivo-conductual",
        }
        api_module.save_checkpoint(run_dir, "a8_manuscript", payload)
        target = tmp_path / "run_unicode" / "checkpoint_a8_manuscript.json"
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["titulo"] == "Revisión sistemática automatizada"
        assert data["autor"] == "García-López"

    def test_load_checkpoint_returns_none_on_corrupt_file(self, tmp_path):
        """load_checkpoint returns None when the JSON is malformed."""
        run_dir = tmp_path / "run_corrupt"
        run_dir.mkdir()
        corrupt = run_dir / "checkpoint_a1_protocol.json"
        corrupt.write_text("{ this is not valid JSON }")
        result = api_module.load_checkpoint(str(run_dir), "a1_protocol")
        assert result is None

    def test_load_checkpoint_returns_none_on_missing_file(self, tmp_path):
        """load_checkpoint returns None if the checkpoint file does not exist."""
        run_dir = str(tmp_path / "run_empty")
        result = api_module.load_checkpoint(run_dir, "a1_protocol")
        assert result is None
