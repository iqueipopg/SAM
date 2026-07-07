# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ARQ6 — CORS_ORIGINS configurable via env var in api.py.

Covers:
- Sin CORS_ORIGINS → defaults a localhost:3000 y localhost:5173.
- CORS_ORIGINS con 1 elemento → lista de 1.
- CORS_ORIGINS con 3 elementos → lista de 3.
- Strings vacíos en el split → filtrados.
- Espacios alrededor de elementos → trimmed.

CPU-only, no GPU, no network.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api as api_module


class TestCorsOrigins:
    """ARQ6: _get_cors_origins resolves allowed origins correctly."""

    def test_no_env_var_returns_default_origins(self):
        """Sin CORS_ORIGINS el default incluye localhost:3000 y localhost:5173."""
        origins = api_module._get_cors_origins(env={})
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins

    def test_no_env_var_returns_exactly_two_defaults(self):
        """Sin CORS_ORIGINS la lista por defecto tiene exactamente 2 elementos."""
        origins = api_module._get_cors_origins(env={})
        assert len(origins) == 2

    def test_single_origin(self):
        """CORS_ORIGINS con un solo valor produce lista de 1 elemento."""
        origins = api_module._get_cors_origins(env={"CORS_ORIGINS": "http://foo.com"})
        assert origins == ["http://foo.com"]
        assert len(origins) == 1

    def test_two_origins(self):
        """CORS_ORIGINS con dos valores separados por coma produce lista de 2."""
        origins = api_module._get_cors_origins(
            env={"CORS_ORIGINS": "https://a.com,https://b.com"}
        )
        assert origins == ["https://a.com", "https://b.com"]

    def test_three_origins(self):
        """CORS_ORIGINS con tres valores produce lista de 3 elementos."""
        origins = api_module._get_cors_origins(
            env={"CORS_ORIGINS": "http://a.com,http://b.com,http://c.com"}
        )
        assert len(origins) == 3
        assert "http://a.com" in origins
        assert "http://b.com" in origins
        assert "http://c.com" in origins

    def test_empty_segments_are_filtered(self):
        """Segmentos vacíos tras el split son ignorados."""
        origins = api_module._get_cors_origins(
            env={"CORS_ORIGINS": "http://a.com,,http://b.com"}
        )
        assert len(origins) == 2
        assert "" not in origins
        assert "http://a.com" in origins
        assert "http://b.com" in origins

    def test_whitespace_trimmed_from_origins(self):
        """Los espacios alrededor de cada origen son eliminados."""
        origins = api_module._get_cors_origins(
            env={"CORS_ORIGINS": " http://a.com , http://b.com "}
        )
        assert "http://a.com" in origins
        assert "http://b.com" in origins
        assert " http://a.com " not in origins

    def test_only_comma_returns_empty_list(self):
        """Una cadena de sólo comas produce lista vacía."""
        origins = api_module._get_cors_origins(env={"CORS_ORIGINS": ",,,,"})
        assert origins == []

    def test_empty_string_returns_empty_list(self):
        """Una cadena vacía produce lista vacía."""
        origins = api_module._get_cors_origins(env={"CORS_ORIGINS": ""})
        assert origins == []

    def test_production_style_origin(self):
        """Un origen de producción HTTPS se devuelve sin modificaciones."""
        origins = api_module._get_cors_origins(
            env={"CORS_ORIGINS": "https://sam.example.com"}
        )
        assert origins == ["https://sam.example.com"]

    def test_uses_os_environ_when_no_env_dict_passed(self, monkeypatch):
        """Sin argumento env=, la función lee os.environ."""
        monkeypatch.setenv("CORS_ORIGINS", "https://env-origin.com")
        origins = api_module._get_cors_origins()
        assert "https://env-origin.com" in origins

    def test_env_dict_takes_precedence_over_os_environ(self, monkeypatch):
        """El argumento env= tiene prioridad sobre os.environ."""
        monkeypatch.setenv("CORS_ORIGINS", "https://from-os-environ.com")
        origins = api_module._get_cors_origins(env={"CORS_ORIGINS": "https://from-dict.com"})
        assert origins == ["https://from-dict.com"]
        assert "https://from-os-environ.com" not in origins
