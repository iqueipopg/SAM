# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/post_run_backup.sh.

Invokes the bash script via subprocess. CPU-only, no GPU/LLM/internet/R required.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "post_run_backup.sh"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _unique_run_name(prefix: str = "run_test_T16_2") -> str:
    """Build a unique run dir name that won't collide with real runs."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _make_synthetic_run(run_name: str, n_files: int = 3) -> Path:
    """Create outputs/<run_name>/ with `n_files` synthetic files. Returns the dir."""
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    for i in range(n_files):
        (run_dir / f"file_{i}.json").write_text(f'{{"index": {i}}}\n')
    # Add a nested file to ensure recursion is correct
    nested = run_dir / "plots"
    nested.mkdir()
    (nested / "forest.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    return run_dir


def _run_script(args: list[str], env_extra: dict[str, str] | None = None,
                check: bool = False) -> subprocess.CompletedProcess:
    """Invoke the script with given args + env overrides."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env=env,
        check=check,
    )


@pytest.fixture(autouse=True)
def _ensure_script_exists():
    """Sanity check: the script under test must exist."""
    assert SCRIPT_PATH.exists(), f"Script not found: {SCRIPT_PATH}"


def test_help_exits_zero():
    """--help prints usage and exits 0."""
    result = _run_script(["--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout or "usage" in result.stdout.lower()


def test_happy_path():
    """Script copies a synthetic run, exits 0, file count matches."""
    run_name = _unique_run_name()
    src_run: Path | None = None
    backup_dir = Path(tempfile.mkdtemp(prefix="t16_2_backup_"))
    try:
        src_run = _make_synthetic_run(run_name, n_files=3)
        result = _run_script([run_name], env_extra={"BACKUP_DIR": str(backup_dir)})

        assert result.returncode == 0, (
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "[post_run_backup] OK" in result.stdout
        assert f"src={src_run}" in result.stdout
        assert f"dst={backup_dir}/{run_name}" in result.stdout

        dst_run = backup_dir / run_name
        assert dst_run.is_dir()

        src_files = sorted(p.relative_to(src_run) for p in src_run.rglob("*") if p.is_file())
        dst_files = sorted(p.relative_to(dst_run) for p in dst_run.rglob("*") if p.is_file())
        assert src_files == dst_files
        assert len(src_files) == 4  # 3 json + 1 png
    finally:
        if src_run is not None and src_run.exists():
            shutil.rmtree(src_run, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def test_backup_dir_not_writable():
    """If BACKUP_DIR exists but is not writable, script exits 3."""
    run_name = _unique_run_name()
    src_run: Path | None = None
    backup_dir = Path(tempfile.mkdtemp(prefix="t16_2_ro_backup_"))
    original_mode = backup_dir.stat().st_mode
    try:
        src_run = _make_synthetic_run(run_name, n_files=2)
        # Make read-only (no write/execute for owner)
        os.chmod(backup_dir, 0o500)
        # Skip the test if we are root — root ignores DAC permissions and the
        # write check would not trip. (Should not happen in CI/dev, but be safe.)
        if os.geteuid() == 0:
            pytest.skip("running as root: DAC write check is bypassed")

        result = _run_script([run_name], env_extra={"BACKUP_DIR": str(backup_dir)})
        assert result.returncode == 3, (
            f"expected exit 3, got {result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "not writable" in result.stderr.lower() or "FAIL" in result.stderr
    finally:
        # Restore mode so cleanup can run
        try:
            os.chmod(backup_dir, original_mode)
        except OSError:
            pass
        if src_run is not None and src_run.exists():
            shutil.rmtree(src_run, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def test_run_not_found():
    """A run name that does not exist exits 2."""
    bogus = _unique_run_name("run_test_T16_2_DOES_NOT_EXIST")
    backup_dir = Path(tempfile.mkdtemp(prefix="t16_2_missing_backup_"))
    try:
        result = _run_script([bogus], env_extra={"BACKUP_DIR": str(backup_dir)})
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert "not found" in result.stderr.lower() or "FAIL" in result.stderr
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def test_autodetect_picks_most_recent():
    """With no argument, the script picks the most recent run_* under outputs/."""
    old_name = _unique_run_name("run_test_T16_2_OLD")
    new_name = _unique_run_name("run_test_T16_2_NEW")
    old_run: Path | None = None
    new_run: Path | None = None
    backup_dir = Path(tempfile.mkdtemp(prefix="t16_2_auto_backup_"))
    try:
        old_run = _make_synthetic_run(old_name, n_files=2)
        # Force old_run to have an older mtime
        old_ts = time.time() - 7 * 24 * 3600  # 7 days ago
        for p in [old_run, *old_run.rglob("*")]:
            os.utime(p, (old_ts, old_ts))

        new_run = _make_synthetic_run(new_name, n_files=3)
        # Touch new_run to "now" to be sure it is more recent
        now_ts = time.time()
        for p in [new_run, *new_run.rglob("*")]:
            os.utime(p, (now_ts, now_ts))

        result = _run_script([], env_extra={"BACKUP_DIR": str(backup_dir)})
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        # The newest synthetic run must have been chosen
        assert f"dst={backup_dir}/{new_name}" in result.stdout
        assert (backup_dir / new_name).is_dir()
        # The older synthetic run must NOT have been copied
        assert not (backup_dir / old_name).exists()

        # NOTE: the autodetect picks the newest run_* in outputs/, which may be
        # one of the real existing runs if their mtime is more recent than our
        # synthetic NEW run. We override mtimes above to make this deterministic
        # within the synthetic set, but if the real outputs/ has runs newer than
        # `now_ts` (impossible in normal conditions), the test would still hold:
        # we only check that our NEW run was copied if it was selected. The
        # assertion above directly verifies the autodetect chose `new_name`.
    finally:
        if old_run is not None and old_run.exists():
            shutil.rmtree(old_run, ignore_errors=True)
        if new_run is not None and new_run.exists():
            shutil.rmtree(new_run, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
