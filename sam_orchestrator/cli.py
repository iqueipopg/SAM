"""argparse CLI for sam_orchestrator (T17.C.0).

Subcommands:
  init   --query "..." [--subset path:split] [--backend claude|qwen]
         -> stdout: single line, run_id
         exit 0 on success, 1 on validation failure (bad subset path, empty query)

  next   <run_id>
         -> stdout: JSON object (single line preferred)
         {"agent": "...", "paper_id": "..."|null, "input_paths": [...], "output_path": "..."}
         or {"done": true}
         exit 0

  commit <run_id> <agent_id> <output_path>
         -> stdout: {"ok": true, ...}
         exit 0 on success, 2 on schema validation failure, 1 on other errors

  audit  <run_id> [--only <agent_id>]
         -> stdout: {"ok": bool, "agents": {...}}
         exit 0 if ok, 1 if any discrepancy
         --only restricts the audit to a single agent_id.

Exit codes (documented):
  0  -> success
  1  -> generic failure (RunNotFound, ManifestCorrupt, audit discrepancies, ...)
  2  -> schema validation failure on commit
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .audit import audit_run
from .commit import commit_output
from .errors import (
    ManifestCorruptError,
    OrchestratorError,
    RunNotFoundError,
    SchemaValidationError,
    SubsetMappingMissingError,
    UnknownAgentError,
)
from .state import init_run, next_work


def _print_json(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _print_err(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        run_id = init_run(
            query=args.query,
            subset_spec=args.subset,
            backend_default=args.backend,
            slug_hint=args.slug_hint,
        )
    except SubsetMappingMissingError as e:
        _print_err(f"subset error: {e}")
        return 1
    except (ValueError, OrchestratorError) as e:
        _print_err(f"init failed: {e}")
        return 1
    sys.stdout.write(run_id + "\n")
    sys.stdout.flush()
    return 0


def _cmd_next(args: argparse.Namespace) -> int:
    try:
        work = next_work(args.run_id)
    except RunNotFoundError as e:
        _print_err(str(e))
        return 1
    except (ManifestCorruptError, OrchestratorError) as e:
        _print_err(f"next failed: {e}")
        return 1
    _print_json(work)
    return 0


def _cmd_commit(args: argparse.Namespace) -> int:
    try:
        result = commit_output(args.run_id, args.agent_id, args.output_path)
    except SchemaValidationError as e:
        _print_err(f"schema validation failed: {e}")
        return 2
    except UnknownAgentError as e:
        _print_err(f"unknown agent: {e}")
        return 1
    except (RunNotFoundError, ManifestCorruptError, OrchestratorError) as e:
        _print_err(f"commit failed: {e}")
        return 1
    _print_json(result)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        report = audit_run(args.run_id, only=args.only)
    except ValueError as e:
        _print_err(f"audit failed: {e}")
        return 1
    except (RunNotFoundError, ManifestCorruptError, OrchestratorError) as e:
        _print_err(f"audit failed: {e}")
        return 1
    _print_json(report)
    return 0 if report.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sam_orchestrator",
        description="SAM deterministic orchestrator (init / next / commit / audit).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Create a new run and emit run_id.")
    init_p.add_argument("--query", required=True, help="User clinical question.")
    init_p.add_argument(
        "--subset",
        default=None,
        help="Optional subset spec 'path/to/partition.json:split'.",
    )
    init_p.add_argument(
        "--backend",
        choices=["claude", "qwen"],
        default="claude",
        help="Default backend for the run (default: claude).",
    )
    init_p.add_argument(
        "--slug-hint",
        default=None,
        help="Optional slug to append after the timestamp in run_id.",
    )
    init_p.set_defaults(func=_cmd_init)

    next_p = sub.add_parser("next", help="Print next work item (or done).")
    next_p.add_argument("run_id")
    next_p.set_defaults(func=_cmd_next)

    commit_p = sub.add_parser(
        "commit", help="Validate + record an agent output."
    )
    commit_p.add_argument("run_id")
    commit_p.add_argument("agent_id")
    commit_p.add_argument("output_path")
    commit_p.set_defaults(func=_cmd_commit)

    audit_p = sub.add_parser("audit", help="Audit a run; exit 1 if discrepancies.")
    audit_p.add_argument("run_id")
    audit_p.add_argument(
        "--only",
        default=None,
        help="Audit only this agent_id (must be a valid agent).",
    )
    audit_p.set_defaults(func=_cmd_audit)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
