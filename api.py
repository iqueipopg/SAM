"""
SAM — FastAPI backend for the React web UI.
Runs the A1→A9 pipeline (9 agents) with SSE progress streaming.

Usage:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents.agent1_protocol import Agent1Protocol
from agents.agent2_search import Agent2Search
from agents.agent3_screening import Agent3Screening
from agents.agent4_fulltext import Agent4FullText
from agents.agent5_extraction import Agent5Extraction
from agents.agent6_quality import Agent6Quality
from agents.agent7_synthesis import Agent7Synthesis
from agents.agent8_manuscript import Agent8Manuscript
from agents.agent9_explainer import Agent9Explainer
from agents.prisma_flow import generate_prisma_flow
from agents.rob_plots import generate_rob_traffic_light, generate_rob_summary
from agents.exporters import export_csv, export_excel, export_revman, export_summary_table
from agents.logger import setup_logger, get_logger
from agents.review_report import ReviewReportGenerator

# ── App ──────────────────────────────────────────────────────────────────────

def _read_version() -> str:
    """Read the SAM version from the VERSION file at project root.

    Falls back to '0.0.0' if the file is missing or unreadable. The path is
    resolved relative to this module file so the function works regardless of
    the caller's working directory.
    """
    version_path = Path(__file__).parent / "VERSION"
    try:
        return version_path.read_text(encoding="utf-8").strip() or "0.0.0"
    except (OSError, FileNotFoundError):
        return "0.0.0"


app = FastAPI(title="SAM API", version=_read_version())


@app.on_event("startup")
async def _startup_validate_env():
    """ARQ3: check required env vars when the server boots.

    Skips validation when running under pytest so individual tests can
    control the environment without the API refusing to load.
    """
    if os.environ.get("SAM_SKIP_ENV_VALIDATION") == "1" or "pytest" in sys.modules:
        return
    validate_env()


def _get_cors_origins(env: Optional[dict] = None) -> list:
    """Resolve the list of allowed CORS origins.

    ARQ6: reads the ``CORS_ORIGINS`` environment variable (comma-separated).
    In production set e.g. ``CORS_ORIGINS=https://sam.example.com``. For local
    development the default allows Vite (5173) and CRA/React (3000).

    Args:
        env: optional env dict (defaults to ``os.environ`` — exposed for tests).

    Returns:
        List of origin strings with whitespace stripped.
    """
    source = env if env is not None else os.environ
    raw = source.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Global state for the running pipeline
_pipeline_state = {
    "running": False,
    "stop_requested": False,
    "kill_requested": False,
    # ARQ8: cooperative cancellation flag checked by long-running agents
    # between iterations so the pipeline can be stopped without killing
    # the Python process.
    "cancel_requested": False,
    "current_run_dir": None,
    # ARQ7: tracks the run_id of the single pipeline currently active
    "active_run_id": None,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def free_vram(agent=None):
    if agent is not None:
        if hasattr(agent, "model") and agent.model is not None:
            del agent.model
            agent.model = None
        if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
            del agent.tokenizer
            agent.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_r_objects(obj):
    if isinstance(obj, dict):
        return {k: _clean_r_objects(v) for k, v in obj.items() if k != "_fit"}
    elif isinstance(obj, list):
        return [_clean_r_objects(item) for item in obj]
    return obj


def save_checkpoint(run_dir, agent_name, data):
    """Persist ``data`` as checkpoint_<agent_name>.json inside ``run_dir``.

    ARQ2: the write is atomic — data is first serialized to a ``.tmp`` file
    in the same directory and then renamed over the target via
    :func:`os.replace` (atomic on POSIX). If any step fails before the rename,
    the previous checkpoint on disk is left untouched, so a partial write
    can never corrupt a prior successful checkpoint.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"checkpoint_{agent_name}.json")
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                # fsync may be unavailable on some filesystems (e.g. tmpfs in
                # CI); the atomic rename below still provides safety.
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Clean up a leftover .tmp so it doesn't accumulate across retries.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(run_dir, agent_name):
    path = os.path.join(run_dir, f"checkpoint_{agent_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def load_config():
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


# ── ARQ3: startup environment validation ────────────────────────────────────

# Required env vars — missing any of these is fatal: the API refuses to start.
_REQUIRED_ENV_VARS = ("ENTREZ_EMAIL", "UNPAYWALL_EMAIL")

# Optional env vars — only logged (warning) if absent. The pipeline can still
# run without them, but some features (Groq backend, extra full-text sources)
# will be disabled.
_OPTIONAL_ENV_VARS = ("GROQ_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "CORE_API_KEY")


def validate_env():
    """Verify required environment variables are present at API startup.

    Raises:
        RuntimeError: in Spanish if any required variable is missing. Message
        includes the list of missing keys so the operator can fix ``.env``.

    Optional variables only emit a logger warning — their absence is not fatal.
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            "Faltan variables de entorno requeridas: "
            + ", ".join(missing)
            + ". Defínelas en .env antes de arrancar la API."
        )
    absent_optional = [v for v in _OPTIONAL_ENV_VARS if not os.environ.get(v)]
    if absent_optional:
        try:
            get_logger("api").warning(
                "Variables opcionales no definidas (la API arranca igualmente): %s",
                ", ".join(absent_optional),
            )
        except Exception:
            pass


# ── Models ───────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    query: str
    max_results: int = 30
    dedup_threshold: float = 0.90
    a3_ensemble: int = 3
    a3_vote_threshold: float = 0.60
    a4_ensemble: int = 1
    a4_vote_threshold: float = 0.60
    # ARQ1: resume from a previous run's checkpoints. If provided, the pipeline
    # reads outputs/{resume_run_id}/checkpoint_*.json and skips any agent whose
    # checkpoint is already present.
    resume_run_id: Optional[str] = None


class StopRequest(BaseModel):
    pass


# ── ARQ1: checkpoint discovery/restoration helpers ──────────────────────────

# Ordered list of per-agent checkpoint names used by the runner. Order must
# mirror the A1->A8 pipeline sequence.
_CHECKPOINT_AGENT_ORDER = [
    "a1_protocol",
    "a2_search",
    "a3_screening",
    "a4_fulltext",
    "a5_extraction",
    "a6_quality",
    "a7_synthesis",
    "a8_manuscript",
]


def _agent_label_for_checkpoint(cp_name: str) -> str:
    """Return the short agent label (e.g. 'A5') for a checkpoint name like 'a5_extraction'."""
    try:
        return cp_name.split("_")[0].upper()
    except Exception:
        return cp_name


def discover_checkpoints(run_dir: str) -> dict:
    """Load all available checkpoint_*.json files from run_dir.

    Returns a dict keyed by agent_name (e.g. 'a1_protocol') with the JSON
    payload as value. Missing/corrupt checkpoints are silently skipped.
    """
    found = {}
    for cp in _CHECKPOINT_AGENT_ORDER:
        data = load_checkpoint(run_dir, cp)
        if data is not None:
            found[cp] = data
    return found


# ── SSE Pipeline Runner ─────────────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _release_pipeline_slot():
    """ARQ7 helper: clear the 'running' / 'active_run_id' flags together.

    Called whenever the pipeline generator exits (success, error, cancel)
    so /api/run can accept the next request.
    """
    _pipeline_state["running"] = False
    _pipeline_state["active_run_id"] = None


def run_pipeline_generator(req: PipelineRequest):
    """Generator that yields SSE events as the pipeline progresses.

    If ``req.resume_run_id`` is set the runner reuses that existing run
    directory, loads any checkpoint_*.json files it finds and skips the
    corresponding agents, emitting a 'resumed' SSE event before continuing.

    ARQ7: the generator maintains ``_pipeline_state["active_run_id"]`` so
    that the /api/run endpoint can reject concurrent requests with 429.
    The flag is always cleared when the generator returns (success, error
    or early cancel) so a failed run never locks out the API.
    """
    _pipeline_state["running"] = True
    _pipeline_state["stop_requested"] = False
    _pipeline_state["kill_requested"] = False
    _pipeline_state["cancel_requested"] = False
    _pipeline_state["active_run_id"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ARQ1: resume support -- reuse existing run_dir if resume_run_id provided
    resume_state = {}
    if req.resume_run_id:
        run_dir = str(OUTPUTS_DIR / req.resume_run_id)
        if not os.path.isdir(run_dir):
            _pipeline_state["running"] = False
            yield _sse_event("pipeline_error", {
                "error": f"resume_run_id '{req.resume_run_id}' not found",
            })
            return
        resume_state = discover_checkpoints(run_dir)
        ts = req.resume_run_id.replace("run_", "")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str(OUTPUTS_DIR / f"run_{ts}")

    os.makedirs(run_dir, exist_ok=True)
    _pipeline_state["current_run_dir"] = run_dir
    # ARQ7: record the active run_id so /api/run can reject collisions
    _pipeline_state["active_run_id"] = os.path.basename(run_dir)

    setup_logger("SAM", log_dir=run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    yield _sse_event("start", {"run_dir": run_dir, "timestamp": ts})

    protocol = None
    pubmed_query = None
    papers = None
    a2_stats = None
    a3_results = None
    a4_result = None
    a5_result = None
    a6_result = None
    a7_result = None
    a8_result = None
    included_papers = []

    # Restore state from checkpoints (if resuming)
    if resume_state:
        if "a1_protocol" in resume_state:
            cp1 = resume_state["a1_protocol"]
            protocol = cp1.get("protocol")
            pubmed_query = cp1.get("pubmed_query")
        if "a2_search" in resume_state:
            cp2 = resume_state["a2_search"]
            papers = cp2.get("papers")
            a2_stats = cp2.get("stats")
        if "a3_screening" in resume_state:
            a3_results = resume_state["a3_screening"]
        if "a4_fulltext" in resume_state:
            a4_result = resume_state["a4_fulltext"]
            included_papers = a4_result.get("include", []) or []
        if "a5_extraction" in resume_state:
            a5_result = resume_state["a5_extraction"]
        if "a6_quality" in resume_state:
            a6_result = resume_state["a6_quality"]
        if "a7_synthesis" in resume_state:
            a7_result = resume_state["a7_synthesis"]
        if "a8_manuscript" in resume_state:
            a8_result = resume_state["a8_manuscript"]

        # Identify the first agent that still needs to run
        completed = [cp for cp in _CHECKPOINT_AGENT_ORDER if cp in resume_state]
        next_idx = len(completed)
        from_agent = (
            _agent_label_for_checkpoint(_CHECKPOINT_AGENT_ORDER[next_idx])
            if next_idx < len(_CHECKPOINT_AGENT_ORDER)
            else "done"
        )
        yield _sse_event("resumed", {
            "run_dir": run_dir,
            "from_agent": from_agent,
            "completed_agents": [_agent_label_for_checkpoint(c) for c in completed],
        })

    def check_stop():
        return (
            _pipeline_state["stop_requested"]
            or _pipeline_state["kill_requested"]
            or _pipeline_state.get("cancel_requested", False)
        )

    # ── A1: Protocol ─────────────────────────────────────────
    if "a1_protocol" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A1", "name": "Protocol", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A1", "name": "Protocol"})
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            a1 = Agent1Protocol()
            protocol = a1.generate_protocol(req.query)
            pubmed_query = a1.generate_pubmed_query(req.query, protocol)
            # PROSPERO draft (non-critical)
            try:
                cfg = load_config()
                prospero = a1.generate_prospero_draft(protocol, config=cfg)
                protocol["prospero_draft"] = prospero
                with open(os.path.join(run_dir, "prospero_draft.txt"), "w") as pf:
                    pf.write(prospero)
            except Exception:
                pass
            free_vram(a1)
            save_checkpoint(run_dir, "a1_protocol", {
                "query": req.query, "protocol": protocol, "pubmed_query": pubmed_query
            })
            yield _sse_event("agent_done", {
                "agent": "A1", "name": "Protocol",
                "summary": {
                    "framework": protocol.get("framework", "?"),
                    "pubmed_query": pubmed_query,
                    "population": protocol.get("population", "?"),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A1", "error": str(e)})
            _pipeline_state["running"] = False
            yield _sse_event("pipeline_error", {"error": f"A1 failed: {e}"})
            return

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A1"})
        _pipeline_state["running"] = False
        return

    # ── A2: Search ───────────────────────────────────────────
    if "a2_search" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A2", "name": "Search", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A2", "name": "Search"})
        try:
            a2 = Agent2Search()
            a2_result = a2.search(protocol, pubmed_query, max_results=req.max_results)
            papers = a2_result["papers"]
            a2_stats = a2_result["stats"]
            save_checkpoint(run_dir, "a2_search", {"papers": papers, "stats": a2_stats})
            yield _sse_event("agent_done", {
                "agent": "A2", "name": "Search",
                "summary": {
                    "total_found": a2_stats.get("total_before_dedup", 0),
                    "after_dedup": a2_stats.get("total_after_dedup", len(papers)),
                    "per_db": {
                        "pubmed": a2_stats.get("pubmed", 0),
                        "europepmc": a2_stats.get("europepmc", 0),
                        "semantic_scholar": a2_stats.get("semantic_scholar", 0),
                    }
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A2", "error": str(e)})
            _pipeline_state["running"] = False
            yield _sse_event("pipeline_error", {"error": f"A2 failed: {e}"})
            return

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A2"})
        _pipeline_state["running"] = False
        return

    # ── A3: Screening ────────────────────────────────────────
    if "a3_screening" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A3", "name": "Screening", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A3", "name": "Screening"})
        try:
            a3 = Agent3Screening(
                ensemble_size=req.a3_ensemble,
                voting_threshold=req.a3_vote_threshold,
            )
            a3_results = a3.screen_batch(papers, protocol)
            free_vram(a3)
            save_checkpoint(run_dir, "a3_screening", a3_results)
            stats3 = a3_results.get("stats", {})
            yield _sse_event("agent_done", {
                "agent": "A3", "name": "Screening",
                "summary": {
                    "included": stats3.get("included", 0),
                    "excluded": stats3.get("excluded", 0),
                    "uncertain": stats3.get("uncertain", 0),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A3", "error": str(e)})
            free_vram(None)
            _pipeline_state["running"] = False
            yield _sse_event("pipeline_error", {"error": f"A3 failed: {e}"})
            return

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A3"})
        _pipeline_state["running"] = False
        return

    # ── A4: Full-text ────────────────────────────────────────
    if "a4_fulltext" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A4", "name": "Full-text", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A4", "name": "Full-text"})
        try:
            a3_included = a3_results.get("include", [])
            a3_uncertain = a3_results.get("uncertain", [])
            ft_papers = a3_included + a3_uncertain

            a4 = Agent4FullText(
                ensemble_size=req.a4_ensemble,
                voting_threshold=req.a4_vote_threshold,
            )
            a4_result = a4.run(ft_papers, papers, protocol)
            free_vram(a4)
            save_checkpoint(run_dir, "a4_fulltext", a4_result)
            included_papers = a4_result.get("include", []) or []
            stats4 = a4_result.get("stats", {})
            yield _sse_event("agent_done", {
                "agent": "A4", "name": "Full-text",
                "summary": {
                    "fulltext_included": stats4.get("fulltext_included", 0),
                    "fulltext_excluded": stats4.get("fulltext_excluded", 0),
                    "retrieval_rate": stats4.get("retrieval_rate", "?"),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A4", "error": str(e)})
            free_vram(None)
            _pipeline_state["running"] = False
            yield _sse_event("pipeline_error", {"error": f"A4 failed: {e}"})
            return

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A4"})
        _pipeline_state["running"] = False
        return

    # ── A5: Extraction ───────────────────────────────────────
    if "a5_extraction" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A5", "name": "Extraction", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A5", "name": "Extraction"})
        try:
            included_papers = a4_result.get("include", []) if a4_result else included_papers
            a5 = Agent5Extraction()
            a5_result = a5.extract_all(included_papers, protocol)
            free_vram(a5)
            save_checkpoint(run_dir, "a5_extraction", a5_result)
            stats5 = a5_result.get("stats", {})
            yield _sse_event("agent_done", {
                "agent": "A5", "name": "Extraction",
                "summary": {
                    "complete": stats5.get("complete_extraction", 0),
                    "partial": stats5.get("partial_extraction", 0),
                    "failed": stats5.get("failed_extraction", 0),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A5", "error": str(e)})
            free_vram(None)
            # Non-fatal: continue with A4 data
            a5_result = None
            yield _sse_event("agent_warning", {"agent": "A5", "warning": str(e)})

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A5"})
        _pipeline_state["running"] = False
        return

    # ── A6: Quality ──────────────────────────────────────────
    if "a6_quality" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A6", "name": "Quality", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A6", "name": "Quality"})
        try:
            quality_papers = (a5_result or {}).get("papers", included_papers)
            a6 = Agent6Quality()
            a6_result = a6.assess_all(quality_papers, protocol)
            free_vram(a6)
            save_checkpoint(run_dir, "a6_quality", a6_result)
            stats6 = a6_result.get("stats", {})

            # Generate RoB plots
            try:
                rob_papers = a6_result.get("papers", [])
                generate_rob_traffic_light(rob_papers, os.path.join(plots_dir, "rob_traffic_light.png"))
                generate_rob_summary(rob_papers, os.path.join(plots_dir, "rob_summary.png"))
            except Exception as e:
                get_logger("api").warning("RoB plot generation failed: %s", e, exc_info=True)

            yield _sse_event("agent_done", {
                "agent": "A6", "name": "Quality",
                "summary": {
                    "rob2": stats6.get("rob2_assessed", 0),
                    "robins_i": stats6.get("robins_i_assessed", 0),
                    "jbi": stats6.get("jbi_assessed", 0),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A6", "error": str(e)})
            free_vram(None)
            a6_result = None
            yield _sse_event("agent_warning", {"agent": "A6", "warning": str(e)})

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A6"})
        _pipeline_state["running"] = False
        return

    # ── A7: Synthesis ────────────────────────────────────────
    if "a7_synthesis" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A7", "name": "Synthesis", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A7", "name": "Synthesis"})
        try:
            a7 = Agent7Synthesis()
            a7_input = a6_result if a6_result else {"papers": included_papers}
            a7_result = a7.synthesize(a7_input, protocol, output_dir=run_dir)
            save_checkpoint(run_dir, "a7_synthesis", _clean_r_objects(a7_result) if a7_result else {})
            stats7 = (a7_result or {}).get("stats", {})
            yield _sse_event("agent_done", {
                "agent": "A7", "name": "Synthesis",
                "summary": {
                    "papers_analyzed": stats7.get("papers_with_usable_data", 0),
                    "pooled_effect": stats7.get("pooled_estimate", "?"),
                    "i_squared": stats7.get("i_squared", "?"),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A7", "error": str(e)})
            a7_result = None
            yield _sse_event("agent_warning", {"agent": "A7", "warning": str(e)})

    if check_stop():
        yield _sse_event("pipeline_stopped", {"last_agent": "A7"})
        _pipeline_state["running"] = False
        return

    # ── A8: Manuscript ───────────────────────────────────────
    if "a8_manuscript" in resume_state:
        yield _sse_event("agent_skipped", {"agent": "A8", "name": "Manuscript", "reason": "resumed"})
    else:
        yield _sse_event("agent_start", {"agent": "A8", "name": "Manuscript"})
        try:
            a8 = Agent8Manuscript()
            a8_result = a8.generate_manuscript(
                protocol=protocol,
                included_papers=included_papers,
                extraction=a5_result,
                quality=a6_result,
                synthesis=_clean_r_objects(a7_result) if a7_result else None,
                output_dir=run_dir,
            )
            free_vram(a8)
            save_checkpoint(run_dir, "a8_manuscript", a8_result)
            stats8 = a8_result.get("stats", {})
            yield _sse_event("agent_done", {
                "agent": "A8", "name": "Manuscript",
                "summary": {
                    "sections": stats8.get("sections_generated", 0),
                    "papers_cited": stats8.get("total_papers_cited", 0),
                    "pdf": bool(a8_result.get("pdf_path")),
                }
            })
        except Exception as e:
            yield _sse_event("agent_error", {"agent": "A8", "error": str(e)})
            free_vram(None)
            a8_result = None
            yield _sse_event("agent_warning", {"agent": "A8", "warning": str(e)})

    # ── Finalize: PRISMA, exports ────────────────────────────
    yield _sse_event("finalizing", {"message": "Building outputs..."})

    stats3 = (a3_results or {}).get("stats", {})
    n_extracted = 0
    if a5_result and a5_result.get("stats"):
        n_extracted = a5_result["stats"].get("complete_extraction", 0) + a5_result["stats"].get("partial_extraction", 0)
    n_quality = 0
    if a6_result and a6_result.get("stats"):
        n_quality = a6_result["stats"].get("rob2_assessed", 0) + a6_result["stats"].get("robins_i_assessed", 0) + a6_result["stats"].get("jbi_assessed", 0)

    _a2s = a2_stats or {}
    prisma = {
        "identified": _a2s.get("total_before_dedup", stats3.get("total", 0)),
        "identified_per_db": {
            "pubmed": _a2s.get("pubmed", 0),
            "europepmc": _a2s.get("europepmc", 0),
            "semantic_scholar": _a2s.get("semantic_scholar", 0),
        },
        "after_dedup": _a2s.get("total_after_dedup", stats3.get("total", 0)),
        "duplicates_removed": _a2s.get("duplicates_removed", 0),
        "screened": stats3.get("total", 0),
        "excluded_title_abstract": stats3.get("excluded", 0),
        "uncertain_title_abstract": stats3.get("uncertain", 0),
        "assessed_fulltext": stats3.get("included", 0) + stats3.get("uncertain", 0),
        "excluded_fulltext": (a4_result or {}).get("stats", {}).get("fulltext_excluded", 0),
        "included_final": (a4_result or {}).get("stats", {}).get("fulltext_included", 0),
        "extracted": n_extracted,
        "quality_assessed": n_quality,
    }

    # PRISMA flow diagram
    try:
        n_synthesis = (a7_result or {}).get("stats", {}).get("papers_with_usable_data", 0)
        diagram_data = {
            "identified": prisma["identified"],
            "identified_per_db": prisma["identified_per_db"],
            "duplicates_removed": prisma["duplicates_removed"],
            "screened": prisma["screened"],
            "excluded_screening": prisma["excluded_title_abstract"],
            "fulltext_assessed": prisma["assessed_fulltext"],
            "excluded_fulltext": prisma["excluded_fulltext"],
            "not_retrieved": 0,
            "included_review": prisma["included_final"],
            "included_synthesis": n_synthesis,
            "quality_assessed": n_quality,
        }
        pf_path = generate_prisma_flow(diagram_data, os.path.join(plots_dir, "prisma_flow.png"))
        if pf_path:
            prisma["prisma_flow_diagram"] = pf_path
    except Exception:
        pass

    a7_clean = _clean_r_objects(a7_result) if a7_result else None

    # Review report
    review_report_data = None
    try:
        interim = {
            "screening": a3_results,
            "fulltext_retrieval": a4_result,
            "data_extraction": a5_result,
            "quality_assessment": a6_result,
            "synthesis": a7_clean,
        }
        rr = ReviewReportGenerator()
        review_report_data = rr.generate(interim)
        with open(os.path.join(run_dir, "review_report.json"), "w") as f:
            json.dump(review_report_data, f, indent=2, ensure_ascii=False)
        rr_txt = rr.generate_text_report(review_report_data)
        with open(os.path.join(run_dir, "review_report.txt"), "w") as f:
            f.write(rr_txt)
    except Exception:
        pass

    # System info
    _sys = {"timestamp": datetime.now().isoformat()}
    try:
        _sys["python"] = sys.version.split()[0]
        _sys["torch"] = torch.__version__
        if torch.cuda.is_available():
            _sys["gpu"] = torch.cuda.get_device_name(0)
            _sys["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    except Exception:
        pass

    # Propagate which databases A2 actually used (for A9 explainability §6).
    # Falls back to deriving from per-DB counts if A2 didn't populate the list.
    _a2_sources = _a2s.get("search_sources") or [
        db for db in ("pubmed", "europepmc", "semantic_scholar", "google_scholar")
        if _a2s.get(db, 0) > 0
    ]

    output = {
        "query": req.query,
        "protocol": protocol,
        "pubmed_query": pubmed_query,
        "config_snapshot": load_config(),
        "system_info": _sys,
        "prisma": prisma,
        "search_stats": _a2s,
        "a2_sources": _a2_sources,
        "screening": a3_results,
        "fulltext_retrieval": a4_result,
        "data_extraction": a5_result,
        "quality_assessment": a6_result,
        "synthesis": a7_clean,
        "manuscript": a8_result,
        "review_report": review_report_data,
    }

    # ── A9: Explainability & Traceability ────────────────────
    yield _sse_event("agent_start", {"agent": "A9", "name": "Explainer"})
    a9_result = None
    try:
        a9 = Agent9Explainer()
        a9_result = a9.run(output, os.path.basename(run_dir), run_dir)
        output["explainability"] = a9_result
        yield _sse_event("agent_done", {
            "agent": "A9", "name": "Explainer",
            "summary": {
                "total_papers": a9_result["stats"]["total_papers"],
                "included": a9_result["stats"]["included"],
                "excluded": a9_result["stats"]["excluded"],
                "elapsed_s": a9_result["stats"]["elapsed_s"],
            }
        })
    except Exception as e:
        yield _sse_event("agent_error", {"agent": "A9", "error": str(e)})
        yield _sse_event("agent_warning", {"agent": "A9", "warning": str(e)})

    # Save pipeline JSON
    out_path = os.path.join(run_dir, "pipeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # Export formats
    try:
        export_csv(output, os.path.join(run_dir, "included_studies.csv"))
    except Exception:
        pass
    try:
        export_excel(output, os.path.join(run_dir, "meta_analysis.xlsx"))
    except Exception:
        pass
    try:
        export_revman(output, os.path.join(run_dir, "revman_export.xml"))
    except Exception:
        pass
    try:
        export_summary_table(output, os.path.join(run_dir, "summary_table.md"))
    except Exception:
        pass

    # Build download manifest
    downloads = {}
    for fname, label in [
        ("pipeline.json", "JSON"),
        ("included_studies.csv", "CSV"),
        ("meta_analysis.xlsx", "Excel"),
        ("revman_export.xml", "RevMan XML"),
        ("summary_table.md", "Summary"),
        ("review_report.txt", "Review Report"),
        ("prospero_draft.txt", "PROSPERO Draft"),
        ("papers_raw.xlsx", "Papers (raw)"),
        ("explainability.md", "Explainability Report"),
    ]:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            downloads[label] = fname

    # Manuscript PDF
    if a8_result and a8_result.get("pdf_path"):
        pdf_path = a8_result["pdf_path"]
        if os.path.exists(pdf_path):
            downloads["Manuscript PDF"] = os.path.relpath(pdf_path, run_dir)

    # Plots
    for plot_name in ["prisma_flow.png", "rob_traffic_light.png", "rob_summary.png"]:
        ppath = os.path.join(plots_dir, plot_name)
        if os.path.exists(ppath):
            downloads[plot_name.replace(".png", "").replace("_", " ").title()] = f"plots/{plot_name}"

    # Forest/funnel plots from A7
    for pname in ["forest_plot.png", "funnel_plot.png"]:
        ppath = os.path.join(plots_dir, pname)
        if os.path.exists(ppath):
            downloads[pname.replace(".png", "").replace("_", " ").title()] = f"plots/{pname}"

    _release_pipeline_slot()
    _pipeline_state["current_run_dir"] = run_dir

    yield _sse_event("pipeline_done", {
        "run_dir": run_dir,
        "prisma": prisma,
        "downloads": downloads,
    })


def _guarded_pipeline_generator(req: PipelineRequest):
    """ARQ7 wrapper: always releases the pipeline slot on exit.

    Wraps :func:`run_pipeline_generator` in a try/finally so that even
    if a streaming client disconnects or an exception bubbles up, the
    ``running`` / ``active_run_id`` flags are cleared and the next
    /api/run call is not blocked by stale state.
    """
    try:
        yield from run_pipeline_generator(req)
    finally:
        _release_pipeline_slot()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run(req: PipelineRequest):
    """Start the pipeline. Returns an SSE stream.

    If ``resume_run_id`` is provided, the pipeline reuses that run's
    directory and restores state from its checkpoint files (skipping
    already-completed agents). Returns 404 if the directory does not
    exist.

    ARQ7: at most one pipeline can run concurrently. If another one is
    already active the endpoint returns HTTP 429 with the ``active_run_id``
    of the in-flight pipeline so the client can subscribe or retry later.
    """
    # ARQ7: single-concurrent-pipeline rate limit
    if _pipeline_state["running"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={
                "error": "Pipeline already running",
                "active_run_id": _pipeline_state.get("active_run_id"),
            },
        )
    # ARQ1: validate resume_run_id before streaming so the client gets
    # a clean HTTP 404 rather than an SSE error event
    if req.resume_run_id:
        resume_dir = OUTPUTS_DIR / req.resume_run_id
        if not resume_dir.is_dir():
            raise HTTPException(
                404,
                f"resume_run_id '{req.resume_run_id}' not found in outputs/",
            )
    return StreamingResponse(
        _guarded_pipeline_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/stop")
async def stop():
    """Request graceful pipeline stop (waits for current agent to finish)."""
    _pipeline_state["stop_requested"] = True
    return {"status": "stop_requested"}


@app.post("/api/kill")
async def kill():
    """Force-kill the pipeline immediately. Frees VRAM and resets state."""
    _pipeline_state["stop_requested"] = True
    _pipeline_state["kill_requested"] = True
    # Force-free any GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _pipeline_state["running"] = False
    return {"status": "killed"}


@app.get("/api/status")
async def status():
    """Check pipeline status.

    ARQ7: the response now includes ``active_run_id`` so the UI can show
    which run is currently in flight when a 429 is returned by /api/run.
    """
    return {
        "running": _pipeline_state["running"],
        "run_dir": _pipeline_state["current_run_dir"],
        "active_run_id": _pipeline_state.get("active_run_id"),
        "cancel_requested": _pipeline_state.get("cancel_requested", False),
    }


@app.get("/api/download/{run_id}/{filepath:path}")
async def download(run_id: str, filepath: str):
    """Download a file from a run directory."""
    run_dir = OUTPUTS_DIR / run_id
    full_path = run_dir / filepath
    if not full_path.exists() or not str(full_path.resolve()).startswith(str(OUTPUTS_DIR.resolve())):
        raise HTTPException(404, "File not found")
    return FileResponse(
        full_path,
        filename=full_path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/runs")
async def list_runs():
    """List completed pipeline runs."""
    runs = []
    if not OUTPUTS_DIR.exists():
        return runs
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        pipeline_json = d / "pipeline.json"
        if pipeline_json.exists():
            try:
                with open(pipeline_json) as f:
                    data = json.load(f)
                runs.append({
                    "id": d.name,
                    "query": data.get("query", "?"),
                    "timestamp": d.name.replace("run_", ""),
                    "prisma": data.get("prisma", {}),
                })
            except Exception:
                pass
    return runs


@app.get("/api/health")
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return {
        "status": "ok",
        "gpu": gpu_name,
        "gpu_available": gpu_available,
    }
