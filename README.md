<p align="center">
  <img src="assets/logo_horizontal_light.svg" alt="SAM — Systematic Agent for Meta-analysis" width="420">
</p>

<p align="center">
  <strong>Automated systematic reviews and meta-analyses from a research question.</strong><br>
  9 AI agents &middot; Multi-database search &middot; R/metafor statistics &middot; Manuscript generation &middot; Explainability report
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#pipeline-overview">Pipeline</a> &middot;
  <a href="#server-deployment-rudiger">Server Deployment</a> &middot;
  <a href="#streamlit-ui">UI</a> &middot;
  <a href="#experiments">Experiments</a>
</p>

---

## What is SAM?

**SAM** (Systematic Agent for Meta-analysis) is a multi-agent pipeline that automates the full systematic review workflow — from protocol design to manuscript generation. Give it a clinical research question and it will search databases, screen papers, retrieve full texts, extract data, assess quality, run meta-analysis, and write a manuscript.

Developed as part of the **Trabajo de Fin de Grado (TFG)** at **Universidad Pontificia Comillas (ICAI)**, Engineering Mathematics (iMAT) program.

### Key Features

- **9 specialized agents** working sequentially (A1→A9)
- **4 search databases**: PubMed, Europe PMC, Semantic Scholar, Google Scholar
- **10-source full-text retrieval cascade**: PMC XML → Unpaywall → Europe PMC → Semantic Scholar → OpenAlex → CrossRef → DOAJ → CORE → Google Scholar → Sci-Hub
- **3 quality assessment tools**: RoB 2.0 (RCTs), ROBINS-I (observational), JBI (qualitative)
- **R/metafor statistical synthesis** — REML, Egger's test, trim-and-fill, Rosenthal's Fail-Safe N, leave-one-out, subgroups, meta-regression, GRADE; no LLM for statistics
- **RAG-based manuscript generation** with FAISS + MiniLM-L6-v2
- **Explainability & traceability** report (A9): `papers_raw.xlsx` + `explainability.md` in Spanish
- **Model-agnostic**: auto-detects chat templates, works with any HuggingFace model
- **Per-agent LLM parameters** in `config/agents.json` (temperature, max_tokens, repetition penalty)
- **Two UIs**: Streamlit (desktop, light/dark mode) and a modern React + Vite + Tailwind web UI backed by a FastAPI/SSE server
- **154+ automated tests** covering all 9 agents plus end-to-end pipeline

---

## Pipeline Overview

| Agent | Role | Method |
|-------|------|--------|
| **A1** | Protocol Design | PICO/SPIDER/PECO/PCC framework detection, criteria generation, PubMed query |
| **A2** | Literature Search | Multi-database search (PubMed, Europe PMC, Semantic Scholar, Google Scholar) + semantic deduplication |
| **A3** | Title/Abstract Screening | Ensemble voting (N votes/paper, configurable majority threshold) |
| **A4** | Full-Text Retrieval & Screening | 10-source retrieval cascade + LLM full-text screening |
| **A5** | Data Extraction | 3 focused prompts per paper (study characteristics, intervention, outcomes) |
| **A6** | Quality Assessment | RoB 2.0 / ROBINS-I / JBI Critical Appraisal with weighted scores |
| **A7** | Statistical Synthesis | R/metafor: random-effects REML, heterogeneity, Egger's test, trim-and-fill, Rosenthal's Fail-Safe N, leave-one-out, subgroups, meta-regression, GRADE, R script export |
| **A8** | Manuscript Generation | RAG-based writing, section-specific retrieval, population descriptives, LaTeX/PDF output |
| **A9** | Explainability & Traceability | `papers_raw.xlsx` (per-paper trace) + `explainability.md` report in Spanish, no GPU |

### Design Principles

- **Single GPU**: Only one model loaded at a time, VRAM freed between agents
- **Conservative fallback**: Parse errors → EXCLUDE (A3/A4) or default quality weights (A6)
- **No LLM for statistics**: A7 uses R/metafor directly via rpy2
- **Model-agnostic**: Chat template auto-detection + token-level output slicing (works with Qwen, LLaMA, Mistral, GLM, etc.)

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU with 8 GB+ VRAM (tested on RTX 4060)
- R with `metafor` package
- `pdflatex` (for PDF manuscript generation)

### Setup

```bash
# 1. Clone
git clone https://github.com/iqueipopg/SAM.git
cd SAM

# 2. Virtual environment
python -m venv venv
source venv/bin/activate

# 3. Python dependencies
pip install -r requirements.txt

# 4. R metafor
Rscript -e "install.packages('metafor', repos='https://cloud.r-project.org', lib='~/R/library')"

# 5. Download a model (default: Qwen2.5-7B)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2.5-7b

# 6. Environment variables
cp .env.example .env
nano .env  # set ENTREZ_EMAIL (required) and optional API keys

# 7. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Quick Start

### Streamlit UI

```bash
source venv/bin/activate
streamlit run app.py --server.headless true
```

Open http://localhost:8501, enter a research question, and SAM runs the full pipeline.

### Command Line (single query)

```bash
./run.sh single "Is cognitive behavioral therapy more effective than medication for treating depression in adults?"
```

### Batch Mode

```bash
./run.sh pipeline   # runs 5 predefined queries
# or
python batch_run.py  # directly via Python
```

### Output

Results are saved to `outputs/run_<timestamp>/`:
- `pipeline_<timestamp>.json` — Full pipeline data
- `manuscript_<timestamp>.pdf` — Generated manuscript
- `plots/` — Forest plots, funnel plots, PRISMA flow, RoB plots

---

## Server Deployment (Rudiger)

The `run.sh` script handles everything for deploying on an Ubuntu server (e.g., the Rudiger research server).

### One-command install

```bash
chmod +x run.sh
./run.sh install
```

This installs system dependencies (Python, R, LaTeX, dev libraries), creates the virtual environment, installs all Python packages, sets up R/metafor, and creates the `.env` file.

### Using a different model

SAM is model-agnostic — it auto-detects chat templates and works with any HuggingFace causal LM. To use a different model on a server with more VRAM:

```bash
# Option 1: use run.sh model command (updates config.yaml automatically)
./run.sh model ./models/qwen3-30b

# Option 2: edit config.yaml manually
# Change all model paths under "models:" section
# Set quantization: "none" for large-VRAM servers, "8bit" or "4bit" for constrained setups
```

Supported quantization modes: `4bit` (NF4, ~5 GB for 7B), `8bit` (~8 GB for 7B), `none` (full precision).

### Multi-model ensemble (multi-GPU servers)

In `config.yaml`, set the screening agent's model entry to a list for model diversity:

```yaml
agent3_screening:
  - "./models/qwen2.5-7b"
  - "./models/glm4-9b"
  - "./models/deepseek-r1-1.5b"
```

Each vote in the ensemble uses a different model (VRAM is freed between swaps).

### Available run.sh commands

| Command | Description |
|---------|-------------|
| `./run.sh install` | Full installation (system + Python + R + .env) |
| `./run.sh pipeline` | Run batch pipeline (5 default queries) |
| `./run.sh single "question"` | Run a single research question |
| `./run.sh ui` | Launch Streamlit interface |
| `./run.sh model <path>` | Switch model (updates config.yaml) |
| `./run.sh gpu` | Check GPU status |
| `./run.sh test` | Run test suite |

---

## Streamlit UI

The desktop interface (`app.py`) features:

- **Chat-based interaction** — describe your question, watch the pipeline run
- **Pipeline progress** — each agent shown as a collapsible status widget
- **Light & Dark mode** — toggle in the sidebar, with themed SVG logos
- **Pipeline History** — click previous runs to review results
- **Compare mode** — side-by-side comparison of two pipeline runs
- **Resume** — restart from checkpoints if a pipeline was interrupted
- **Configurable settings** — manuscript language, max papers, ensemble sizes, voting thresholds
- **Exports** — PDF manuscript, CSV, Excel, RevMan XML, summary tables

---

## React Web UI + FastAPI Backend

A modern web UI lives in `web/` (React + Vite + Tailwind) and talks to a FastAPI
backend (`api.py`) that streams pipeline progress over Server-Sent Events (SSE).

```bash
# 1. Launch the backend (A1-A9 pipeline over HTTP)
uvicorn api:app --host 0.0.0.0 --port 8000

# 2. In another terminal, launch the React dev server
cd web && npm install && npm run dev
```

Vite proxies `/api` to port 8000. The React client subscribes to `/api/run`
and renders `agent_start` / `agent_done` / `agent_warning` events live, then
consumes the `pipeline_done` manifest to expose downloads (PDF, CSV, Excel,
plots, RevMan XML, `papers_raw.xlsx`, `explainability.md`).

Key endpoints: `/api/run`, `/api/stop`, `/api/kill`, `/api/status`, `/api/runs`,
`/api/download/{run_id}/{filepath}`, `/api/health`.

---

## Project Structure

```
SAM/
├── app.py                     # Streamlit UI
├── api.py                     # FastAPI backend (SSE streaming) for the React web UI
├── config.yaml                # Central configuration
├── config/agents.json         # Per-agent LLM parameters (temp, max_tokens, rep_penalty, ...)
├── requirements.txt           # Python dependencies
├── run.sh                     # Server deployment script
├── batch_run.py               # Batch pipeline runner
├── .env.example               # Environment variable template
├── ARCHITECTURE.md            # Pipeline diagram, agent table, VRAM flow, API endpoints
│
├── agents/                    # Core multi-agent pipeline
│   ├── base.py                # BaseAgent: model loading, chat template detection
│   ├── agent1_protocol.py     # A1: Protocol (PICO/SPIDER/PECO/PCC)
│   ├── agent2_search.py       # A2: Multi-database search + dedup
│   ├── agent3_screening.py    # A3: Ensemble title/abstract screening
│   ├── agent4_fulltext.py     # A4: 10-source retrieval + LLM screening
│   ├── agent5_extraction.py   # A5: Structured data extraction
│   ├── agent6_quality.py      # A6: RoB 2.0 / ROBINS-I / JBI
│   ├── agent7_synthesis.py    # A7: R/metafor statistical synthesis
│   ├── agent8_manuscript.py   # A8: RAG manuscript generation
│   ├── agent9_explainer.py    # A9: papers_raw.xlsx + explainability.md (ES)
│   ├── groq_adapter.py        # Optional Groq cloud LLM provider
│   ├── prompts.py             # All LLM prompt templates
│   ├── effect_converters.py   # Effect size conversion utilities
│   ├── exporters.py           # CSV, Excel, RevMan XML exporters
│   ├── prisma_flow.py         # PRISMA flow diagram generator
│   ├── rob_plots.py           # Risk of bias plot generators
│   ├── review_report.py       # Review report generator
│   ├── logger.py              # Logging setup
│   └── retry.py               # Exponential backoff retry logic
│
├── web/                       # React + Vite + Tailwind web UI
│   ├── src/App.jsx            # Main app component
│   ├── src/components/        # Hero, Pipeline, RunPipeline, Features, Stats, ...
│   └── vite.config.js         # Proxies /api to FastAPI on :8000
│
├── assets/                    # SVG logos (light + dark variants)
├── .streamlit/config.toml     # Streamlit theme configuration
│
├── tests/                     # 154+ tests (per-agent + end-to-end pipeline)
├── experiments/               # Validation and benchmarking scripts (exp1-exp8)
├── docs/                      # Documentation & presentation
│   ├── presentacion_sam.tex   # Beamer presentation (LaTeX)
│
├── models/                    # Local LLM weights (not in git)
└── outputs/                   # Generated results (not in git)
```

---

## Configuration

SAM reads configuration from two files:

- **`config.yaml`** — pipeline-level defaults:
  - **Models**: paths per agent, quantization mode (`4bit`/`8bit`/`none`)
  - **Search**: database toggles, rate limits, dedup threshold
  - **Screening**: ensemble size, voting threshold
  - **Retrieval**: enable/disable each of the 10 full-text sources
  - **Quality**: RoB 2.0 / ROBINS-I / JBI weights, design keywords
  - **Synthesis**: REML method, heterogeneity thresholds, GRADE settings
  - **Manuscript**: language, RAG chunk sizes, word limits
  - **Pipeline**: checkpointing, resume, cleanup settings
  - **Groq provider**: `groq.provider: local | groq` (cloud LLM fallback)

- **`config/agents.json`** — per-agent LLM parameters (temperature, max_tokens,
  max_input_chars, repetition_penalty), with per-prompt granularity for A5
  (3 prompt types) and A8 (6 sections). Overrides `config.yaml` defaults when
  present; backward compatible if absent.

- **`.env`** — API keys (`ENTREZ_EMAIL`, `UNPAYWALL_EMAIL`, `GROQ_API_KEY`, ...),
  loaded via `python-dotenv` with fallback to `config.yaml`.

---

## Experiments

The `experiments/` directory contains validation scripts:

| Script | Purpose |
|--------|---------|
| `exp1_framework_selection.py` | Framework detection accuracy (PICO vs SPIDER vs PECO) |
| `exp2_screening_gold_standard.py` | Screening against gold-standard Cochrane reviews |
| `exp3_ensemble_ablation.py` | Ensemble size vs accuracy trade-off |
| `exp4_model_comparison.py` | Cross-model performance comparison |
| `exp5_end_to_end_benchmark.py` | Full pipeline benchmark |
| `exp6_fulltext_impact.py` | Impact of full-text vs abstract-only screening |
| `exp7_vram_profiling.py` | VRAM + time profiling per agent |
| `exp8_gold_standard_eval.py` | A3 / A3+A4 vs human gold standard |

Run all experiments:

```bash
python experiments/run_all.py
```

---

## Technologies

| Category | Stack |
|----------|-------|
| **LLM** | Qwen2.5-7B (default), any HuggingFace causal LM, 4/8-bit quantization (BitsAndBytes) |
| **Embeddings** | MiniLM-L6-v2 (sentence-transformers) |
| **RAG** | LangChain + FAISS |
| **Statistics** | R / metafor (via rpy2) |
| **Search APIs** | Biopython (Entrez), Europe PMC, Semantic Scholar, Google Scholar |
| **Full-text** | PMC, Unpaywall, OpenAlex, CrossRef, DOAJ, CORE, Sci-Hub |
| **PDF** | PyMuPDF, pdfplumber, pdfminer, pypdfium2 |
| **UI** | Streamlit |
| **Output** | LaTeX (pdflatex), CSV, Excel, RevMan XML |

---

## Utilities

Helper scripts live in `scripts/` and are intended for manual, operator-driven use (no cron, no CI).

### `scripts/cleanup_old_runs.sh`

Removes old pipeline runs from `outputs/` (directories named `run_*`). Useful when `outputs/` grows over time on a shared server.

```bash
# Preview what would be deleted (default: runs older than 30 days)
scripts/cleanup_old_runs.sh --dry-run

# Custom retention window
scripts/cleanup_old_runs.sh --days 60 --dry-run

# Actually delete runs older than 30 days
scripts/cleanup_old_runs.sh

# Delete runs older than 7 days
scripts/cleanup_old_runs.sh --days 7
```

Flags:

| Flag | Description |
|------|-------------|
| `--days N` | Retention threshold in days (default: `30`). Only directories with mtime older than N days are targeted. |
| `--dry-run` | Print the directories that would be deleted without removing them. Recommended before a real run. |

The script only touches `outputs/run_*` directories directly under the project root and never follows into subdirectories. It is intentionally not scheduled in CI or cron — always run it manually after confirming with `--dry-run`.

### `scripts/post_run_backup.sh`

Copies a single pipeline run from `outputs/` to a backup directory (default `~/backup_outputs`), preserving timestamps and permissions. The script verifies the file count after copy and exits non-zero if integrity fails.

```bash
# Backup a specific run by name
./scripts/post_run_backup.sh run_20260507_081359

# Backup the most recent run (auto-detect)
./scripts/post_run_backup.sh

# Use a custom backup directory (e.g. external drive)
BACKUP_DIR=/external/sam ./scripts/post_run_backup.sh run_20260507_081359
```

Flags / arguments:

| Argument / env | Description |
|----------------|-------------|
| `<run_name>` | Run directory name (e.g. `run_20260507_081359`), relative to `outputs/`. |
| `<abs_path>` | Absolute path to a run directory inside `outputs/`. |
| `(no args)` | Auto-detects the most recent `run_*` under `outputs/`. |
| `BACKUP_DIR` | Env variable. Destination directory. Default: `~/backup_outputs`. |
| `--help` | Prints usage and exits 0. |

Exit codes: `0` success, `1` invalid args, `2` run not found, `3` `BACKUP_DIR` not writable, `4` post-copy integrity check failed.

Recommended `.env` entry (operator preference, optional): `BACKUP_DIR=/path/to/your/backup/dir`. Retention/cleanup of the backup dir is out of scope for this script — use `cleanup_old_runs.sh` on `outputs/` separately.

---

## Limitations

- **Single-GPU constraint**: Only one LLM fits in VRAM at a time (8 GB). Ensemble votes are sequential with the same model. Multi-model ensemble supported but adds latency.
- **Full-text access**: Many articles are publisher-restricted. The 10-source cascade helps, but some papers still fall back to abstract-only.
- **Statistical synthesis**: Requires papers to report compatible quantitative outcomes. If extracted data lacks effect sizes/CIs, no meta-analysis is produced.
- **LLM accuracy**: A 7B model can generate incorrect extractions or quality judgments. Human validation is recommended.
- **Google Scholar**: Aggressive rate limiting and CAPTCHAs may block access after many requests.

---

## Credits

**Trabajo de Fin de Grado (TFG)** — Universidad Pontificia Comillas, ICAI

### Author
- **Ignacio Queipo de Llano Perez-Gascon** — [iqueipo.pg24@gmail.com](mailto:iqueipo.pg24@gmail.com)

### Acknowledgements
- Qwen Team (Alibaba) for Qwen2.5-7B
- Hugging Face for the Transformers ecosystem
- Wolfgang Viechtbauer for R/metafor
- Open-source communities: LangChain, FAISS, Streamlit, Biopython

---

**Version:** v0.6.0 — 9 agents (A1-A9), React + Vite + Tailwind web UI, FastAPI/SSE backend, `config/agents.json`, 154+ tests.
