# SAM — Architecture

## Pipeline Overview

```
User query
    |
    v
+-------------+     +------------+     +--------------+     +--------------+
| A1 Protocol |---->| A2 Search  |---->| A3 Screening |---->| A4 FullText  |
| [LLM]       |     | [CPU]      |     | [LLM]        |     | [LLM]        |
+-------------+     +------------+     +--------------+     +--------------+
    |                    |                   |                     |
    | protocol           | papers            | include/exclude     | include + text
    | pubmed_query        | stats             | stats               | text_source
    v                    v                   v                     v
+--------------+     +--------------+     +--------------+     +--------------+
| A5 Extract   |---->| A6 Quality   |---->| A7 Synthesis |---->| A8 Manuscript|
| [LLM]        |     | [LLM]        |     | [R/metafor]  |     | [LLM+RAG]   |
+--------------+     +--------------+     +--------------+     +--------------+
    |                    |                   |                     |
    | study_chars        | quality_weight    | pooled estimates    | sections
    | intervention       | RoB domains       | forest/funnel PNGs  | references
    | outcomes           |                   | GRADE               | LaTeX PDF
    v                    v                   v                     v
                                                             +--------------+
                                                             | A9 Explainer |
                                                             | [CPU]        |
                                                             +--------------+
                                                                  |
                                                                  | papers_raw.xlsx
                                                                  | explainability.md
```

## Agent Table

| Agent | File | Function | GPU | Model/Backend |
|-------|------|----------|-----|---------------|
| A1 Protocol | `agent1_protocol.py` | PICO/SPIDER/PECO/PCC framework detection, criteria generation, PubMed query | Yes | Qwen2.5-7B (4-bit) or Groq |
| A2 Search | `agent2_search.py` | PubMed + Europe PMC + Semantic Scholar + Google Scholar, semantic dedup | No | MiniLM-L6-v2 (CPU) |
| A3 Screening | `agent3_screening.py` | Title/abstract ensemble voting (N votes/paper, majority rule) | Yes | Qwen2.5-7B (4-bit) or Groq |
| A4 FullText | `agent4_fulltext.py` | Multi-source cascade retrieval + LLM full-text screening | Yes | Qwen2.5-7B (4-bit) or Groq |
| A5 Extraction | `agent5_extraction.py` | 3 focused prompts per paper + numeric recovery 4-pass + author propagation | Yes | Qwen2.5-7B (4-bit) or Groq |
| A6 Quality | `agent6_quality.py` | RoB 2.0 (RCTs) / ROBINS-I (observational) / JBI (qualitative) | Yes | Qwen2.5-7B (4-bit) or Groq |
| A7 Synthesis | `agent7_synthesis.py` | R/metafor: REML, heterogeneity, Egger, Fail-Safe N, leave-one-out, subgroups, meta-regression, GRADE, R script export | No | R via rpy2 |
| A8 Manuscript | `agent8_manuscript.py` | RAG (FAISS + MiniLM) + section-by-section generation, population descriptives, LaTeX/PDF | Yes | Qwen2.5-7B (4-bit) or Groq + MiniLM-L6-v2 (CPU) |
| A9 Explainer | `agent9_explainer.py` | Excel export (papers_raw.xlsx) + Spanish explainability report | No | Pure Python (openpyxl) |

## VRAM Flow

Only one model fits in the RTX 4060 (8 GB) at a time. Each LLM agent loads,
runs, then calls `free_vram()` before the next agent.

```
A1(load,gen) free -> A2(CPU) -> A3(load,screen) free ->
A4(HTTP,load,screen) free -> A5(load,extract) free ->
A6(load,quality) free -> A7(R,noGPU) -> A8(load,RAG,gen) free -> A9(CPU)
```

Non-GPU agents: A2 (search + dedup), A7 (R/metafor), A9 (Excel + report).

## Data Flow

Each agent receives the previous agent's output and adds its own keys:

```
A1  ->  protocol: {framework, population, intervention, comparator, outcome,
                    inclusion_criteria, exclusion_criteria}
        pubmed_query: str

A2  ->  papers: [{pmid, title, abstract, authors, year, journal, doi, source}]
        search_stats: {pubmed, europepmc, semantic_scholar, google_scholar,
                       total_before_dedup, duplicates_removed, total_after_dedup}

A3  ->  screening: {include: [...], exclude: [...], uncertain: [...],
                     stats: {total, included, excluded, ensemble_size, voting_threshold}}

A4  ->  fulltext_retrieval: {include: [{...paper, full_text, text_source}],
                              exclude: [...], stats: {pmc_fulltext, unpaywall_pdf, ...}}

A5  ->  data_extraction: {papers: [{pmid, study_characteristics, intervention,
                           outcomes: [{name, measure_type, effect_size_value,
                           ci_lower, ci_upper, ...}], authors, year, journal}]}

A6  ->  quality_assessment: {papers: [{pmid, quality_assessment: {tool, domains,
                              overall}, quality_weight}], stats: {rob2_assessed, ...}}

A7  ->  synthesis: {analyses: {<measure>: {model, heterogeneity, pub_bias,
                     failsafe_n, sensitivity, grade}}, plots: {forest, funnel},
                    r_script_path: str}

A8  ->  manuscript: {sections: {abstract, introduction, methods, results,
                      discussion, conclusion}, references: [...],
                     pdf_path: str, stats: {chunks_per_section, ...}}

A9  ->  explainability: {excel_path, report_path, stats: {total_papers,
                          included, excluded, elapsed_s}}
```

## External Dependencies

| Service | Agent | Key required | Purpose |
|---------|-------|--------------|---------|
| PubMed/NCBI Entrez | A2, A4 | ENTREZ_EMAIL | Literature search, full-text XML |
| Europe PMC | A2, A4 | None | Search + full-text |
| Semantic Scholar | A2, A4 | None (basic) | Search + OA PDFs |
| Google Scholar | A2, A4 | None (scraping) | Search + PDF links |
| Unpaywall | A4 | UNPAYWALL_EMAIL | OA PDF lookup |
| OpenAlex | A4 | None | OA URL lookup |
| CrossRef | A4 | None | DOI-based PDF links |
| DOAJ | A4 | None | OA article links |
| The Lens | A4 | LENS_API_KEY | Full-text search |
| CORE | A4 | CORE_API_KEY | OA full text |
| Groq | A1,A3-A6,A8 | GROQ_API_KEY | Cloud LLM (optional) |
| R/metafor | A7 | N/A (local) | Statistical analysis |

## Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Central config: model paths, thresholds, DB settings, API keys fallback |
| `config/agents.json` | Per-agent LLM params: temperature, max_tokens, max_input_chars, repetition_penalty |
| `.env` | API keys (ENTREZ_EMAIL, UNPAYWALL_EMAIL, GROQ_API_KEY, etc.) |

`agents.json` overrides `config.yaml` defaults when present. Supports per-prompt
granularity for A5 (3 prompt types) and A8 (6 sections).

## User Interfaces

SAM ships with two front-ends that invoke the same pipeline:

| UI | Entrypoint | Stack | Purpose |
|----|------------|-------|---------|
| Streamlit | `app.py` | Streamlit (teal/white/slate theme, light/dark mode) | Local desktop UI, chat-like pipeline progress, history, compare mode |
| Web | `web/` | React + Vite + Tailwind | Modern web UI; talks to the FastAPI backend |

Both UIs render the same agent outputs (A1-A9), PRISMA flow, RoB plots, forest
and funnel plots, and the generated LaTeX/PDF manuscript.

## FastAPI Backend

`api.py` exposes the A1-A9 pipeline over HTTP for the React web UI:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/run` | POST | Starts a pipeline run and streams progress as Server-Sent Events (SSE) |
| `/api/stop` | POST | Requests a graceful stop after the current agent finishes |
| `/api/kill` | POST | Force-kills the run and frees VRAM |
| `/api/status` | GET | Returns `{running, run_dir}` |
| `/api/runs` | GET | Lists past pipeline runs stored under `outputs/` |
| `/api/download/{run_id}/{filepath}` | GET | Downloads a file from a run directory |
| `/api/health` | GET | GPU/CPU health check |

SSE events emitted per agent: `agent_start`, `agent_done`, `agent_warning`,
`agent_error`, plus pipeline-level `start`, `finalizing`, `pipeline_done`,
`pipeline_stopped`, `pipeline_error`. The React client subscribes to
`/api/run` and renders each event live. Vite proxies `/api` to the FastAPI
server on port 8000 (`web/vite.config.js`).

Launch:

```
uvicorn api:app --host 0.0.0.0 --port 8000   # backend
cd web && npm run dev                         # React dev server
```
