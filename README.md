# SAM - Systematic Agent for Meta-analysis

**SAM** is an automated multi-agent pipeline that conducts systematic reviews and meta-analyses from a natural language research question. It orchestrates 8 specialized AI agents — from protocol design to manuscript generation — using a local LLM (Qwen2.5-7B, 4-bit quantization) and R/metafor for statistical synthesis.

This project was developed as part of the **Trabajo de Fin de Grado (TFG)** at **Universidad Pontificia Comillas, ICAI**, within the **Engineering Mathematics (iMAT)** program.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Pipeline Details](#pipeline-details)
- [Credits](#credits)

---

## Project Overview

SAM automates the full systematic review workflow through 8 sequential agents:

| Agent | Role | Method |
|-------|------|--------|
| **A1** | Protocol Design | PICO/SPIDER/PECO/PCC framework detection, inclusion/exclusion criteria, PubMed query generation |
| **A2** | Literature Search | PubMed search via Entrez API + semantic deduplication |
| **A3** | Title/Abstract Screening | Ensemble voting (N votes/paper, majority threshold) |
| **A4** | Full-Text Retrieval & Screening | PMC XML → Unpaywall → Europe PMC → Semantic Scholar → Sci-Hub → abstract fallback + LLM screening |
| **A5** | Data Extraction | 3 focused prompts per paper (study characteristics, intervention, outcomes) |
| **A6** | Quality Assessment | RoB 2.0 (RCTs) / NOS (observational), keyword-based design detection with LLM fallback |
| **A7** | Statistical Synthesis | R/metafor random-effects models, heterogeneity analysis, publication bias, sensitivity analysis, GRADE |
| **A8** | Manuscript Generation | RAG-based writing with FAISS index, section-specific retrieval, LaTeX/PDF output |

Key design constraints:
- **Single GPU**: Only one model loaded at a time (RTX 4060, 8GB VRAM), with VRAM freed between agents
- **Conservative fallback**: Parse errors default to EXCLUDE (A3/A4) or default quality weights (A6)
- **No LLM for statistics**: A7 uses R/metafor directly via rpy2

---

## Installation

### Prerequisites
- **Python 3.12+**
- **CUDA-capable GPU** with 8GB+ VRAM (RTX 4060 or equivalent)
- **R** with the `metafor` package installed
- **pdflatex** (for PDF manuscript generation)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd meta-analisis-automatizado
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install R metafor package:**
   ```r
   install.packages("metafor", lib = "~/R/library")
   ```

5. **Download the Qwen2.5-7B model** into `models/qwen2.5-7b/`:
   ```bash
   huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/qwen2.5-7b
   ```

6. **Verify GPU:**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
   ```

---

## Quick Start

### Launch the Streamlit UI

```bash
source venv/bin/activate
streamlit run app.py --server.headless true
```

Enter a research question (e.g., *"Is cognitive behavioral therapy more effective than medication for treating depression in adults?"*) and SAM will run the full A1→A8 pipeline automatically.

### Pipeline output

Results are saved to `outputs/` as:
- `pipeline_<timestamp>.json` — Full pipeline data (protocol, screening, extraction, quality, synthesis, manuscript)
- `manuscript_<timestamp>.pdf` — Generated LaTeX manuscript
- `outputs/plots/` — Forest plots, funnel plots

---

## Project Structure

```
meta-analisis-automatizado/
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml                # Model paths, parameters, retrieval config
├── .gitignore                 # Excluded files
│
├── agents/                    # Core multi-agent pipeline
│   ├── base.py                # BaseAgent: load_model(), generate(), _clean_json()
│   ├── agent1_protocol.py     # A1: Protocol design (PICO/SPIDER/PECO/PCC)
│   ├── agent2_search.py       # A2: PubMed search + deduplication
│   ├── agent3_screening.py    # A3: Ensemble title/abstract screening
│   ├── agent4_fulltext.py     # A4: Full-text retrieval + LLM screening
│   ├── agent5_extraction.py   # A5: Structured data extraction
│   ├── agent6_quality.py      # A6: RoB 2.0 / NOS quality assessment
│   ├── agent7_synthesis.py    # A7: R/metafor statistical synthesis
│   ├── agent8_manuscript.py   # A8: RAG-based manuscript generation
│   └── prompts.py             # All LLM prompt templates
│
├── app.py                     # Streamlit UI (SAM interface)
├── .streamlit/
│   └── config.toml            # Streamlit theme (light, teal accent)
│
├── tests/                     # Agent and pipeline tests
│   ├── test_agent1_complete.py
│   ├── test_agent4.py
│   ├── test_a4_sources.py
│   ├── test_agent5.py
│   ├── test_agent6.py
│   ├── test_agent7.py
│   ├── test_agent8.py
│   ├── test_a6_a7_pipeline.py
│   └── test_pipeline_complete.py
│
├── experiments/                # Validation and benchmarking scripts
│   ├── exp1_framework_selection.py
│   ├── exp2_screening_gold_standard.py
│   ├── exp3_ensemble_ablation.py
│   ├── exp4_model_comparison.py
│   ├── exp5_end_to_end_benchmark.py
│   ├── exp6_fulltext_impact.py
│   └── run_all.py
│
├── docs/                      # Documentation
│   └── DIARIO_DESARROLLO_COMPLETO.txt
│
├── models/                    # Local LLM weights (not tracked in git)
│   └── qwen2.5-7b/
│
└── outputs/                   # Generated results (not tracked in git)
    ├── pipeline_<timestamp>.json
    ├── manuscript_<timestamp>.pdf
    └── plots/
```

---

## Technologies Used

### LLM & ML
- **Qwen2.5-7B** — Local LLM for all AI agents (4-bit quantization via BitsAndBytes)
- **PyTorch** — Deep learning backend
- **Hugging Face Transformers** — Model loading and inference
- **Sentence-Transformers (MiniLM-L6-v2)** — Embeddings for RAG and semantic deduplication

### RAG & Retrieval
- **LangChain + FAISS** — Vector store for manuscript generation
- **Biopython (Entrez)** — PubMed API access
- **Unpaywall / Europe PMC / Semantic Scholar / Sci-Hub** — Multi-source full-text retrieval

### Statistical Analysis
- **R / metafor** (via rpy2) — Random-effects meta-analysis, heterogeneity, publication bias, sensitivity analysis
- **GRADE assessment** — Evidence quality grading

### UI & Output
- **Streamlit** — Interactive web interface
- **LaTeX (pdflatex)** — Manuscript PDF generation
- **Matplotlib** — Fallback plot generation

### Key Techniques
- **4-bit NF4 Quantization** — Fits 7B model in 8GB VRAM
- **Ensemble Voting** — Multiple LLM votes per paper for robust screening
- **Conservative Fallback** — Parse errors → EXCLUDE to minimize false inclusions
- **VRAM Management** — Sequential model loading with explicit GPU memory release

---

## Pipeline Details

### Screening Strategy
- A3 and A4 use configurable `ensemble_size` with `voting_threshold` (default 0.60)
- A4 retrieves full text through a 6-source cascade before LLM screening
- **Multi-model ensemble**: Both A3 and A4 support using different models per vote. In `config.yaml`, set the agent's model entry to a list instead of a string. Each vote rotates through the model list, with VRAM freed between swaps. On a single GPU (8GB), this adds latency per model swap; on a multi-GPU server it enables true model diversity for more robust screening

### Quality Assessment
- **RCTs**: RoB 2.0 (5 domains: randomization, deviations, missing data, measurement, selection)
- **Observational**: Newcastle-Ottawa Scale (3 domains, 0-9 stars)
- Quality weights: Low risk/Good → 1.0, Some concerns/Fair → 0.6, High risk/Poor → 0.3

### Statistical Synthesis
- Random-effects REML models grouped by outcome measure (SMD, OR, RR, HR, MD, PROP)
- Heterogeneity: I², tau², Cochran's Q
- Publication bias: Egger's test + trim-and-fill
- Sensitivity: Leave-one-out + exclude-high-risk-of-bias

### Manuscript Generation
- RAG with variable-k retrieval per section (results=10, discussion=8, intro=6, others=4)
- Section-specific temperatures (0.3 factual, 0.6 narrative)
- Anti-hallucination rules in all prompts
- `[Author, Year]` citation format

---

## Limitations and Future Work

- **Single-GPU constraint**: Only one LLM fits in VRAM at a time (8GB). The ensemble currently votes N times with the same model via stochastic sampling. Multi-model ensemble is supported in config but adds significant latency due to sequential model swaps. A multi-GPU server would enable true parallel multi-model voting.
- **Full-text access**: Many PMC articles are publisher-restricted (no `<body>` in XML). The 6-source retrieval cascade mitigates this, but some papers still fall back to abstract-only screening.
- **Statistical synthesis**: A7 requires papers to report compatible quantitative outcomes (effect sizes, CIs, SDs). When extracted data lacks these, no meta-analysis is produced.
- **LLM hallucination**: Despite anti-hallucination prompts and conservative fallbacks, a 7B model can still generate incorrect extractions or quality judgments. Human validation of results is recommended.

---

## Credits

This project was developed as part of the **Trabajo de Fin de Grado (TFG)** at **Universidad Pontificia Comillas, ICAI**.

### Author
- **Ignacio Queipo de Llano Pérez-Gascón** ([iqueipo.pg24@gmail.com](mailto:iqueipo.pg24@gmail.com))

### Acknowledgements
- **Qwen Team (Alibaba)** for the Qwen2.5-7B model
- **Hugging Face** for the Transformers ecosystem
- **Wolfgang Viechtbauer** for the R metafor package
- Open-source communities behind LangChain, FAISS, Streamlit, and Biopython
