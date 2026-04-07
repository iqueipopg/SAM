# SAM Architecture

## 1. Pipeline Diagram

```
 USER QUERY (research question)
       |
       v
 +-------------+     +-------------+     +----------------+
 | A1 Protocol |---->| A2 Search   |---->| A3 Screening   |
 |   [LLM]     |     |   [CPU]     |     |   [LLM]        |
 | PICO/SPIDER |     | PubMed      |     | Ensemble voting|
 | PECO/PCC    |     | Europe PMC  |     | title/abstract |
 | + criteria  |     | Semantic S. |     | majority rule  |
 | + PubMed Q  |     | Google Sch. |     |                |
 +------+------+     | + dedup     |     +-------+--------+
        |             +-------------+             |
        |  free_vram()                free_vram() |
        v                                         v
 +----------------+     +------------------+     +----------------+
 | A4 Full-Text   |<----| A3 INCLUDE list  |     | A5 Extraction  |
 |   [LLM]        |     +------------------+     |   [LLM]        |
 | 10-source      |                               | 3 prompts/     |
 | retrieval      |------------------------------>| paper + numeric|
 | + FT screening |     (included papers)         | recovery       |
 +-------+--------+                               +-------+--------+
         |  free_vram()                    free_vram()     |
         v                                                 v
 +----------------+     +------------------+     +------------------+
 | A6 Quality     |<----|  A5 output       |     | A7 Synthesis     |
 |   [LLM]        |     +------------------+     |   [R/metafor]    |
 | RoB 2.0 (RCT)  |                               | REML pooling    |
 | ROBINS-I (obs.) |------------------------------>| I2, tau2, Egger |
 | JBI (qual.)     |     (quality weights)         | GRADE, SoF      |
 +-------+---------+                               | forest/funnel   |
         |  free_vram()                             +--------+--------+
         v                                                   |
 +------------------+                                        |
 | A8 Manuscript    |<---------------------------------------+
 |   [LLM + RAG]   |     (all pipeline data)
 | FAISS + MiniLM   |
 | section-by-sect. |
 | LaTeX -> PDF     |
 +------------------+
         |  free_vram()
         v
   OUTPUT: JSON + PDF + plots
```

VRAM flow (only one model in GPU at a time):

```
A1(load,gen) -> free -> A2(CPU) -> A3(load,screen) -> free ->
A4(HTTP,load,screen) -> free -> A5(load,extract) -> free ->
A6(load,quality) -> free -> A7(R, no GPU) -> A8(load,RAG,gen) -> free
```

## 2. Agent Reference Table

| Agent | Class | Description | Input | Output | GPU | Est. Time |
|-------|-------|-------------|-------|--------|-----|-----------|
| A1 | `Agent1Protocol` | Protocol generation: framework detection (PICO/SPIDER/PECO/PCC), inclusion/exclusion criteria, PubMed query | User query (str) | Protocol dict (framework, criteria, PubMed query) | Yes | ~15 s |
| A2 | `Agent2Search` | Multi-database search (PubMed, Europe PMC, Semantic Scholar, Google Scholar) + 3-tier semantic deduplication | Protocol dict, PubMed query | Papers list + stats dict | No (CPU) | 30-120 s |
| A3 | `Agent3Screening` | Title/abstract screening with ensemble voting (N votes/paper, majority threshold) | Papers list, protocol | Include/exclude/uncertain lists + stats | Yes | ~5 s/paper |
| A4 | `Agent4FullText` | Full-text retrieval (10-source cascade) + LLM full-text PRISMA screening | Included papers, protocol | Include/exclude lists with text_source | Yes | ~10 s/paper |
| A5 | `Agent5Extraction` | Structured data extraction: 3 focused prompts (study chars, intervention, outcomes) + numeric recovery | Included papers with text, protocol | Papers with study_characteristics, intervention, outcomes | Yes | ~15 s/paper |
| A6 | `Agent6Quality` | Quality assessment: RoB 2.0 (RCTs, 5 domains), ROBINS-I (observational, 7 domains), JBI (qualitative, 10 items) | Extracted papers, protocol | Papers with quality_assessment + quality_weight | Yes | ~10 s/paper |
| A7 | `Agent7Synthesis` | Statistical meta-analysis: random-effects REML, heterogeneity, Egger's test, trim-and-fill, leave-one-out, GRADE, SoF | Extraction data, quality data | Analyses dict (per outcome group) + forest/funnel PNGs | No (R) | 5-30 s |
| A8 | `Agent8Manuscript` | RAG-based manuscript generation: FAISS + MiniLM-L6-v2 for retrieval, Qwen2.5-7B for prose, LaTeX/PDF output | All pipeline data | Manuscript sections, references, LaTeX PDF | Yes | ~60 s |

Supporting modules (no pipeline step):

| Module | Description |
|--------|-------------|
| `base.py` | BaseAgent: model loading, chat template detection, generate(), free_vram() |
| `prompts.py` | All LLM prompt templates (A1, A3, A5, A6, A8) centralized |
| `exporters.py` | CSV, Excel (openpyxl), RevMan XML, Markdown summary table exports |
| `effect_converters.py` | Effect size conversions (MD-SMD, OR-RR, Hedges' g) for A7 pooling |
| `review_report.py` | Borderline review report: flags uncertain decisions for optional human review |
| `rob_plots.py` | RoB traffic light + summary bar plots (matplotlib) |
| `prisma_flow.py` | PRISMA 2020 flow diagram (matplotlib) |
| `logger.py` | Structured rotating logger (console + file) |
| `retry.py` | Exponential backoff with jitter for all HTTP calls |

## 3. External Dependencies

| Library | Version | Function in SAM |
|---------|---------|-----------------|
| `torch` | >= 2.5.0 | GPU tensor operations for LLM inference |
| `transformers` | >= 5.0.0 | HuggingFace model loading, tokenization, generation |
| `bitsandbytes` | >= 0.49.0 | 4-bit and 8-bit quantization (NF4) for VRAM-constrained GPUs |
| `accelerate` | >= 1.12.0 | Device mapping and mixed-precision support |
| `safetensors` | >= 0.7.0 | Safe model weight serialization |
| `sentence-transformers` | >= 5.2.0 | MiniLM-L6-v2 embeddings for deduplication (A2) and RAG (A8) |
| `langchain` | >= 1.2.0 | RAG orchestration and document splitting (A8) |
| `langchain-community` | >= 0.4.0 | Community integrations for FAISS vector store |
| `faiss-cpu` | >= 1.13.0 | FAISS vector index for RAG retrieval (A8) |
| `biopython` | >= 1.86 | Entrez API for PubMed search (A2) and PMC retrieval (A4) |
| `requests` | >= 2.32.0 | HTTP client for Europe PMC, Semantic Scholar, Unpaywall, etc. |
| `beautifulsoup4` | >= 4.14.0 | HTML parsing for Google Scholar scraping (A2) |
| `lxml` | >= 6.0.0 | XML parsing for PMC JATS full-text extraction (A4) |
| `rpy2` | >= 3.6.0 | R/metafor bridge for statistical meta-analysis (A7) |
| `pdfplumber` | >= 0.11.0 | PDF text extraction for full-text retrieval (A4) |
| `PyMuPDF` | >= 1.27.0 | PDF text extraction (alternative backend for A4) |
| `pdfminer.six` | >= 20251230 | PDF text extraction (fallback backend for A4) |
| `pypdfium2` | >= 5.5.0 | PDF rendering and text extraction (fallback for A4) |
| `streamlit` | >= 1.54.0 | Web UI framework for the SAM interface |
| `numpy` | >= 2.4.0 | Numerical operations for embeddings and statistics |
| `pandas` | >= 2.3.0 | Data manipulation for export and analysis |
| `scipy` | >= 1.17.0 | Statistical functions used by A7 and experiments |
| `scikit-learn` | >= 1.8.0 | ML utilities for clustering and evaluation in experiments |
| `PyYAML` | >= 6.0.0 | Configuration file parsing (config.yaml) |
| `python-dotenv` | >= 1.2.0 | Environment variable loading from .env files |
| `pillow` | >= 12.1.0 | Image processing for plot generation |
| `tqdm` | >= 4.67.0 | Progress bars for batch operations |
| `openpyxl` | (transitive) | Excel file generation in exporters.py |

R dependencies (installed separately):

| Package | Function |
|---------|----------|
| `metafor` | Random-effects meta-analysis, forest/funnel plots, Egger's test, trim-and-fill |
