"""Agent 8: Manuscript Generation with RAG (Qwen2.5-7B + MiniLM + FAISS)

Generates a publishable systematic review manuscript using:
  - RAG over full texts of included papers (FAISS + MiniLM-L6-v2 on CPU)
  - Qwen2.5-7B for text generation per section
  - Citation tracking via chunk metadata
  - Pre-built data blocks so the LLM writes prose, not numbers
"""

import sys, os, shutil, subprocess, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from agents.prompts import (
    get_a8_rag_query, get_a8_section_prompt, A8_RAG_K,
    get_a8_section_prompt_localized, BABEL_LANGUAGES, LANGUAGE_NAMES,
)
import json, torch, re, time, yaml
from typing import List, Dict, Optional, Tuple


# Section generation order
SECTION_ORDER = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]


class _EmbeddingWrapper:
    """Wraps SentenceTransformer to match langchain's Embeddings interface.

    Inherits from langchain_core Embeddings so FAISS isinstance() check passes.
    """

    def __init__(self, model):
        self._model = model

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False, batch_size=64).tolist()


# Register as Embeddings subclass so isinstance() check works in FAISS
try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
    _LCEmbeddings.register(_EmbeddingWrapper)
except Exception:
    pass


class Agent8Manuscript(BaseAgent):
    """RAG-based manuscript generator producing a systematic review document.

    Input: protocol, included papers, extraction, quality, and synthesis data.
    Output: dict with sections (abstract through conclusion), references, and LaTeX/PDF paths.
    GPU: yes -- uses MiniLM-L6-v2 (CPU) for RAG embeddings and Qwen2.5-7B for prose generation.
    """

    def __init__(self, language: str = None):
        super().__init__("agent8_manuscript")
        self._faiss_index = None
        self._paper_lookup = {}  # pmid -> citation info
        self._references = []    # ordered list of cited papers
        self._cited_pmids = set()
        self._citation_index = {}  # (surname_lower, year_str) -> paper info
        self._verification_reports = {}  # section_name -> verification report

        # Language: explicit arg > config.yaml > "en"
        if language is None:
            ms_cfg = self.config.get("manuscript", {})
            language = ms_cfg.get("language", "en")
        self._language = language if language in LANGUAGE_NAMES else "en"

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def generate_manuscript(
        self,
        protocol: Dict,
        included_papers: List[Dict],
        extraction: Optional[Dict] = None,
        quality: Optional[Dict] = None,
        synthesis: Optional[Dict] = None,
        output_dir: str = "outputs",
        prisma: Optional[Dict] = None,
    ) -> Dict:
        """Generate full manuscript with RAG citations. Returns sections + references.

        Args:
            protocol: A1 protocol dict (research_question, population, etc.)
            included_papers: A4 included papers list
            extraction: A5 extraction dict (optional)
            quality: A6 quality dict (optional)
            synthesis: A7 synthesis dict (optional)
            output_dir: output directory path
            prisma: PRISMA flow dict (optional, with identified/screened/excluded/included counts)
        """
        print(f"\n{'='*60}")
        print(f"AGENT 8 - MANUSCRIPT GENERATION (RAG) [lang={self._language}]")
        print(f"{'='*60}")

        start = time.time()
        self._output_dir = output_dir

        # Store pipeline data for data-block builders
        self._protocol = protocol
        self._extraction = extraction
        self._quality = quality
        self._synthesis = synthesis
        self._prisma = prisma or {}
        self._included_papers = included_papers

        # Build RAG corpus from papers
        n_chunks = self._build_rag_corpus(included_papers, extraction)
        print(f"  RAG corpus: {n_chunks} chunks from {len(included_papers)} papers")

        # Build citation index for verification
        self._citation_index = self._build_citation_index(included_papers)
        self._verification_reports = {}
        print(f"  Citation index: {len(self._citation_index)} author-year entries")

        # Build data summaries
        synthesis_summary = self._build_synthesis_summary(synthesis) if synthesis else ""
        quality_summary = self._build_quality_summary(quality) if quality else ""
        n_studies = len(included_papers)

        # Pre-build data blocks for injection into prompts
        self._prisma_text = self._build_prisma_text()
        self._study_list_text = self._build_study_list_text()
        self._methods_data_block = self._build_methods_data_block()

        # Load LLM
        if self.model is None:
            self.load_model()

        # Generate each section
        manuscript = {}
        chunks_per_section = {}

        for section_name in SECTION_ORDER:
            print(f"\n  Generating: {section_name}...")
            text, n_used = self._generate_section(
                section_name, protocol, synthesis_summary, quality_summary, n_studies
            )
            manuscript[section_name] = text
            chunks_per_section[section_name] = n_used
            print(f"    -> {len(text)} chars, {n_used} RAG chunks used")

        # Cross-section coherence check
        coherence_report = self._check_coherence(manuscript, synthesis)
        if coherence_report["warnings"]:
            print(f"\n  Coherence check: {len(coherence_report['warnings'])} warning(s)")
            for w in coherence_report["warnings"][:5]:
                print(f"    - {w}")

        # Ensure ALL included papers appear in references (systematic review standard)
        for pmid, info in self._paper_lookup.items():
            if pmid not in self._cited_pmids:
                self._cited_pmids.add(pmid)
                self._references.append(info)

        elapsed = round(time.time() - start, 1)

        # Build citation verification summary
        verification_summary = self._build_verification_summary()
        if verification_summary["total_removed"] > 0:
            print(f"\n  Citation verification: {verification_summary['total_verified']}/{verification_summary['total_citations']} "
                  f"verified, {verification_summary['total_removed']} phantom(s) removed")
        elif verification_summary["total_citations"] > 0:
            print(f"\n  Citation verification: all {verification_summary['total_citations']} citation(s) verified")

        # Generate PDF
        print("\n  Generating PDF via LaTeX...")
        pdf_path = self._generate_pdf(
            manuscript, self._references, protocol,
            extraction, quality, synthesis,
        )

        result = {
            "manuscript": manuscript,
            "sections": manuscript,  # alias for consistent access
            "references": self._references,
            "pdf_path": pdf_path,
            "citation_verification": verification_summary,
            "coherence_check": coherence_report,
            "stats": {
                "total_chunks": n_chunks,
                "chunks_per_section": chunks_per_section,
                "total_papers_cited": len(self._cited_pmids),
                "elapsed_s": elapsed,
            },
        }

        self._print_summary(result)
        return result

    # ================================================================== #
    #  DATA BLOCK BUILDERS — pre-write factual text for prompt injection  #
    # ================================================================== #

    def _build_prisma_text(self) -> str:
        """Build a plain-text PRISMA flow from pipeline data."""
        p = self._prisma
        if not p:
            # Try to infer from available data
            n_inc = len(self._included_papers)
            return f"{n_inc} studies were included in the final review."

        identified = p.get("identified", "?")
        excl_ta = p.get("excluded_title_abstract", "?")
        uncertain = p.get("uncertain_title_abstract", 0)
        assessed_ft = p.get("assessed_fulltext", "?")
        excl_ft = p.get("excluded_fulltext", "?")
        included = p.get("included_final", len(self._included_papers))

        lines = [
            f"{identified} records were identified from PubMed.",
            f"{excl_ta} were excluded during title/abstract screening.",
        ]
        if uncertain:
            lines.append(f"{uncertain} were classified as uncertain and forwarded for full-text review.")
        lines.append(f"{assessed_ft} articles were assessed at full-text level.")
        lines.append(f"{excl_ft} were excluded after full-text review.")
        lines.append(f"{included} studies met all inclusion criteria and were included in the review.")

        extracted = p.get("extracted", 0)
        qa = p.get("quality_assessed", 0)
        if extracted:
            lines.append(f"Data were extracted from {extracted} studies.")
        if qa:
            lines.append(f"Quality assessment was completed for {qa} studies.")

        return " ".join(lines)

    def _build_study_list_text(self) -> str:
        """Build a readable list of included studies with characteristics."""
        ext_lookup = {}
        if self._extraction and self._extraction.get("papers"):
            for ep in self._extraction["papers"]:
                ext_lookup[str(ep.get("pmid", ""))] = ep

        lines = []
        for p in self._included_papers:
            pmid = str(p.get("pmid", ""))
            info = self._paper_lookup.get(pmid, {})
            cite = info.get("citation", "Unknown")

            ext = ext_lookup.get(pmid, {})
            sc = ext.get("study_characteristics", {})
            ie = ext.get("intervention_exposure", {})
            outcomes = ext.get("outcomes", [])

            design = sc.get("study_design") or "not reported"
            sample = sc.get("sample_size")
            sample_str = str(sample) if sample else "NR"
            # Try both field names for intervention
            intervention = ie.get("name") or ie.get("intervention_name") or "not reported"
            comparator = ie.get("comparator") or ie.get("control_description") or "NR"
            followup = sc.get("followup_duration") or "NR"

            outcome_parts = []
            for o in outcomes[:2]:
                oname = o.get("name") or o.get("outcome_name") or "outcome"
                val = o.get("value") or o.get("effect_size")
                measure = o.get("measure_type", "")
                if val is not None:
                    ci_lo = o.get("ci_lower", "?")
                    ci_hi = o.get("ci_upper", "?")
                    outcome_parts.append(f"{oname}: {measure}={val} (95% CI: {ci_lo} to {ci_hi})")
                else:
                    outcome_parts.append(oname)

            outcome_str = "; ".join(outcome_parts) if outcome_parts else "not reported"

            lines.append(
                f"- [{cite}] (PMID {pmid}): {design}, N={sample_str}, "
                f"intervention={intervention}, comparator={comparator}, "
                f"follow-up={followup}. Outcomes: {outcome_str}"
            )

        return "\n".join(lines) if lines else "No detailed study data available."

    def _build_methods_data_block(self) -> str:
        """Build a pre-written methods description from actual pipeline data."""
        protocol = self._protocol
        pop = protocol.get("population", "the target population")
        intv = protocol.get("intervention", protocol.get("exposure", "the intervention"))
        comp = protocol.get("comparator", "standard care or placebo")
        outc = protocol.get("outcome", "the primary outcome")

        # Get pubmed query if available
        pubmed_query = protocol.get("pubmed_query", "")
        # It might also be at top level of pipeline data — the caller should pass it via protocol
        # but we use what we have

        # Quality tools used
        qual_tools = []
        if self._quality and self._quality.get("stats"):
            qs = self._quality["stats"]
            if qs.get("rob2_assessed", 0) > 0:
                qual_tools.append("Risk of Bias 2 (RoB 2) for randomized trials")
            # NOS legacy fallback (old pipeline runs)
            if qs.get("nos_assessed", 0) > 0:
                qual_tools.append("Newcastle-Ottawa Scale (NOS) for observational studies")
            if qs.get("robins_i_assessed", 0) > 0:
                qual_tools.append("ROBINS-I for non-randomized studies")
            if qs.get("jbi_assessed", 0) > 0:
                qual_tools.append("JBI Critical Appraisal for qualitative studies")
        if not qual_tools:
            qual_tools = ["standard quality assessment tools"]

        # Inclusion/exclusion
        inc = protocol.get("inclusion_criteria", [])
        exc = protocol.get("exclusion_criteria", [])
        inc_str = "; ".join(inc) if inc else "relevant study designs reporting the outcome of interest"
        exc_str = "; ".join(exc) if exc else "studies not meeting inclusion criteria"

        # Synthesis info
        has_synthesis = bool(self._synthesis and self._synthesis.get("analyses"))

        parts = [
            "This systematic review followed PRISMA guidelines.",
            f"The PubMed database was searched for studies examining {intv} in {pop} with {outc} as the outcome.",
        ]
        if pubmed_query:
            parts.append(f"The search query used was: {pubmed_query}.")

        parts.append(
            f"Eligibility criteria: Population: {pop}; Intervention/Exposure: {intv}; "
            f"Comparator: {comp}; Outcome: {outc}."
        )
        parts.append(f"Inclusion criteria: {inc_str}.")
        parts.append(f"Exclusion criteria: {exc_str}.")

        parts.append(
            "Study selection was performed in two phases: automated title/abstract screening "
            "using an ensemble-based approach with majority voting, followed by full-text screening "
            "against eligibility criteria."
        )
        parts.append(
            "Data extraction captured study characteristics (design, sample size, setting, follow-up), "
            "intervention details, and outcome measures including effect sizes and confidence intervals."
        )
        parts.append(f"Quality was assessed using {'; '.join(qual_tools)}.")

        if has_synthesis:
            parts.append(
                "Statistical analysis used random-effects meta-analysis with restricted maximum likelihood (REML) "
                "estimation via R/metafor. Heterogeneity was assessed using I-squared, tau-squared, and Cochran's Q test. "
                "Sensitivity analyses included leave-one-out analysis and exclusion of high risk-of-bias studies. "
                "Publication bias was evaluated using Egger's regression test and trim-and-fill method."
            )
        else:
            parts.append(
                "A narrative synthesis was performed as quantitative meta-analysis was not feasible."
            )

        return " ".join(parts)

    # ================================================================== #
    #  RAG CORPUS                                                          #
    # ================================================================== #

    def _build_rag_corpus(
        self,
        papers: List[Dict],
        extraction: Optional[Dict] = None,
    ) -> int:
        """Chunk paper texts, embed with MiniLM on CPU, build FAISS index."""
        from sentence_transformers import SentenceTransformer
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        # Build paper lookup for citations
        self._paper_lookup = {}
        for p in papers:
            pmid = str(p.get("pmid", ""))
            authors = p.get("authors", [])
            first_author = authors[0].split(",")[0].split()[-1] if authors else "Unknown"
            year = p.get("year", "")
            citation_key = f"{first_author} et al., {year}" if year else f"{first_author} et al."

            self._paper_lookup[pmid] = {
                "pmid": pmid,
                "citation": citation_key,
                "title": p.get("title", ""),
                "journal": p.get("journal", ""),
                "year": year,
            }

        # Merge extraction data if available
        ext_lookup = {}
        if extraction and extraction.get("papers"):
            for ep in extraction["papers"]:
                ext_lookup[str(ep.get("pmid", ""))] = ep

        # Chunk texts
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
        )

        texts = []
        metadatas = []

        for p in papers:
            pmid = str(p.get("pmid", ""))

            # Get text: prefer sections, then full_text, then abstract
            sections = p.get("sections") or {}
            if sections:
                for sec_name, sec_text in sections.items():
                    if not sec_text:
                        continue
                    chunks = splitter.split_text(sec_text)
                    for chunk in chunks:
                        texts.append(chunk)
                        metadatas.append({
                            "pmid": pmid,
                            "section": sec_name,
                            "citation_key": self._paper_lookup[pmid]["citation"],
                        })
            else:
                full_text = p.get("full_text", "") or p.get("abstract", "") or ""
                if full_text:
                    chunks = splitter.split_text(full_text)
                    for chunk in chunks:
                        texts.append(chunk)
                        metadatas.append({
                            "pmid": pmid,
                            "section": "full",
                            "citation_key": self._paper_lookup[pmid]["citation"],
                        })

            # Add extraction data as chunks
            ext = ext_lookup.get(pmid)
            if ext:
                ext_text = self._extraction_to_text(ext)
                if ext_text:
                    chunks = splitter.split_text(ext_text)
                    for chunk in chunks:
                        texts.append(chunk)
                        metadatas.append({
                            "pmid": pmid,
                            "section": "extraction",
                            "citation_key": self._paper_lookup[pmid]["citation"],
                        })

        if not texts:
            print("    [!] No text chunks to index")
            return 0

        # Embed on CPU
        print(f"  Embedding {len(texts)} chunks with MiniLM (CPU)...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = _EmbeddingWrapper(embed_model)

        self._faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return len(texts)

    def _extraction_to_text(self, ext: Dict) -> str:
        """Convert extraction data to readable text for RAG."""
        parts = []
        sc = ext.get("study_characteristics", {})
        if sc:
            parts.append(
                f"Study design: {sc.get('study_design', '?')}, "
                f"Sample: {sc.get('sample_size', '?')}, "
                f"Setting: {sc.get('setting', '?')}, "
                f"Follow-up: {sc.get('followup_duration', '?')}"
            )
        ie = ext.get("intervention_exposure", {})
        if ie:
            # Support both field name variants
            intv_name = ie.get("name") or ie.get("intervention_name") or "?"
            ctrl = ie.get("comparator") or ie.get("control_description") or "?"
            dur = ie.get("dose_duration") or ie.get("duration") or "?"
            parts.append(
                f"Intervention: {intv_name}, "
                f"Control: {ctrl}, "
                f"Duration: {dur}"
            )
        outcomes = ext.get("outcomes", [])
        for o in outcomes:
            oname = o.get("name") or o.get("outcome_name") or "?"
            val = o.get("value") or o.get("effect_size") or "?"
            ci_lo = o.get("ci_lower", "?")
            ci_hi = o.get("ci_upper", "?")
            parts.append(
                f"Outcome: {oname} = {val} "
                f"(95% CI: {ci_lo}-{ci_hi}), "
                f"p={o.get('p_value', '?')}, measure={o.get('measure_type', '?')}"
            )
        return "\n".join(parts)

    def _get_extracted_outcome_names(self) -> List[str]:
        """Collect all outcome names from A5 extraction results."""
        names = []
        if self._extraction and self._extraction.get("papers"):
            for paper in self._extraction["papers"]:
                for outcome in paper.get("outcomes", []):
                    name = outcome.get("name") or outcome.get("outcome_name")
                    if name and name.strip():
                        names.append(name.strip())
        return names

    # ================================================================== #
    #  RAG RETRIEVAL                                                       #
    # ================================================================== #

    def _retrieve_chunks(self, query: str, k: int = 6) -> List[Tuple[str, Dict]]:
        """Retrieve top-k relevant chunks from FAISS index."""
        if not self._faiss_index or not query:
            return []
        results = self._faiss_index.similarity_search_with_score(query, k=k)
        return [(doc.page_content, doc.metadata) for doc, score in results]

    # ================================================================== #
    #  SECTION GENERATION                                                  #
    # ================================================================== #

    # Section-specific temperatures: lower for fact-heavy, higher for narrative
    _SECTION_TEMPS = {
        "abstract": 0.2,
        "methods": 0.2,
        "results": 0.2,
        "conclusion": 0.2,
        "introduction": 0.3,
        "discussion": 0.3,
    }

    # Max new tokens per section
    _SECTION_MAX_TOKENS = {
        "abstract": 400,
        "introduction": 800,
        "methods": 1000,
        "results": 1200,
        "discussion": 1000,
        "conclusion": 500,
    }

    def _generate_section(
        self,
        section_name: str,
        protocol: Dict,
        synthesis_summary: str,
        quality_summary: str,
        n_studies: int,
    ) -> Tuple[str, int]:
        """Generate one manuscript section using RAG + LLM."""
        # Build outcome-aware RAG query
        outcome_names = self._get_extracted_outcome_names()
        synthesis_keys = list((self._synthesis or {}).get("analyses", {}).keys())
        rag_query = get_a8_rag_query(
            section_name, protocol,
            outcome_names=outcome_names,
            synthesis_keys=synthesis_keys,
        )
        k = A8_RAG_K.get(section_name, 4)
        chunks = self._retrieve_chunks(rag_query, k=k) if rag_query else []

        # Track citations from retrieved chunks
        for _, meta in chunks:
            pmid = meta.get("pmid", "")
            if pmid and pmid not in self._cited_pmids:
                self._cited_pmids.add(pmid)
                if pmid in self._paper_lookup:
                    self._references.append(self._paper_lookup[pmid])

        # Format RAG context
        rag_context = ""
        if chunks:
            rag_parts = []
            for text, meta in chunks:
                cite = meta.get("citation_key", "")
                rag_parts.append(f"[{cite}]: {text}")
            rag_context = "\n\n".join(rag_parts)

        # Build prompt with pre-built data blocks (language-aware)
        prompt = get_a8_section_prompt_localized(
            section_name, protocol, rag_context,
            synthesis_summary, quality_summary, n_studies,
            prisma_text=self._prisma_text,
            study_list_text=self._study_list_text,
            methods_data_block=self._methods_data_block,
            language=self._language,
        )

        # Generate with section-specific temperature and token limit
        temp = self._SECTION_TEMPS.get(section_name, 0.5)
        max_tok = self._SECTION_MAX_TOKENS.get(section_name, 1000)
        text = self._simple_generate(prompt, max_tokens=max_tok, temperature=temp)

        # Clean output
        text = self._clean_section_text(text, section_name)

        # Fix citation format (numeric -> named, normalize)
        text = self._fix_citation_format(text, self._references)

        # Verify citations — remove phantoms
        text, verification_report = self._verify_citations(text, self._references)
        self._verification_reports[section_name] = verification_report
        if verification_report["removed"] > 0:
            print(f"    [!] Citation verification: {verification_report['removed']} phantom citation(s) removed")
            for d in verification_report.get("details", []):
                if "removed" in d.get("status", ""):
                    print(f"        - Removed: {d['citation']}")

        # Post-processing: flag insufficient data in results/abstract
        if section_name in ("results", "abstract"):
            _synth_lower = synthesis_summary.lower() if synthesis_summary else ""
            if any(marker in _synth_lower for marker in [
                "no quantitative synthesis", "insufficient data", "0 analyses",
                "no meta-analysis", "no pooled", "could not be pooled",
            ]):
                _insuf_note = "Due to insufficient quantitative data, no pooled estimates could be calculated."
                if _insuf_note.lower() not in text.lower():
                    text = text.rstrip()
                    if not text.endswith('.'):
                        text += '.'
                    text += f"\n\n{_insuf_note}"

        return text, len(chunks)

    # ================================================================== #
    #  CITATION VERIFICATION                                               #
    # ================================================================== #

    def _build_citation_index(self, papers: List[Dict]) -> Dict:
        """Build lookup index for citation verification.

        Maps (first_author_surname_lower, year_str) -> paper info dict.
        Also stores alternate keys for common variants.

        Args:
            papers: list of included paper dicts (from A4)

        Returns:
            dict mapping (surname_lower, year_str) to paper info
        """
        index = {}

        for p in papers:
            pmid = str(p.get("pmid", ""))
            authors = p.get("authors", [])
            year = str(p.get("year", "")).strip()
            title = p.get("title", "")

            if not year:
                continue

            # Extract surnames from all authors
            surnames = []
            for author in authors:
                # Handle "LastName, FirstName" or "FirstName LastName" formats
                if "," in author:
                    surname = author.split(",")[0].strip()
                else:
                    parts = author.strip().split()
                    surname = parts[-1] if parts else ""
                if surname:
                    surnames.append(surname)

            if not surnames:
                continue

            first_surname = surnames[0].lower()
            paper_info = {
                "pmid": pmid,
                "first_author": surnames[0],
                "all_surnames": [s.lower() for s in surnames],
                "year": year,
                "title": title,
                "citation": self._paper_lookup.get(pmid, {}).get("citation", f"{surnames[0]} et al., {year}"),
            }

            # Primary key: (first_author_surname, year)
            key = (first_surname, year)
            index[key] = paper_info

            # Also index with just surname (no year) for fuzzy matching
            # and with all author surnames for multi-author citations
            if len(surnames) >= 2:
                second_surname = surnames[1].lower()
                # "(Smith & Jones, 2020)" pattern
                index[(first_surname, second_surname, year)] = paper_info

        return index

    def _verify_citations(
        self,
        section_text: str,
        references_list: List[Dict],
    ) -> Tuple[str, Dict]:
        """Verify that all in-text citations match real papers. Remove phantoms.

        Parses citation patterns:
          - (Author, Year) / (Author et al., Year) / (Author & Other, Year)
          - (Author1, Year1; Author2, Year2)   — multi-citations
          - [1], [2,3], [1-3]                  — numeric references

        For each citation, checks against self._citation_index.
        Unmatched citations are replaced with neutral phrasing.

        Args:
            section_text: the generated section text
            references_list: current ordered reference list

        Returns:
            (cleaned_text, verification_report)
            verification_report: {total, verified, removed, details: [...]}
        """
        if not self._citation_index:
            return section_text, {"total_citations": 0, "verified": 0, "removed": 0, "details": []}

        total = 0
        verified = 0
        removed = 0
        details = []

        # ----------------------------------------------------------
        # Pattern 1: Parenthetical author-year citations
        #   (Smith, 2020)  (Smith et al., 2020)  (Smith & Jones, 2020)
        #   (Smith et al., 2020; Jones et al., 2021)  — semicolon-separated
        # ----------------------------------------------------------
        # Match the entire parenthetical group
        paren_citation_re = re.compile(
            r'\(([A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-zA-Z\u00C0-\u024F\'-]+))?'
            r',?\s*\d{4}'
            r'(?:\s*;\s*[A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-zA-Z\u00C0-\u024F\'-]+))?'
            r',?\s*\d{4})*)\)'
        )

        # Parse individual citation within a parenthetical group
        single_cite_re = re.compile(
            r'([A-Z][a-zA-Z\u00C0-\u024F\'-]+)'           # first author surname
            r'(?:\s+et\s+al\.?)?'                           # optional "et al."
            r'(?:\s*&\s*([A-Z][a-zA-Z\u00C0-\u024F\'-]+))?' # optional "& SecondAuthor"
            r',?\s*(\d{4})'                                  # year
        )

        def _check_single_citation(surname: str, second_author: Optional[str], year: str) -> Tuple[bool, float]:
            """Check if a single author-year citation matches the index.

            Returns (matched, confidence) where confidence is 0.0-1.0:
            - 1.0: exact (surname, year) match
            - 0.9: exact (surname, second_author, year) match
            - 0.7: surname found in any author list for that year
            - 0.0: no match found
            """
            s_lower = surname.lower()
            y = year.strip()

            # Direct match: (surname, year)
            if (s_lower, y) in self._citation_index:
                return True, 1.0

            # Try with second author: (surname, second, year)
            if second_author:
                s2_lower = second_author.lower()
                if (s_lower, s2_lower, y) in self._citation_index:
                    return True, 0.9

            # Fuzzy: check if surname appears in any paper's author list for that year
            for (key_surname, *rest), info in self._citation_index.items():
                if len(rest) >= 1 and rest[-1] == y:
                    # Check if surname matches first author or any author
                    if s_lower == key_surname:
                        return True, 0.8
                    if s_lower in info.get("all_surnames", []):
                        return True, 0.7

            return False, 0.0

        def _replace_paren_group(match):
            nonlocal total, verified, removed, details
            full_match = match.group(0)   # includes parens
            inner = match.group(1)        # inside parens

            # Split on semicolons for multi-citations
            parts = [p.strip() for p in inner.split(';')]
            verified_parts = []
            any_removed = False

            for part in parts:
                m = single_cite_re.search(part)
                if not m:
                    # Can't parse — keep conservatively
                    verified_parts.append(part)
                    total += 1
                    verified += 1
                    details.append({"citation": part, "status": "kept_unparseable"})
                    continue

                surname = m.group(1)
                second_author = m.group(2)
                year = m.group(3)
                total += 1

                matched, confidence = _check_single_citation(surname, second_author, year)
                if matched:
                    verified_parts.append(part)
                    verified += 1
                    status = "verified" if confidence >= 0.8 else "verified_low_confidence"
                    details.append({"citation": part, "status": status, "confidence": confidence})
                    if confidence < 0.8:
                        print(f"    [citation] Low confidence ({confidence:.1f}): {part}")
                else:
                    removed += 1
                    any_removed = True
                    details.append({"citation": part, "status": "removed_phantom", "confidence": 0.0})

            if not verified_parts:
                # All citations in the group were phantom — replace with neutral text
                return "previous research"
            elif any_removed:
                # Some removed — rebuild parenthetical with only verified
                return "(" + "; ".join(verified_parts) + ")"
            else:
                return full_match

        section_text = paren_citation_re.sub(_replace_paren_group, section_text)

        # ----------------------------------------------------------
        # Pattern 1b: Bracket author-year citations (non-numeric)
        #   [Smith, 2020]  [Smith et al., 2020]  [@Smith et al., 2020]
        #   [Smith et al., 2020; Jones et al., 2021]
        # ----------------------------------------------------------
        bracket_citation_re = re.compile(
            r'\[@?([A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-zA-Z\u00C0-\u024F\'-]+))?'
            r',?\s*\d{4}'
            r'(?:\s*;\s*@?[A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-zA-Z\u00C0-\u024F\'-]+))?'
            r',?\s*\d{4})*)\]'
        )

        def _replace_bracket_group(match):
            nonlocal total, verified, removed, details
            match.group(0)
            inner = match.group(1)

            parts = [p.strip().lstrip('@') for p in inner.split(';')]
            verified_parts = []
            any_removed = False

            for part in parts:
                m = single_cite_re.search(part)
                if not m:
                    verified_parts.append(part)
                    total += 1
                    verified += 1
                    details.append({"citation": part, "status": "kept_unparseable"})
                    continue

                surname = m.group(1)
                second_author = m.group(2)
                year = m.group(3)
                total += 1

                matched, confidence = _check_single_citation(surname, second_author, year)
                if matched:
                    verified_parts.append(part)
                    verified += 1
                    status = "verified" if confidence >= 0.8 else "verified_low_confidence"
                    details.append({"citation": part, "status": status, "confidence": confidence})
                else:
                    removed += 1
                    any_removed = True
                    details.append({"citation": part, "status": "removed_phantom_bracket", "confidence": 0.0})

            if not verified_parts:
                return "previous research"
            elif any_removed:
                return "(" + "; ".join(verified_parts) + ")"
            else:
                # Convert valid bracket citations to parenthetical format
                return "(" + "; ".join(verified_parts) + ")"

        section_text = bracket_citation_re.sub(_replace_bracket_group, section_text)

        # ----------------------------------------------------------
        # Pattern 2: Numeric references [1], [2,3], [1-3], [1, 2, 3]
        # ----------------------------------------------------------
        numeric_cite_re = re.compile(r'\[(\d+(?:\s*[,\-]\s*\d+)*)\]')

        def _replace_numeric(match):
            nonlocal total, verified, removed, details
            inner = match.group(1)
            full_match = match.group(0)

            # Parse all numbers from the bracket
            nums = []
            # Handle ranges like 1-3
            range_parts = re.split(r'[,\s]+', inner)
            for rp in range_parts:
                rp = rp.strip()
                if '-' in rp:
                    try:
                        lo, hi = rp.split('-', 1)
                        nums.extend(range(int(lo), int(hi) + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        nums.append(int(rp))
                    except ValueError:
                        pass

            if not nums:
                return full_match

            valid_nums = []
            for n in nums:
                total += 1
                # Check if reference number is within bounds
                if 1 <= n <= len(references_list):
                    verified += 1
                    valid_nums.append(n)
                    details.append({"citation": f"[{n}]", "status": "verified"})
                else:
                    removed += 1
                    details.append({"citation": f"[{n}]", "status": "removed_out_of_range"})

            if not valid_nums:
                return "previous studies"
            elif len(valid_nums) == len(nums):
                return full_match  # all valid, keep original
            else:
                return "[" + ", ".join(str(n) for n in valid_nums) + "]"

        section_text = numeric_cite_re.sub(_replace_numeric, section_text)

        # ----------------------------------------------------------
        # Clean up artifacts from removal
        # ----------------------------------------------------------
        # Double spaces
        section_text = re.sub(r'  +', ' ', section_text)
        # "( )" or empty parens
        section_text = re.sub(r'\(\s*\)', '', section_text)
        # Orphaned punctuation: ", ." or "; ."
        section_text = re.sub(r'[,;]\s*\.', '.', section_text)
        # Double periods
        section_text = re.sub(r'\.\.+', '.', section_text)
        # Space before period/comma
        section_text = re.sub(r'\s+([.,;])', r'\1', section_text)

        report = {
            "total_citations": total,
            "verified": verified,
            "removed": removed,
            "details": details,
        }
        return section_text, report

    def _fix_citation_format(self, text: str, references: List[Dict]) -> str:
        """Standardize citation formats in text.

        - Convert numeric [N] references to (Author et al., Year) if possible
        - Normalize spacing and punctuation in existing author-year citations

        Args:
            text: section text with citations
            references: ordered reference list

        Returns:
            text with standardized citation format
        """
        # Build numeric-to-named mapping from references list
        num_to_named = {}
        for i, ref in enumerate(references):
            citation = ref.get("citation", "")
            if citation:
                num_to_named[i + 1] = citation  # 1-indexed

        # Replace [N] with (Author et al., Year) where possible
        def _replace_single_numeric(match):
            inner = match.group(1)
            nums = []
            range_parts = re.split(r'[,\s]+', inner)
            for rp in range_parts:
                rp = rp.strip()
                if '-' in rp:
                    try:
                        lo, hi = rp.split('-', 1)
                        nums.extend(range(int(lo), int(hi) + 1))
                    except ValueError:
                        nums.append(-1)
                else:
                    try:
                        nums.append(int(rp))
                    except ValueError:
                        nums.append(-1)

            named_parts = []
            all_resolved = True
            for n in nums:
                if n in num_to_named:
                    named_parts.append(num_to_named[n])
                else:
                    all_resolved = False
                    break

            if all_resolved and named_parts:
                return "(" + "; ".join(named_parts) + ")"
            else:
                return match.group(0)  # keep numeric if can't resolve all

        numeric_re = re.compile(r'\[(\d+(?:\s*[,\-]\s*\d+)*)\]')
        text = numeric_re.sub(_replace_single_numeric, text)

        # Normalize "et al" variants: "et al" -> "et al."
        text = re.sub(r'\bet al(?!\.)(\s*[,;)])', r'et al.\1', text)

        # Ensure comma between author and year: "(Smith et al. 2020)" -> "(Smith et al., 2020)"
        text = re.sub(
            r'(et\s+al\.)\s+(\d{4})',
            r'\1, \2',
            text,
        )

        return text

    def _build_verification_summary(self) -> Dict:
        """Build aggregate citation verification summary across all sections.

        Returns:
            dict with total_citations, total_verified, total_removed,
            and per-section breakdown including removed citation details.
        """
        total_citations = 0
        total_verified = 0
        total_removed = 0
        per_section = {}
        removed_details = []

        for section_name, report in self._verification_reports.items():
            sc = report.get("total_citations", 0)
            sv = report.get("verified", 0)
            sr = report.get("removed", 0)
            total_citations += sc
            total_verified += sv
            total_removed += sr
            per_section[section_name] = {
                "total": sc,
                "verified": sv,
                "removed": sr,
            }
            # Collect removed citation details
            for d in report.get("details", []):
                if "removed" in d.get("status", ""):
                    removed_details.append({
                        "section": section_name,
                        "citation": d["citation"],
                        "reason": d["status"],
                    })

        return {
            "total_citations": total_citations,
            "total_verified": total_verified,
            "total_removed": total_removed,
            "per_section": per_section,
            "removed_citations": removed_details,
        }

    # ================================================================== #
    #  CROSS-SECTION COHERENCE CHECK                                       #
    # ================================================================== #

    def _check_coherence(self, manuscript: Dict, synthesis: Optional[Dict]) -> Dict:
        """Validate cross-section coherence of the generated manuscript.

        Checks:
        1. Results claims should appear in Discussion/Conclusion
        2. Figure/Table references should be valid
        3. Key statistics in Results should match synthesis data
        4. Abstract should not contain claims absent from Results

        Returns dict with warnings list and pass/fail status.
        """
        warnings = []
        results_text = (manuscript.get("results") or "").lower()
        discussion_text = (manuscript.get("discussion") or "").lower()
        conclusion_text = (manuscript.get("conclusion") or "").lower()
        abstract_text = (manuscript.get("abstract") or "").lower()

        # 1. Check figure/table references exist
        all_text = " ".join(manuscript.get(s, "") for s in manuscript)
        fig_refs = re.findall(r'[Ff]igure\s+(\d+)', all_text)
        re.findall(r'[Tt]able\s+(\d+)', all_text)

        # Count actual figures (from synthesis plots)
        n_actual_figures = 0
        if synthesis and synthesis.get("analyses"):
            for analysis in synthesis["analyses"].values():
                plots = analysis.get("plots", {})
                if plots.get("forest"):
                    n_actual_figures += 1
                if plots.get("funnel"):
                    n_actual_figures += 1

        for fig_num in fig_refs:
            n = int(fig_num)
            if n > n_actual_figures + 2:  # +2 for PRISMA + possible extra
                warnings.append(f"Reference to Figure {n} but only ~{n_actual_figures} figures exist")

        # 2. Check that key pooled estimates from Results appear in Discussion
        # Extract numbers that look like effect sizes from Results
        es_pattern = re.compile(r'(?:pooled|overall)\s+(?:estimate|effect)[^.]*?(-?\d+\.?\d*)', re.IGNORECASE)
        results_estimates = es_pattern.findall(results_text)
        for est in results_estimates[:3]:  # check top 3
            if est not in discussion_text and est not in conclusion_text:
                warnings.append(f"Pooled estimate {est} in Results not mentioned in Discussion/Conclusion")

        # 3. Check abstract consistency — abstract shouldn't have numbers not in results
        abstract_numbers = re.findall(r'(?:pooled|estimate|effect)\s*=?\s*(-?\d+\.\d+)', abstract_text)
        for num in abstract_numbers:
            if num not in results_text:
                warnings.append(f"Abstract mentions effect size {num} not found in Results section")

        # 4. Check that synthesis group keys are mentioned in Results
        if synthesis and synthesis.get("analyses"):
            for group_key in synthesis["analyses"]:
                # Check if category name or related term appears in results
                key_lower = group_key.lower().replace("_", " ")
                if key_lower not in results_text and key_lower not in "smd md or rr hr prop":
                    # Try partial match
                    key_tokens = key_lower.split()
                    if not any(t in results_text for t in key_tokens if len(t) > 3):
                        warnings.append(f"Synthesis group '{group_key}' not clearly discussed in Results")

        return {
            "warnings": warnings,
            "n_warnings": len(warnings),
            "passed": len(warnings) == 0,
        }

    # ================================================================== #
    #  SUMMARIES                                                           #
    # ================================================================== #

    def _build_synthesis_summary(self, synthesis: Dict) -> str:
        """Build compact, pre-written summary of A7 synthesis results.

        This text is meant to be directly usable by the LLM — no interpretation needed.
        """
        parts = []
        analyses = synthesis.get("analyses", {})
        stats = synthesis.get("stats", {})

        usable = stats.get("papers_with_usable_data", 0)
        n_analyses = stats.get("analyses_performed", 0)

        if not analyses or n_analyses == 0:
            if usable == 0:
                return "No quantitative meta-analysis was performed because no studies had usable numeric outcome data for pooling."
            return f"{usable} papers had usable data but no meta-analyses were completed."

        parts.append(f"{usable} studies contributed usable data to {n_analyses} meta-analysis(es).")

        for measure, analysis in analyses.items():
            model = analysis.get("model", {})
            het = analysis.get("heterogeneity", {})
            grade = analysis.get("grade", {})
            pub = analysis.get("publication_bias", {})
            sens = analysis.get("sensitivity", {})

            k = model.get("k_studies", "?")
            est = model.get("pooled_estimate", "?")
            ci_lo = model.get("ci_lower", "?")
            ci_hi = model.get("ci_upper", "?")
            pval = model.get("p_value", "?")

            parts.append(
                f"For {measure} ({k} studies): the pooled estimate was {est} "
                f"(95% CI: {ci_lo} to {ci_hi}, p={pval})."
            )

            i2 = het.get("I2", "?")
            tau2 = het.get("tau2", "?")
            q = het.get("Q", "?")
            q_p = het.get("Q_pvalue", "?")
            interp = het.get("interpretation", "")
            parts.append(
                f"Heterogeneity: I-squared={i2}%, tau-squared={tau2}, "
                f"Q={q} (p={q_p}). {interp}"
            )

            if grade:
                gl = grade.get("final_level", "?")
                dg = grade.get("total_downgrades", 0)
                parts.append(f"GRADE certainty: {gl} ({dg} downgrade(s)).")

            egger = pub.get("egger_test", {}) if isinstance(pub, dict) else {}
            if egger and not egger.get("error"):
                ez = egger.get("z_value", "?")
                ep = egger.get("p_value", "?")
                sig = egger.get("significant", False)
                sig_str = "significant" if sig else "not significant"
                parts.append(f"Egger's test: z={ez}, p={ep} ({sig_str}).")

            tf = pub.get("trim_and_fill", {}) if isinstance(pub, dict) else {}
            if tf and not tf.get("error"):
                miss = tf.get("estimated_missing", 0)
                adj = tf.get("adjusted_estimate", "?")
                parts.append(f"Trim-and-fill: {miss} missing studies estimated, adjusted estimate={adj}.")

            excl = sens.get("exclude_high_risk", {}) if isinstance(sens, dict) else {}
            if excl and not excl.get("error"):
                kr = excl.get("k_remaining", "?")
                se = excl.get("pooled_estimate", "?")
                parts.append(f"Sensitivity (excluding high risk-of-bias): {kr} studies remaining, estimate={se}.")

        return " ".join(parts)

    def _build_quality_summary(self, quality: Dict) -> str:
        """Build compact summary of A6 quality results."""
        parts = []
        stats = quality.get("stats", {})

        rob2 = stats.get("rob2_assessed", 0)
        robins = stats.get("robins_i_assessed", 0)
        jbi = stats.get("jbi_assessed", 0)
        failed = stats.get("failed", 0)

        tool_parts = []
        if rob2:
            tool_parts.append(f"{rob2} with RoB 2")
        if robins:
            tool_parts.append(f"{robins} with ROBINS-I")
        if jbi:
            tool_parts.append(f"{jbi} with JBI")

        if tool_parts:
            parts.append(f"Quality assessed: {', '.join(tool_parts)}.")
        if failed:
            parts.append(f"{failed} assessments failed.")

        # Summarize overall judgments
        papers = quality.get("papers", [])
        judgment_counts = {}
        for p in papers:
            qa = p.get("quality_assessment", {})
            overall = qa.get("overall", "Unknown")
            judgment_counts[overall] = judgment_counts.get(overall, 0) + 1

        if judgment_counts:
            jc_str = ", ".join(f"{v} {k}" for k, v in sorted(judgment_counts.items(), key=lambda x: -x[1]))
            parts.append(f"Overall judgments: {jc_str}.")

        return " ".join(parts)

    # ================================================================== #
    #  LLM HELPERS                                                         #
    # ================================================================== #

    def _simple_generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.7) -> str:
        """Generate text, truncating input if needed. Uses chat template when available."""
        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=5000
            ).to(self.device)
        else:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=5000
            ).to(self.device)

        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(outputs, input_length)

    def _remove_prompt_leakage(self, text: str) -> str:
        """Remove imperative instruction lines that leaked from the prompt."""
        # Remove code fences
        text = re.sub(r'```\w*\s*', '', text)

        # Remove common LLM preamble phrases at the very start
        preamble_re = re.compile(
            r'^(Certainly:?\s*|Sure[.!:]\s*|Of course[.!:]\s*|'
            r'No additional (comments?|prompts?)[^.]*[.!]?\s*|'
            r'I\'d be happy to[^.]*[.!]?\s*|'
            r'Here\'?s?\s+(is\s+)?(the|your|a)[^.]*[.!:]\s*|'
            r'Below is[^.]*[.!:]\s*|'
            r'The following[^.]*[.!:]\s*)',
            re.IGNORECASE,
        )
        # Apply up to 3 times to catch stacked preambles
        for _ in range(3):
            text = preamble_re.sub('', text)

        # Remove markdown headers (###, ##, #)
        text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE)

        # Remove leading section title if model repeats it
        section_titles = [
            r'^\*?\*?(Structured\s+)?Abstract\*?\*?\s*$',
            r'^\*?\*?Introduction\*?\*?\s*$',
            r'^\*?\*?Methods?\*?\*?\s*$',
            r'^\*?\*?Results?\*?\*?\s*$',
            r'^\*?\*?Discussion\*?\*?\s*$',
            r'^\*?\*?Conclusions?\*?\*?\s*$',
            r'^(Here is the|Below is the|The following is|Here\'s).*:?\s*$',
        ]
        title_re = re.compile('|'.join(section_titles), re.IGNORECASE | re.MULTILINE)
        text = title_re.sub('', text)

        leakage_patterns = [
            r'^(Include the following|Avoid repetition|Write a\b|Ensure\b|Make sure\b|Do NOT\b|'
            r'CRITICAL RULES|You are a systematic|Remember to\b|Please\b|Note:|Be sure to\b|'
            r'Keep (under|the|it)|Use (precise|citations|academic)|Respond ONLY|'
            r'RULES:|FORMAT|RESEARCH QUESTION|DATA TO USE|CONTEXT FROM|'
            r'Paragraph \d+:|P[aá]rrafo \d+:)',
        ]
        combined = re.compile('|'.join(leakage_patterns), re.IGNORECASE)
        lines = text.split('\n')
        cleaned = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Scan all lines for leakage (not just first 15)
            if stripped and combined.search(stripped):
                continue
            cleaned.append(line)
        text = '\n'.join(cleaned)

        # Remove all-caps section labels leaked into prose
        text = re.sub(
            r'^(SUMMARY OF MAIN FINDINGS|COMPARISON WITH EXISTING EVIDENCE|'
            r'LIMITATIONS INCLUDE THE FOLLOWING POINTS|STrengths|STRENGTHS|'
            r'IMPLICATIONS FOR CLINICAL PRACTICE|FUTURE RESEARCH DIRECTIONS|'
            r'KEY RESULTS|MAIN FINDINGS|ELIGIBILITY CRITERIA)[:\s]*',
            '', text, flags=re.MULTILINE | re.IGNORECASE,
        )

        # Remove "Background is mandatory" and similar meta-commentary
        text = re.sub(r'Background is mandatory.*?(?:\.\s*|\n)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Paragraph breaks are included.*?(?:\.\s*|\n)', '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove repeated "previous research" filler
        text = re.sub(r'\s*previous research\s*', ' ', text, flags=re.IGNORECASE)
        # Clean up double spaces left behind
        text = re.sub(r'  +', ' ', text)

        return text

    def _truncate_repetitions(self, text: str) -> str:
        """Detect and truncate repetition loops and word-salad degeneration."""
        # 1) Remove exact consecutive sentence duplicates
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text

        cleaned = [sentences[0]]
        repeat_count = 1
        for i in range(1, len(sentences)):
            if sentences[i] == sentences[i - 1] and len(sentences[i]) > 20:
                repeat_count += 1
                if repeat_count >= 3:
                    continue
            else:
                repeat_count = 1
            cleaned.append(sentences[i])

        result = ' '.join(cleaned)

        # 2) Catch repeated phrases within running text
        result = re.sub(r'(.{20,}?)\1{2,}', r'\1', result)

        # 3) Detect word-salad degeneration: long runs of words with no
        #    sentence-ending punctuation (.!?) anywhere in a 80-word window.
        #    Normal academic text always has periods within 80 words.
        words = result.split()
        if len(words) > 100:
            for i in range(60, len(words) - 80):
                window = words[i:i + 80]
                window_text = ' '.join(window)
                has_sentence_end = bool(re.search(r'[.!?]', window_text))
                if not has_sentence_end:
                    # Truncate: find last good sentence end before this point
                    prefix = ' '.join(words[:i])
                    last_period = prefix.rfind('.')
                    if last_period > len(prefix) * 0.3:
                        result = prefix[:last_period + 1]
                    break

        return result

    # Max words per section (abstract is strict, others are soft caps)
    _SECTION_WORD_LIMITS = {
        "abstract": 350,
        "introduction": 600,
        "methods": 800,
        "results": 800,
        "discussion": 800,
        "conclusion": 500,
    }

    def _enforce_word_limit(self, text: str, section_name: str) -> str:
        """Truncate text to word limit, cutting at last sentence boundary."""
        limit = self._SECTION_WORD_LIMITS.get(section_name)
        if not limit:
            return text
        words = text.split()
        if len(words) <= limit:
            return text
        # Truncate at word limit, then find last sentence end
        truncated = ' '.join(words[:limit])
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.6:
            return truncated[:last_period + 1]
        return truncated + '.'

    def _enforce_paragraphs(self, text: str, section_name: str) -> str:
        """Ensure text has paragraph breaks. Insert them if missing."""
        # Count existing paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) >= 3:
            return text  # Already well-structured

        # If it's a wall of text, insert paragraph breaks at sentence boundaries
        # Target: ~4-5 sentences per paragraph
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 6:
            return text  # Too short to split

        # Determine target paragraph size
        target_para_size = max(3, min(5, len(sentences) // 4))

        new_paragraphs = []
        current = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if len(current) >= target_para_size and i < len(sentences) - 1:
                new_paragraphs.append(' '.join(current))
                current = []
        if current:
            new_paragraphs.append(' '.join(current))

        return '\n\n'.join(new_paragraphs)

    def _clean_section_text(self, text: str, section_name: str = "") -> str:
        """Clean generated section text — leakage, repetitions, LaTeX artifacts."""
        # Step 1: Remove prompt leakage
        text = self._remove_prompt_leakage(text)

        # Step 2: Truncate repetitions
        text = self._truncate_repetitions(text)

        # Step 3: Remove markdown headers (### Title) including any preceding punctuation
        text = re.sub(r'^[.\s]*#{1,4}\s+.*$', '', text, flags=re.MULTILINE)
        # Remove standalone numbered section headers (e.g., "1. Protocol and Registration")
        text = re.sub(r'^\d+\.\s+[A-Z][a-zA-Z\s]+(?:and\s+\w+)?\s*$', '', text, flags=re.MULTILINE)
        # Remove leading colons or dashes left after cleanup
        text = re.sub(r'^[:\-]+\s*', '', text)

        # Step 4: Remove image refs, horizontal rules, trailing meta-blocks
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # ![](...)
        text = re.sub(r'^-{3,}\s*', '', text, flags=re.MULTILINE)
        # Remove markdown bold markers **text** -> text
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*Note:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\*\*References:\*\*.*', '', text, flags=re.DOTALL)
        text = re.sub(r'(Feel free to|Let me know).*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Step 4b: Clean HTML tags (LLM sometimes generates <sup>, <sub>, etc.)
        text = re.sub(r'<sup>(.*?)</sup>', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'<sub>(.*?)</sub>', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'</?(?:b|i|em|strong|br|p|div|span|a)[^>]*>', '', text, flags=re.IGNORECASE)

        # Step 4c: Remove markdown tables (lines with |---|--- or | col | col |)
        text = re.sub(r'^[|\s]*[-:]+[|\s:\-]+$', '', text, flags=re.MULTILINE)  # separator rows
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)  # data rows

        # Step 4d: Remove URLs and hallucinated placeholders
        text = re.sub(r'\(https?://[^\)]+\)', '', text)  # (https://...)
        text = re.sub(r'https?://\S+', '', text)  # bare URLs
        text = re.sub(r'\[?\*?Note\*?:\s*Missing reference\]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Study [A-Z],\s*Study [A-Z],?\s*(?:etc\.?|\.\.\.)', '', text)
        text = re.sub(r'\*Study [A-Z]\*', '', text)  # *Study A*
        text = re.sub(r'\*?(?:min|max)_[*\w]+_[*\w]+\*?', 'various', text)
        text = re.sub(r'\*?final_number\*?', 'the final number', text)

        # Step 4e: Remove inline markdown tables (| col | col | patterns)
        text = re.sub(r'\|[\s\-:]+\|', ' ', text)  # |---| separators
        text = re.sub(r'\|[^|\n]{0,50}\|', ' ', text)  # |content| cells (repeat to handle multi-col)
        text = re.sub(r'\|[^|\n]{0,50}\|', ' ', text)  # second pass
        text = re.sub(r'\|', '', text)  # any remaining pipes

        # Step 4f-pre: Remove markdown-style citation artifacts [@Author et al., Year]
        text = re.sub(r'\[@[^\]]*?\]', '', text)
        # Remove leaked paragraph labels: "Paragraph 1:", "Párrafo 2:", etc.
        text = re.sub(r'^(?:Paragraph|Párrafo|Parágrafo)\s*\d+\s*:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Step 4f-pre2: Remove sentences with obvious hallucination markers
        # Sentences with "X participants" or "X studies" where X is a single letter (not a number)
        text = re.sub(r'[^.!?]*\b[A-Z]\s+(?:participants|studies|patients|subjects|trials)\b[^.!?]*[.!?]', '', text, flags=re.IGNORECASE)
        # Sentences referencing "Author Y" as a name (single-letter surname)
        text = re.sub(r'[^.!?]*\bAuthor\s+[A-Z]\b[^.!?]*[.!?]', '', text)

        # Step 4f: Remove LaTeX-like commands the LLM may generate
        text = re.sub(r'\\citep?\{[^}]*\}', '', text)  # \cite{...}, \citep{...}
        text = re.sub(r'\\citet?\{[^}]*\}', '', text)  # \citet{...}
        text = re.sub(r'\\textgreater\{\}', '>', text)
        text = re.sub(r'\\textless\{\}', '<', text)
        text = re.sub(r'\\textasciitilde\{\}', '~', text)
        text = re.sub(r'\\textasciicircum\{\}', '^', text)
        text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\underline\{([^}]*)\}', r'\1', text)
        # Remove other stray LaTeX commands: \command{} or \command
        text = re.sub(r'\\(?:ref|label|footnote|footnotesize|small|large|Large|normalsize)\{[^}]*\}', '', text)
        text = re.sub(r'\\(?:newline|linebreak|pagebreak|clearpage|newpage|par|noindent|bigskip|medskip|smallskip)\b', '', text)
        text = re.sub(r'\\\\(?=\s|$)', '', text)  # stray \\

        # Step 4g: Remove placeholder values (\\XXX, =XX, XX%)
        text = re.sub(r'\\\\XXX\b', '', text)
        text = re.sub(r'\bXX%', '', text)
        text = re.sub(r'=XX\b', '', text)
        # Remove [citation missing] or <citation missing> patterns
        text = re.sub(r'[\[<]citation\s*missing[\]>]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\[<]insert\s+\w+[\]>]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\[<]ref[\]>]', '', text, flags=re.IGNORECASE)

        # Step 4h: Remove markdown link syntax [text](#anchor) or [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)

        # Step 4i: Fix statistics broken across lines (LLM puts newlines inside parens)
        # Collapse multi-line parenthetical expressions: "(\nMD = 0.86\n)" -> "(MD = 0.86)"
        def _fix_multiline_parens(m):
            inner = m.group(0).replace('\n', ' ')
            return re.sub(r'\s{2,}', ' ', inner)
        text = re.sub(r'\([^)]{0,200}\n[^)]{0,200}\)', _fix_multiline_parens, text)
        # Second pass for deeply broken ones
        text = re.sub(r'\([^)]{0,200}\n[^)]{0,200}\)', _fix_multiline_parens, text)

        # Step 4j: Remove LLM meta-commentary / prompt leakage about placeholders
        text = re.sub(r':?\s*(?:Please\s+)?[Nn]ote\s+that\s+certain\s+placeholder.*?(?:\.\s*|\n)', '', text, flags=re.DOTALL)
        text = re.sub(r'\*?\*?(?:Note|Nota|NB)\s*:\s*.*?(?:placeholder|retained|symbol|figure|weren.?t specified).*?(?:\.\s*|\n)', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'[Pp]lease\s+note\s+that.*?(?:today|addressed|specified)\.?\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'since\s+actual\s+figures\s+weren.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'as\s+(?:requested|instructed).*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        # Step 4k: Fix stray asterisks around "et al." and author names
        text = re.sub(r'\*et\s+al\.\*', 'et al.', text)
        text = re.sub(r'\*\(', '(', text)  # *(text) -> (text)
        text = re.sub(r'\)\*', ')', text)  # (text)* -> (text)
        # Fix lone asterisks that aren't markdown bold: "Wang *et" -> "Wang et"
        text = re.sub(r'(\w)\s*\*(\w)', r'\1 \2', text)
        text = re.sub(r'(\w)\*\s*(\w)', r'\1 \2', text)
        # Clean remaining lone asterisks
        text = re.sub(r'(?<!\*)\*(?!\*)', '', text)

        # Step 4l: Fix caret/hat notation for tau-squared, I-squared, etc.
        text = re.sub(r'\^2\s*=', 'tau-squared =', text)
        text = re.sub(r'I\^2', 'I-squared', text)
        text = re.sub(r'I-Sq=', 'I-squared = ', text)
        text = re.sub(r'Tau\s*Sq=', 'tau-squared = ', text)

        # Step 4m: Standardize inline statistics formatting
        # "MD = +1.79, 95%CI[+1.03,+2.55]" -> clean format
        text = re.sub(r'95%CI\[', '95% CI [', text)
        text = re.sub(r'(\d),\+', r'\1, +', text)
        # Fix P<>.05 notation
        text = re.sub(r'P<>\.', 'P < .', text)
        text = re.sub(r'P=\*>', 'P > ', text)
        text = re.sub(r'P\*?=\*?>', 'P > ', text)

        # Step 5: Collapse excess newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 6: Remove trailing incomplete sentences
        if text and not text.rstrip().endswith(('.', '!', '?')):
            last_period = text.rfind('.')
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]

        text = text.strip()

        # Step 7: Enforce word limits (abstract, conclusion)
        text = self._enforce_word_limit(text, section_name)

        # Step 8: Ensure paragraph breaks
        text = self._enforce_paragraphs(text, section_name)

        return text

    def _clean_json(self, text: str) -> str:
        """Clean JSON from LLM output."""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)
        start = cleaned.find("{")
        if start < 0:
            return "{}"
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start: i + 1]
        return cleaned[start:] + "}"

    # ================================================================== #
    #  LATEX / PDF GENERATION                                              #
    # ================================================================== #

    def _latex_escape(self, text: str) -> str:
        """Escape LaTeX special characters in plain text.

        This should be called exactly ONCE on each piece of text before
        embedding it in the LaTeX document.
        """
        if not text:
            return ""
        # First strip any LaTeX commands the LLM may have generated
        # (these would break if we escape their backslashes)
        text = re.sub(r'\\citep?\{[^}]*\}', '', text)
        text = re.sub(r'\\citet?\{[^}]*\}', '', text)
        text = re.sub(r'\\textgreater\{\}', '>', text)
        text = re.sub(r'\\textless\{\}', '<', text)
        text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\\\XXX\b', '', text)
        text = re.sub(r'\\\\(?=\s|$)', '', text)
        # Remove any other remaining \command{...} patterns
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        # Remove stray backslash-commands without braces
        text = re.sub(r'\\[a-zA-Z]+\b', '', text)

        # Now escape special chars (order matters: & first since \& would be re-escaped)
        for ch in ['\\', '&', '%', '$', '#', '_', '{', '}']:
            text = text.replace(ch, f'\\{ch}')
        text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        return text

    def _build_latex_document(self, manuscript: Dict, references: List[Dict],
                              synthesis: Optional[Dict], extraction: Optional[Dict],
                              quality: Optional[Dict], protocol: Dict,
                              tex_dir: str) -> str:
        """Assemble complete .tex document."""
        rq = self._latex_escape(protocol.get("research_question", "Systematic Review"))
        date_str = time.strftime("%B %d, %Y")

        # Babel language for LaTeX
        babel_lang = BABEL_LANGUAGES.get(self._language, "english")
        babel_line = f"\\usepackage[{babel_lang}]{{babel}}" if babel_lang != "english" else ""

        preamble = rf"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
{babel_line}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{array}}
\usepackage{{parskip}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small Systematic Review}}
\fancyhead[R]{{\small Generated by SAM}}
\fancyfoot[C]{{\thepage}}

"""
        title_page = f"""\\title{{{rq}}}
\\author{{Generated by SAM --- Systematic Agent for Meta-analysis}}
\\date{{{date_str}}}

\\begin{{document}}
\\maketitle
\\thispagestyle{{empty}}
\\newpage
\\tableofcontents
\\newpage

"""
        # Sections
        section_map = {
            "abstract": ("Abstract", False),
            "introduction": ("Introduction", True),
            "methods": ("Methods", True),
            "results": ("Results", True),
            "discussion": ("Discussion", True),
            "conclusion": ("Conclusion", True),
        }

        body = ""
        for sec_key in SECTION_ORDER:
            label, numbered = section_map.get(sec_key, (sec_key.title(), True))
            text = manuscript.get(sec_key, "")
            # Pre-clean: strip HTML, markdown tables, URLs BEFORE LaTeX escaping
            text = re.sub(r'<sup>(.*?)</sup>', r'\1', text, flags=re.IGNORECASE)
            text = re.sub(r'<sub>(.*?)</sub>', r'\1', text, flags=re.IGNORECASE)
            text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
            text = re.sub(r'^[|\s]*[-:]+[|\s:\-]+$', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\(https?://[^\)]+\)', '', text)
            text = re.sub(r'https?://\S+', '', text)
            # Apply LaTeX escaping exactly once
            escaped = self._latex_escape(text)
            # Convert paragraph breaks to LaTeX paragraph spacing
            escaped = re.sub(r'\n\n+', r'\n\n\\medskip' + '\n\n', escaped)
            # Single newlines -> space (avoid LaTeX treating them oddly)
            escaped = re.sub(r'(?<!\n)\n(?!\n)', ' ', escaped)
            if numbered:
                body += f"\\section{{{label}}}\n\n{escaped}\n\n"
            else:
                body += f"\\section*{{{label}}}\n\\addcontentsline{{toc}}{{section}}{{{label}}}\n\n{escaped}\n\n"

            # After Results: insert forest plots inline, then funnel plots
            if sec_key == "results":
                forest_tex = self._build_forest_plots_inline(synthesis, tex_dir)
                if forest_tex:
                    body += forest_tex
                funnel_tex = self._build_funnel_plots_subsection(synthesis, tex_dir)
                if funnel_tex:
                    body += funnel_tex

        # PRISMA flow diagram (end of methods or start of results)
        prisma_fig_tex = self._build_prisma_figure(tex_dir)
        if prisma_fig_tex:
            body += prisma_fig_tex

        # Appendix
        body += "\\appendix\n\n"

        # Study Characteristics table
        if extraction:
            table = self._build_study_characteristics_table(extraction)
            if table:
                body += "\\section{Study Characteristics}\n\n" + table + "\n\n"

        # Quality Assessment table
        if quality:
            table = self._build_quality_table(quality)
            if table:
                body += "\\section{Quality Assessment}\n\n" + table + "\n\n"

        # RoB plots (traffic light + summary)
        rob_plots_tex = self._build_rob_plots_section(quality, tex_dir)
        if rob_plots_tex:
            body += rob_plots_tex

        # GRADE table
        if synthesis:
            table = self._build_grade_table(synthesis)
            if table:
                body += "\\section{GRADE Summary}\n\n" + table + "\n\n"

        # References
        refs_tex = self._build_references_section(references)
        body += refs_tex

        body += "\n\\end{document}\n"

        return preamble + title_page + body

    # Friendly display names for clinical outcome categories
    _CATEGORY_DISPLAY = {
        "weight": "Weight Change",
        "waist": "Waist Circumference",
        "glycemic": "Glycemic Control",
        "lipids": "Lipid Profile",
        "blood_pressure": "Blood Pressure",
        "dietary_intake": "Dietary Intake",
        "physical_activity": "Physical Activity",
        "adherence": "Treatment Adherence",
        "quality_of_life": "Quality of Life",
        "other": "Other Outcomes",
    }

    def _category_label(self, key: str) -> str:
        """Map outcome group key to a human-readable label."""
        # Try clinical category display name first, then uppercase the key
        return self._CATEGORY_DISPLAY.get(key, key.replace("_", " ").title())

    def _build_forest_plots_inline(self, synthesis: Optional[Dict], tex_dir: str) -> str:
        """Build LaTeX for forest plots placed inline after Results text."""
        if not synthesis or not synthesis.get("analyses"):
            return ""

        parts = []
        fig_num = 1
        for group_key, analysis in synthesis.get("analyses", {}).items():
            plots = analysis.get("plots", {})
            plot_path = plots.get("forest")
            if not plot_path or not os.path.exists(plot_path):
                continue
            fname = os.path.basename(plot_path)
            label = f"forest_{group_key}".replace(" ", "_").lower()
            display = self._category_label(group_key)
            measure = analysis.get("measure_type", analysis.get("model", {}).get("measure", group_key))
            k = analysis.get("model", {}).get("k_studies", "")
            caption = f"Forest plot: {display} ({measure}, k={k} studies)"
            # Scale down large plots (>20 studies) to fit on page
            width = "0.85" if k and int(k) > 20 else "0.9"
            parts.append(f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width={width}\\textwidth]{{{fname}}}
\\caption{{{self._latex_escape(caption)}}}
\\label{{fig:{label}}}
\\end{{figure}}

""")
            fig_num += 1
        return "".join(parts)

    def _build_funnel_plots_subsection(self, synthesis: Optional[Dict], tex_dir: str) -> str:
        """Build LaTeX subsection for funnel plots (publication bias assessment)."""
        if not synthesis or not synthesis.get("analyses"):
            return ""

        parts = []
        for group_key, analysis in synthesis.get("analyses", {}).items():
            plots = analysis.get("plots", {})
            plot_path = plots.get("funnel")
            if not plot_path or not os.path.exists(plot_path):
                continue
            fname = os.path.basename(plot_path)
            label = f"funnel_{group_key}".replace(" ", "_").lower()
            display = self._category_label(group_key)
            caption = f"Funnel plot: {display} (publication bias assessment)"
            parts.append(f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.75\\textwidth]{{{fname}}}
\\caption{{{self._latex_escape(caption)}}}
\\label{{fig:{label}}}
\\end{{figure}}

""")

        if not parts:
            return ""
        return "\\subsection{Publication Bias}\n\n" + "".join(parts)

    def _build_plots_section(self, synthesis: Optional[Dict], tex_dir: str) -> str:
        """Legacy method — plots are now inline in Results. Returns empty string."""
        return ""

    def _build_prisma_figure(self, tex_dir: str) -> str:
        """Build LaTeX figure for PRISMA flow diagram if it exists."""
        # Look for prisma_flow.png in the output directory's plots subfolder
        out = getattr(self, '_output_dir', 'outputs')
        candidates = [
            os.path.join(out, "plots", "prisma_flow.png"),
            os.path.join(out, "prisma_flow.png"),
        ]
        for prisma_path in candidates:
            if os.path.exists(prisma_path):
                fname = "prisma_flow.png"
                dest = os.path.join(tex_dir, fname)
                shutil.copy2(prisma_path, dest)
                return f"""\\subsection{{PRISMA Flow Diagram}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{{fname}}}
\\caption{{PRISMA 2020 Flow Diagram}}
\\label{{fig:prisma_flow}}
\\end{{figure}}

"""
        return ""

    def _build_rob_plots_section(self, quality: Optional[Dict], tex_dir: str) -> str:
        """Build LaTeX section for RoB traffic light and summary plots."""
        if not quality:
            return ""

        parts = []

        # Look for RoB plots
        out = getattr(self, '_output_dir', 'outputs')
        rob_tl_path = quality.get("rob_traffic_light_plot") or os.path.join(out, "plots", "rob_traffic_light.png")
        rob_sum_path = quality.get("rob_summary_plot") or os.path.join(out, "plots", "rob_summary.png")

        has_plots = False
        if os.path.exists(rob_tl_path):
            fname = "rob_traffic_light.png"
            shutil.copy2(rob_tl_path, os.path.join(tex_dir, fname))
            parts.append(f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{fname}}}
\\caption{{Risk of Bias Traffic Light Plot}}
\\label{{fig:rob_traffic_light}}
\\end{{figure}}

""")
            has_plots = True

        if os.path.exists(rob_sum_path):
            fname = "rob_summary.png"
            shutil.copy2(rob_sum_path, os.path.join(tex_dir, fname))
            parts.append(f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{{{fname}}}
\\caption{{Risk of Bias Summary}}
\\label{{fig:rob_summary}}
\\end{{figure}}

""")
            has_plots = True

        if has_plots:
            return "\\section{Risk of Bias Figures}\n\n" + "".join(parts)
        return ""

    def _copy_plots_to_texdir(self, synthesis: Optional[Dict], tex_dir: str):
        """Copy plot PNGs to the LaTeX build directory."""
        if not synthesis or not synthesis.get("analyses"):
            return
        for measure, analysis in synthesis.get("analyses", {}).items():
            plots = analysis.get("plots", {})
            for plot_type in ("forest", "funnel"):
                plot_path = plots.get(plot_type)
                if plot_path and os.path.exists(plot_path):
                    dest = os.path.join(tex_dir, os.path.basename(plot_path))
                    shutil.copy2(plot_path, dest)

    def _build_study_characteristics_table(self, extraction: Dict) -> str:
        """Build longtable for study characteristics from A5 data."""
        papers = extraction.get("papers", [])
        if not papers:
            return ""

        rows = []
        for p in papers:
            sc = p.get("study_characteristics", {}) or {}
            ie = p.get("intervention_exposure", {}) or {}
            outcomes = p.get("outcomes", []) or []

            # Outcome names — try both field name variants
            outcome_parts = []
            for o in outcomes[:2]:
                oname = o.get("name") or o.get("outcome_name") or ""
                if oname and oname != "?":
                    outcome_parts.append(oname)
            outcome_str = ", ".join(outcome_parts) if outcome_parts else "Not reported"
            if len(outcomes) > 2:
                outcome_str += f" (+{len(outcomes)-2})"

            # Intervention — try both field name variants
            intervention = ie.get("name") or ie.get("intervention_name") or "Not reported"

            # Sample size — handle None/null
            sample = sc.get("sample_size")
            sample_str = str(sample) if sample and sample != "?" else "NR"

            # Design
            design = sc.get("study_design") or "NR"

            # Follow-up
            followup = sc.get("followup_duration") or "NR"

            row = [
                str(p.get("pmid", "?")),
                design,
                sample_str,
                intervention,
                outcome_str,
                followup,
            ]
            rows.append(" & ".join(self._latex_escape(str(c))[:40] for c in row) + " \\\\")

        header = (
            "\\begin{longtable}{p{1.5cm}p{2cm}p{1cm}p{3cm}p{3.5cm}p{2cm}}\n"
            "\\toprule\n"
            "PMID & Design & N & Intervention & Outcome & Follow-up \\\\\n"
            "\\midrule\n"
            "\\endhead\n"
        )
        footer = "\\bottomrule\n\\end{longtable}\n"
        return header + "\n".join(rows) + "\n" + footer

    def _build_quality_table(self, quality: Dict) -> str:
        """Build longtable for quality assessment from A6 data."""
        papers = quality.get("papers", [])
        if not papers:
            return ""

        rows = []
        for p in papers:
            qa = p.get("quality_assessment", {})
            row = [
                str(p.get("pmid", "?")),
                qa.get("tool", "?"),
                qa.get("overall", "?"),
                str(p.get("quality_weight", "?")),
                (qa.get("overall_justification", "") or "")[:50],
            ]
            rows.append(" & ".join(self._latex_escape(str(c)) for c in row) + " \\\\")

        header = (
            "\\begin{longtable}{p{1.5cm}p{1.5cm}p{2.5cm}p{1.5cm}p{5cm}}\n"
            "\\toprule\n"
            "PMID & Tool & Overall & Weight & Justification \\\\\n"
            "\\midrule\n"
            "\\endhead\n"
        )
        footer = "\\bottomrule\n\\end{longtable}\n"
        return header + "\n".join(rows) + "\n" + footer

    def _build_grade_table(self, synthesis: Dict) -> str:
        """Build longtable for GRADE summary from A7 data."""
        analyses = synthesis.get("analyses", {})
        if not analyses:
            return ""

        rows = []
        for measure, analysis in analyses.items():
            model = analysis.get("model", {})
            het = analysis.get("heterogeneity", {})
            grade = analysis.get("grade", {})
            pooled = f'{model.get("pooled_estimate", "?")} [{model.get("ci_lower", "?")}, {model.get("ci_upper", "?")}]'
            i2 = f'{het.get("I2", "?")}\\%'
            grade_level = grade.get("final_level", "?")
            downgrades = str(grade.get("total_downgrades", 0))
            row = [
                measure,
                str(model.get("k_studies", "?")),
                pooled,
                i2,
                grade_level,
                downgrades,
            ]
            rows.append(" & ".join(self._latex_escape(str(c)) for c in row) + " \\\\")

        header = (
            "\\begin{longtable}{p{2cm}p{1.5cm}p{3.5cm}p{1.5cm}p{2cm}p{2cm}}\n"
            "\\toprule\n"
            "Measure & Studies & Pooled [95\\% CI] & I$^2$ & GRADE & Downgrades \\\\\n"
            "\\midrule\n"
            "\\endhead\n"
        )
        footer = "\\bottomrule\n\\end{longtable}\n"
        return header + "\n".join(rows) + "\n" + footer

    def _build_references_section(self, references: List[Dict]) -> str:
        """Build numbered references list."""
        if not references:
            return "\\section*{References}\n\\addcontentsline{toc}{section}{References}\n\nNo references.\n"

        items = []
        for ref in references:
            cite = self._latex_escape(ref.get("citation", "?"))
            title = self._latex_escape(ref.get("title", "?"))
            journal = self._latex_escape(ref.get("journal", "?"))
            pmid = ref.get("pmid", "?")
            items.append(f"  \\item {cite}. \\textit{{{title}}}. {journal}. PMID: {pmid}")

        return (
            "\\section*{References}\n"
            "\\addcontentsline{toc}{section}{References}\n\n"
            "\\begin{enumerate}\n"
            + "\n".join(items) + "\n"
            "\\end{enumerate}\n"
        )

    def _generate_pdf(self, manuscript: Dict, references: List[Dict],
                      protocol: Dict, extraction: Optional[Dict],
                      quality: Optional[Dict], synthesis: Optional[Dict]) -> Optional[str]:
        """Write .tex, compile with pdflatex, return PDF path or None."""
        try:
            # Create build directory
            out = getattr(self, '_output_dir', 'outputs')
            os.makedirs(out, exist_ok=True)
            tex_dir = tempfile.mkdtemp(prefix="sam_latex_", dir=out)

            # Copy plots
            self._copy_plots_to_texdir(synthesis, tex_dir)

            # Build LaTeX source
            tex_source = self._build_latex_document(
                manuscript, references, synthesis, extraction, quality, protocol, tex_dir
            )

            tex_path = os.path.join(tex_dir, "manuscript.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_source)

            # Compile (2 passes for TOC)
            for pass_num in range(2):
                try:
                    result = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
                        cwd=tex_dir,
                        capture_output=True,
                        timeout=60,
                    )
                except subprocess.TimeoutExpired:
                    print(f"    [!] pdflatex pass {pass_num+1} timed out after 60s")
                    break
                if result.returncode != 0 and pass_num == 1:
                    print(f"    [!] pdflatex pass {pass_num+1} returned {result.returncode}")
                    # Still try to use the PDF if it was generated
                    log_path = os.path.join(tex_dir, "manuscript.log")
                    if os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as lf:
                            log_tail = lf.read()[-500:]
                        print(f"    [!] Log tail: {log_tail}")

            # Move PDF to output dir
            pdf_src = os.path.join(tex_dir, "manuscript.pdf")
            if os.path.exists(pdf_src):
                pdf_dest = os.path.join(out, "manuscript.pdf")
                shutil.move(pdf_src, pdf_dest)
                print(f"  PDF generated: {pdf_dest} ({os.path.getsize(pdf_dest)} bytes)")

                # Keep .tex for debugging, cleanup temp dir
                shutil.copy2(tex_path, os.path.join(out, "manuscript.tex"))
                shutil.rmtree(tex_dir, ignore_errors=True)

                return pdf_dest
            else:
                print("    [!] PDF not generated — pdflatex may have failed")
                return None

        except Exception as e:
            print(f"    [!] PDF generation failed: {e}")
            return None

    # ================================================================== #
    #  SUMMARY                                                             #
    # ================================================================== #

    def _print_summary(self, result: Dict):
        stats = result["stats"]
        cv = result.get("citation_verification", {})
        print(f"\n{'='*60}")
        print("AGENT 8 - MANUSCRIPT SUMMARY")
        print(f"{'='*60}")
        print(f"  RAG chunks total : {stats['total_chunks']}")
        print(f"  Papers cited     : {stats['total_papers_cited']}")
        for sec, n in stats["chunks_per_section"].items():
            word_count = len(result["manuscript"].get(sec, "").split())
            sec_cv = cv.get("per_section", {}).get(sec, {})
            cv_str = ""
            if sec_cv.get("total", 0) > 0:
                cv_str = f", {sec_cv['verified']}/{sec_cv['total']} citations verified"
                if sec_cv.get("removed", 0) > 0:
                    cv_str += f" ({sec_cv['removed']} removed)"
            print(f"    {sec:15s}: {word_count:4d} words, {n} RAG chunks{cv_str}")
        print(f"  References       : {len(result['references'])}")
        if cv:
            print(f"  Citations total  : {cv.get('total_citations', 0)} "
                  f"(verified: {cv.get('total_verified', 0)}, "
                  f"removed: {cv.get('total_removed', 0)})")
        print(f"  Total time       : {stats['elapsed_s']}s")


# ====================================================================== #
#  STANDALONE TEST                                                         #
# ====================================================================== #
if __name__ == "__main__":
    print("Use test_agent8.py for standalone testing.")
