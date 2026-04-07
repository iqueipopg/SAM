"""Agent 4: Full-Text Retrieval + PRISMA Full-Text Screening (Qwen2.5-7B)

Two phases:
  Phase A — Retrieve full text: PMC XML → Unpaywall PDF → Europe PMC → Semantic Scholar
            → OpenAlex → CrossRef → DOAJ → The Lens → CORE → Sci-Hub → abstract fallback
  Phase B — Full-text PRISMA screening via LLM (ensemble voting, configurable)
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from agents.retry import retry_with_backoff
from Bio import Entrez
import json, torch, re, time, io, requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple

# Entrez config: env var takes priority, then fallback
Entrez.email = os.environ.get("ENTREZ_EMAIL", "meta-analysis@example.com")

# Section title mapping for PMC JATS XML
_SECTION_MAP = {
    "background": ["background", "introduction", "intro"],
    "methods": [
        "methods",
        "method",
        "materials and methods",
        "study design",
        "participants",
        "procedure",
        "statistical analysis",
    ],
    "results": ["results", "findings", "outcome"],
    "discussion": ["discussion", "interpretation"],
    "conclusions": ["conclusions", "conclusion", "summary"],
}

# Sections to skip
_SKIP_SECTIONS = {
    "references",
    "bibliography",
    "supplementary",
    "supplement",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "competing interests",
    "conflict of interest",
    "author contributions",
    "data availability",
}


class Agent4FullText(BaseAgent):
    """Full-text retrieval (10-source cascade) and LLM-based full-text screening.

    Input: list of included papers (from A3) and protocol (from A1).
    Output: dict with 'include' and 'exclude' lists, each paper annotated with text_source.
    GPU: yes -- uses LLM for full-text PRISMA screening (Phase B).
    """

    def __init__(
        self,
        ensemble_size: int = 1,
        voting_threshold: float = 0.60,
    ):
        super().__init__("agent4_validation")  # config.yaml key
        self.ensemble_size = ensemble_size
        self.voting_threshold = voting_threshold
        self._last_ncbi_call = 0.0
        self._last_unpaywall_call = 0.0
        self._last_semanticscholar_call = 0.0

        # Enhanced retrieval config
        retrieval_cfg = self.config.get("retrieval", {})
        self.europepmc_enabled = retrieval_cfg.get("europepmc_enabled", True)
        self.semanticscholar_enabled = retrieval_cfg.get("semanticscholar_enabled", True)
        self.openalex_enabled = retrieval_cfg.get("openalex_enabled", True)
        self.crossref_enabled = retrieval_cfg.get("crossref_enabled", True)
        self.doaj_enabled = retrieval_cfg.get("doaj_enabled", True)
        self.lens_enabled = retrieval_cfg.get("lens_enabled", True)
        self.core_enabled = retrieval_cfg.get("core_enabled", True)
        self.scihub_enabled = retrieval_cfg.get("scihub_enabled", True)
        self.scihub_domain = retrieval_cfg.get("scihub_domain", "sci-hub.se")
        self.google_scholar_enabled = retrieval_cfg.get("google_scholar_enabled", True)

        # API keys for token-gated sources (env var takes priority)
        self._lens_api_token = os.environ.get("LENS_API_TOKEN") or retrieval_cfg.get("lens_api_token", "")
        self._core_api_key = os.environ.get("CORE_API_KEY") or retrieval_cfg.get("core_api_key", "")

        # Rate limit timestamps for new sources
        self._last_openalex_call = 0.0
        self._last_crossref_call = 0.0
        self._last_doaj_call = 0.0
        self._last_lens_call = 0.0
        self._last_core_call = 0.0
        self._last_google_scholar_call = 0.0

        # Rate limit intervals (from config or defaults)
        rate_cfg = retrieval_cfg.get("rate_limits", {})
        self._openalex_interval = rate_cfg.get("openalex_seconds", 0.2)
        self._crossref_interval = rate_cfg.get("crossref_seconds", 0.5)
        self._doaj_interval = rate_cfg.get("doaj_seconds", 0.5)
        self._lens_interval = rate_cfg.get("lens_seconds", 1.0)
        self._core_interval = rate_cfg.get("core_seconds", 0.1)
        self._google_scholar_interval = rate_cfg.get("google_scholar_seconds", 3.0)

    # ================================================================== #
    #  PUBLIC API                                                          #
    # ================================================================== #

    def run(
        self,
        included_papers: List[Dict],
        a2_papers: List[Dict],
        protocol: Dict,
    ) -> Dict:
        """
        Full pipeline: retrieve full texts → screen with LLM.
        Returns enriched papers with fulltext_decision.
        """
        print(f"\n{'='*60}")
        print("AGENT 4 - FULL-TEXT RETRIEVAL + PRISMA SCREENING")
        print(f"{'='*60}")
        print(f"  Papers to process: {len(included_papers)}")

        start = time.time()

        # Phase A: Retrieve full texts (no GPU needed)
        papers = self.retrieve_fulltext(included_papers, a2_papers)

        # Phase B: Full-text screening (needs GPU)
        result = self.screen_fulltext(papers, protocol)

        elapsed = round(time.time() - start, 1)
        result["stats"]["elapsed_s"] = elapsed

        self._print_summary(result)
        return result

    # ================================================================== #
    #  PHASE A: FULL-TEXT RETRIEVAL                                        #
    # ================================================================== #

    def retrieve_fulltext(
        self,
        included_papers: List[Dict],
        a2_papers: List[Dict],
    ) -> List[Dict]:
        """Enrich included papers with full text from PMC / Unpaywall / abstract."""
        print("\n--- Phase A: Full-Text Retrieval ---")

        # Build lookup: pmid → a2 paper (has abstract, authors, etc.)
        a2_lookup = {str(p.get("pmid", "")): p for p in a2_papers}

        # Collect PMIDs
        pmids = [str(p.get("pmid", "")) for p in included_papers if p.get("pmid")]

        # Batch-fetch metadata (DOI, PMCID) from PubMed
        metadata = self._batch_fetch_metadata(pmids)

        enriched = []
        stats = {
            "pmc_fulltext": 0, "unpaywall_pdf": 0,
            "europepmc_fulltext": 0, "semanticscholar_pdf": 0,
            "openalex_pdf": 0, "crossref_pdf": 0, "doaj_pdf": 0,
            "lens_fulltext": 0, "core_fulltext": 0,
            "google_scholar_pdf": 0,
            "scihub_pdf": 0, "abstract_only": 0,
        }

        for paper in included_papers:
            pmid = str(paper.get("pmid", ""))
            a2 = a2_lookup.get(pmid, {})

            # Merge A3 screening data with A2 paper data
            enriched_paper = {
                "pmid": pmid,
                "title": paper.get("title", a2.get("title", "")),
                "abstract": a2.get("abstract", ""),
                "authors": a2.get("authors", []),
                "journal": a2.get("journal", ""),
                "year": a2.get("year", ""),
                "doi": metadata.get(pmid, {}).get("doi", ""),
                "pmcid": metadata.get(pmid, {}).get("pmcid", ""),
                "full_text": None,
                "text_source": "abstract_only",
                "full_text_length": 0,
                "sections": {},
                "oa_url": None,
            }

            try:
                self._retrieve_single(enriched_paper, stats)
            except Exception as e:
                print(f"    [!] Error retrieving {pmid}: {e}")

            # Parse sections from plain text if source didn't provide them
            if (enriched_paper.get("full_text")
                    and not enriched_paper.get("sections")
                    and enriched_paper.get("text_source") != "abstract_only"):
                sections = self._parse_sections_from_text(enriched_paper["full_text"])
                if sections:
                    enriched_paper["sections"] = sections
                    n_secs = [k for k in sections if k not in ("references",)]
                    print(f"    [{pmid}] Parsed {len(n_secs)} sections: {', '.join(n_secs)}")

            enriched.append(enriched_paper)

        print(
            f"\n  Retrieval summary: "
            f"PMC={stats['pmc_fulltext']}, "
            f"Unpaywall={stats['unpaywall_pdf']}, "
            f"EuropePMC={stats['europepmc_fulltext']}, "
            f"S2={stats['semanticscholar_pdf']}, "
            f"OpenAlex={stats['openalex_pdf']}, "
            f"CrossRef={stats['crossref_pdf']}, "
            f"DOAJ={stats['doaj_pdf']}, "
            f"Lens={stats['lens_fulltext']}, "
            f"CORE={stats['core_fulltext']}, "
            f"GScholar={stats['google_scholar_pdf']}, "
            f"SciHub={stats['scihub_pdf']}, "
            f"Abstract-only={stats['abstract_only']}"
        )

        return enriched

    def _retrieve_single(self, paper: Dict, stats: Dict):
        """Try PMC → Unpaywall → abstract fallback for a single paper."""
        pmid = paper["pmid"]
        paper["title"][:60]

        # 1. Try PMC full-text XML
        pmcid = paper.get("pmcid", "")
        if pmcid:
            print(f"  [{pmid}] Trying PMC ({pmcid})... ", end="", flush=True)
            full_text, sections = self._fetch_pmc_fulltext(pmcid)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "pmc"
                paper["full_text_length"] = len(full_text)
                paper["sections"] = sections or {}
                stats["pmc_fulltext"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("no body (publisher restricted)")

        # 2. Try Unpaywall PDF
        doi = paper.get("doi", "")
        if doi:
            print(f"  [{pmid}] Trying Unpaywall ({doi})... ", end="", flush=True)
            full_text, oa_url = self._fetch_unpaywall_pdf(doi)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "unpaywall_pdf"
                paper["full_text_length"] = len(full_text)
                paper["oa_url"] = oa_url
                stats["unpaywall_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 3. Try Europe PMC full-text XML
        if self.europepmc_enabled:
            print(f"  [{pmid}] Trying Europe PMC... ", end="", flush=True)
            full_text, sections = self._fetch_europepmc_fulltext(pmid)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "europepmc"
                paper["full_text_length"] = len(full_text)
                paper["sections"] = sections or {}
                stats["europepmc_fulltext"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 4. Try Semantic Scholar open-access PDF
        if self.semanticscholar_enabled:
            print(f"  [{pmid}] Trying Semantic Scholar... ", end="", flush=True)
            full_text = self._fetch_semanticscholar_pdf(pmid)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "semanticscholar_pdf"
                paper["full_text_length"] = len(full_text)
                stats["semanticscholar_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 5. Try OpenAlex
        if self.openalex_enabled:
            print(f"  [{pmid}] Trying OpenAlex... ", end="", flush=True)
            full_text = self._try_openalex(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "openalex_pdf"
                paper["full_text_length"] = len(full_text)
                stats["openalex_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 6. Try CrossRef
        if self.crossref_enabled and doi:
            print(f"  [{pmid}] Trying CrossRef ({doi})... ", end="", flush=True)
            full_text = self._try_crossref(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "crossref_pdf"
                paper["full_text_length"] = len(full_text)
                stats["crossref_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 7. Try DOAJ
        if self.doaj_enabled and doi:
            print(f"  [{pmid}] Trying DOAJ ({doi})... ", end="", flush=True)
            full_text = self._try_doaj(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "doaj_pdf"
                paper["full_text_length"] = len(full_text)
                stats["doaj_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 8. Try The Lens
        if self.lens_enabled:
            print(f"  [{pmid}] Trying The Lens... ", end="", flush=True)
            full_text = self._try_lens(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "lens_fulltext"
                paper["full_text_length"] = len(full_text)
                stats["lens_fulltext"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 9. Try CORE
        if self.core_enabled:
            print(f"  [{pmid}] Trying CORE... ", end="", flush=True)
            full_text = self._try_core(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "core_fulltext"
                paper["full_text_length"] = len(full_text)
                stats["core_fulltext"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 10. Try Google Scholar (search by title → find PDF link)
        if self.google_scholar_enabled:
            print(f"  [{pmid}] Trying Google Scholar... ", end="", flush=True)
            full_text = self._try_google_scholar(paper)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "google_scholar_pdf"
                paper["full_text_length"] = len(full_text)
                stats["google_scholar_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 11. Try Sci-Hub PDF
        if self.scihub_enabled and doi:
            print(f"  [{pmid}] Trying Sci-Hub ({doi})... ", end="", flush=True)
            full_text = self._fetch_scihub_pdf(doi)
            if full_text:
                paper["full_text"] = full_text
                paper["text_source"] = "scihub_pdf"
                paper["full_text_length"] = len(full_text)
                stats["scihub_pdf"] += 1
                print(f"OK ({len(full_text)} chars)")
                return
            else:
                print("not available")

        # 11. Abstract fallback
        abstract = paper.get("abstract", "")
        if abstract:
            paper["full_text"] = abstract
            paper["full_text_length"] = len(abstract)
        stats["abstract_only"] += 1
        print(f"  [{pmid}] Using abstract fallback ({len(abstract)} chars)")

    # ------------------------------------------------------------------ #
    #  Section parsing from plain text                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_sections_from_text(text: str) -> Dict[str, str]:
        """Parse academic paper sections from plain text (PDF-extracted).

        Detects common section headings (Introduction, Methods, Results, etc.)
        and splits the text into a dict of {section_name: content}.
        """
        if not text or len(text) < 500:
            return {}

        import re

        # Patterns for section headings (case-insensitive)
        # Matches: "3. Results", "RESULTS", "Results", "3 Results", "III. Results"
        # Only match major section headings (not sub-headings like Participants)
        heading_pattern = re.compile(
            r'(?:^|\n)'                          # start of line
            r'[ \t]*'                            # optional leading whitespace
            r'(?:\d+\.?\s*|[IVX]+\.?\s*)?'       # optional numbering
            r'(Introduction|Background|'
            r'Methods?|Materials?\s*(?:and|&)\s*Methods?|'
            r'Results?|Findings|'
            r'Discussion|'
            r'Conclusions?|'
            r'References|Bibliography|'
            r'Abstract)'
            r'[ \t]*\n',
            re.IGNORECASE | re.MULTILINE,
        )

        # Map detected headings to canonical section names
        canonical_map = {
            "introduction": "introduction", "background": "introduction",
            "method": "methods", "methods": "methods",
            "materials and methods": "methods", "materials & methods": "methods",
            "result": "results", "results": "results", "findings": "results",
            "discussion": "discussion",
            "conclusion": "conclusion", "conclusions": "conclusion",
            "abstract": "abstract",
            "references": "references", "bibliography": "references",
        }

        matches = list(heading_pattern.finditer(text))
        if not matches:
            return {}

        sections = {}
        for i, match in enumerate(matches):
            heading = match.group(1).strip().lower()
            canonical = canonical_map.get(heading, heading)

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            # If same canonical name appears multiple times, append
            if canonical in sections:
                sections[canonical] += "\n\n" + content
            else:
                sections[canonical] = content

        # Skip if we only found references or abstract
        meaningful = {k: v for k, v in sections.items()
                      if k not in ("references", "abstract", "bibliography")}
        if not meaningful:
            return {}

        return sections

    # ------------------------------------------------------------------ #
    #  Metadata fetch (DOI + PMCID)                                        #
    # ------------------------------------------------------------------ #

    def _batch_fetch_metadata(self, pmids: List[str]) -> Dict[str, Dict]:
        """Fetch DOI and PMCID for a list of PMIDs from PubMed XML."""
        if not pmids:
            return {}

        print(f"  Fetching metadata for {len(pmids)} PMIDs...")
        self._rate_limit_ncbi()

        try:
            def _fetch_meta():
                h = Entrez.efetch(
                    db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml"
                )
                data = h.read()
                h.close()
                return data

            xml_bytes = retry_with_backoff(
                _fetch_meta,
                retryable_exceptions=(Exception,),
                on_retry=lambda a, e, d: print(f"    [!] Metadata fetch retry {a+1}: {e}"),
            )
        except Exception as e:
            print(f"    [!] Metadata fetch failed: {e}")
            return {}

        result = {}
        try:
            root = ET.fromstring(xml_bytes)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None:
                    continue
                pmid = pmid_el.text

                meta = {"doi": "", "pmcid": ""}

                # Search ArticleIdList for DOI and PMC
                for aid in article.findall(".//ArticleId"):
                    id_type = aid.get("IdType", "")
                    if id_type == "doi" and aid.text:
                        meta["doi"] = aid.text
                    elif id_type == "pmc" and aid.text:
                        meta["pmcid"] = aid.text

                result[pmid] = meta
                print(
                    f"    {pmid}: DOI={meta['doi'] or '—'}, PMCID={meta['pmcid'] or '—'}"
                )
        except ET.ParseError as e:
            print(f"    [!] XML parse error: {e}")

        return result

    # ------------------------------------------------------------------ #
    #  PMC Full-Text                                                       #
    # ------------------------------------------------------------------ #

    def _fetch_pmc_fulltext(self, pmcid: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Fetch full-text from PMC via Entrez, parse JATS XML."""
        self._rate_limit_ncbi()
        try:
            def _pmc_fetch():
                h = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
                data = h.read()
                h.close()
                return data

            xml_bytes = retry_with_backoff(
                _pmc_fetch,
                retryable_exceptions=(Exception,),
                on_retry=lambda a, e, d: print(f"PMC retry {a+1}: {e} "),
            )
            return self._parse_pmc_sections(xml_bytes)
        except Exception as e:
            print(f"PMC fetch error: {e}")
            return None, None

    def _parse_pmc_sections(
        self, xml_bytes: bytes
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse PMC JATS XML into structured sections."""
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return None, None

        # Find <body> element
        body = root.find(".//body")
        if body is None:
            return None, None

        sections = {}
        all_text_parts = []

        for sec in body.findall(".//sec"):
            title_el = sec.find("title")
            sec_title = (
                (title_el.text or "").strip().lower() if title_el is not None else ""
            )

            # Skip non-content sections
            if any(skip in sec_title for skip in _SKIP_SECTIONS):
                continue

            # Extract all text from this section (including nested elements)
            sec_text = "".join(sec.itertext()).strip()
            if not sec_text:
                continue

            all_text_parts.append(sec_text)

            # Map to standard section name
            for standard_name, aliases in _SECTION_MAP.items():
                if any(alias in sec_title for alias in aliases):
                    if standard_name in sections:
                        sections[standard_name] += "\n\n" + sec_text
                    else:
                        sections[standard_name] = sec_text
                    break

        if not all_text_parts:
            # Try getting all text from body as fallback
            body_text = "".join(body.itertext()).strip()
            if body_text:
                all_text_parts.append(body_text)

        full_text = "\n\n".join(all_text_parts) if all_text_parts else None
        return full_text, sections if sections else None

    # ------------------------------------------------------------------ #
    #  Unpaywall PDF                                                       #
    # ------------------------------------------------------------------ #

    def _fetch_unpaywall_pdf(self, doi: str) -> Tuple[Optional[str], Optional[str]]:
        """Query Unpaywall for OA PDF, download and extract text."""
        oa_info = self._query_unpaywall(doi)
        if not oa_info or not oa_info.get("url_for_pdf"):
            return None, None

        pdf_url = oa_info["url_for_pdf"]
        try:
            resp = requests.get(
                pdf_url,
                timeout=30,
                headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if resp.status_code != 200:
                return None, None
            text = self._extract_pdf_text(resp.content)
            return (text, pdf_url) if text else (None, None)
        except Exception as e:
            print(f"PDF download error: {e}")
            return None, None

    def _query_unpaywall(self, doi: str) -> Optional[Dict]:
        """Query Unpaywall API for OA availability."""
        self._rate_limit_unpaywall()
        _uw_email = os.environ.get("UNPAYWALL_EMAIL") or self.config.get("retrieval", {}).get("unpaywall_email", "meta-analysis@example.com")
        url = f"https://api.unpaywall.org/v2/{doi}?email={_uw_email}"
        try:
            def _unpaywall_req():
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                return r.json()

            data = retry_with_backoff(
                _unpaywall_req,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"Unpaywall retry {a+1}: {e} "),
            )
            best = data.get("best_oa_location") or {}
            return {
                "is_oa": data.get("is_oa", False),
                "url_for_pdf": best.get("url_for_pdf"),
                "oa_url": best.get("url"),
            }
        except Exception:
            return None

    def _extract_pdf_text(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes. Tries PyMuPDF (fitz) first, falls back to pdfplumber.

        If the primary backend yields fewer than 500 characters (common with
        scanned / image-heavy PDFs), the fallback backend is attempted.
        """
        MIN_CHARS = 500
        text = None

        # Backend 1: PyMuPDF (fitz) — fast, handles most PDFs well
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    pages_text.append(page_text)
            doc.close()
            if pages_text:
                text = "\n\n".join(pages_text)
        except Exception as e:
            print(f"PyMuPDF error: {e} ", end="", flush=True)

        # Backend 2: pdfplumber — better with complex layouts / tables
        if not text or len(text) < MIN_CHARS:
            try:
                import pdfplumber

                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    if pages_text:
                        plumber_text = "\n\n".join(pages_text)
                        # Use pdfplumber result if it's longer
                        if not text or len(plumber_text) > len(text):
                            text = plumber_text
            except Exception as e:
                print(f"pdfplumber error: {e} ", end="", flush=True)

        # Backend 3: Extract structured tables and append
        # (pdfplumber table extraction preserves column alignment that raw text loses)
        if text and len(text) >= MIN_CHARS:
            try:
                import pdfplumber

                table_parts = []
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_tables = page.extract_tables()
                        if not page_tables:
                            continue
                        for table in page_tables:
                            if not table or len(table) < 2:
                                continue
                            rows = []
                            for row in table:
                                cells = [str(c).strip() if c else "" for c in row]
                                if any(cells):
                                    rows.append(" | ".join(cells))
                            if rows:
                                table_parts.append(
                                    f"\nTABLE (page {page_num}):\n" + "\n".join(rows)
                                )
                if table_parts:
                    text += "\n\n--- EXTRACTED TABLES ---\n" + "\n".join(table_parts)
            except Exception:
                pass  # Table extraction is best-effort

        return text if text and len(text.strip()) > 0 else None

    # ------------------------------------------------------------------ #
    #  Europe PMC Full-Text                                                #
    # ------------------------------------------------------------------ #

    def _fetch_europepmc_fulltext(self, pmid: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Fetch full-text XML from Europe PMC REST API, reuse _parse_pmc_sections."""
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/fullTextXML"
        try:
            def _epmc_req():
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                return r.content

            xml_bytes = retry_with_backoff(
                _epmc_req,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"Europe PMC retry {a+1}: {e} "),
            )
            if not xml_bytes or len(xml_bytes) < 200:
                return None, None
            return self._parse_pmc_sections(xml_bytes)
        except Exception as e:
            print(f"Europe PMC error: {e}")
            return None, None

    # ------------------------------------------------------------------ #
    #  Semantic Scholar PDF                                                #
    # ------------------------------------------------------------------ #

    def _fetch_semanticscholar_pdf(self, pmid: str) -> Optional[str]:
        """Query Semantic Scholar for open-access PDF, download and extract text."""
        self._rate_limit_semanticscholar()
        url = f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}?fields=openAccessPdf"
        try:
            def _s2_api():
                r = requests.get(url, timeout=15, headers={"User-Agent": "meta-analysis-bot/1.0"})
                r.raise_for_status()
                return r.json()

            data = retry_with_backoff(
                _s2_api,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"S2 API retry {a+1}: {e} "),
            )
            oa = data.get("openAccessPdf")
            if not oa or not oa.get("url"):
                return None
            pdf_url = oa["url"]

            def _s2_pdf():
                r = requests.get(
                    pdf_url, timeout=30,
                    headers={"User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"},
                )
                r.raise_for_status()
                return r.content

            pdf_content = retry_with_backoff(
                _s2_pdf,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"S2 PDF retry {a+1}: {e} "),
            )
            return self._extract_pdf_text(pdf_content)
        except Exception as e:
            print(f"S2 error: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  OpenAlex                                                            #
    # ------------------------------------------------------------------ #

    def _try_openalex(self, paper: Dict) -> Optional[str]:
        """Query OpenAlex for an open-access URL, download and extract text.

        OpenAlex is free with no API key. We try filtering by PMID first,
        then by DOI. The response includes open_access.oa_url.
        """
        self._rate_limit_openalex()
        pmid = paper.get("pmid", "")
        doi = paper.get("doi", "")
        headers = {
            "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)",
            "Accept": "application/json",
        }

        oa_url = None
        try:
            def _openalex_query(filter_str):
                u = f"https://api.openalex.org/works?filter={filter_str}"
                r = requests.get(u, timeout=15, headers=headers)
                r.raise_for_status()
                return r.json()

            # Try PMID first
            if pmid:
                try:
                    data = retry_with_backoff(
                        lambda: _openalex_query(f"pmid:{pmid}"),
                        retryable_exceptions=(requests.exceptions.RequestException,),
                        on_retry=lambda a, e, d: print(f"OpenAlex retry {a+1}: {e} "),
                    )
                    results = data.get("results", [])
                    if results:
                        oa_url = results[0].get("open_access", {}).get("oa_url")
                except Exception:
                    pass

            # Fallback to DOI
            if not oa_url and doi:
                try:
                    data = retry_with_backoff(
                        lambda: _openalex_query(f"doi:{doi}"),
                        retryable_exceptions=(requests.exceptions.RequestException,),
                        on_retry=lambda a, e, d: print(f"OpenAlex DOI retry {a+1}: {e} "),
                    )
                    results = data.get("results", [])
                    if results:
                        oa_url = results[0].get("open_access", {}).get("oa_url")
                except Exception:
                    pass

            if not oa_url:
                return None

            # Download the resource (could be PDF or HTML)
            dl_resp = requests.get(
                oa_url, timeout=30, headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if dl_resp.status_code != 200:
                return None

            content_type = dl_resp.headers.get("Content-Type", "")
            if "pdf" in content_type or oa_url.endswith(".pdf"):
                return self._extract_pdf_text(dl_resp.content)
            elif "html" in content_type or "xml" in content_type:
                return self._extract_html_text(dl_resp.text)
            else:
                # Try as PDF anyway
                text = self._extract_pdf_text(dl_resp.content)
                if text:
                    return text
                return self._extract_html_text(dl_resp.text)

        except Exception as e:
            print(f"OpenAlex error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  CrossRef                                                            #
    # ------------------------------------------------------------------ #

    def _try_crossref(self, paper: Dict) -> Optional[str]:
        """Query CrossRef for PDF links via the works API.

        CrossRef provides a link[] array with content-type and URL.
        We look for application/pdf entries.
        """
        doi = paper.get("doi", "")
        if not doi:
            return None

        self._rate_limit_crossref()
        headers = {
            "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)",
            "Accept": "application/json",
        }

        try:
            url = f"https://api.crossref.org/works/{doi}"

            def _crossref_req():
                r = requests.get(url, timeout=15, headers=headers)
                r.raise_for_status()
                return r.json()

            data = retry_with_backoff(
                _crossref_req,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"CrossRef retry {a+1}: {e} "),
            )
            message = data.get("message", {})
            links = message.get("link", [])

            # Look for PDF links
            pdf_url = None
            for link in links:
                ct = link.get("content-type", "")
                link_url = link.get("URL", "")
                if "pdf" in ct.lower() and link_url:
                    pdf_url = link_url
                    break

            if not pdf_url:
                return None

            pdf_resp = requests.get(
                pdf_url, timeout=30, headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if pdf_resp.status_code != 200:
                return None

            return self._extract_pdf_text(pdf_resp.content)

        except Exception as e:
            print(f"CrossRef error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  DOAJ (Directory of Open Access Journals)                            #
    # ------------------------------------------------------------------ #

    def _try_doaj(self, paper: Dict) -> Optional[str]:
        """Query DOAJ for open-access article links.

        DOAJ indexes open-access journals and provides direct article URLs.
        API: GET https://doaj.org/api/search/articles/doi:{doi}
        """
        doi = paper.get("doi", "")
        if not doi:
            return None

        self._rate_limit_doaj()
        headers = {
            "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)",
            "Accept": "application/json",
        }

        try:
            url = f"https://doaj.org/api/search/articles/doi:{doi}"
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            # Extract fulltext URL from DOAJ bibjson
            bibjson = results[0].get("bibjson", {})
            article_links = bibjson.get("link", [])

            fulltext_url = None
            for link in article_links:
                link_type = link.get("type", "")
                link_url = link.get("url", "")
                if link_url and link_type == "fulltext":
                    fulltext_url = link_url
                    break

            # Fallback: any link with a URL
            if not fulltext_url:
                for link in article_links:
                    link_url = link.get("url", "")
                    if link_url:
                        fulltext_url = link_url
                        break

            if not fulltext_url:
                return None

            dl_resp = requests.get(
                fulltext_url, timeout=30, headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if dl_resp.status_code != 200:
                return None

            content_type = dl_resp.headers.get("Content-Type", "")
            if "pdf" in content_type or fulltext_url.endswith(".pdf"):
                return self._extract_pdf_text(dl_resp.content)
            elif "html" in content_type:
                return self._extract_html_text(dl_resp.text)
            else:
                # Try PDF first, then HTML
                text = self._extract_pdf_text(dl_resp.content)
                if text:
                    return text
                return self._extract_html_text(dl_resp.text)

        except Exception as e:
            print(f"DOAJ error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  The Lens (lens.org)                                                 #
    # ------------------------------------------------------------------ #

    def _try_lens(self, paper: Dict) -> Optional[str]:
        """Query The Lens scholarly API for open-access full text.

        Requires a free API token from lens.org. If no token is configured,
        the method skips gracefully.
        """
        if not self._lens_api_token:
            print("no API token, skipping ", end="", flush=True)
            return None

        doi = paper.get("doi", "")
        pmid = paper.get("pmid", "")
        if not doi and not pmid:
            return None

        self._rate_limit_lens()

        headers = {
            "Authorization": f"Bearer {self._lens_api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            # Build query: prefer DOI, fall back to PMID
            if doi:
                query_body = {
                    "query": {"match": {"external_ids.value": doi}},
                    "include": ["external_ids", "open_access", "source_urls"],
                    "size": 1,
                }
            else:
                query_body = {
                    "query": {"match": {"external_ids.value": pmid}},
                    "include": ["external_ids", "open_access", "source_urls"],
                    "size": 1,
                }

            timeout = self.config.get("retrieval", {}).get("request_timeout", 30)

            def _lens_req():
                r = requests.post(
                    "https://api.lens.org/scholarly/search",
                    json=query_body,
                    headers=headers,
                    timeout=timeout,
                )
                if r.status_code == 429:
                    raise requests.exceptions.RequestException("Lens rate limited (429)")
                r.raise_for_status()
                return r.json()

            data = retry_with_backoff(
                _lens_req,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"Lens retry {a+1}: {e} ", end="", flush=True),
            )

            results = data.get("data", [])
            if not results:
                return None

            record = results[0]

            # Try open_access URL first
            oa_info = record.get("open_access") or {}
            oa_url = oa_info.get("url") or oa_info.get("oa_url")

            # Fallback: source_urls
            if not oa_url:
                source_urls = record.get("source_urls") or []
                for su in source_urls:
                    if isinstance(su, str) and su:
                        oa_url = su
                        break
                    elif isinstance(su, dict):
                        oa_url = su.get("url", "")
                        if oa_url:
                            break

            if not oa_url:
                return None

            # Download the resource
            dl_resp = requests.get(
                oa_url,
                timeout=timeout,
                headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if dl_resp.status_code != 200:
                return None

            content_type = dl_resp.headers.get("Content-Type", "")
            if "pdf" in content_type or oa_url.endswith(".pdf"):
                return self._extract_pdf_text(dl_resp.content)
            elif "html" in content_type or "xml" in content_type:
                return self._extract_html_text(dl_resp.text)
            else:
                text = self._extract_pdf_text(dl_resp.content)
                if text:
                    return text
                return self._extract_html_text(dl_resp.text)

        except Exception as e:
            print(f"Lens error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  CORE (core.ac.uk)                                                   #
    # ------------------------------------------------------------------ #

    def _try_core(self, paper: Dict) -> Optional[str]:
        """Query CORE API v3 for open-access full text or PDF.

        Requires a free API key from core.ac.uk. If no key is configured,
        the method skips gracefully.
        CORE sometimes includes fullText directly in the response.
        """
        if not self._core_api_key:
            print("no API key, skipping ", end="", flush=True)
            return None

        doi = paper.get("doi", "")
        title = paper.get("title", "")
        if not doi and not title:
            return None

        self._rate_limit_core()

        headers = {
            "Authorization": f"Bearer {self._core_api_key}",
            "Accept": "application/json",
        }
        timeout = self.config.get("retrieval", {}).get("request_timeout", 30)

        try:
            # Build search query: prefer DOI, fall back to title
            if doi:
                query = f'doi:"{doi}"'
            else:
                # Use first 100 chars of title to avoid overly long queries
                safe_title = title[:100].replace('"', '\\"')
                query = f'title:"{safe_title}"'

            url = f"https://api.core.ac.uk/v3/search/works?q={requests.utils.quote(query)}&limit=1"

            def _core_req():
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code == 429:
                    raise requests.exceptions.RequestException("CORE rate limited (429)")
                r.raise_for_status()
                return r.json()

            data = retry_with_backoff(
                _core_req,
                retryable_exceptions=(requests.exceptions.RequestException,),
                on_retry=lambda a, e, d: print(f"CORE retry {a+1}: {e} ", end="", flush=True),
            )

            results = data.get("results", [])
            if not results:
                return None

            record = results[0]

            # CORE sometimes has the full text directly in the response
            full_text = record.get("fullText") or record.get("fulltext")
            if full_text and len(full_text.strip()) >= 500:
                return full_text.strip()

            # Try downloadUrl for PDF
            download_url = record.get("downloadUrl") or record.get("download_url")
            if not download_url:
                # Try sourceFulltextUrls or links
                links = record.get("sourceFulltextUrls") or record.get("links") or []
                if isinstance(links, list) and links:
                    download_url = links[0] if isinstance(links[0], str) else links[0].get("url", "")

            if not download_url:
                return None

            dl_resp = requests.get(
                download_url,
                timeout=timeout,
                headers={
                    "User-Agent": "meta-analysis-bot/1.0 (mailto:meta-analysis@example.com)"
                },
            )
            if dl_resp.status_code != 200:
                return None

            content_type = dl_resp.headers.get("Content-Type", "")
            if "pdf" in content_type or download_url.endswith(".pdf"):
                return self._extract_pdf_text(dl_resp.content)
            elif "html" in content_type or "xml" in content_type:
                return self._extract_html_text(dl_resp.text)
            else:
                text = self._extract_pdf_text(dl_resp.content)
                if text:
                    return text
                return self._extract_html_text(dl_resp.text)

        except Exception as e:
            print(f"CORE error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  HTML text extraction helper                                         #
    # ------------------------------------------------------------------ #

    def _extract_html_text(self, html: str) -> Optional[str]:
        """Extract readable text from an HTML page (article body)."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, header, footer
            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Try common article body selectors
            body_el = (
                soup.find("article")
                or soup.find("div", class_=re.compile(r"article|fulltext|body|content", re.I))
                or soup.find("main")
                or soup.find("body")
            )

            if body_el is None:
                return None

            text = body_el.get_text(separator="\n", strip=True)

            # Only accept if substantial text extracted
            if text and len(text) >= 500:
                return text
            return None

        except Exception as e:
            print(f"HTML parse error: {e} ", end="", flush=True)
            return None

    # ------------------------------------------------------------------ #
    #  Google Scholar PDF (search by title → find OA PDF link)             #
    # ------------------------------------------------------------------ #

    def _try_google_scholar(self, paper: Dict) -> Optional[str]:
        """Search Google Scholar for the paper title and try to find an OA PDF.

        Google Scholar often links to free versions (author copies, institutional
        repositories, ResearchGate, preprint servers) even when the publisher
        version is paywalled. We search by exact title, look for [PDF] links,
        download and extract text.
        """
        title = paper.get("title", "")
        if not title or len(title) < 10:
            return None

        try:
            from bs4 import BeautifulSoup
            import urllib.parse
        except ImportError:
            return None

        # Rate limit
        elapsed = time.time() - self._last_google_scholar_call
        if elapsed < self._google_scholar_interval:
            time.sleep(self._google_scholar_interval - elapsed)
        self._last_google_scholar_call = time.time()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            # Search with exact title in quotes for precision
            query = f'"{title}"'
            params = {"q": query, "hl": "en"}
            url = "https://scholar.google.com/scholar?" + urllib.parse.urlencode(params)

            resp = requests.get(url, headers=headers, timeout=20)

            # Check for CAPTCHA/rate limiting
            if resp.status_code == 429 or "CAPTCHA" in resp.text or "unusual traffic" in resp.text.lower():
                return None

            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Strategy 1: Look for [PDF] link on the right side of results
            for pdf_link in soup.select("div.gs_ggs a"):
                href = pdf_link.get("href", "")
                if href and ("pdf" in href.lower() or pdf_link.get_text(strip=True).upper() == "[PDF]"):
                    text = self._download_and_extract_pdf(href, headers)
                    if text:
                        return text

            # Strategy 2: Look for links in result entries that point to PDFs
            for result in soup.select("div.gs_r.gs_or.gs_scl"):
                # Check the top-right PDF link
                pdf_span = result.select_one("div.gs_or_ggsm a, div.gs_ggs a")
                if pdf_span:
                    href = pdf_span.get("href", "")
                    if href:
                        text = self._download_and_extract_pdf(href, headers)
                        if text:
                            return text

                # Check the title link itself (might point to a free HTML version)
                title_link = result.select_one("h3.gs_rt a")
                if title_link:
                    href = title_link.get("href", "")
                    if href and ".pdf" in href.lower():
                        text = self._download_and_extract_pdf(href, headers)
                        if text:
                            return text

            return None

        except Exception:
            return None

    def _download_and_extract_pdf(self, url: str, headers: Dict) -> Optional[str]:
        """Download a PDF from URL, extract text. Returns None if not a valid PDF."""
        try:
            resp = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
            if resp.status_code != 200:
                return None

            content_type = resp.headers.get("Content-Type", "").lower()

            # If it's a PDF, extract text
            if "pdf" in content_type or url.lower().endswith(".pdf") or resp.content[:5] == b"%PDF-":
                text = self._extract_pdf_text(resp.content)
                if text and len(text) >= 500:
                    return text

            # If it's HTML, try to extract article body (e.g., ResearchGate, PubMed Central)
            elif "html" in content_type:
                text = self._extract_html_text(resp.text)
                if text and len(text) >= 500:
                    return text

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Sci-Hub PDF                                                         #
    # ------------------------------------------------------------------ #

    def _fetch_scihub_pdf(self, doi: str) -> Optional[str]:
        """Fetch PDF from Sci-Hub, parse HTML for embed/iframe PDF URL."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return None

        try:
            url = f"https://{self.scihub_domain}/{doi}"
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            pdf_url = None

            # Try <embed> or <iframe> with PDF src
            for tag in soup.find_all(["embed", "iframe"]):
                src = tag.get("src", "")
                if src and (".pdf" in src or "sci-hub" in src):
                    pdf_url = src
                    break

            if not pdf_url:
                return None

            # Fix protocol-relative URLs
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url

            pdf_resp = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if pdf_resp.status_code != 200:
                return None
            return self._extract_pdf_text(pdf_resp.content)
        except Exception as e:
            print(f"Sci-Hub error: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Rate limiting                                                       #
    # ------------------------------------------------------------------ #

    def _rate_limit_semanticscholar(self):
        """Semantic Scholar: 1 RPS for unauthenticated."""
        elapsed = time.time() - self._last_semanticscholar_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_semanticscholar_call = time.time()

    def _rate_limit_ncbi(self):
        """NCBI allows 3 req/s without API key; be conservative with 0.4s gap."""
        elapsed = time.time() - self._last_ncbi_call
        if elapsed < 0.4:
            time.sleep(0.4 - elapsed)
        self._last_ncbi_call = time.time()

    def _rate_limit_unpaywall(self):
        """Unpaywall: 100K/day, be polite with 1s gap."""
        elapsed = time.time() - self._last_unpaywall_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_unpaywall_call = time.time()

    def _rate_limit_openalex(self):
        """OpenAlex: polite pool allows 10 RPS with email, conservative 0.2s gap."""
        elapsed = time.time() - self._last_openalex_call
        if elapsed < self._openalex_interval:
            time.sleep(self._openalex_interval - elapsed)
        self._last_openalex_call = time.time()

    def _rate_limit_crossref(self):
        """CrossRef: polite pool with mailto, 0.5s gap."""
        elapsed = time.time() - self._last_crossref_call
        if elapsed < self._crossref_interval:
            time.sleep(self._crossref_interval - elapsed)
        self._last_crossref_call = time.time()

    def _rate_limit_doaj(self):
        """DOAJ: no official rate limit, be polite with 0.5s gap."""
        elapsed = time.time() - self._last_doaj_call
        if elapsed < self._doaj_interval:
            time.sleep(self._doaj_interval - elapsed)
        self._last_doaj_call = time.time()

    def _rate_limit_lens(self):
        """The Lens: 1 request per second for free tier."""
        elapsed = time.time() - self._last_lens_call
        if elapsed < self._lens_interval:
            time.sleep(self._lens_interval - elapsed)
        self._last_lens_call = time.time()

    def _rate_limit_core(self):
        """CORE: 10 requests per second for free tier, 0.1s gap."""
        elapsed = time.time() - self._last_core_call
        if elapsed < self._core_interval:
            time.sleep(self._core_interval - elapsed)
        self._last_core_call = time.time()

    # ================================================================== #
    #  PHASE B: FULL-TEXT PRISMA SCREENING                                 #
    # ================================================================== #

    def screen_fulltext(self, papers: List[Dict], protocol: Dict) -> Dict:
        """Screen papers using full text + LLM. Returns include/exclude lists + stats."""
        print("\n--- Phase B: Full-Text PRISMA Screening ---")
        print(f"  Ensemble size: {self.ensemble_size}")
        if len(self.model_paths) > 1:
            print(f"  Multi-model: {len(self.model_paths)} models")

        if self.model is None:
            self.load_model()

        include_list = []
        exclude_list = []

        for i, paper in enumerate(papers):
            title_short = paper.get("title", "")[:70]
            print(f"\n  [{i+1}/{len(papers)}] {title_short}...")
            print(
                f"    Source: {paper['text_source']} ({paper['full_text_length']} chars)"
            )

            result = self._screen_single_fulltext(paper, protocol)

            paper["fulltext_decision"] = result["decision"]
            paper["fulltext_reasoning"] = result["reasoning"]
            paper["exclusion_reason"] = result.get("exclusion_reason")
            paper["fulltext_votes"] = result["votes"]
            paper["fulltext_confidence"] = result["confidence"]
            paper["screening_confidence"] = result["screening_confidence"]
            paper["borderline"] = result["borderline"]

            icon = {"INCLUDE": "✅", "EXCLUDE": "❌"}[result["decision"]]
            [v["decision"] for v in result["votes"]]
            print(
                f"    -> {icon} {result['decision']}"
                f" | screen_conf: {result['screening_confidence']}"
                f"{' [BORDERLINE]' if result['borderline'] else ''}"
            )
            if result.get("exclusion_reason"):
                print(f"      Reason: {result['exclusion_reason']}")

            if result["decision"] == "INCLUDE":
                include_list.append(paper)
            else:
                exclude_list.append(paper)

        stats = {
            "total_retrieved": len(papers),
            "pmc_fulltext": sum(1 for p in papers if p["text_source"] == "pmc"),
            "unpaywall_pdf": sum(
                1 for p in papers if p["text_source"] == "unpaywall_pdf"
            ),
            "europepmc_fulltext": sum(
                1 for p in papers if p["text_source"] == "europepmc"
            ),
            "semanticscholar_pdf": sum(
                1 for p in papers if p["text_source"] == "semanticscholar_pdf"
            ),
            "openalex_pdf": sum(
                1 for p in papers if p["text_source"] == "openalex_pdf"
            ),
            "crossref_pdf": sum(
                1 for p in papers if p["text_source"] == "crossref_pdf"
            ),
            "doaj_pdf": sum(
                1 for p in papers if p["text_source"] == "doaj_pdf"
            ),
            "lens_fulltext": sum(
                1 for p in papers if p["text_source"] == "lens_fulltext"
            ),
            "core_fulltext": sum(
                1 for p in papers if p["text_source"] == "core_fulltext"
            ),
            "google_scholar_pdf": sum(
                1 for p in papers if p["text_source"] == "google_scholar_pdf"
            ),
            "scihub_pdf": sum(
                1 for p in papers if p["text_source"] == "scihub_pdf"
            ),
            "abstract_only": sum(
                1 for p in papers if p["text_source"] == "abstract_only"
            ),
            "fulltext_included": len(include_list),
            "fulltext_excluded": len(exclude_list),
        }

        return {
            "papers": papers,
            "include": include_list,
            "exclude": exclude_list,
            "stats": stats,
        }

    def _screen_single_fulltext(self, paper: Dict, protocol: Dict) -> Dict:
        """Ensemble voting on a single paper's full text."""
        vote_details = []  # list of dicts with full vote info
        multi_model = len(self.model_paths) > 1

        for i in range(self.ensemble_size):
            model_name = "default"
            if multi_model:
                model_path = self.model_paths[i % len(self.model_paths)]
                self.load_model(model_path)
                model_name = os.path.basename(model_path)
            vote = self._single_fulltext_vote(paper, protocol)
            vote_details.append({
                "decision": vote["decision"],
                "reasoning": vote["reasoning"],
                "exclusion_reason": vote.get("exclusion_reason", ""),
                "model": model_name,
            })

        raw_votes = [v["decision"] for v in vote_details]
        decision, confidence = self._aggregate_votes(raw_votes)

        # Pick first reasoning/exclusion_reason from matching vote
        reasoning = vote_details[0]["reasoning"] if vote_details else ""
        exclusion_reason = None
        if decision == "EXCLUDE":
            for vd in vote_details:
                if vd["decision"] == "EXCLUDE" and vd.get("exclusion_reason"):
                    reasoning = vd["reasoning"]
                    exclusion_reason = vd["exclusion_reason"]
                    break
            if not exclusion_reason:
                exclusion_reason = "excluded on full-text review"

        # Screening confidence: 0 = maximally uncertain, 1 = unanimous
        n = len(raw_votes)
        include_frac = raw_votes.count("INCLUDE") / n if n > 0 else 0.5
        screening_confidence = round(abs(include_frac - 0.5) * 2, 2)
        is_borderline = screening_confidence < 0.3

        return {
            "decision": decision,
            "confidence": round(confidence, 2),
            "votes": vote_details,
            "reasoning": reasoning,
            "exclusion_reason": exclusion_reason if decision == "EXCLUDE" else None,
            "screening_confidence": screening_confidence,
            "borderline": is_borderline,
        }

    def _single_fulltext_vote(self, paper: Dict, protocol: Dict) -> Dict:
        """One LLM inference on full text → INCLUDE / EXCLUDE decision."""
        prompt = self._build_fulltext_prompt(paper, protocol)
        response = self._simple_generate(prompt, max_tokens=200)

        try:
            cleaned = self._clean_json(response)
            parsed = json.loads(cleaned)
            decision = parsed.get("decision", "EXCLUDE").upper().strip()
            if decision not in ("INCLUDE", "EXCLUDE"):
                decision = "EXCLUDE"
            return {
                "decision": decision,
                "reasoning": parsed.get("reasoning", ""),
                "exclusion_reason": parsed.get("exclusion_reason", ""),
            }
        except Exception:
            return {
                "decision": "EXCLUDE",
                "reasoning": "JSON parse error - conservative EXCLUDE",
                "exclusion_reason": "unable to parse LLM response",
            }

    def _build_fulltext_prompt(self, paper: Dict, protocol: Dict) -> str:
        """Build the full-text screening prompt, adapting to the review framework."""
        incl = "\n".join(f"  - {c}" for c in protocol.get("inclusion_criteria", []))
        excl = "\n".join(f"  - {c}" for c in protocol.get("exclusion_criteria", []))

        # Use sections if available, otherwise raw text
        text_for_review = self._prepare_text_for_prompt(paper)

        framework = protocol.get("framework", "PICO")
        pop = protocol.get("population", "")
        outcome = protocol.get("outcome", "")

        # Build framework-specific focus rules and decision criteria
        if framework == "PECO":
            intv = protocol.get("exposure", protocol.get("intervention", ""))
            protocol.get("comparator", "")
            focus_rule = (
                f'CRITICAL RULE — PRIMARY EXPOSURE:\n'
                f'The paper must study "{intv}" as its PRIMARY exposure in "{pop}" with '
                f'"{outcome}" as a primary or secondary outcome.\n'
                f'EXCLUDE if "{intv}" only appears as:\n'
                f'  - A covariate or confounder adjusted for in statistical models\n'
                f'  - A baseline demographic characteristic in Table 1\n'
                f'  - A control variable without being the main predictor\n'
                f'  - A brief mention in the discussion without being the focus'
            )
            design_focus = (
                "- Actual study design (cohort, case-control, cross-sectional?)\n"
                "- Exposure measurement and definition\n"
                "- Outcome measures used\n"
                "- Population characteristics and sample size\n"
                "- Confounders controlled for"
            )
            decision_include = f'paper studies "{intv}" as a primary exposure and meets ALL inclusion criteria'
            decision_exclude = f'"{intv}" is NOT a primary exposure, OR paper fails any inclusion criterion, OR triggers any exclusion criterion'

        elif framework == "SPIDER":
            intv = protocol.get("phenomenon_of_interest", protocol.get("intervention", protocol.get("interest", "")))
            design_type = protocol.get("design", "qualitative")
            eval_type = protocol.get("evaluation", outcome)
            focus_rule = (
                f'SCREENING FOCUS — QUALITATIVE / MIXED-METHODS REVIEW:\n'
                f'This is a SPIDER-framework review. Do NOT require a treatment or intervention.\n'
                f'The paper must focus on "{intv}" as its central phenomenon of interest in "{pop}".\n'
                f'INCLUDE if the paper:\n'
                f'  - Explores, investigates, or examines "{intv}" as a main topic\n'
                f'  - Studies the perspectives, experiences, perceptions, or barriers related to "{intv}"\n'
                f'  - Uses qualitative, mixed-methods, or survey methodology relevant to "{intv}"\n'
                f'EXCLUDE only if:\n'
                f'  - "{intv}" is only mentioned peripherally or in passing\n'
                f'  - The paper is a systematic review, editorial, commentary, or protocol\n'
                f'  - The population is clearly different from "{pop}"'
            )
            design_focus = (
                f"- Study design ({design_type}: interviews, focus groups, surveys, ethnography, etc.)\n"
                "- Whether the phenomenon of interest is a central focus\n"
                f"- How findings relate to \"{eval_type}\"\n"
                "- Population and setting\n"
                "- Data collection and analysis methods"
            )
            decision_include = f'paper explores or investigates "{intv}" in "{pop}" and meets ALL inclusion criteria'
            decision_exclude = f'"{intv}" is NOT a central focus, OR population differs from "{pop}", OR paper is a review/editorial/protocol'

        elif framework == "PCC":
            intv = protocol.get("concept", protocol.get("intervention", ""))
            context = protocol.get("context", "")
            focus_rule = (
                f'SCREENING FOCUS — SCOPING REVIEW (PCC):\n'
                f'This is a PCC-framework scoping review. Do NOT require an intervention or treatment.\n'
                f'The paper must address "{intv}" as its central concept in the context of "{context}" for "{pop}".\n'
                f'INCLUDE if the paper:\n'
                f'  - Directly addresses, maps, or explores "{intv}"\n'
                f'  - Is relevant to the context of "{context}"\n'
                f'  - Any study design is acceptable (quantitative, qualitative, mixed-methods)\n'
                f'EXCLUDE only if:\n'
                f'  - "{intv}" is only mentioned peripherally\n'
                f'  - The paper is a systematic review, editorial, or protocol\n'
                f'  - The context or population clearly does not match'
            )
            design_focus = (
                "- Study design (any design is acceptable for scoping reviews)\n"
                f"- Whether \"{intv}\" is a central concept\n"
                f"- Relevance to context: \"{context}\"\n"
                "- Population and setting"
            )
            decision_include = f'paper addresses "{intv}" in "{pop}" within the context of "{context}" and meets ALL inclusion criteria'
            decision_exclude = f'"{intv}" is NOT a central concept, OR context/population clearly does not match, OR paper is a review/editorial/protocol'

        else:
            # PICO (default) — intervention/treatment studies
            intv = protocol.get("intervention", "")
            protocol.get("comparator", "")
            focus_rule = (
                f'SCREENING FOCUS:\n'
                f'The paper must use "{intv}" as one of its main interventions or treatments.\n'
                f'INCLUDE if "{intv}" is tested, compared, or evaluated as an active treatment arm.\n'
                f'The outcome should relate to "{outcome}" in "{pop}".\n'
                f'EXCLUDE only if "{intv}" is not an active treatment in this study, or if the '
                f'paper is a review, editorial, or case report.'
            )
            design_focus = (
                "- Actual study design (RCT, quasi-experimental, observational?)\n"
                "- Specific outcome measures used\n"
                "- Population characteristics and sample size\n"
                "- Follow-up duration\n"
                "- Whether intervention matches protocol criteria"
            )
            decision_include = f'paper studies "{intv}" as an active intervention and meets ALL inclusion criteria'
            decision_exclude = f'"{intv}" is NOT studied as a treatment/intervention, OR paper fails any inclusion criterion, OR triggers any exclusion criterion'

        return f"""You are a systematic review expert performing FULL-TEXT screening.
You have already screened this paper by title/abstract and it passed. Now review the full text against the protocol criteria.

REVIEW FRAMEWORK: {framework}

INCLUSION CRITERIA:
{incl}

EXCLUSION CRITERIA:
{excl}

PAPER:
Title: {paper.get('title', '')}
Journal: {paper.get('journal', '')} ({paper.get('year', '')})

FULL TEXT (may be truncated):
{text_for_review}

SCREENING FOCUS (criteria hard to verify from abstract alone):
{design_focus}

{focus_rule}

DECISION RULES:
- INCLUDE: {decision_include}
- EXCLUDE: {decision_exclude}

Respond ONLY with valid JSON:
{{"decision": "INCLUDE or EXCLUDE", "reasoning": "2-3 sentences", "exclusion_reason": "brief reason if EXCLUDE, empty string if INCLUDE"}}"""

    def _prepare_text_for_prompt(self, paper: Dict) -> str:
        """Prepare and truncate text for the screening prompt."""
        max_chars = (
            6000  # ~1500 tokens, leaves room for prompt + response in 7B context
        )

        sections = paper.get("sections") or {}
        if sections:
            # Prioritize methods + results (most relevant for screening)
            priority = ["methods", "results", "background", "conclusions", "discussion"]
            parts = []
            total = 0
            for sec_name in priority:
                if sec_name in sections and total < max_chars:
                    text = sections[sec_name]
                    remaining = max_chars - total
                    if len(text) > remaining:
                        text = text[:remaining] + "..."
                    parts.append(f"[{sec_name.upper()}]\n{text}")
                    total += len(text)
            return "\n\n".join(parts)

        # Raw text fallback
        full_text = paper.get("full_text", "") or ""
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n[...truncated...]"
        return full_text

    # ------------------------------------------------------------------ #
    #  VOTING AGGREGATION (same pattern as A3)                             #
    # ------------------------------------------------------------------ #

    def _aggregate_votes(self, votes: List[str]) -> Tuple[str, float]:
        """Aggregate ensemble votes. UNCERTAIN treated as EXCLUDE for full-text."""
        n = len(votes)
        include_n = votes.count("INCLUDE")
        confidence = include_n / n

        if include_n / n >= self.voting_threshold:
            return "INCLUDE", confidence
        else:
            return "EXCLUDE", confidence

    # ------------------------------------------------------------------ #
    #  CALIBRATION & SENSITIVITY ANALYSIS                                  #
    # ------------------------------------------------------------------ #

    def analyze_voting_sensitivity(self, results: dict) -> dict:
        """Post-hoc analysis of how different thresholds would change results."""
        papers = results.get("papers", [])
        thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        analysis = {}
        for t in thresholds:
            included = 0
            excluded = 0
            for p in papers:
                votes = p.get("fulltext_votes", [])
                if not votes:
                    # Backward compatibility: try inline votes
                    votes = p.get("votes", [])
                if not votes:
                    continue
                # votes may be list of dicts or list of strings
                decisions = []
                for v in votes:
                    if isinstance(v, dict):
                        decisions.append(v.get("decision", "EXCLUDE"))
                    else:
                        decisions.append(v)
                n = len(decisions)
                include_frac = sum(1 for d in decisions if d == "INCLUDE") / n
                if include_frac >= t:
                    included += 1
                else:
                    excluded += 1
            analysis[str(t)] = {
                "included": included,
                "excluded": excluded,
                "threshold": t,
            }

        # Calculate inter-rater agreement (Fleiss' kappa)
        kappa = self._calculate_fleiss_kappa(papers)

        # Borderline papers needing human review
        borderline_papers = [
            {"pmid": p.get("pmid", ""), "title": p.get("title", ""), "screening_confidence": p.get("screening_confidence", 0)}
            for p in papers if p.get("borderline", False)
        ]

        return {
            "threshold_sensitivity": analysis,
            "current_threshold": self.voting_threshold,
            "fleiss_kappa": round(kappa, 3),
            "interpretation": self._interpret_kappa(kappa),
            "total_papers": len(papers),
            "borderline_count": len(borderline_papers),
            "borderline_papers": borderline_papers,
        }

    def _calculate_fleiss_kappa(self, papers: list) -> float:
        """Calculate Fleiss' kappa for inter-rater agreement.

        Fleiss' kappa formula: kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
        """
        categories = ["INCLUDE", "EXCLUDE"]
        n_raters = self.ensemble_size

        if n_raters < 2:
            return 0.0  # kappa undefined for single rater

        vote_matrix = []  # each entry: [n_include, n_exclude]
        n_subjects = 0
        for p in papers:
            votes = p.get("fulltext_votes", p.get("votes", []))
            if not votes:
                continue
            counts = [0, 0]
            for v in votes:
                decision = v.get("decision", v) if isinstance(v, dict) else v
                if decision == "INCLUDE":
                    counts[0] += 1
                else:  # EXCLUDE (and anything else)
                    counts[1] += 1
            vote_matrix.append(counts)
            n_subjects += 1

        if n_subjects == 0:
            return 0.0

        # P_i for each subject
        p_values = []
        for counts in vote_matrix:
            n = sum(counts)
            if n < 2:
                continue
            sum_sq = sum(c * c for c in counts)
            p_i = (sum_sq - n) / (n * (n - 1))
            p_values.append(p_i)

        if not p_values:
            return 0.0

        P_bar = sum(p_values) / len(p_values)

        # P_bar_e: expected agreement by chance
        total_votes = sum(sum(counts) for counts in vote_matrix)
        if total_votes == 0:
            return 0.0

        P_bar_e = 0.0
        for j in range(len(categories)):
            p_j = sum(counts[j] for counts in vote_matrix) / total_votes
            P_bar_e += p_j * p_j

        if P_bar_e >= 1.0:
            return 1.0

        kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)
        return kappa

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Fleiss' kappa value using Landis & Koch scale."""
        if kappa < 0:
            return "Poor"
        if kappa < 0.20:
            return "Slight"
        if kappa < 0.40:
            return "Fair"
        if kappa < 0.60:
            return "Moderate"
        if kappa < 0.80:
            return "Substantial"
        return "Almost perfect"

    # ------------------------------------------------------------------ #
    #  HELPERS (same patterns as A3)                                        #
    # ------------------------------------------------------------------ #

    def _simple_generate(self, prompt: str, max_tokens: int = 200) -> str:
        if self._is_groq():
            return self.generate(prompt, max_new_tokens=max_tokens, temperature=0.2)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self._decode_response(outputs, input_length)

    def _clean_json(self, text: str) -> str:
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
                    return cleaned[start : i + 1]
        return cleaned[start:] + "}"

    def _print_summary(self, result: Dict):
        s = result["stats"]
        total_fulltext = (
            s['pmc_fulltext'] + s['unpaywall_pdf']
            + s.get('europepmc_fulltext', 0) + s.get('semanticscholar_pdf', 0)
            + s.get('openalex_pdf', 0) + s.get('crossref_pdf', 0)
            + s.get('doaj_pdf', 0) + s.get('lens_fulltext', 0)
            + s.get('core_fulltext', 0) + s.get('scihub_pdf', 0)
        )
        total_papers = total_fulltext + s['abstract_only']
        pct = round(100 * total_fulltext / total_papers, 1) if total_papers else 0

        print(f"\n{'='*60}")
        print("AGENT 4 - SUMMARY")
        print(f"{'='*60}")
        print(f"  Retrieval ({total_fulltext}/{total_papers} full-text, {pct}%):")
        print(f"    PMC full-text   : {s['pmc_fulltext']}")
        print(f"    Unpaywall PDF   : {s['unpaywall_pdf']}")
        print(f"    Europe PMC      : {s.get('europepmc_fulltext', 0)}")
        print(f"    Semantic Scholar: {s.get('semanticscholar_pdf', 0)}")
        print(f"    OpenAlex        : {s.get('openalex_pdf', 0)}")
        print(f"    CrossRef        : {s.get('crossref_pdf', 0)}")
        print(f"    DOAJ            : {s.get('doaj_pdf', 0)}")
        print(f"    The Lens        : {s.get('lens_fulltext', 0)}")
        print(f"    CORE            : {s.get('core_fulltext', 0)}")
        print(f"    Sci-Hub PDF     : {s.get('scihub_pdf', 0)}")
        print(f"    Abstract only   : {s['abstract_only']}")
        print("  Full-text screening:")
        print(f"    INCLUDE        : {s['fulltext_included']}")
        print(f"    EXCLUDE        : {s['fulltext_excluded']}")
        print(f"  Total time       : {s.get('elapsed_s', 0)}s")


# ====================================================================== #
#  STANDALONE TEST                                                         #
# ====================================================================== #
if __name__ == "__main__":
    # Quick test: metadata fetch only (no GPU)
    a4 = Agent4FullText(ensemble_size=1)

    test_pmids = ["41135204", "40930184", "40379363"]
    print("Testing metadata fetch...")
    meta = a4._batch_fetch_metadata(test_pmids)
    for pmid, m in meta.items():
        print(f"  {pmid}: DOI={m['doi']}, PMCID={m['pmcid']}")

    print("\nDone. Run test_agent4.py for full test with screening.")
