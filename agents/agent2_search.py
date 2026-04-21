"""Agent 2: Multi-Database Search (PubMed + Europe PMC + Semantic Scholar + Google Scholar)"""

from Bio import Entrez
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
import re
import requests
import yaml
import os
import urllib.parse

from agents.retry import retry_with_backoff


class Agent2Search:
    """Multi-database literature search with semantic deduplication.

    Input: protocol (dict from A1) and pubmed_query (str).
    Output: dict with 'papers' (list of unique papers) and 'stats' (per-DB counts).
    GPU: no -- uses CPU-only MiniLM-L6-v2 for deduplication embeddings.
    """

    def __init__(self, min_per_db: int = None):
        self.dedup_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        search_cfg = cfg.get("search", {})

        # Entrez email: env var takes priority, then config.yaml
        Entrez.email = os.environ.get("ENTREZ_EMAIL") or search_cfg.get("entrez_email", "meta-analysis@example.com")
        db_cfg = search_cfg.get("databases", {})
        self.db_pubmed = db_cfg.get("pubmed", True)
        self.db_europepmc = db_cfg.get("europepmc", True)
        self.db_semantic_scholar = db_cfg.get("semantic_scholar", True)
        self.db_google_scholar = db_cfg.get("google_scholar", True)
        self.max_per_db = search_cfg.get("max_results_per_database", 500)
        self.dedup_threshold = search_cfg.get("dedup_similarity_threshold", 0.95)
        self.rate_limit = search_cfg.get("rate_limit_seconds", 1.0)
        self.google_scholar_rate_limit = search_cfg.get("google_scholar_rate_limit_seconds", 3.0)

        # CFG1: floor of papers per DB. Precedence:
        #   1. constructor kwarg (min_per_db, for tests / programmatic override)
        #   2. config/agents.json -> agents.agent2_search.min_per_db
        #   3. hardcoded default 10 (backward compat)
        if min_per_db is not None:
            self.min_per_db = int(min_per_db)
        else:
            self.min_per_db = int(self._load_min_per_db_from_agents_json(default=10))

        # Semantic Scholar API key (optional — strongly recommended to avoid 429).
        # Env var priority over config. S2 accepts header "x-api-key".
        self.s2_api_key = (
            os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            or search_cfg.get("semantic_scholar_api_key", "")
            or ""
        ).strip()
        if not self.s2_api_key:
            print(
                "[A2] WARNING: SEMANTIC_SCHOLAR_API_KEY no configurada. "
                "Semantic Scholar impondra rate limit estricto (429). "
                "Solicita clave gratuita en https://www.semanticscholar.org/product/api#api-key-form"
            )

    @staticmethod
    def _load_min_per_db_from_agents_json(default: int = 10) -> int:
        """Read `agents.agent2_search.min_per_db` from config/agents.json.

        Returns `default` (10) if the file is missing, malformed, or the key
        isn't set — preserves backward compatibility with the previous
        hardcoded floor of 10 papers per database.
        """
        path = os.path.join(os.path.dirname(__file__), "..", "config", "agents.json")
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return default
        agent_cfg = (data.get("agents") or {}).get("agent2_search") or {}
        if "min_per_db" in agent_cfg:
            try:
                return int(agent_cfg["min_per_db"])
            except (TypeError, ValueError):
                return default
        defaults = data.get("defaults") or {}
        if "min_per_db" in defaults:
            try:
                return int(defaults["min_per_db"])
            except (TypeError, ValueError):
                return default
        return default

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, protocol, pubmed_query, max_results=500):
        """Search all enabled databases, merge, deduplicate, return with stats.

        Uses dynamic redistribution: if one DB returns fewer results than its
        share, the remaining DBs get a higher cap to reach max_results total.
        Requests ~20% extra to compensate for deduplication losses.
        """
        # CFG1: be defensive if callers instantiated Agent2Search bypassing
        # __init__ (e.g. some older tests patch __init__ to a no-op).
        if not hasattr(self, "min_per_db"):
            self.min_per_db = 10

        search_terms = self._build_search_terms(protocol)

        # Request 20% extra to compensate for deduplication losses
        target = int(max_results * 1.2)

        stats = {
            "pubmed": 0,
            "europepmc": 0,
            "semantic_scholar": 0,
            "google_scholar": 0,
            "total_before_dedup": 0,
            "total_after_dedup": 0,
            "duplicates_removed": 0,
        }

        all_papers = []
        remaining = target  # how many we still need

        # Count remaining active DBs for fair distribution
        dbs_left = sum([self.db_pubmed, self.db_europepmc, self.db_semantic_scholar, self.db_google_scholar])

        # 1. PubMed
        if self.db_pubmed:
            cap = min(max(remaining // max(dbs_left, 1), self.min_per_db), self.max_per_db)
            dbs_left -= 1
            try:
                pm = self._search_pubmed(pubmed_query, max_results=cap)
                stats["pubmed"] = len(pm)
                all_papers.extend(pm)
                remaining -= len(pm)
            except Exception as e:
                print(f"[A2] PubMed search failed: {e}")

        # 2. Europe PMC
        if self.db_europepmc:
            cap = min(max(remaining // max(dbs_left, 1), self.min_per_db), self.max_per_db)
            dbs_left -= 1
            try:
                epmc = self._search_europepmc(search_terms, max_results=cap)
                stats["europepmc"] = len(epmc)
                all_papers.extend(epmc)
                remaining -= len(epmc)
            except Exception as e:
                print(f"[A2] Europe PMC search failed: {e}")

        # 3. Semantic Scholar
        if self.db_semantic_scholar:
            cap = min(max(remaining, self.min_per_db), self.max_per_db)
            dbs_left -= 1
            try:
                s2 = self._search_semantic_scholar(search_terms, max_results=cap)
                stats["semantic_scholar"] = len(s2)
                all_papers.extend(s2)
                remaining -= len(s2)
            except Exception as e:
                print(f"[A2] Semantic Scholar search failed: {e}")

        # 4. Google Scholar
        if self.db_google_scholar:
            cap = min(max(remaining, self.min_per_db), self.max_per_db)
            dbs_left -= 1
            try:
                gs = self._search_google_scholar(search_terms, max_results=cap)
                stats["google_scholar"] = len(gs)
                all_papers.extend(gs)
                remaining -= len(gs)
            except Exception as e:
                print(f"[A2] Google Scholar search failed: {e}")

        # 5. Backfill: if still under target, retry with broader queries
        if len(all_papers) < max_results:
            shortfall = max_results - len(all_papers)
            broad_query = protocol.get("research_question", pubmed_query)
            print(f"[A2] Backfill: {len(all_papers)} papers < {max_results} target, requesting {shortfall} more with broader query")

            # Try Europe PMC with broader query (research question instead of PICO terms)
            if self.db_europepmc and shortfall > 0:
                try:
                    extra = self._search_europepmc(broad_query, max_results=shortfall)
                    new_count = len(extra)
                    stats["europepmc"] += new_count
                    all_papers.extend(extra)
                    shortfall -= new_count
                    print(f"[A2] Backfill Europe PMC: +{new_count}")
                except Exception as e:
                    print(f"[A2] Backfill Europe PMC failed: {e}")

            # Try PubMed with broader terms (drop "randomized controlled trial" etc.)
            if self.db_pubmed and shortfall > 0:
                broad_pm_query = " ".join(pubmed_query.replace("[MeSH]", "").split()[:6])
                try:
                    extra = self._search_pubmed(broad_pm_query, max_results=shortfall)
                    new_count = len(extra)
                    stats["pubmed"] += new_count
                    all_papers.extend(extra)
                    shortfall -= new_count
                    print(f"[A2] Backfill PubMed: +{new_count}")
                except Exception as e:
                    print(f"[A2] Backfill PubMed failed: {e}")

            # Try Semantic Scholar with just the research question
            if self.db_semantic_scholar and shortfall > 0:
                try:
                    extra = self._search_semantic_scholar(broad_query, max_results=shortfall)
                    new_count = len(extra)
                    stats["semantic_scholar"] += new_count
                    all_papers.extend(extra)
                    print(f"[A2] Backfill S2: +{new_count}")
                except Exception as e:
                    print(f"[A2] Backfill S2 failed: {e}")

        stats["total_before_dedup"] = len(all_papers)

        # Record which databases actively contributed at least one result.
        # Consumed by A9 (explainability report §6 "procedencia de datos").
        search_sources = [
            db for db in ("pubmed", "europepmc", "semantic_scholar", "google_scholar")
            if stats.get(db, 0) > 0
        ]
        stats["search_sources"] = search_sources

        # 4. Deduplicate (3-tier)
        unique = self.deduplicate(all_papers, threshold=self.dedup_threshold)
        stats["total_after_dedup"] = len(unique)
        stats["duplicates_removed"] = len(all_papers) - len(unique)

        print(
            f"[A2] Summary: PubMed={stats['pubmed']} | "
            f"Europe PMC={stats['europepmc']} | "
            f"S2={stats['semantic_scholar']} | "
            f"GScholar={stats['google_scholar']} | "
            f"Before dedup={stats['total_before_dedup']} | "
            f"After dedup={stats['total_after_dedup']}"
        )

        return {"papers": unique, "stats": stats}

    # ── PubMed ────────────────────────────────────────────────────────────────

    def search_pubmed(self, query, max_results=1000):
        """Legacy public method — kept for backward compatibility."""
        return self._search_pubmed(query, max_results)

    def _search_pubmed(self, query, max_results=1000):
        """Busca en PubMed/MEDLINE (gratis)"""
        print(f"[A2] Searching PubMed: '{query}' (max {max_results})...")

        def _esearch():
            h = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            return Entrez.read(h)

        results = retry_with_backoff(
            _esearch,
            retryable_exceptions=(Exception,),
            on_retry=lambda a, e, d: print(f"[A2] PubMed esearch retry {a+1}: {e}"),
        )
        pmids = results["IdList"]
        print(f"[A2] PubMed: {len(pmids)} hits")

        if not pmids:
            print("[A2] PubMed returned 0 results.")
            return []

        # Fetch abstracts in batches of 200
        papers = []
        for i in range(0, len(pmids), 200):
            batch = pmids[i : i + 200]

            def _efetch(b=batch):
                h = Entrez.efetch(db="pubmed", id=b, rettype="abstract", retmode="xml")
                return Entrez.read(h)

            records = retry_with_backoff(
                _efetch,
                retryable_exceptions=(Exception,),
                on_retry=lambda a, e, d: print(f"[A2] PubMed efetch retry {a+1}: {e}"),
            )

            for record in records["PubmedArticle"]:
                try:
                    article = record["MedlineCitation"]["Article"]
                    # Extract DOI
                    doi = ""
                    for eid in article.get("ELocationID", []):
                        if hasattr(eid, "attributes") and eid.attributes.get("EIdType") == "doi":
                            doi = str(eid)
                            break
                    if not doi:
                        art_ids = record.get("PubmedData", {}).get("ArticleIdList", [])
                        for aid in art_ids:
                            if hasattr(aid, "attributes") and aid.attributes.get("IdType") == "doi":
                                doi = str(aid)
                                break

                    papers.append(
                        {
                            "pmid": str(record["MedlineCitation"]["PMID"]),
                            "doi": doi,
                            "title": str(article["ArticleTitle"]),
                            "abstract": str(
                                article.get("Abstract", {}).get(
                                    "AbstractText", [""]
                                )[0]
                            ),
                            "authors": [
                                a.get("LastName", "")
                                for a in article.get("AuthorList", [])[:3]
                            ],
                            "journal": str(article["Journal"]["Title"]),
                            "year": str(
                                article["Journal"]["JournalIssue"]
                                .get("PubDate", {})
                                .get("Year", "")
                            ),
                            "sources": ["pubmed"],
                        }
                    )
                except Exception:
                    continue

            if i + 200 < len(pmids):
                time.sleep(0.4)  # NCBI rate limit

        print(f"[A2] PubMed: parsed {len(papers)} papers")
        return papers

    # ── Europe PMC ────────────────────────────────────────────────────────────

    def _search_europepmc(self, search_terms, max_results=500):
        """Search Europe PMC REST API (free, no key)."""
        print(f"[A2] Searching Europe PMC: '{search_terms[:80]}...' (max {max_results})...")
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        papers = []
        cursor = "*"
        page_size = min(100, max_results)

        while len(papers) < max_results:
            params = {
                "query": search_terms,
                "resultType": "core",
                "pageSize": page_size,
                "format": "json",
                "cursorMark": cursor,
            }
            try:
                def _epmc_request(p=params):
                    r = requests.get(base_url, params=p, timeout=30)
                    r.raise_for_status()
                    return r.json()

                data = retry_with_backoff(
                    _epmc_request,
                    retryable_exceptions=(requests.exceptions.RequestException,),
                    on_retry=lambda a, e, d: print(f"[A2] Europe PMC retry {a+1}: {e}"),
                )
            except Exception as e:
                print(f"[A2] Europe PMC request failed: {e}")
                break

            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for r in results:
                if len(papers) >= max_results:
                    break
                pmid = str(r.get("pmid", "")) if r.get("pmid") else ""
                doi = str(r.get("doi", "")) if r.get("doi") else ""
                title = str(r.get("title", ""))
                abstract = str(r.get("abstractText", ""))
                # Skip entries with no title
                if not title or title.lower() == "none":
                    continue
                author_str = r.get("authorString", "")
                authors = [a.strip().split()[-1] for a in author_str.split(",")[:3]] if author_str else []
                journal = str(r.get("journalTitle", ""))
                year = str(r.get("pubYear", ""))

                papers.append({
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "year": year,
                    "sources": ["europepmc"],
                })

            next_cursor = data.get("nextCursorMark", "")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(self.rate_limit)

        print(f"[A2] Europe PMC: {len(papers)} papers")
        return papers

    # ── Semantic Scholar ──────────────────────────────────────────────────────

    def _sanitize_s2_query(self, search_terms):
        """Clean PubMed-style syntax that S2 rejects / ignores.

        S2 expects free-text keywords. Tags like [MeSH], [Publication Type],
        boolean operators in caps (AND/OR), quoted phrases and parentheses
        either return 0 hits or a 400. We strip them and keep the content words.
        """
        if not search_terms:
            return ""
        q = str(search_terms)
        # Drop PubMed-style field tags like [MeSH], [Publication Type], [tiab]
        q = re.sub(r"\[[^\]]*\]", " ", q)
        # Drop boolean operators and parentheses
        q = re.sub(r"\b(AND|OR|NOT)\b", " ", q)
        q = q.replace("(", " ").replace(")", " ").replace('"', " ").replace("*", " ")
        # Collapse whitespace and cap length (S2 silently truncates anyway)
        q = " ".join(q.split())
        return q[:300]

    def _search_semantic_scholar(self, search_terms, max_results=500):
        """Search Semantic Scholar Academic Graph API.

        Uses SEMANTIC_SCHOLAR_API_KEY (env or config) when available for higher
        rate limits. Without a key S2 returns 429 frequently; we honor the
        Retry-After header and log a clear warning instead of silently
        returning 0 results.
        """
        clean_query = self._sanitize_s2_query(search_terms)
        if clean_query != search_terms:
            print(f"[A2] S2 query sanitized: '{search_terms[:80]}' -> '{clean_query[:80]}'")
        if not clean_query:
            print("[A2] Semantic Scholar: empty query after sanitization, skipping")
            return []

        print(f"[A2] Searching Semantic Scholar: '{clean_query[:80]}...' (max {max_results})...")
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {
            "User-Agent": "SAM-MetaAnalysis/0.6 (+https://github.com/iqueipopg/SAM)",
            "Accept": "application/json",
        }
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key
        papers = []
        offset = 0
        page_size = min(100, max_results)

        while len(papers) < max_results:
            params = {
                "query": clean_query,
                "limit": page_size,
                "offset": offset,
                "fields": "paperId,externalIds,title,abstract,authors,year,journal,citationCount",
            }
            try:
                def _s2_request(p=params, h=headers):
                    r = requests.get(base_url, params=p, headers=h, timeout=30)
                    # Honor Retry-After on 429 by sleeping before the retry
                    # helper even sees the exception; the helper still applies
                    # its own backoff on top for additional safety.
                    if r.status_code == 429:
                        retry_after = r.headers.get("Retry-After", "")
                        try:
                            wait = min(float(retry_after), 30.0) if retry_after else 0.0
                        except ValueError:
                            wait = 0.0
                        if wait > 0:
                            print(f"[A2] S2 429: honoring Retry-After={wait}s")
                            time.sleep(wait)
                    r.raise_for_status()
                    return r.json()

                data = retry_with_backoff(
                    _s2_request,
                    retryable_exceptions=(requests.exceptions.RequestException,),
                    max_retries=4,
                    base_delay=2.0,
                    max_delay=60.0,
                    on_retry=lambda a, e, d: print(f"[A2] Semantic Scholar retry {a+1}: {e}"),
                )
            except requests.exceptions.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429:
                    print(
                        "[A2] Semantic Scholar 429 Too Many Requests tras reintentos. "
                        "Configura SEMANTIC_SCHOLAR_API_KEY en .env para evitarlo "
                        "(free key en https://www.semanticscholar.org/product/api#api-key-form). "
                        f"Papers acumulados hasta ahora: {len(papers)}."
                    )
                else:
                    print(f"[A2] Semantic Scholar HTTP {status}: {e}")
                break
            except Exception as e:
                print(f"[A2] Semantic Scholar request failed: {e}")
                break

            results = data.get("data", [])
            total = data.get("total", 0)
            if not results:
                break

            for r in results:
                if len(papers) >= max_results:
                    break
                ext_ids = r.get("externalIds") or {}
                pmid = str(ext_ids.get("PubMed", "")) if ext_ids.get("PubMed") else ""
                doi = str(ext_ids.get("DOI", "")) if ext_ids.get("DOI") else ""
                title = str(r.get("title", ""))
                abstract = str(r.get("abstract") or "")
                if not title or title.lower() == "none":
                    continue
                authors_list = r.get("authors") or []
                authors = [a.get("name", "").split()[-1] for a in authors_list[:3]]
                journal_info = r.get("journal") or {}
                journal = str(journal_info.get("name", ""))
                year = str(r.get("year") or "")
                citation_count = r.get("citationCount", 0)

                papers.append({
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "year": year,
                    "citation_count": citation_count,
                    "sources": ["semantic_scholar"],
                })

            offset += page_size
            if offset >= total or offset >= max_results:
                break
            time.sleep(self.rate_limit)

        print(f"[A2] Semantic Scholar: {len(papers)} papers")
        return papers

    # ── Google Scholar ─────────────────────────────────────────────────────────

    def _search_google_scholar(self, search_terms, max_results=100):
        """Search Google Scholar via scraping with polite rate limiting.

        Google Scholar has no official API. We scrape the HTML results page,
        extracting titles, snippets, authors, years, and links. We respect
        rate limits (default 3s between requests) and use a realistic
        User-Agent to avoid blocks. If blocked (CAPTCHA/429), we stop
        gracefully and return whatever we found so far.

        Max practical limit: ~100 results (10 pages of 10).
        """
        print(f"[A2] Searching Google Scholar: '{search_terms[:80]}...' (max {max_results})...")
        papers = []
        # Google Scholar returns 10 results per page
        max_pages = min(max_results // 10 + 1, 10)  # cap at 10 pages (100 results)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        from bs4 import BeautifulSoup

        for page in range(max_pages):
            if len(papers) >= max_results:
                break

            start = page * 10
            params = {
                "q": search_terms,
                "start": start,
                "hl": "en",
            }
            url = "https://scholar.google.com/scholar?" + urllib.parse.urlencode(params)

            try:
                resp = requests.get(url, headers=headers, timeout=20)

                # Check for CAPTCHA or rate limiting
                if resp.status_code == 429 or "CAPTCHA" in resp.text or "unusual traffic" in resp.text.lower():
                    print(f"[A2] Google Scholar: rate limited/CAPTCHA at page {page+1}, stopping")
                    break

                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                results = soup.select("div.gs_r.gs_or.gs_scl")
                if not results:
                    # Try alternative selector
                    results = soup.select("div.gs_ri")
                    if not results:
                        break

                for result in results:
                    if len(papers) >= max_results:
                        break

                    try:
                        # Title and link — use separator=" " so text in nested
                        # tags (<b>, <span>) stays word-separated ("...therapyfor..."
                        # vs "...therapy for..."). Then collapse runs of whitespace.
                        title_elem = result.select_one("h3.gs_rt a") or result.select_one("h3.gs_rt")
                        if not title_elem:
                            continue
                        title = title_elem.get_text(separator=" ", strip=True)
                        title = re.sub(r"\s+", " ", title).strip()
                        # Clean [PDF], [HTML], [BOOK] prefixes
                        title = re.sub(r'^\[(PDF|HTML|BOOK|CITATION)\]\s*', '', title)
                        if not title:
                            continue

                        link = title_elem.get("href", "") if title_elem.name == "a" else ""

                        # Authors, journal, year line (gs_a cite row)
                        meta_elem = result.select_one("div.gs_a")
                        authors = []
                        journal = ""
                        year = ""
                        pmid = ""
                        gs_doi = ""
                        if meta_elem:
                            meta_text = meta_elem.get_text(separator=" ", strip=True)
                            meta_text = re.sub(r"\s+", " ", meta_text).strip()
                            # Year: 4-digit token anywhere in the cite row
                            year_match = re.search(r'\b(19|20)\d{2}\b', meta_text)
                            if year_match:
                                year = year_match.group()
                            # PMID: "PMID: 12345678" or "pubmed/12345678"
                            pmid_match = re.search(
                                r'(?:PMID[:\s]+|pubmed/)(\d{5,9})', meta_text, flags=re.IGNORECASE
                            )
                            if pmid_match:
                                pmid = pmid_match.group(1)
                            # DOI inside cite row: "10.XXXX/..."
                            doi_match_meta = re.search(r'\b(10\.\d{4,9}/[^\s,]+)', meta_text)
                            if doi_match_meta:
                                gs_doi = doi_match_meta.group(1).rstrip(".,;)")
                            # Format: "Author1, Author2 - Journal, Year - Publisher"
                            parts = meta_text.split(" - ")
                            if parts:
                                author_str = parts[0].strip()
                                authors = [a.strip().split()[-1] for a in author_str.split(",")[:3]
                                           if a.strip() and not a.strip().startswith("...")]
                            if len(parts) >= 2:
                                journal_year = parts[1].strip()
                                jy_year_match = re.search(r'\b(19|20)\d{2}\b', journal_year)
                                if jy_year_match:
                                    journal = journal_year[:jy_year_match.start()].strip().rstrip(",").strip()
                                else:
                                    journal = journal_year
                            # Venue = publisher part if present (last chunk)
                            venue = parts[-1].strip() if len(parts) >= 3 else journal

                        # Abstract/snippet — also separator-aware
                        snippet_elem = result.select_one("div.gs_rs")
                        if snippet_elem:
                            abstract = snippet_elem.get_text(separator=" ", strip=True)
                            abstract = re.sub(r"\s+", " ", abstract).strip()
                        else:
                            abstract = ""

                        # Try to extract DOI from link (overrides cite-row DOI
                        # only when link contains a doi.org URL)
                        doi = gs_doi
                        if link and "doi.org/" in link:
                            doi_match = re.search(r'doi\.org/(10\.\S+)', link)
                            if doi_match:
                                doi = doi_match.group(1).rstrip(".,;)")

                        papers.append({
                            "pmid": pmid,
                            "doi": doi,
                            "title": title,
                            "abstract": abstract,
                            "authors": authors,
                            "journal": journal,
                            "venue": venue if meta_elem else "",
                            "year": year,
                            "link": link,
                            "sources": ["google_scholar"],
                        })
                    except Exception:
                        continue

            except requests.exceptions.RequestException as e:
                print(f"[A2] Google Scholar page {page+1} failed: {e}")
                break

            # Polite rate limiting — Google Scholar is aggressive about blocking
            if page < max_pages - 1:
                time.sleep(self.google_scholar_rate_limit)

        print(f"[A2] Google Scholar: {len(papers)} papers")
        return papers

    # ── Search term builder ───────────────────────────────────────────────────

    def _build_search_terms(self, protocol):
        """Build free-text search terms from protocol for non-PubMed databases."""
        parts = []
        for key in ("population", "sample", "intervention", "exposure", "outcome",
                     "phenomenon", "comparison"):
            val = protocol.get(key, "")
            if val and val not in ("?", "Primary outcome", "Intervention"):
                parts.append(str(val))
        if parts:
            return " ".join(parts)
        # Fallback: use the research question
        return protocol.get("research_question", protocol.get("query", ""))

    # ── 3-tier deduplication ──────────────────────────────────────────────────

    def deduplicate(self, papers, threshold=0.95):
        """3-tier deduplication: PMID match -> DOI match -> semantic similarity."""
        if not papers:
            print("[A2] No papers to deduplicate.")
            return []

        print(f"[A2] Deduplicating {len(papers)} papers (threshold {threshold})...")

        # --- Tier 1 & 2: exact PMID / DOI merge ---
        seen_pmid = {}   # pmid -> index in unique list
        seen_doi = {}     # doi -> index in unique list
        unique = []

        for p in papers:
            pmid = p.get("pmid", "")
            doi = p.get("doi", "").lower().strip() if p.get("doi") else ""
            merged = False

            # Tier 1: PMID match
            if pmid and pmid in seen_pmid:
                idx = seen_pmid[pmid]
                self._merge_paper(unique[idx], p)
                # Also register DOI if new
                if doi and doi not in seen_doi:
                    seen_doi[doi] = idx
                merged = True

            # Tier 2: DOI match
            elif doi and doi in seen_doi:
                idx = seen_doi[doi]
                self._merge_paper(unique[idx], p)
                # Also register PMID if new
                if pmid and pmid not in seen_pmid:
                    seen_pmid[pmid] = idx
                merged = True

            if not merged:
                idx = len(unique)
                unique.append(p)
                if pmid:
                    seen_pmid[pmid] = idx
                if doi:
                    seen_doi[doi] = idx

        exact_dedup_removed = len(papers) - len(unique)
        if exact_dedup_removed:
            print(f"[A2] Exact ID dedup removed {exact_dedup_removed} duplicates ({len(unique)} remain)")

        # --- Tier 3: semantic similarity ---
        if len(unique) < 2:
            print(f"[A2] {len(unique)} unique papers (no semantic dedup needed)")
            return unique

        texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in unique]
        embeddings = self.dedup_model.encode(texts, show_progress_bar=True)

        keep_indices = []
        for i in range(len(embeddings)):
            is_duplicate = False
            for j in keep_indices:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                if sim > threshold:
                    # Merge sources into the kept paper
                    self._merge_paper(unique[j], unique[i])
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)

        deduplicated = [unique[i] for i in keep_indices]
        sem_removed = len(unique) - len(deduplicated)
        if sem_removed:
            print(f"[A2] Semantic dedup removed {sem_removed} more duplicates")
        print(f"[A2] Final: {len(deduplicated)} unique papers")
        return deduplicated

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_paper(target, source):
        """Merge source paper into target, keeping richer metadata and merging sources."""
        # Merge sources list
        for s in source.get("sources", []):
            if s not in target.get("sources", []):
                target.setdefault("sources", []).append(s)
        # Fill missing fields from source
        for key in ("pmid", "doi", "abstract", "journal", "year"):
            if not target.get(key) and source.get(key):
                target[key] = source[key]
        # Keep higher citation count
        if source.get("citation_count", 0) > target.get("citation_count", 0):
            target["citation_count"] = source["citation_count"]

    def fetch_by_pmids(self, pmids: list) -> list:
        """Recupera abstracts directamente por lista de PMIDs."""
        if not pmids:
            return []
        handle = Entrez.efetch(
            db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml"
        )
        records = Entrez.read(handle)
        papers = []
        for record in records.get("PubmedArticle", []):
            try:
                article = record["MedlineCitation"]["Article"]
                papers.append({
                    "pmid": str(record["MedlineCitation"]["PMID"]),
                    "doi": "",
                    "title": str(article["ArticleTitle"]),
                    "abstract": str(
                        article.get("Abstract", {}).get("AbstractText", [""])[0]
                    ),
                    "authors": [
                        a.get("LastName", "")
                        for a in article.get("AuthorList", [])[:3]
                    ],
                    "journal": str(article["Journal"]["Title"]),
                    "year": str(
                        article["Journal"]["JournalIssue"]
                        .get("PubDate", {})
                        .get("Year", "")
                    ),
                    "sources": ["pubmed"],
                })
            except Exception:
                continue
        return papers

    def count_pubmed_results(self, query: str) -> int:
        """Devuelve el numero de hits de una query sin descargar abstracts."""
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        record = Entrez.read(handle)
        return int(record.get("Count", 0))


# Test
if __name__ == "__main__":
    agent = Agent2Search()

    # Test with legacy single-DB method
    papers = agent.search_pubmed(
        "cognitive behavioral therapy depression RCT", max_results=20
    )
    unique = agent.deduplicate(papers)
    print(f"\nLegacy result: {len(unique)} papers")

    # Test with multi-DB search
    protocol = {
        "population": "adults with depression",
        "intervention": "cognitive behavioral therapy",
        "outcome": "depression symptoms",
    }
    result = agent.search(
        protocol,
        pubmed_query="cognitive behavioral therapy depression RCT",
        max_results=20,
    )
    print(f"\nMulti-DB result: {len(result['papers'])} papers")
    print(f"Stats: {json.dumps(result['stats'], indent=2)}")
    if result["papers"]:
        print(f"Sample sources: {result['papers'][0].get('sources')}")
