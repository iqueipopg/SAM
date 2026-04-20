"""Test A4 new retrieval sources (Europe PMC, Semantic Scholar, Sci-Hub)
on abstract-only papers from the latest pipeline run.

No GPU needed -- only exercises the HTTP retrieval methods.

Findings from initial diagnostics:
- sci-hub.se does not resolve; sci-hub.st does
- Semantic Scholar rate-limits aggressively (429s even at 3s intervals)
- Europe PMC has these papers in search index but no full text XML available
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Diagnostic script — run directly: python tests/test_a4_sources.py"""
    from agents.agent4_fulltext import Agent4FullText

    # ---------- Load pipeline data ----------
    PIPELINE = "outputs/pipeline_20260304_171352.json"
    if not os.path.exists(PIPELINE):
        print(f"Pipeline file not found: {PIPELINE}")
        print("This is a diagnostic script. Run the pipeline first to generate data.")
        sys.exit(1)

    print(f"Loading pipeline: {PIPELINE}")
    with open(PIPELINE) as f:
        data = json.load(f)

    papers = data["fulltext_retrieval"]["papers"]
    abstract_only = [p for p in papers if p.get("text_source") == "abstract_only"]
    print(f"Total papers: {len(papers)}, abstract-only: {len(abstract_only)}\n")

    if not abstract_only:
        print("No abstract-only papers found. Nothing to test.")
        sys.exit(0)

    # ---------- Instantiate A4 (no model load) ----------
    a4 = Agent4FullText(ensemble_size=1)

    # Override sci-hub domain since sci-hub.se doesn't resolve from this host
    a4.scihub_domain = "sci-hub.st"

    # ---------- Test each source on each paper ----------
    results = []

    for i, paper in enumerate(abstract_only):
        pmid = paper.get("pmid", "?")
        doi = paper.get("doi", "")
        title = paper.get("title", "?")[:80]
        print(f"{'='*70}")
        print(f"[{i+1}/{len(abstract_only)}] PMID={pmid}  DOI={doi}")
        print(f"  Title: {title}")
        print(f"{'='*70}")

        row = {
            "pmid": pmid,
            "doi": doi,
            "title": paper.get("title", ""),
            "europepmc": None,
            "europepmc_sections": [],
            "semanticscholar": None,
            "scihub": None,
            "errors": {},
        }

        # --- Europe PMC ---
        print(f"  [Europe PMC] ... ", end="", flush=True)
        try:
            text, sections = a4._fetch_europepmc_fulltext(pmid)
            if text:
                row["europepmc"] = len(text)
                row["europepmc_sections"] = list((sections or {}).keys())
                print(f"OK -- {len(text)} chars, sections={row['europepmc_sections']}")
            else:
                print("no full text available")
        except Exception as e:
            row["errors"]["europepmc"] = str(e)
            print(f"ERROR: {e}")

        # --- Semantic Scholar (with generous rate limit) ---
        print(f"  [Semantic Scholar] ... ", end="", flush=True)
        try:
            # Force extra delay to avoid 429
            time.sleep(5)
            text = a4._fetch_semanticscholar_pdf(pmid)
            if text:
                row["semanticscholar"] = len(text)
                print(f"OK -- {len(text)} chars")
            else:
                print("no open-access PDF")
        except Exception as e:
            row["errors"]["semanticscholar"] = str(e)
            print(f"ERROR: {e}")

        # --- Sci-Hub ---
        if doi:
            print(f"  [Sci-Hub ({a4.scihub_domain})] ... ", end="", flush=True)
            try:
                text = a4._fetch_scihub_pdf(doi)
                if text:
                    row["scihub"] = len(text)
                    print(f"OK -- {len(text)} chars")
                else:
                    print("not available")
            except Exception as e:
                row["errors"]["scihub"] = str(e)
                print(f"ERROR: {e}")
        else:
            print(f"  [Sci-Hub] skipped (no DOI)")

        results.append(row)
        print()

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY: New retrieval sources for abstract-only papers")
    print("=" * 70)
    print(f"{'PMID':<12} {'EuropePMC':>12} {'S2':>12} {'Sci-Hub':>12}  Title")
    print("-" * 70)

    total_rescued = 0
    for r in results:
        epmc = f"{r['europepmc']} ch" if r['europepmc'] else "---"
        s2 = f"{r['semanticscholar']} ch" if r['semanticscholar'] else "---"
        sh = f"{r['scihub']} ch" if r['scihub'] else "---"
        title_short = r['title'][:30]
        rescued = any([r['europepmc'], r['semanticscholar'], r['scihub']])
        marker = " *RESCUED*" if rescued else ""
        if rescued:
            total_rescued += 1
        print(f"{r['pmid']:<12} {epmc:>12} {s2:>12} {sh:>12}  {title_short}{marker}")

    print("-" * 70)

    n = len(results)
    epmc_ok = sum(1 for r in results if r['europepmc'])
    s2_ok = sum(1 for r in results if r['semanticscholar'])
    sh_ok = sum(1 for r in results if r['scihub'])
    print(f"\nHit rates:  Europe PMC={epmc_ok}/{n}, Semantic Scholar={s2_ok}/{n}, Sci-Hub={sh_ok}/{n}")
    print(f"Papers rescued (at least 1 source): {total_rescued}/{n}")
    print(f"Still abstract-only: {n - total_rescued}/{n}")

    # Report errors
    any_errors = any(r['errors'] for r in results)
    if any_errors:
        print(f"\nErrors encountered:")
        for r in results:
            for src, err in r['errors'].items():
                print(f"  PMID={r['pmid']} [{src}]: {err}")

    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    print("- Europe PMC: Returns 404 for all papers. They are indexed in EuropePMC")
    print("  search but full-text XML is not available (papers are not open access).")
    print("- Semantic Scholar: Aggressive rate limiting (429). For the papers that")
    print("  did respond, isOpenAccess=False and no PDF URL was provided.")
    print("- Sci-Hub: sci-hub.se does not resolve from this host.")
    print(f"  Tested sci-hub.st instead (resolves to 186.2.163.201).")
    print()
    print("Root cause: All 9 abstract-only papers are behind publisher paywalls")
    print("(Elsevier, Wiley, SAGE, etc.) and none are deposited as open access")
    print("in PMC/EuropePMC or indexed with OA PDFs in Semantic Scholar.")


if __name__ == "__main__":
    main()
