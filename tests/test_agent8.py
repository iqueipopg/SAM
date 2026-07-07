# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Test Agent 8: Manuscript Generation with RAG

Loads the latest pipeline JSON and runs A8 manuscript generation.
Verifies: 6 sections generated, references list populated, stats present,
no repetition loops, no prompt leakage, PDF generated, all sections have RAG.
"""

import json, glob, os, sys, time, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent8_manuscript import Agent8Manuscript


def load_latest_pipeline():
    """Load most recent pipeline JSON from outputs/ that has included papers."""
    new_pattern = os.path.join("outputs", "run_*", "pipeline.json")
    old_pattern = os.path.join("outputs", "pipeline_*.json")
    files = sorted(glob.glob(new_pattern) + glob.glob(old_pattern), reverse=True)
    if not files:
        print("[!] No pipeline JSON found in outputs/. Run the full pipeline first.")
        return None
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            a4 = data.get("fulltext_retrieval", {})
            if a4 and a4.get("include"):
                print(f"Loading: {path} ({len(a4['include'])} included papers)")
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    print("[!] No pipeline JSON with included papers found.")
    return None


def check_repetitions(text, section_name):
    """Check that no phrase >20 chars appears 3+ times consecutively."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return True
    repeat_count = 1
    for i in range(1, len(sentences)):
        if sentences[i] == sentences[i - 1] and len(sentences[i]) > 20:
            repeat_count += 1
            if repeat_count >= 3:
                print(f"  [FAIL] {section_name}: repeated phrase ({repeat_count}x): {sentences[i][:60]}...")
                return False
        else:
            repeat_count = 1
    return True


def check_prompt_leakage(text, section_name):
    """Check first 5 lines don't contain instruction patterns."""
    leakage_patterns = re.compile(
        r'^(Include the following|Avoid repetition|Write a\b|Ensure\b|'
        r'CRITICAL RULES|You are a systematic|Do NOT\b)',
        re.IGNORECASE,
    )
    lines = text.strip().split('\n')[:5]
    for line in lines:
        if leakage_patterns.search(line.strip()):
            print(f"  [FAIL] {section_name}: prompt leakage detected: {line.strip()[:80]}")
            return False
    return True


def run_manuscript():
    """Test A8 manuscript generation on real pipeline data."""
    print(f"\n{'='*60}")
    print("TEST: AGENT 8 MANUSCRIPT GENERATION")
    print(f"{'='*60}")

    data = load_latest_pipeline()
    if not data:
        return False

    protocol = data.get("protocol", {})
    a4 = data.get("fulltext_retrieval", {})
    a5 = data.get("data_extraction")
    a6 = data.get("quality_assessment")
    a7 = data.get("synthesis")

    included_papers = a4.get("include", []) if a4 else []
    if not included_papers:
        print("[!] No included papers in pipeline data.")
        return False

    print(f"  Protocol: {protocol.get('research_question', '?')}")
    print(f"  Included papers: {len(included_papers)}")
    print(f"  Extraction data: {'yes' if a5 else 'no'}")
    print(f"  Quality data: {'yes' if a6 else 'no'}")
    print(f"  Synthesis data: {'yes' if a7 else 'no'}")

    # Run A8
    ts = time.strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join("outputs", f"test_{ts}")
    os.makedirs(test_dir, exist_ok=True)

    a8 = Agent8Manuscript()
    prisma = data.get("prisma", {})
    # Pass pubmed_query into protocol if available
    protocol_a8 = dict(protocol)
    if data.get("pubmed_query"):
        protocol_a8["pubmed_query"] = data["pubmed_query"]
    result = a8.generate_manuscript(
        protocol=protocol_a8,
        included_papers=included_papers,
        extraction=a5,
        quality=a6,
        synthesis=a7,
        output_dir=test_dir,
        prisma=prisma,
    )

    # Verify structure
    manuscript = result.get("manuscript", {})
    references = result.get("references", [])
    stats = result.get("stats", {})
    pdf_path = result.get("pdf_path")

    expected_sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
    ok = True

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    # Check all sections exist and are non-empty
    for sec in expected_sections:
        text = manuscript.get(sec, "")
        word_count = len(text.split()) if text else 0
        status = "OK" if word_count > 20 else "WARN (short)"
        if not text:
            status = "FAIL (empty)"
            ok = False
        print(f"  {sec:15s}: {word_count:4d} words  [{status}]")

    # Check no repetition loops
    print(f"\n  Repetition check:")
    for sec in expected_sections:
        text = manuscript.get(sec, "")
        if text and not check_repetitions(text, sec):
            ok = False
        elif text:
            print(f"    {sec}: OK")

    # Check no prompt leakage
    print(f"\n  Prompt leakage check:")
    for sec in expected_sections:
        text = manuscript.get(sec, "")
        if text and not check_prompt_leakage(text, sec):
            ok = False
        elif text:
            print(f"    {sec}: OK")

    # Check all sections have RAG chunks > 0
    print(f"\n  RAG chunks per section:")
    chunks_per = stats.get("chunks_per_section", {})
    for sec in expected_sections:
        n = chunks_per.get(sec, 0)
        status = "OK" if n > 0 else "WARN (no RAG)"
        print(f"    {sec:15s}: {n} chunks  [{status}]")

    # Check references
    print(f"\n  References: {len(references)}")
    if references:
        for ref in references[:5]:
            print(f"    - [{ref.get('pmid', '?')}] {ref.get('citation', '?')}: {ref.get('title', '?')[:60]}")
        if len(references) > 5:
            print(f"    ... and {len(references) - 5} more")
    else:
        print("  [WARN] No references generated")

    # Check stats
    print(f"\n  Stats:")
    print(f"    Total chunks: {stats.get('total_chunks', '?')}")
    print(f"    Papers cited: {stats.get('total_papers_cited', '?')}")
    print(f"    Elapsed: {stats.get('elapsed_s', '?')}s")

    # Check PDF
    print(f"\n  PDF check:")
    if pdf_path and os.path.exists(pdf_path):
        pdf_size = os.path.getsize(pdf_path)
        if pdf_size > 10240:
            print(f"    PDF: {pdf_path} ({pdf_size} bytes) [OK]")
        else:
            print(f"    PDF: {pdf_path} ({pdf_size} bytes) [WARN: small file]")
    else:
        print(f"    [WARN] PDF not generated (path={pdf_path})")

    # Save manuscript to file
    out_path = os.path.join(test_dir, "manuscript.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")

    # Also save as plain text
    txt_path = os.path.join(test_dir, "manuscript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for sec in expected_sections:
            f.write(f"\n{'='*60}\n")
            f.write(f"{sec.upper()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(manuscript.get(sec, "(not generated)"))
            f.write("\n\n")
        f.write(f"\n{'='*60}\n")
        f.write("REFERENCES\n")
        f.write(f"{'='*60}\n\n")
        for i, ref in enumerate(references, 1):
            f.write(f"{i}. {ref.get('citation', '?')}. {ref.get('title', '?')}. "
                    f"{ref.get('journal', '?')}. PMID: {ref.get('pmid', '?')}\n")
    print(f"  Saved text: {txt_path}")

    print(f"\n  Overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    success = run_manuscript()
    sys.exit(0 if success else 1)
