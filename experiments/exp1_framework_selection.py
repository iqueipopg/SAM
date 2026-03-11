"""
Exp 1 — Framework Classification Accuracy (A1)
Mide: Accuracy y confusion matrix 4x4 (PICO/SPIDER/PECO/PCC).
Dataset: 12 queries anotadas manualmente (3 por framework).
"""
import sys, json, gc, time, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agent1_protocol import Agent1Protocol

QUERIES = [
    # PICO — interventions with measurable outcome
    {"query": "Is CBT more effective than antidepressants for depression in adults?", "expected": "PICO"},
    {"query": "Does aerobic exercise reduce HbA1c in type 2 diabetic patients?", "expected": "PICO"},
    {"query": "Effectiveness of mindfulness vs medication for anxiety disorders", "expected": "PICO"},
    # SPIDER — qualitative experiences
    {"query": "What are nurses experiences with burnout in ICU settings?", "expected": "SPIDER"},
    {"query": "How do cancer patients perceive palliative care quality?", "expected": "SPIDER"},
    {"query": "What barriers do GPs face when prescribing opioids for chronic pain?", "expected": "SPIDER"},
    # PECO — etiology / risk factors
    {"query": "Does smoking increase the risk of myocardial infarction in type 2 diabetics?", "expected": "PECO"},
    {"query": "Is air pollution associated with increased asthma incidence in children?", "expected": "PECO"},
    {"query": "Does sleep deprivation increase obesity risk in adolescents?", "expected": "PECO"},
    # PCC — prevalence / scoping
    {"query": "What is the prevalence of anxiety disorders among university students?", "expected": "PCC"},
    {"query": "How common is burnout among medical residents worldwide?", "expected": "PCC"},
    {"query": "What is the global prevalence of type 2 diabetes in adults over 60?", "expected": "PCC"},
]

FRAMEWORKS = ["PICO", "SPIDER", "PECO", "PCC"]
RESULTS_DIR = Path("experiments/results")


def free_vram(agent):
    if hasattr(agent, "model") and agent.model is not None:
        del agent.model
        agent.model = None
    if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
        del agent.tokenizer
        agent.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Exp 1] Framework Selection Accuracy")
    print("=" * 55)
    a1 = Agent1Protocol()
    rows = []
    confusion = {e: {p: 0 for p in FRAMEWORKS} for e in FRAMEWORKS}

    t0 = time.time()
    for item in QUERIES:
        protocol = a1.generate_protocol(item["query"])
        predicted = protocol.get("framework", "UNKNOWN").upper()
        expected = item["expected"]
        correct = predicted == expected
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
        rows.append({
            "query": item["query"], "expected": expected,
            "predicted": predicted, "correct": correct,
        })
        icon = "OK" if correct else "XX"
        print(f"  [{icon}] {expected}->{predicted}  {item['query'][:55]}...")

    free_vram(a1)
    elapsed = round(time.time() - t0, 1)

    accuracy = sum(1 for r in rows if r["correct"]) / len(rows)
    print(f"\n  Accuracy: {sum(r['correct'] for r in rows)}/{len(rows)} = {accuracy:.1%}")
    print(f"\n  Confusion matrix (row=expected, col=predicted):")
    print(f"  {'':>8}", "  ".join(f"{f:>6}" for f in FRAMEWORKS))
    for e in FRAMEWORKS:
        print(f"  {e:>8}", "  ".join(f"{confusion[e].get(p,0):>6}" for p in FRAMEWORKS))

    output = {
        "experiment": "exp1_framework_selection",
        "n_queries": len(QUERIES),
        "accuracy": round(accuracy, 4),
        "elapsed_s": elapsed,
        "rows": rows,
        "confusion_matrix": confusion,
    }
    out_path = RESULTS_DIR / "exp1_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {out_path}")
    return output


if __name__ == "__main__":
    run()
