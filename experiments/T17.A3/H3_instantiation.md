# T17.A3 H.3.0 — Instantiation of operating thresholds

**Timestamp**: 2026-05-20 02:57 UTC (post first iteration run)
**Pre-commit anchor**: `4dc75f4` (tag `pre-T17.A3`)

## What H.3.0 specifies (pre-registered)

From `docs/iterations/T17.A3_hypothesis.md` §8:
- `floor = max(V0_macro + 0.05, 0.50)`
- `ambitious = V0_macro + 0.10`

The formulas were frozen before any data was generated. The post-V0 instantiation step (H.3.0) is the pre-registered moment to plug in the measured `V0_macro`.

## V0 baseline measured (DEV N=20, seed=42)

| Metric | Value |
|---|---|
| `V0_macro` (primary) | **0.7028** |
| `pct_samples_3_of_4_above_0.5` (secondary) | 0.650 |
| Per-element P / I / C / O | 0.866 / 0.873 / 0.521 / 0.551 |
| Median per-sample mean cosine | 0.7034 |

Source: `experiments/T17.A3/cache/V0/eval.json` (eval ran with `metadata_prompt_template=V0.txt`, which renders bit-identically to the hardcoded baseline of `_generate_metadata` when `prompt_template=None`).

## Operating thresholds instantiated

| Threshold | Formula | Value |
|---|---|---|
| Floor (`--target-macro`) | `max(0.7028 + 0.05, 0.50)` | **0.7528** |
| Ambitious (`--ambitious-macro`) | `0.7028 + 0.10` | **0.8028** |

## Why the first iteration run is discarded for the canonical trajectory

The first run used `--ambitious-macro 0.60` (placeholder default), which triggered early termination after only 2 variants (V0 + V1) because both happened to exceed 0.60. This is **not** the pre-registered behaviour — the pre-registration explicitly defers operating thresholds to this H.3.0 step.

V0 and V1 results are preserved via:
- `experiments/T17.A3/iteration_log_h3_calibration.json` — first-run log
- `experiments/T17.A3/best_variant_h3_calibration.json` — first-run best
- `experiments/T17.A3/cache/V0/eval.json` — V0 raw eval (reused by second run)
- `experiments/T17.A3/cache/V1/eval.json` — V1 raw eval (reused by second run)

The second run, launched with the H.3.0 thresholds above, is the canonical trajectory. V0 and V1 are reused from cache (no re-execution); the orchestrator proceeds from V2 onward.

## Anti-HARKing trace

- Δ-based thresholds (+0.05 macro, +0.10 secondary) were pre-registered in `T17.A3_hypothesis.md` §7.2-§7.3 **before** any data was generated.
- The deferral to H.3.0 was explicit in §8.
- The instantiation `floor=0.7528, ambitious=0.8028` is a deterministic function of the pre-registered formula applied to the measured V0.
- No threshold formula has been rewritten post-hoc.

This document is committed alongside the canonical run to make the H.3.0 step auditable.
