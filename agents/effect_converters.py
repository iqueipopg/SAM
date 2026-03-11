"""Effect size conversion functions for meta-analysis pooling.

Enables Agent 7 to combine studies that report different but compatible
measures (e.g. MD + SMD, OR + RR) by converting to a common scale.

All conversions propagate uncertainty — SE and CI are recalculated after
transformation.  Edge cases (None, zero, negative where invalid) return None.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


# ------------------------------------------------------------------ #
#  Continuous-scale conversions                                        #
# ------------------------------------------------------------------ #

def md_to_smd(md: float, sd_pooled: float) -> Optional[float]:
    """Convert raw Mean Difference to Standardised Mean Difference.

    SMD = MD / SD_pooled
    """
    if md is None or sd_pooled is None:
        return None
    if sd_pooled <= 0:
        return None
    return md / sd_pooled


def smd_to_md(smd: float, sd_pooled: float) -> Optional[float]:
    """Convert SMD back to raw Mean Difference.

    MD = SMD * SD_pooled
    """
    if smd is None or sd_pooled is None:
        return None
    if sd_pooled <= 0:
        return None
    return smd * sd_pooled


# ------------------------------------------------------------------ #
#  Binary-outcome conversions                                          #
# ------------------------------------------------------------------ #

def or_to_logit(or_val: float) -> Optional[Tuple[float, float]]:
    """Convert Odds Ratio to log-odds (logit scale).

    logit = ln(OR)
    Returns (logit, None) — SE must be propagated separately.
    """
    if or_val is None or or_val <= 0:
        return None
    return (math.log(or_val), None)


def rr_to_logit(rr_val: float) -> Optional[float]:
    """Convert Risk Ratio to log-RR.

    log_RR = ln(RR)
    """
    if rr_val is None or rr_val <= 0:
        return None
    return math.log(rr_val)


def or_to_rr(or_val: float, baseline_risk: float) -> Optional[float]:
    """Convert Odds Ratio to Risk Ratio given a baseline risk (p0).

    RR = OR / (1 - p0 + p0 * OR)
    baseline_risk must be in (0, 1).
    """
    if or_val is None or baseline_risk is None:
        return None
    if or_val <= 0 or baseline_risk <= 0 or baseline_risk >= 1:
        return None
    denom = 1.0 - baseline_risk + baseline_risk * or_val
    if denom <= 0:
        return None
    return or_val / denom


def rr_to_or(rr_val: float, baseline_risk: float) -> Optional[float]:
    """Convert Risk Ratio to Odds Ratio given a baseline risk (p0).

    OR = RR * (1 - p0) / (1 - RR * p0)
    """
    if rr_val is None or baseline_risk is None:
        return None
    if rr_val <= 0 or baseline_risk <= 0 or baseline_risk >= 1:
        return None
    denom = 1.0 - rr_val * baseline_risk
    if denom <= 0:
        return None
    return rr_val * (1.0 - baseline_risk) / denom


# ------------------------------------------------------------------ #
#  Correlation / Cohen's d conversions                                 #
# ------------------------------------------------------------------ #

def d_to_r(d: float) -> Optional[float]:
    """Convert Cohen's d to Pearson r.

    r = d / sqrt(d^2 + 4)
    """
    if d is None:
        return None
    denom = math.sqrt(d * d + 4.0)
    if denom == 0:
        return None
    return d / denom


def r_to_d(r: float) -> Optional[float]:
    """Convert Pearson r to Cohen's d.

    d = 2r / sqrt(1 - r^2)
    """
    if r is None:
        return None
    if abs(r) >= 1.0:
        return None
    return 2.0 * r / math.sqrt(1.0 - r * r)


# ------------------------------------------------------------------ #
#  From raw group statistics to SMD (Hedges' g)                        #
# ------------------------------------------------------------------ #

def means_sds_to_smd(
    m1: float, sd1: float, n1: int,
    m2: float, sd2: float, n2: int,
) -> Optional[Tuple[float, float]]:
    """Compute Hedges' g (bias-corrected SMD) from group-level statistics.

    Returns (g, se_g) or None if inputs are invalid.

    Steps:
      1. SD_pooled = sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1+n2-2))
      2. d = (m1 - m2) / SD_pooled
      3. SE_d = sqrt(1/n1 + 1/n2 + d^2 / (2*(n1+n2)))
      4. Hedges' J = 1 - 3 / (4*(n1+n2-2) - 1)
      5. g = J * d,  se_g = J * SE_d
    """
    if any(v is None for v in (m1, sd1, n1, m2, sd2, n2)):
        return None
    try:
        n1, n2 = int(n1), int(n2)
    except (ValueError, TypeError):
        return None
    if n1 < 2 or n2 < 2:
        return None
    if sd1 <= 0 or sd2 <= 0:
        return None

    df = n1 + n2 - 2
    sd_pooled = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df)
    if sd_pooled <= 0:
        return None

    d = (m1 - m2) / sd_pooled
    se_d = math.sqrt(1.0 / n1 + 1.0 / n2 + d ** 2 / (2.0 * (n1 + n2)))

    # Hedges' g correction
    j_corr = 4.0 * df - 1.0
    if j_corr <= 0:
        return None
    j = 1.0 - 3.0 / j_corr
    g = j * d
    se_g = j * se_d

    return (g, se_g)


# ------------------------------------------------------------------ #
#  CI ↔ SE helpers                                                     #
# ------------------------------------------------------------------ #

def ci_to_se(lower: float, upper: float, ci_level: float = 0.95) -> Optional[float]:
    """Recover SE from a symmetric confidence interval.

    SE = (upper - lower) / (2 * z)
    where z corresponds to ci_level (default 1.96 for 95 %).
    """
    if lower is None or upper is None:
        return None
    width = upper - lower
    if width <= 0:
        return None
    z = _z_for_ci(ci_level)
    return width / (2.0 * z)


def propagate_ci(
    value: float, se: float, ci_level: float = 0.95
) -> Optional[Tuple[float, float]]:
    """Build a symmetric CI from a point estimate and SE.

    Returns (lower, upper).
    """
    if value is None or se is None:
        return None
    if se < 0:
        return None
    z = _z_for_ci(ci_level)
    return (value - z * se, value + z * se)


# ------------------------------------------------------------------ #
#  High-level conversion helpers (used by Agent 7)                     #
# ------------------------------------------------------------------ #

def convert_md_to_smd_study(
    yi: float,
    vi: float,
    outcome: dict,
) -> Optional[dict]:
    """Try to convert a single MD study to SMD.

    Looks for sd_pooled or group-level stats in the outcome dict.
    Returns dict with converted yi, vi, se, ci, and metadata, or None.
    """
    if yi is None or vi is None:
        return None

    se_original = math.sqrt(vi) if vi > 0 else None

    # Strategy 1: sd_pooled available directly
    sd_pooled = _get_float(outcome, "sd_pooled")
    if sd_pooled and sd_pooled > 0:
        smd = md_to_smd(yi, sd_pooled)
        if smd is None:
            return None
        # SE transforms the same way: se_smd = se_md / sd_pooled
        se_smd = se_original / sd_pooled if se_original else None
        return _build_conversion_result(
            smd, se_smd,
            f"Converted from MD to SMD using SDpooled={sd_pooled:.3f}",
            yi, se_original,
        )

    # Strategy 2: group-level means, SDs, Ns
    npg = outcome.get("n_per_group") or {}
    groups = list(npg.values()) if isinstance(npg, dict) else []
    if len(groups) >= 2:
        try:
            n1, n2 = int(groups[0]), int(groups[1])
        except (ValueError, TypeError):
            n1 = n2 = None

        # We need group SDs — check for them in the outcome
        sd1 = _get_float(outcome, "sd1") or _get_float(outcome, "sd_group1")
        sd2 = _get_float(outcome, "sd2") or _get_float(outcome, "sd_group2")

        if n1 and n2 and sd1 and sd2 and n1 >= 2 and n2 >= 2:
            df = n1 + n2 - 2
            sd_pooled = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / df)
            if sd_pooled > 0:
                smd = md_to_smd(yi, sd_pooled)
                if smd is None:
                    return None
                se_smd = se_original / sd_pooled if se_original else None
                # Apply Hedges' g correction
                j_denom = 4.0 * df - 1.0
                if j_denom > 0:
                    j = 1.0 - 3.0 / j_denom
                    smd = j * smd
                    se_smd = j * se_smd if se_smd else None
                return _build_conversion_result(
                    smd, se_smd,
                    f"Converted from MD to SMD (Hedges' g) using SDpooled={sd_pooled:.3f} "
                    f"(n1={n1}, n2={n2})",
                    yi, se_original,
                )

    # Strategy 3: rough estimate — use SE of MD as a proxy for SD_pooled
    # This is a last-resort heuristic: if we have the sample sizes we can
    # approximate SD_pooled ≈ SE_md * sqrt(n1*n2/(n1+n2)).
    # Only valid when both group Ns are known.
    if len(groups) >= 2 and se_original:
        try:
            n1, n2 = int(groups[0]), int(groups[1])
        except (ValueError, TypeError):
            n1 = n2 = None
        if n1 and n2 and n1 >= 2 and n2 >= 2:
            sd_pooled_approx = se_original * math.sqrt(n1 * n2 / (n1 + n2))
            if sd_pooled_approx > 0:
                smd = md_to_smd(yi, sd_pooled_approx)
                if smd is None:
                    return None
                se_smd = se_original / sd_pooled_approx
                return _build_conversion_result(
                    smd, se_smd,
                    f"Converted from MD to SMD (approx) using estimated SDpooled={sd_pooled_approx:.3f} "
                    f"from SE and group Ns (n1={n1}, n2={n2})",
                    yi, se_original,
                )

    # Strategy 4: n_total available but no per-group Ns — assume 50/50 split
    # Reasonable heuristic for RCTs. Use SE to approximate SD_pooled.
    n_total = _get_float(outcome, "n_total")
    if n_total and n_total >= 4 and se_original and se_original > 0:
        n1_est = int(n_total / 2)
        n2_est = int(n_total) - n1_est
        if n1_est >= 2 and n2_est >= 2:
            sd_pooled_approx = se_original * math.sqrt(n1_est * n2_est / (n1_est + n2_est))
            if sd_pooled_approx > 0:
                smd = md_to_smd(yi, sd_pooled_approx)
                if smd is None:
                    return None
                se_smd = se_original / sd_pooled_approx
                return _build_conversion_result(
                    smd, se_smd,
                    f"Converted from MD to SMD (approx, 50/50 split) using estimated SDpooled="
                    f"{sd_pooled_approx:.3f} from SE and N_total={int(n_total)} "
                    f"(assumed n1={n1_est}, n2={n2_est})",
                    yi, se_original,
                )

    return None


def convert_rr_to_or_study(
    yi: float,
    vi: float,
    baseline_risk: float = 0.1,
) -> Optional[dict]:
    """Convert a study on the RR scale to OR scale.

    Works on the log scale: log(OR) from log(RR).
    If yi is already a raw RR (> 0), we log-transform first.
    baseline_risk is the assumed control-group event rate (0 < p0 < 1).

    Returns dict with converted yi (log-OR), vi, se, ci, and metadata, or None.
    """
    if yi is None or vi is None:
        return None
    if baseline_risk is None or baseline_risk <= 0 or baseline_risk >= 1:
        return None

    se_rr = math.sqrt(vi) if vi > 0 else None

    # Determine if yi is on log scale or raw scale
    # Heuristic: if |yi| < 5 and can be negative, likely log scale already
    # If yi > 0 and looks like a ratio, convert
    if yi > 0 and yi != 1.0:
        # Could be raw RR — check if it looks like a ratio
        # log(RR) typically in range (-3, 3), raw RR in (0.01, 100)
        if yi > 5 or (0 < yi < 0.2):
            # Almost certainly a raw RR
            log_rr = math.log(yi)
            # Delta method: se_log_rr = se_rr / rr
            se_log_rr = se_rr / yi if se_rr else None
        else:
            # Ambiguous — assume log scale if negative possible
            log_rr = yi
            se_log_rr = se_rr
    else:
        log_rr = yi
        se_log_rr = se_rr

    # Convert log(RR) to log(OR)
    # OR = RR * (1 - p0) / (1 - RR * p0)
    # log(OR) = log(RR) + log((1-p0)/(1-RR*p0))
    rr = math.exp(log_rr)
    or_val = rr_to_or(rr, baseline_risk)
    if or_val is None or or_val <= 0:
        return None
    log_or = math.log(or_val)

    # Approximate SE on log-OR scale using delta method
    # d(log_or)/d(log_rr) ≈ 1 + p0*rr/(1 - p0*rr)  (via chain rule)
    denom = 1.0 - baseline_risk * rr
    if denom <= 0:
        return None
    derivative = 1.0 + baseline_risk * rr / denom
    se_log_or = se_log_rr * abs(derivative) if se_log_rr else None
    vi_log_or = se_log_or ** 2 if se_log_or else None

    return _build_conversion_result(
        log_or, se_log_or,
        f"Converted from RR to OR (log scale) using baseline_risk={baseline_risk:.3f}",
        yi, se_rr,
        vi_override=vi_log_or,
    )


def convert_or_to_rr_study(
    yi: float,
    vi: float,
    baseline_risk: float = 0.1,
) -> Optional[dict]:
    """Convert a study on the OR scale to RR scale.

    Analogous to convert_rr_to_or_study but in reverse direction.
    Returns dict with converted yi (log-RR), vi, se, ci, and metadata, or None.
    """
    if yi is None or vi is None:
        return None
    if baseline_risk is None or baseline_risk <= 0 or baseline_risk >= 1:
        return None

    se_or = math.sqrt(vi) if vi > 0 else None

    # Determine log vs raw scale (same heuristic)
    if yi > 0 and yi != 1.0 and (yi > 5 or 0 < yi < 0.2):
        log_or = math.log(yi)
        se_log_or = se_or / yi if se_or else None
    else:
        log_or = yi
        se_log_or = se_or

    or_raw = math.exp(log_or)
    rr_val = or_to_rr(or_raw, baseline_risk)
    if rr_val is None or rr_val <= 0:
        return None
    log_rr = math.log(rr_val)

    # Delta method for SE
    denom = 1.0 - baseline_risk + baseline_risk * or_raw
    if denom <= 0:
        return None
    # d(log_rr)/d(log_or) = or_raw * d(rr)/d(or) / rr
    # d(rr)/d(or) = (1-p0) / (1-p0+p0*or)^2  ... simplified
    derivative_rr_or = (1.0 - baseline_risk) / (denom ** 2)
    derivative_log = or_raw * derivative_rr_or / rr_val if rr_val > 0 else None
    se_log_rr = se_log_or * abs(derivative_log) if (se_log_or and derivative_log) else None
    vi_log_rr = se_log_rr ** 2 if se_log_rr else None

    return _build_conversion_result(
        log_rr, se_log_rr,
        f"Converted from OR to RR (log scale) using baseline_risk={baseline_risk:.3f}",
        yi, se_or,
        vi_override=vi_log_rr,
    )


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _z_for_ci(ci_level: float) -> float:
    """Return z-value for a given CI level.  Supports common levels."""
    # Avoid scipy dependency — use lookup for common values
    table = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}
    return table.get(ci_level, 1.96)


def _get_float(d: dict, key: str) -> Optional[float]:
    """Safely extract a float from a dict."""
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _build_conversion_result(
    yi_new: float,
    se_new: Optional[float],
    note: str,
    original_yi: float,
    original_se: Optional[float],
    vi_override: Optional[float] = None,
) -> dict:
    """Package a conversion result with metadata."""
    vi_new = vi_override if vi_override is not None else (se_new ** 2 if se_new else None)
    ci = propagate_ci(yi_new, se_new) if se_new else None

    result = {
        "yi": yi_new,
        "vi": vi_new,
        "se": se_new,
        "ci_lower": ci[0] if ci else None,
        "ci_upper": ci[1] if ci else None,
        "conversion_note": note,
        "converted": True,
        "original_effect_size": original_yi,
        "original_se": original_se,
    }
    # Compute original CI for record-keeping
    if original_se:
        orig_ci = propagate_ci(original_yi, original_se)
        if orig_ci:
            result["original_ci_lower"] = orig_ci[0]
            result["original_ci_upper"] = orig_ci[1]
    return result
