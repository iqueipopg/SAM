# Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
# SPDX-License-Identifier: Apache-2.0
"""Agent4FullTextSplit — A4 fulltext con Phase B dividida en reporter + decider.

Decisión arquitectónica T17.B.1: el screening de fulltext se ejecuta en dos
pasadas LLM secuenciales sobre Qwen 2.5-7B 4-bit:
1. Reporter: describe el paper (population, intervention, comparator, outcomes,
   study_design, publication_type, language, notes) sin emitir decisión.
2. Decider: recibe el reporte JSON del reporter + criterios PICO y emite la
   decisión INCLUDE/EXCLUDE sin ver el paper original.

Phase A (retrieval cascada) se hereda intacta de Agent4FullText. Phase B
(screen_fulltext) se sobrescribe. A4 monolítico permanece intacto.

La flag `a4_implementation` no se introduce en config.yaml en T17.B.1 (se
difiere hasta que el experimento pase el gate optimistic — Phase 4). El
evaluador experimental instancia esta clase directamente.

Referencia: docs/iterations/T17.B.1_hypothesis.md sección 6.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from agents.agent4_fulltext import Agent4FullText
from agents.prompts import (
    AGENT4_DECIDER_TEMPLATE_V1,
    AGENT4_REPORTER_TEMPLATE_V1,
    format_agent4_decider_prompt,
    format_agent4_reporter_prompt,
)


# Schemas (typed shapes). Used by the parsers for shallow validation.
# T17.B.1 hypothesis §11.3.
REPORTER_REQUIRED_KEYS_V1: Tuple[str, ...] = (
    "population",
    "intervention",
    "comparator",
    "outcomes",
    "study_design",
    "publication_type",
    "language",
    "notes",
)
REPORTER_REQUIRED_KEYS_V2: Tuple[str, ...] = REPORTER_REQUIRED_KEYS_V1 + (
    "exclusion_signals",
)

DECIDER_REQUIRED_KEYS: Tuple[str, ...] = ("decision", "reason", "confidence")
DECIDER_VALID_DECISIONS = ("INCLUDE", "EXCLUDE")
DECIDER_VALID_CONFIDENCE = ("high", "medium", "low")


class Agent4FullTextSplit(Agent4FullText):
    """A4 fulltext with reporter+decider split (Phase B). Phase A is inherited.

    Per-paper Phase B pipeline:
        1. Render reporter prompt (paper + PICO reference, no judgement).
        2. Generate reporter output (LLM call #1, same Qwen instance).
        3. Parse + validate reporter JSON. On failure → conservative
           EXCLUDE fallback with raw output logged in `fulltext_votes`.
        4. Render decider prompt (reporter JSON + criteria; NO paper).
        5. Generate decider output (LLM call #2, same Qwen instance).
        6. Parse + validate decider JSON. On failure → EXCLUDE fallback.
        7. Aggregate per-paper result with both sub-outputs.

    The split shares the model slot — no free_vram() between reporter and
    decider of the same paper. Repetition_penalty=1.3 (project rule #5).
    Temperatures 0.2 (A4 family). Single-vote (no ensemble) per T17.B.1
    lock-in #10.
    """

    # The split is single-vote by construction (lock-in #10). Ensemble is
    # intentionally not applied here.
    def __init__(
        self,
        ensemble_size: int = 1,
        voting_threshold: float = 0.60,
        reporter_prompt_template: str = AGENT4_REPORTER_TEMPLATE_V1,
        decider_prompt_template: str = AGENT4_DECIDER_TEMPLATE_V1,
        reporter_temperature: float = 0.2,
        decider_temperature: float = 0.2,
        reporter_max_tokens: int = 500,
        decider_max_tokens: int = 200,
    ):
        # Init parent: registers `agent4_validation` against config.yaml,
        # loads agents.json defaults + agent4_validation overrides, and
        # configures Phase A retrieval state.
        super().__init__(
            ensemble_size=ensemble_size,
            voting_threshold=voting_threshold,
        )

        # Overlay split-specific params from agents.json (entry
        # `agent4_validation_split`). Backward-compatible: if the entry is
        # missing, the constructor defaults persist.
        split_params = self._load_agent_params("agent4_validation_split")
        if split_params:
            # Merge over existing params so callers using
            # self.get_agent_param(...) still find the split values.
            self.agent_params = {**self.agent_params, **split_params}

        # Store split-specific runtime knobs. Constructor values win over
        # agents.json values to keep call-site overrides explicit.
        self.reporter_prompt_template = reporter_prompt_template
        self.decider_prompt_template = decider_prompt_template
        self.reporter_temperature = float(
            reporter_temperature
            if reporter_temperature is not None
            else self.get_agent_param("reporter_temperature", 0.2)
        )
        self.decider_temperature = float(
            decider_temperature
            if decider_temperature is not None
            else self.get_agent_param("decider_temperature", 0.2)
        )
        self.reporter_max_tokens = int(
            reporter_max_tokens
            if reporter_max_tokens is not None
            else self.get_agent_param("reporter_max_tokens", 500)
        )
        self.decider_max_tokens = int(
            decider_max_tokens
            if decider_max_tokens is not None
            else self.get_agent_param("decider_max_tokens", 200)
        )
        # repetition_penalty: read from agents.json split entry, fallback to
        # BaseAgent default (1.3 via config.yaml generation block).
        self.reporter_repetition_penalty = float(
            self.get_agent_param(
                "reporter_repetition_penalty", self.repetition_penalty or 1.3
            )
            or 1.3
        )
        self.decider_repetition_penalty = float(
            self.get_agent_param(
                "decider_repetition_penalty", self.repetition_penalty or 1.3
            )
            or 1.3
        )

        # Identifier surfaced in `fulltext_votes` for traceability.
        self.model_name = os.path.basename(str(self.model_path or "qwen2.5-7b"))

    # ================================================================== #
    #  PHASE B OVERRIDE                                                   #
    # ================================================================== #

    def screen_fulltext(self, papers: List[Dict], protocol: Dict) -> Dict:
        """Screen retrieved papers via reporter→decider split.

        Returns the same shape as `Agent4FullText.screen_fulltext` so the
        downstream pipeline (A5+) and the experiment evaluator can consume
        either implementation interchangeably.
        """
        print("\n--- Phase B (SPLIT): Reporter + Decider ---")
        print(f"  Reporter: max_tokens={self.reporter_max_tokens}, "
              f"temperature={self.reporter_temperature}, "
              f"repetition_penalty={self.reporter_repetition_penalty}")
        print(f"  Decider:  max_tokens={self.decider_max_tokens}, "
              f"temperature={self.decider_temperature}, "
              f"repetition_penalty={self.decider_repetition_penalty}")

        if self.model is None:
            self.load_model()

        include_list: List[Dict] = []
        exclude_list: List[Dict] = []

        for i, paper in enumerate(papers):
            title_short = (paper.get("title", "") or "")[:70]
            print(f"\n  [{i + 1}/{len(papers)}] {title_short}...")
            print(
                f"    Source: {paper.get('text_source', '?')} "
                f"({paper.get('full_text_length', 0)} chars)"
            )

            result = self._screen_single_paper_split(paper, protocol)

            paper["fulltext_decision"] = result["fulltext_decision"]
            paper["fulltext_votes"] = result["fulltext_votes"]
            # Mirror the monolithic-output keys so downstream code (A5+,
            # exporters, review report) sees the same shape.
            primary_vote = result["fulltext_votes"][0] if result["fulltext_votes"] else {}
            paper["fulltext_reasoning"] = primary_vote.get("reasoning", "")
            paper["exclusion_reason"] = (
                primary_vote.get("exclusion_reason", "")
                if result["fulltext_decision"] == "EXCLUDE"
                else None
            )
            paper["fulltext_confidence"] = primary_vote.get("confidence", "")
            paper["screening_confidence"] = result["screening_confidence"]
            paper["borderline"] = result["borderline"]
            # Split-specific debugging flags.
            paper["reporter_ok"] = result["reporter_ok"]
            paper["decider_ok"] = result["decider_ok"]

            icon = {"INCLUDE": "[INCLUDE]", "EXCLUDE": "[EXCLUDE]"}.get(
                result["fulltext_decision"], "[?]"
            )
            print(
                f"    -> {icon} | screen_conf: {result['screening_confidence']}"
                f"{' [BORDERLINE]' if result['borderline'] else ''}"
                f" | reporter_ok={result['reporter_ok']}"
                f" | decider_ok={result['decider_ok']}"
            )
            if paper.get("exclusion_reason"):
                print(f"      Reason: {paper['exclusion_reason']}")

            if result["fulltext_decision"] == "INCLUDE":
                include_list.append(paper)
            else:
                exclude_list.append(paper)

        stats = self._build_stats(papers, include_list, exclude_list)

        return {
            "papers": papers,
            "include": include_list,
            "exclude": exclude_list,
            "stats": stats,
        }

    # ================================================================== #
    #  PER-PAPER SPLIT PIPELINE                                           #
    # ================================================================== #

    def _screen_single_paper_split(self, paper: Dict, protocol: Dict) -> Dict:
        """Run reporter -> decider on one paper. Conservative EXCLUDE on errors."""
        # ----- Reporter pass -----
        reporter_prompt = self._build_reporter_prompt(paper, protocol)
        reporter_raw = self._generate_split(
            reporter_prompt,
            max_tokens=self.reporter_max_tokens,
            temperature=self.reporter_temperature,
            repetition_penalty=self.reporter_repetition_penalty,
        )
        reporter_json = self._parse_reporter(reporter_raw)

        if reporter_json is None:
            # Conservative fallback: parse error -> EXCLUDE (project A4 policy
            # mirrored: parser failures must not silently include). Log the
            # raw output for forensics.
            vote = {
                "decision": "EXCLUDE",
                "reasoning": "reporter parse failed",
                "exclusion_reason": "reporter_parse_error",
                "model": self.model_name,
                "reporter_raw": reporter_raw,
                "reporter": None,
                "decider": None,
                "confidence": "low",
            }
            return {
                "fulltext_decision": "EXCLUDE",
                "fulltext_votes": [vote],
                "screening_confidence": 0.0,
                "borderline": True,
                "reporter_ok": False,
                "decider_ok": False,
            }

        # ----- Decider pass -----
        decider_prompt = self._build_decider_prompt(reporter_json, protocol)
        decider_raw = self._generate_split(
            decider_prompt,
            max_tokens=self.decider_max_tokens,
            temperature=self.decider_temperature,
            repetition_penalty=self.decider_repetition_penalty,
        )
        decider_json = self._parse_decider(decider_raw)

        if decider_json is None:
            # Same conservative policy: parse error -> EXCLUDE.
            vote = {
                "decision": "EXCLUDE",
                "reasoning": "decider parse failed",
                "exclusion_reason": "decider_parse_error",
                "model": self.model_name,
                "reporter": reporter_json,
                "reporter_raw": reporter_raw,
                "decider": None,
                "decider_raw": decider_raw,
                "confidence": "low",
            }
            return {
                "fulltext_decision": "EXCLUDE",
                "fulltext_votes": [vote],
                "screening_confidence": 0.0,
                "borderline": True,
                "reporter_ok": True,
                "decider_ok": False,
            }

        decision = decider_json["decision"]
        reason = decider_json.get("reason", "")
        confidence = decider_json.get("confidence", "medium")

        # Map confidence to numeric screening_confidence used by the
        # downstream review-report and borderline flagging.
        if confidence == "high":
            screening_confidence = 1.0
        elif confidence == "medium":
            screening_confidence = 0.6
        else:  # "low"
            screening_confidence = 0.3
        borderline = confidence == "low"

        vote = {
            "decision": decision,
            "reasoning": reason,
            "exclusion_reason": reason if decision == "EXCLUDE" else "",
            "model": self.model_name,
            "reporter": reporter_json,
            "reporter_raw": reporter_raw,
            "decider": decider_json,
            "decider_raw": decider_raw,
            "confidence": confidence,
        }

        return {
            "fulltext_decision": decision,
            "fulltext_votes": [vote],
            "screening_confidence": screening_confidence,
            "borderline": borderline,
            "reporter_ok": True,
            "decider_ok": True,
        }

    # ================================================================== #
    #  PROMPT BUILDERS                                                    #
    # ================================================================== #

    def _build_reporter_prompt(self, paper: Dict, protocol: Dict) -> str:
        """Render the reporter prompt with truncated paper text + PICO ref."""
        # Reuse the parent's section-aware truncation so the reporter sees
        # the same text the monolithic A4 would have seen.
        paper_text = self._prepare_text_for_prompt(paper)
        return format_agent4_reporter_prompt(
            paper=paper,
            protocol=protocol,
            paper_text=paper_text,
            template=self.reporter_prompt_template,
        )

    def _build_decider_prompt(self, reporter_json: Dict, protocol: Dict) -> str:
        """Render the decider prompt. The decider does NOT receive the paper."""
        return format_agent4_decider_prompt(
            reporter_json=reporter_json,
            protocol=protocol,
            template=self.decider_prompt_template,
        )

    # ================================================================== #
    #  PARSERS                                                            #
    # ================================================================== #

    def _parse_reporter(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse + validate reporter JSON. Return None on any failure."""
        if not raw:
            return None
        try:
            cleaned = self._clean_json(raw)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None

        # notes es opcional en el contrato del prompt: el template dice
        # "Si nada, 'ninguna'". El modelo lo omite bajo prompt V2 largo;
        # restauramos el default para no descartar outputs por lo demás válidos.
        if "notes" not in parsed:
            parsed["notes"] = "ninguna"

        # Required keys vary with template (V1 vs V2). We pick the schema
        # based on the active reporter template to keep validation strict.
        required_keys = self._reporter_required_keys()
        for key in required_keys:
            if key not in parsed:
                return None
            value = parsed[key]
            # Coerce non-string scalars to string; reject containers.
            if isinstance(value, (dict, list)):
                return None
            if not isinstance(value, str):
                parsed[key] = str(value)
        return parsed

    def _parse_decider(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse + validate decider JSON. Return None on any failure.

        V3 emits 1-2 reasoning sentences before the JSON block; `_clean_json`
        (inherited) extracts the JSON between the first `{` and the matching
        `}`, so V3 is parsed transparently.
        """
        if not raw:
            return None
        try:
            cleaned = self._clean_json(raw)
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None

        for key in DECIDER_REQUIRED_KEYS:
            if key not in parsed:
                return None

        decision = parsed.get("decision")
        if not isinstance(decision, str):
            return None
        decision = decision.strip().upper()
        if decision not in DECIDER_VALID_DECISIONS:
            return None
        parsed["decision"] = decision

        reason = parsed.get("reason")
        if not isinstance(reason, str):
            if reason is None:
                return None
            reason = str(reason)
        parsed["reason"] = reason

        confidence = parsed.get("confidence")
        if not isinstance(confidence, str):
            return None
        confidence = confidence.strip().lower()
        if confidence not in DECIDER_VALID_CONFIDENCE:
            return None
        parsed["confidence"] = confidence

        return parsed

    def _reporter_required_keys(self) -> Tuple[str, ...]:
        """Return the schema keys for the active reporter template (V1/V2)."""
        try:
            from agents.prompts import AGENT4_REPORTER_TEMPLATE_V2 as _v2
        except ImportError:
            _v2 = None
        if _v2 is not None and self.reporter_prompt_template is _v2:
            return REPORTER_REQUIRED_KEYS_V2
        # Fallback: detect by literal substring if identity check fails
        # (e.g., template was reloaded). Cheap and robust.
        if "exclusion_signals" in (self.reporter_prompt_template or ""):
            return REPORTER_REQUIRED_KEYS_V2
        return REPORTER_REQUIRED_KEYS_V1

    # ================================================================== #
    #  GENERATION HELPER (per-call temperature + rep_penalty)             #
    # ================================================================== #

    def _generate_split(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
    ) -> str:
        """Single LLM forward pass with explicit decoding knobs.

        Bypasses `_simple_generate` because that helper hard-codes
        temperature=0.2 and does not pass repetition_penalty. The split
        needs explicit control over both for reporter vs decider.
        """
        if self._is_groq():
            # Groq adapter: temperature passed through, rep_penalty unsupported.
            return self.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=self.do_sample,
            temperature=temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if repetition_penalty and repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        return self._decode_response(outputs, input_length)

    # ================================================================== #
    #  STATS                                                              #
    # ================================================================== #

    def _build_stats(
        self,
        papers: List[Dict],
        include_list: List[Dict],
        exclude_list: List[Dict],
    ) -> Dict[str, Any]:
        """Build a stats dict shape-compatible with `Agent4FullText`."""
        source_keys = (
            "pmc",
            "unpaywall_pdf",
            "europepmc",
            "semanticscholar_pdf",
            "openalex_pdf",
            "crossref_pdf",
            "doaj_pdf",
            "lens_fulltext",
            "core_fulltext",
            "google_scholar_pdf",
            "scihub_pdf",
            "abstract_only",
        )
        # Mirror the monolithic stats key naming (pmc_fulltext, europepmc_fulltext)
        # to keep the downstream consumer agnostic.
        stats: Dict[str, Any] = {"total_retrieved": len(papers)}
        for key in source_keys:
            count = sum(1 for p in papers if p.get("text_source") == key)
            if key == "pmc":
                stats["pmc_fulltext"] = count
            elif key == "europepmc":
                stats["europepmc_fulltext"] = count
            else:
                stats[key] = count

        # Split-specific telemetry (informational, not consumed by exporters).
        stats["fulltext_included"] = len(include_list)
        stats["fulltext_excluded"] = len(exclude_list)
        stats["reporter_failures"] = sum(
            1 for p in papers if p.get("reporter_ok") is False
        )
        stats["decider_failures"] = sum(
            1 for p in papers
            if p.get("reporter_ok") is True and p.get("decider_ok") is False
        )
        return stats


# ====================================================================== #
#  STANDALONE SANITY CHECK (no GPU, no I/O)                               #
# ====================================================================== #

if __name__ == "__main__":
    print("Agent4FullTextSplit module imported successfully.")
    print(f"Reporter required keys (V1): {REPORTER_REQUIRED_KEYS_V1}")
    print(f"Reporter required keys (V2): {REPORTER_REQUIRED_KEYS_V2}")
    print(f"Decider required keys: {DECIDER_REQUIRED_KEYS}")
    print("Use the experimental evaluator to exercise reporter/decider with a GPU.")
