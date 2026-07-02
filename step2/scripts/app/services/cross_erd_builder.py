"""
cross_erd_builder.py
~~~~~~~~~~~~~~~~~~~~
Three integration strategies for the cross-ERD task.

All three functions receive the RAW ground-truth model dicts
(ground_truth_model_aligned.json / ground_truth_model_B_aligned.json),
which have the "concepts" key with the full canonical structure.

  cross_rule_based   – deterministic normalised-name match (no LLM)
  cross_llm          – LLM with both ground truth models as text
  cross_llm_brief    – LLM with both models + expert domain brief
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from app.services.vocab_builder import (
    _levenshtein_distance,
    _token_jaccard_similarity,
    _value_overlap_ratio,
)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", s.strip().lower())


# ---------------------------------------------------------------------------
# Compact text serialisation of a ground-truth model for the LLM prompt
# ---------------------------------------------------------------------------

def _model_to_text(model: dict[str, Any], label: str) -> str:
    """
    Render a ground-truth model (concepts list) as compact readable text.
    Shows: entity -> canonical  [category]  + key attributes/relations.
    """
    lines = [f"{label} (source ERD: {model.get('source_erd', '?')})"]
    for concept in model.get("concepts", []):
        canon   = concept.get("canonical_name", "")
        src     = concept.get("source_entity", "")
        cat     = concept.get("ontological_category", "")
        ref     = concept.get("refinement", {})
        ref_str = f"  [{ref.get('type','')}]" if isinstance(ref, dict) and ref.get("type") else ""

        lines.append(f"  CONCEPT {src} -> {canon} ({cat}){ref_str}")

        for attr in concept.get("attributes", []):
            mt = attr.get("maps_to", {})
            mtype = mt.get("type", "")
            if mtype == "relation_role":
                rel    = mt.get("relation", "")
                target = mt.get("target_concept", "")
                lines.append(f"    {attr.get('source','')}  FK -> {rel} ({target})")
            else:
                cname = mt.get("canonical_name", "")
                role  = mt.get("role", "")
                lines.append(f"    {attr.get('source','')}  -> {cname}  [{role}]")

    # Relations (if present)
    for rel in model.get("relations", []):
        rname  = rel.get("canonical_name", "")
        domain = rel.get("domain", "")
        rrange = rel.get("range", "")
        lines.append(f"  RELATION {domain} --{rname}--> {rrange}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario 0 – Deterministic rule-based baseline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Classic deterministic matcher: lexical + structural + instance similarity
# ---------------------------------------------------------------------------

# Combination weights and the minimum score to accept an alignment.
_W_LEXICAL = 0.30
_W_STRUCTURAL = 0.50
_W_INSTANCE = 0.20
_MATCH_THRESHOLD = 0.35


def _concept_profile(c: dict[str, Any]) -> dict[str, Any]:
    """Extract the structural fingerprint of a gold-model concept."""
    props, rel_targets = set(), set()
    for a in c.get("attributes", []) or []:
        mt = a.get("maps_to", {}) or {}
        if mt.get("type") == "relation_role":
            if mt.get("target_concept"):
                rel_targets.add(_norm(mt["target_concept"]))
        else:
            if mt.get("canonical_name"):
                props.add(_norm(mt["canonical_name"]))
    return {
        "canonical": c.get("canonical_name", ""),
        "source_entity": c.get("source_entity", ""),
        "category": _norm(c.get("ontological_category", "")),
        "props": props,
        "rel_targets": rel_targets,
    }


def _jaccard(s1: set, s2: set) -> float:
    u = s1 | s2
    return len(s1 & s2) / len(u) if u else 0.0


def _lexical_sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    edit = 1.0 - _levenshtein_distance(na, nb) / max(len(na), len(nb))
    return 0.5 * max(0.0, edit) + 0.5 * _token_jaccard_similarity(a, b)


def _structural_sim(pa: dict, pb: dict) -> float:
    cat = 1.0 if pa["category"] and pa["category"] == pb["category"] else 0.0
    return (0.34 * cat
            + 0.33 * _jaccard(pa["rel_targets"], pb["rel_targets"])
            + 0.33 * _jaccard(pa["props"], pb["props"]))


def merge_rule_based(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    instances_a: dict[str, set] | None = None,
    instances_b: dict[str, set] | None = None,
) -> dict[str, Any]:
    """Deterministic cross-ERD matcher in the classic schema/ontology-matching
    style (COMA-like): for each concept pair it combines

      - lexical similarity (edit distance + token Jaccard on canonical names),
      - structural signals (ontological category, shared relation targets and
        shared property names),
      - instance overlap (value overlap of the underlying data),

    then greedily assigns 1:1 alignments above a threshold. No world knowledge
    and no LLM. Unlike plain string equality it can bridge name divergence
    (e.g. SeatAssignment vs Booking) via structure and instances.

    instances_a/b map a lower-cased source_entity to its set of data values
    (optional; the instance signal is skipped when absent).
    """
    instances_a = instances_a or {}
    instances_b = instances_b or {}
    profs_a = [_concept_profile(c) for c in model_a.get("concepts", [])]
    profs_b = [_concept_profile(c) for c in model_b.get("concepts", [])]

    # Score every cross pair.
    scored: list[tuple[float, int, int]] = []
    for i, pa in enumerate(profs_a):
        va = instances_a.get(pa["source_entity"].lower(), set())
        for j, pb in enumerate(profs_b):
            vb = instances_b.get(pb["source_entity"].lower(), set())
            inst = _value_overlap_ratio(list(va), list(vb)) if va and vb else 0.0
            score = (_W_LEXICAL * _lexical_sim(pa["canonical"], pb["canonical"])
                     + _W_STRUCTURAL * _structural_sim(pa, pb)
                     + _W_INSTANCE * inst)
            scored.append((score, i, j))

    # Greedy 1:1 assignment, highest score first, above threshold.
    scored.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    alignments: list[dict] = []
    for score, i, j in scored:
        if score < _MATCH_THRESHOLD or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        alignments.append({
            "canonical": profs_a[i]["canonical"],
            "erd_a": profs_a[i]["source_entity"],
            "erd_b": profs_b[j]["source_entity"],
            "score": round(score, 4),
        })

    unique_to_a = [p["canonical"] for k, p in enumerate(profs_a) if k not in used_a]
    unique_to_b = [p["canonical"] for k, p in enumerate(profs_b) if k not in used_b]
    integrated_concepts = (
        [al["canonical"] for al in alignments] + unique_to_a + unique_to_b
    )
    return {
        "alignments": alignments,
        "unique_to_a": unique_to_a,
        "unique_to_b": unique_to_b,
        "integrated_concepts": integrated_concepts,
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, temperature: float = 0.2) -> dict[str, Any]:
    from app.services.llm import LLMClient
    client = LLMClient()
    raw = client.generate_json(prompt, temperature=temperature, schema=None)
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"LLM returned non-dict: {type(raw)}")


def _build_prompt(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    expert_brief: str | None = None,
) -> str:
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(base_dir, "helpers", "base_prompt_crosserd.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    evidence = (
        _model_to_text(model_a, "MODEL_A") +
        "\n\n" +
        _model_to_text(model_b, "MODEL_B")
    )

    if expert_brief:
        evidence += (
            "\n\nEXPERT DOMAIN BRIEF (general modelling guidance):\n"
            + expert_brief
        )

    return base_prompt.replace("{{EVIDENCE_BLOCK}}", evidence)


# ---------------------------------------------------------------------------
# Scenario 1 – LLM (2 ground-truth models only)
# ---------------------------------------------------------------------------

def merge_llm(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    temperature: float = 0.2,
) -> dict[str, Any]:
    prompt = _build_prompt(model_a, model_b, expert_brief=None)
    return _call_llm(prompt, temperature=temperature)


# ---------------------------------------------------------------------------
# Scenario 2 – LLM (2 ground-truth models + expert brief)
# ---------------------------------------------------------------------------

def merge_llm_brief(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    expert_brief: str,
    temperature: float = 0.2,
) -> dict[str, Any]:
    prompt = _build_prompt(model_a, model_b, expert_brief=expert_brief)
    return _call_llm(prompt, temperature=temperature)
