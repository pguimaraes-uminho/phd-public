"""Score a predicted Step-1 ERD against a ground-truth ERD.

Unlike the app's intrinsic validator (ERD-vs-DATA), this compares the reconstructed
ERD to the EXPERT ERD, so it measures how much of the true structure each method
recovers. Entities are matched by attribute-set overlap (names differ across
methods). Reports, per scenario:

  entity_precision/recall/f1   — did we recover the right entities (attribute groups)?
  attribute_coverage           — of the truth's attributes, how many appear anywhere?
  attribute_placement          — how many are in the RIGHT (matched) entity?
  pk_accuracy                  — matched entities whose PK equals the truth PK
  relationship_precision/recall/f1 — FK edges recovered (endpoints matched + same fk)
  type_accuracy                — datatype agreement on shared attributes (broad classes)
"""
from __future__ import annotations

import re
from typing import Any


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _attrset(entity: dict) -> set:
    return {_norm(a.get("name")) for a in (entity.get("attributes") or []) if isinstance(a, dict) and a.get("name")}


def _broad_type(t: Any) -> str:
    s = str(t or "").lower()
    if any(k in s for k in ("char", "text", "string", "clob")):
        return "string"
    if "bool" in s:
        return "boolean"
    if any(k in s for k in ("date", "time", "timestamp")):
        return "datetime"
    if any(k in s for k in ("int", "serial")):
        return "numeric"
    if any(k in s for k in ("float", "real", "double", "decimal", "numeric", "number")):
        return "numeric"
    return "string"


def _f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def _match_entities(pred: list, truth: list) -> dict:
    """Greedy 1:1 match truth→pred by fraction of the truth entity's attributes the
    predicted entity covers. Returns {truth_idx: (pred_idx, coverage)}."""
    pairs = []
    for ti, t in enumerate(truth):
        ts = _attrset(t)
        if not ts:
            continue
        for pi, p in enumerate(pred):
            shared = len(ts & _attrset(p))
            if shared:
                pairs.append((shared / len(ts), shared, ti, pi))
    pairs.sort(reverse=True)
    t2p: dict[int, tuple] = {}
    used_t, used_p = set(), set()
    for cov, shared, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        t2p[ti] = (pi, cov)
    return t2p


def evaluate_erd_vs_truth(pred: dict, truth: dict, match_threshold: float = 0.5) -> dict:
    pred_ents = [e for e in (pred.get("entities") or []) if isinstance(e, dict)]
    truth_ents = [e for e in (truth.get("entities") or []) if isinstance(e, dict)]
    t2p = _match_entities(pred_ents, truth_ents)

    # --- entities: a truth entity is "recovered" when matched with enough coverage
    recovered = {ti: (pi, cov) for ti, (pi, cov) in t2p.items() if cov >= match_threshold}
    ent_recall = len(recovered) / len(truth_ents) if truth_ents else 0.0
    ent_precision = len(recovered) / len(pred_ents) if pred_ents else 0.0

    # --- attributes: coverage (anywhere) + placement (right entity)
    all_pred_attrs = set().union(*[_attrset(e) for e in pred_ents]) if pred_ents else set()
    total_truth_attrs = 0
    covered = 0
    placed = 0
    type_hits = type_total = 0
    pk_correct = 0
    for ti, t in enumerate(truth_ents):
        ts = _attrset(t)
        total_truth_attrs += len(ts)
        covered += len(ts & all_pred_attrs)
        if ti in recovered:
            p = pred_ents[recovered[ti][0]]
            ps = _attrset(p)
            placed += len(ts & ps)
            # type accuracy on shared attributes
            p_types = {_norm(a.get("name")): a.get("data_type") for a in (p.get("attributes") or []) if isinstance(a, dict)}
            for a in (t.get("attributes") or []):
                an = _norm(a.get("name"))
                if an in p_types:
                    type_total += 1
                    if _broad_type(a.get("data_type")) == _broad_type(p_types[an]):
                        type_hits += 1
            # pk accuracy
            if {_norm(x) for x in (t.get("primary_key") or [])} == {_norm(x) for x in (p.get("primary_key") or [])}:
                pk_correct += 1

    attribute_coverage = covered / total_truth_attrs if total_truth_attrs else 0.0
    attribute_placement = placed / total_truth_attrs if total_truth_attrs else 0.0
    pk_accuracy = pk_correct / len(recovered) if recovered else 0.0
    type_accuracy = type_hits / type_total if type_total else 0.0

    # --- relationships: a truth FK edge is recovered if endpoints matched + same fk
    def _matched_pred_attrs_by_truth_entity(ti):
        return recovered.get(ti)
    truth_name_to_idx = {_norm(e.get("name")): i for i, e in enumerate(truth_ents)}
    pred_matched = {recovered[ti][0] for ti in recovered}
    # index predicted rels by (frozenset(pred endpoint idx pair), norm fk) for lookup
    pred_by_name = {_norm(e.get("name")): pi for pi, e in enumerate(pred_ents)}
    pred_rel_keys = set()
    for r in (pred.get("relationships") or []):
        if not isinstance(r, dict):
            continue
        a = pred_by_name.get(_norm(r.get("from_entity")))
        b = pred_by_name.get(_norm(r.get("to_entity")))
        if a is not None and b is not None:
            pred_rel_keys.add((frozenset((a, b)), _norm(r.get("fk_attribute"))))

    truth_rels = [r for r in (truth.get("relationships") or []) if isinstance(r, dict)]
    rel_recovered = 0
    for r in truth_rels:
        ta = truth_name_to_idx.get(_norm(r.get("from_entity")))
        tb = truth_name_to_idx.get(_norm(r.get("to_entity")))
        if ta is None or tb is None or ta not in recovered or tb not in recovered:
            continue
        pa, pb = recovered[ta][0], recovered[tb][0]
        if (frozenset((pa, pb)), _norm(r.get("fk_attribute"))) in pred_rel_keys:
            rel_recovered += 1
    rel_recall = rel_recovered / len(truth_rels) if truth_rels else 0.0
    total_pred_rels = len([r for r in (pred.get("relationships") or []) if isinstance(r, dict)])
    rel_precision = rel_recovered / total_pred_rels if total_pred_rels else 0.0

    return {
        "pred_entities": len(pred_ents), "truth_entities": len(truth_ents), "matched_entities": len(recovered),
        "entity_precision": round(ent_precision, 4), "entity_recall": round(ent_recall, 4),
        "entity_f1": _f1(ent_precision, ent_recall),
        "attribute_coverage": round(attribute_coverage, 4), "attribute_placement": round(attribute_placement, 4),
        "pk_accuracy": round(pk_accuracy, 4), "type_accuracy": round(type_accuracy, 4),
        "relationship_precision": round(rel_precision, 4), "relationship_recall": round(rel_recall, 4),
        "relationship_f1": _f1(rel_precision, rel_recall),
    }
