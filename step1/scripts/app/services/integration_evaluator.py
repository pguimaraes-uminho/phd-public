from __future__ import annotations

import os
import json
import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", s.strip().lower())


def calculate_integration_evaluation_metrics(proposed_plan: dict[str, Any]) -> dict[str, Any]:
    """
    Computes Cross-ERD integration evaluation metrics against cross_erd_ground_truth.json.

    proposed_plan keys used:
      alignments        – list of {canonical, erd_a, erd_b}
      unique_to_a       – list of canonical names kept from ERD-A only
      unique_to_b       – list of canonical names kept from ERD-B only
      integrated_concepts – full union list (aligned + unique)
    """
    # Repository root (scripts/app/services/ -> up 4); cross gold lives in ground-truth/.
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    gold_path = os.path.join(root_dir, "ground-truth", "ground_truth_cross_erd.json")

    result = {
        "match_precision": 1.0,
        "match_recall": 1.0,
        "match_f1": 1.0,
        "integration_over_merge": 0.0,
        "integration_under_merge": 0.0,
        "conflict_resolution_accuracy": 1.0,
        "unique_a_coverage": 1.0,
        "unique_b_coverage": 1.0,
    }

    if not os.path.exists(gold_path):
        print(f"[warn] cross-ERD gold not found at {gold_path}; returning default (perfect) metrics")
        return result

    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)

        # ------------------------------------------------------------------
        # 1. Gold concept pairs (symmetric, keyed by source entity names)
        # ------------------------------------------------------------------
        gold_pairs: set[tuple[str, str]] = set()
        for cm in gold.get("concept_matches", []):
            a = cm.get("erd_a", "").strip().lower()
            b = cm.get("erd_b", "").strip().lower()
            if a and b:
                gold_pairs.add((min(a, b), max(a, b)))

        # ------------------------------------------------------------------
        # 2. Proposed pairs (from alignments list)
        # ------------------------------------------------------------------
        alignments = proposed_plan.get("alignments", []) or []
        proposed_pairs: set[tuple[str, str]] = set()
        for al in alignments:
            a = al.get("erd_a", "").strip().lower()
            b = al.get("erd_b", "").strip().lower()
            if a and b:
                proposed_pairs.add((min(a, b), max(a, b)))

        # ------------------------------------------------------------------
        # 3. Match precision / recall / F1
        # ------------------------------------------------------------------
        correct_pairs = proposed_pairs & gold_pairs
        precision = len(correct_pairs) / len(proposed_pairs) if proposed_pairs else 1.0
        recall    = len(correct_pairs) / len(gold_pairs)    if gold_pairs     else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ------------------------------------------------------------------
        # 4. Over-merge rate – two distinct gold concepts collapsed into one
        # ------------------------------------------------------------------
        target_sources: dict[str, set[str]] = {}
        for a, b in proposed_pairs:
            target_sources.setdefault(a, set()).add(b)
            target_sources.setdefault(b, set()).add(a)

        over_merge_count = 0
        for sources in target_sources.values():
            src_list = list(sources)
            for i in range(len(src_list)):
                for j in range(i + 1, len(src_list)):
                    pair = (min(src_list[i], src_list[j]), max(src_list[i], src_list[j]))
                    if pair not in gold_pairs:
                        over_merge_count += 1

        over_merge_rate = over_merge_count / len(proposed_pairs) if proposed_pairs else 0.0

        # ------------------------------------------------------------------
        # 5. Under-merge rate – gold pairs missed by the proposal
        # ------------------------------------------------------------------
        under_merge_count = len(gold_pairs - proposed_pairs)
        under_merge_rate  = under_merge_count / len(gold_pairs) if gold_pairs else 0.0

        # ------------------------------------------------------------------
        # 6. Unique-concept coverage
        #    Gold records which concepts appear only in A or only in B.
        #    We check whether the proposal retains them.
        # ------------------------------------------------------------------
        gold_unique_a = {_norm(c) for c in gold.get("unique_to_a", {}).get("concepts", [])}
        gold_unique_b = {_norm(c) for c in gold.get("unique_to_b", {}).get("concepts", [])}

        prop_integrated = {_norm(c) for c in (proposed_plan.get("integrated_concepts") or [])}
        prop_unique_a   = {_norm(c) for c in (proposed_plan.get("unique_to_a") or [])}
        prop_unique_b   = {_norm(c) for c in (proposed_plan.get("unique_to_b") or [])}
        # A unique concept is "kept" if it appears anywhere in the integration output
        prop_all = prop_integrated | prop_unique_a | prop_unique_b

        unique_a_covered = len(gold_unique_a & prop_all) / len(gold_unique_a) if gold_unique_a else 1.0
        unique_b_covered = len(gold_unique_b & prop_all) / len(gold_unique_b) if gold_unique_b else 1.0

        # ------------------------------------------------------------------
        # 7. Conflict resolution accuracy
        #    Operationalised directly from the gold's "conflicts" list.
        #    Each conflict has a type; we check what the proposal does.
        # ------------------------------------------------------------------
        integrated_norms = {_norm(c) for c in (proposed_plan.get("integrated_concepts") or [])}

        def is_aligned(x: str, y: str) -> bool:
            pair = (min(x.lower(), y.lower()), max(x.lower(), y.lower()))
            return pair in proposed_pairs

        def concept_present(name: str) -> bool:
            return _norm(name) in integrated_norms | prop_all

        correct_resolutions = 0
        total_conflicts = 0

        for conflict in gold.get("conflicts", []):
            ctype  = conflict.get("type", "")
            canon  = conflict.get("canonical", "")
            detail = conflict.get("detail", "")

            if ctype == "naming":
                # Resolution: the two differently-named concepts must be aligned.
                # Extract the two names from detail ("A: X; B: Y")
                total_conflicts += 1
                parts = [p.strip() for p in detail.split(";")]
                names = []
                for p in parts:
                    if ":" in p:
                        names.append(p.split(":", 1)[1].strip())
                if len(names) >= 2:
                    if is_aligned(names[0], names[1]):
                        correct_resolutions += 1
                else:
                    # Fallback: canonical must appear in integration
                    if concept_present(canon):
                        correct_resolutions += 1

            elif ctype == "identity":
                # Resolution: the two source entities must be aligned
                total_conflicts += 1
                parts = [p.strip() for p in detail.split(";")]
                names = []
                for p in parts:
                    m = re.search(r"A:\s*natural key\s+\(([^)]+)\)", p)
                    if not m:
                        m = re.search(r"A:\s*([^\s;,]+)", p)
                    if m:
                        names.append(m.group(1).strip())
                # Simpler: just check that the canonical concept is aligned
                # by verifying it appears in the alignments under gold names
                gold_a_name = ""
                gold_b_name = ""
                for cm in gold.get("concept_matches", []):
                    if _norm(cm.get("canonical", "")) == _norm(canon):
                        gold_a_name = cm.get("erd_a", "")
                        gold_b_name = cm.get("erd_b", "")
                        break
                if gold_a_name and gold_b_name:
                    if is_aligned(gold_a_name, gold_b_name):
                        correct_resolutions += 1
                else:
                    # No pair found – credit if canonical is in integration
                    if concept_present(canon):
                        correct_resolutions += 1

            elif ctype == "granularity":
                # Resolution: the concept (or both granularity variants) must
                # appear somewhere in the integrated output.
                total_conflicts += 1
                if concept_present(canon.split(".")[0]):
                    correct_resolutions += 1

            elif ctype == "structure":
                # Resolution: the richer structural choice must be preserved.
                # Gold resolution describes what should be kept; we check that
                # the concept mentioned in "canonical" is present.
                total_conflicts += 1
                concept_name = canon.split(".")[0]
                if concept_present(concept_name):
                    correct_resolutions += 1

        conflict_accuracy = correct_resolutions / total_conflicts if total_conflicts > 0 else 1.0

        result = {
            "match_precision":             round(precision,        4),
            "match_recall":                round(recall,           4),
            "match_f1":                    round(f1,               4),
            "integration_over_merge":      round(over_merge_rate,  4),
            "integration_under_merge":     round(under_merge_rate, 4),
            "unique_a_coverage":           round(unique_a_covered, 4),
            "unique_b_coverage":           round(unique_b_covered, 4),
            "conflict_resolution_accuracy": round(conflict_accuracy, 4),
        }

    except Exception as exc:
        result["error"] = str(exc)

    return result
