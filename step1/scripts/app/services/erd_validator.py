from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from app.models.erd import ERDModel


VALID_CARDINALITIES = {"1", "0..1", "0..N", "1..N"}

MAX_FD_LHS = 3
MAX_KEY_SIZE = 4
MAX_ATTR_FOR_FD = 12
MAX_ROWS_FOR_PROFILE = 5000
MIN_SUPPORT_RATIO = 0.9
# "the sample must be sufficient": an FD is only trusted with enough rows AND a
# determinant that actually REPEATS. A (near-)unique determinant trivially
# determines everything on a small/distinct sample → a coincidental FD, not evidence.
MIN_ROWS_FOR_FD = 12
MAX_DETERMINANT_DISTINCT_RATIO = 0.9


@dataclass(frozen=True)
class FDResult:
    lhs: tuple[str, ...]
    rhs: str
    strength: float
    support_ratio: float
    row_count: int


def validate_erd(dfs: list[pd.DataFrame] | pd.DataFrame, erd: ERDModel) -> dict[str, Any]:
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        
    # Aggregate all unique columns across all CSVs
    all_csv_cols_raw = []
    for df in dfs:
        all_csv_cols_raw.extend([str(c) for c in df.columns.tolist()])
    
    csv_norm_map: dict[str, str] = {}
    for col in all_csv_cols_raw:
        csv_norm_map.setdefault(_norm(col), col)
    csv_cols = set(csv_norm_map.keys())

    erd_attrs_raw = [str(attr.name) for ent in erd.entities for attr in ent.attributes]
    erd_norm_map: dict[str, str] = {}
    for attr in erd_attrs_raw:
        erd_norm_map.setdefault(_norm(attr), attr)
    erd_attrs = set(erd_norm_map.keys())

    tp = len(csv_cols & erd_attrs)
    fp = len(erd_attrs - csv_cols)
    fn = len(csv_cols - erd_attrs)

    # A degenerate ERD (no entities → nothing modeled) must not earn vacuous 1.0s.
    _has_entities = bool(erd.entities)

    precision = tp / (tp + fp) if (tp + fp) else 0.0  # no attributes modeled → 0, not vacuous 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    relationship_checks = _validate_relationships(erd)
    integrity_summary = _summarize_integrity(erd, relationship_checks)
    relationship_score = (
        sum(1 for c in relationship_checks if c["valid"]) / len(relationship_checks)
        if relationship_checks
        else (1.0 if _has_entities else 0.0)  # a single-entity ERD legitimately has none
    )

    # 3NF validation: For each entity, find the best matching DataFrame
    nf3_checks = _validate_3nf_multi(dfs, erd)
    nf3_summary = _summarize_nf3(nf3_checks)
    nf3_score = (
        sum(1 for c in nf3_checks if c["valid"]) / len(nf3_checks)
        if nf3_checks
        else (1.0 if _has_entities else 0.0)
    )

    total_score = (precision * 0.35) + (recall * 0.35) + (relationship_score * 0.2) + (nf3_score * 0.1)

    return {
        "attribute_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "csv_column_count": len(csv_cols),
            "erd_attribute_count": len(erd_attrs),
            "matched_columns": sorted(csv_norm_map[n] for n in (csv_cols & erd_attrs)),
            "missing_columns": sorted(csv_norm_map[n] for n in (csv_cols - erd_attrs)),
            "extra_attributes": sorted(erd_norm_map[n] for n in (erd_attrs - csv_cols)),
        },
        "integrity_summary": integrity_summary,
        "relationship_checks": relationship_checks,
        "nf3_checks": nf3_checks,
        "nf3_summary": nf3_summary,
        "scores": {
            "relationship_score": round(relationship_score, 4),
            "nf3_score": round(nf3_score, 4),
            "total_score": round(total_score, 4),
        },
    }


def _validate_relationships(erd: ERDModel) -> list[dict[str, Any]]:
    entity_names = {ent.name for ent in erd.entities}
    results = []

    for rel in erd.relationships:
        issues = []
        if rel.from_entity not in entity_names:
            issues.append("from_entity_missing")
        if rel.to_entity not in entity_names:
            issues.append("to_entity_missing")
        if rel.from_cardinality not in VALID_CARDINALITIES:
            issues.append("invalid_from_cardinality")
        if rel.to_cardinality not in VALID_CARDINALITIES:
            issues.append("invalid_to_cardinality")

        from_many = "N" in (rel.from_cardinality or "")
        to_many = "N" in (rel.to_cardinality or "")
        expected_fk_entity = None
        expected_pk_entity = None

        if from_many and to_many:
            issues.append("many_to_many_requires_bridge")
        elif from_many != to_many:
            expected_fk_entity = rel.from_entity if from_many else rel.to_entity
            expected_pk_entity = rel.to_entity if from_many else rel.from_entity

        if not rel.fk_attribute:
            issues.append("fk_attribute_missing")
        else:
            fk_in_from = _attribute_exists(erd, rel.from_entity, rel.fk_attribute)
            fk_in_to = _attribute_exists(erd, rel.to_entity, rel.fk_attribute)
            if expected_fk_entity:
                if expected_fk_entity == rel.from_entity and not fk_in_from and fk_in_to:
                    issues.append("fk_on_wrong_entity")
                elif expected_fk_entity == rel.to_entity and not fk_in_to and fk_in_from:
                    issues.append("fk_on_wrong_entity")
                elif not fk_in_from and not fk_in_to:
                    issues.append("fk_attribute_missing")
            else:
                if not fk_in_from and not fk_in_to:
                    issues.append("fk_attribute_missing")

        if not rel.pk_attribute:
            issues.append("pk_attribute_missing")
        else:
            pk_in_from = _attribute_exists(erd, rel.from_entity, rel.pk_attribute)
            pk_in_to = _attribute_exists(erd, rel.to_entity, rel.pk_attribute)
            if expected_pk_entity:
                if expected_pk_entity == rel.from_entity and not pk_in_from and pk_in_to:
                    issues.append("pk_on_wrong_entity")
                elif expected_pk_entity == rel.to_entity and not pk_in_to and pk_in_from:
                    issues.append("pk_on_wrong_entity")
                elif not pk_in_from and not pk_in_to:
                    issues.append("pk_attribute_missing")
            else:
                if not pk_in_from and not pk_in_to:
                    issues.append("pk_attribute_missing")

        results.append(
            {
                "relationship": rel.model_dump(),
                "expected_fk_entity": expected_fk_entity,
                "expected_pk_entity": expected_pk_entity,
                "valid": len(issues) == 0,
                "issues": issues,
            }
        )

    return results


def _attribute_exists(erd: ERDModel, entity_name: str, attr_name: str) -> bool:
    for ent in erd.entities:
        if ent.name != entity_name:
            continue
        for attr in ent.attributes:
            if _norm(attr.name) == _norm(attr_name):
                return True
    return False


def _validate_3nf_multi(dfs: list[pd.DataFrame], erd: ERDModel) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for ent in erd.entities:
        attrs = [a.name for a in ent.attributes]
        pk = ent.primary_key or [a.name for a in ent.attributes if a.is_primary_key]

        # Find the best matching DataFrame for this entity
        best_df = None
        best_coverage = -1
        
        for df in dfs:
            aligned_attrs, _ = _align_columns(df, attrs)
            coverage = len(aligned_attrs)
            if coverage > best_coverage:
                best_coverage = coverage
                best_df = df
        
        if best_df is None or best_coverage == 0:
            results.append({
                "entity": ent.name, 
                "valid": False, 
                "issues": ["no_entity_data_in_any_file"], 
                "report": {}
            })
            continue

        issues: list[str] = []
        aligned_attrs, missing_attrs = _align_columns(best_df, attrs)
        aligned_pk, missing_pk = _align_columns(best_df, pk)
        entity_df = best_df[aligned_attrs].copy()

        profile = _profile_entity(
            entity_df,
            aligned_attrs,
            aligned_pk,
            attrs_original=attrs,
            pk_original=pk,
            missing_attrs=missing_attrs,
            missing_pk=missing_pk,
        )
        issues.extend(profile["issues"])

        valid = len(issues) == 0
        results.append({"entity": ent.name, "valid": valid, "issues": issues, "report": profile["report"]})

    return results


def _summarize_integrity(erd: ERDModel, rel_checks: list[dict[str, Any]]) -> dict[str, Any]:
    entity_count = len(erd.entities)
    entities_with_pk = 0
    for ent in erd.entities:
        pk = ent.primary_key or [a.name for a in ent.attributes if a.is_primary_key]
        if pk:
            entities_with_pk += 1

    issue_counts: dict[str, int] = {}
    for check in rel_checks:
        for issue in check.get("issues", []) or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    relationships_count = len(rel_checks)
    relationships_valid = sum(1 for c in rel_checks if c.get("valid"))
    relationships_missing_fk = issue_counts.get("fk_attribute_missing", 0)
    relationships_fk_wrong = issue_counts.get("fk_on_wrong_entity", 0)
    relationships_missing_pk = issue_counts.get("pk_attribute_missing", 0)
    relationships_pk_wrong = issue_counts.get("pk_on_wrong_entity", 0)
    relationships_invalid_card = issue_counts.get("invalid_from_cardinality", 0) + issue_counts.get(
        "invalid_to_cardinality", 0
    )
    relationships_many_to_many = issue_counts.get("many_to_many_requires_bridge", 0)

    return {
        "entity_count": entity_count,
        "entities_with_pk": entities_with_pk,
        "entities_missing_pk": entity_count - entities_with_pk,
        "relationships_count": relationships_count,
        "relationships_valid": relationships_valid,
        "relationships_missing_fk": relationships_missing_fk,
        "relationships_fk_wrong_entity": relationships_fk_wrong,
        "relationships_missing_pk": relationships_missing_pk,
        "relationships_pk_wrong_entity": relationships_pk_wrong,
        "relationships_invalid_cardinality": relationships_invalid_card,
        "relationships_many_to_many": relationships_many_to_many,
        "relationship_issue_counts": issue_counts,
    }


def _summarize_nf3(nf3_checks: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for check in nf3_checks:
        for issue in check.get("issues", []) or []:
            issue_key = issue.split(":", 1)[0]
            issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
    return {"issue_counts": issue_counts}


def _profile_entity(
    df: pd.DataFrame,
    attrs: list[str],
    pk: list[str],
    attrs_original: list[str] | None = None,
    pk_original: list[str] | None = None,
    missing_attrs: list[str] | None = None,
    missing_pk: list[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    report: dict[str, Any] = {}

    working_df = _sample_df(df)
    report["row_count"] = len(df)
    report["sampled_rows"] = len(working_df)
    report["sampled"] = len(working_df) != len(df)
    attrs_original = attrs_original or attrs
    pk_original = pk_original or pk

    if missing_attrs is None or missing_pk is None:
        col_norms = {_norm(c) for c in working_df.columns}
        missing_attrs = [a for a in attrs_original if _norm(a) not in col_norms]
        missing_pk = [a for a in pk_original if _norm(a) not in col_norms]

    report["attributes"] = attrs_original
    report["attributes_present"] = attrs
    report["attributes_missing"] = missing_attrs
    report["primary_key"] = pk_original
    report["primary_key_present"] = pk
    report["primary_key_missing"] = missing_pk

    if not pk_original:
        issues.append("missing_primary_key")
        report["candidate_keys"] = []
        report["prime_attributes"] = []
        report["fds"] = []
        return {"issues": issues, "report": report}

    if missing_pk:
        issues.append(f"pk_missing_columns:{','.join(missing_pk)}")

    candidate_keys = _find_candidate_keys(working_df, attrs, MAX_KEY_SIZE)
    prime_attrs = sorted({a for key in candidate_keys for a in key})
    report["candidate_keys"] = candidate_keys
    report["prime_attributes"] = prime_attrs

    pk_unique = _is_unique(working_df, pk) if pk else False
    report["pk_unique"] = pk_unique
    if pk and not pk_unique:
        issues.append("pk_not_unique")

    pk_minimal = True
    if pk:
        for subset_size in range(1, len(pk)):
            for subset in itertools.combinations(pk, subset_size):
                if _is_unique(working_df, list(subset)):
                    pk_minimal = False
                    issues.append(f"pk_not_minimal:{'+'.join(subset)}")
                    break
            if not pk_minimal:
                break
    report["pk_minimal"] = pk_minimal

    non_key_attrs = [a for a in attrs if a not in pk]

    if pk:
        for nk in non_key_attrs:
            if not _is_fd(working_df, pk, nk):
                issues.append(f"non_key_not_dependent_on_pk:{nk}")

        if len(pk) > 1:
            for subset_size in range(1, len(pk)):
                for subset in itertools.combinations(pk, subset_size):
                    for nk in non_key_attrs:
                        if _is_fd(working_df, list(subset), nk):
                            issues.append(f"partial_dependency:{'+'.join(subset)}->{nk}")

    fd_results, warnings = _discover_fds(working_df, attrs)
    report["fds"] = [
        {
            "lhs": list(fd.lhs),
            "rhs": fd.rhs,
            "strength": round(fd.strength, 4),
            "support_ratio": round(fd.support_ratio, 4),
            "row_count": fd.row_count,
        }
        for fd in fd_results
    ]
    if warnings:
        report["warnings"] = warnings

    for fd in fd_results:
        if fd.rhs in prime_attrs:
            continue
        if not _is_unique(working_df, list(fd.lhs)):
            issues.append(f"3nf_violation:{'+'.join(fd.lhs)}->{fd.rhs}")

    if pk:
        for fd in fd_results:
            if fd.rhs in non_key_attrs and fd.lhs[0] in non_key_attrs:
                if _is_fd(working_df, pk, fd.lhs[0]):
                    issues.append(f"transitive_dependency:{'+'.join(fd.lhs)}->{fd.rhs}")

    return {"issues": issues, "report": report}


def _discover_fds(df: pd.DataFrame, attrs: list[str]) -> tuple[list[FDResult], list[str]]:
    warnings: list[str] = []
    results: list[FDResult] = []

    # Insufficient sample → do NOT trust discovered FDs (they would be coincidental
    # and produce false 3NF violations). The caller then treats the entity as
    # unviolated rather than penalising it on flimsy evidence.
    if len(df) < MIN_ROWS_FOR_FD:
        return results, ["fd_checks_skipped:insufficient_sample"]

    if len(attrs) > MAX_ATTR_FOR_FD:
        warnings.append("fd_search_truncated:too_many_attributes")
        attrs = attrs[:MAX_ATTR_FOR_FD]

    for size in range(1, min(MAX_FD_LHS, len(attrs)) + 1):
        for lhs in itertools.combinations(attrs, size):
            # Skip a (near-)unique determinant: it determines everything trivially,
            # so any FD from it is coincidental, not real evidence of a dependency.
            lhs_df = df[list(lhs)].dropna()
            if lhs_df.empty:
                continue
            if lhs_df.drop_duplicates().shape[0] > len(lhs_df) * MAX_DETERMINANT_DISTINCT_RATIO:
                continue
            for rhs in attrs:
                if rhs in lhs:
                    continue
                strength, support_ratio = _fd_strength(df, list(lhs), rhs)
                if support_ratio < MIN_SUPPORT_RATIO:
                    continue
                if strength >= 1.0:
                    results.append(
                        FDResult(
                            lhs=lhs,
                            rhs=rhs,
                            strength=strength,
                            support_ratio=support_ratio,
                            row_count=len(df),
                        )
                    )

    return results, warnings


def _fd_strength(df: pd.DataFrame, determinants: list[str], dependent: str) -> tuple[float, float]:
    if not determinants:
        return 0.0, 0.0
    missing = [c for c in determinants + [dependent] if c not in df.columns]
    if missing:
        return 0.0, 0.0

    subset = df[determinants + [dependent]].copy()
    subset = subset.dropna(subset=determinants)
    if subset.empty:
        return 0.0, 0.0

    total_groups = subset[determinants].drop_duplicates().shape[0]
    if total_groups == 0:
        return 0.0, 0.0

    nunique = subset.groupby(determinants, dropna=False)[dependent].nunique(dropna=False)
    consistent = (nunique <= 1).sum()
    strength = consistent / total_groups
    support_ratio = len(subset) / len(df) if len(df) else 0.0

    return strength, support_ratio


def _is_fd(df: pd.DataFrame, determinants: list[str], dependent: str) -> bool:
    strength, support_ratio = _fd_strength(df, determinants, dependent)
    return strength >= 1.0 and support_ratio >= MIN_SUPPORT_RATIO


def _find_candidate_keys(df: pd.DataFrame, attrs: list[str], max_size: int) -> list[list[str]]:
    keys: list[list[str]] = []

    for size in range(1, min(max_size, len(attrs)) + 1):
        for combo in itertools.combinations(attrs, size):
            if _is_unique(df, list(combo)):
                if not any(set(existing).issubset(combo) for existing in keys):
                    keys.append(list(combo))

    return keys


def _is_unique(df: pd.DataFrame, cols: list[str]) -> bool:
    if not cols:
        return False
    if any(c not in df.columns for c in cols):
        return False
    subset = df[cols].copy()
    subset = subset.dropna(subset=cols)
    if subset.empty:
        return False
    return subset.drop_duplicates().shape[0] == len(subset)


def _sample_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_ROWS_FOR_PROFILE:
        return df
    return df.sample(MAX_ROWS_FOR_PROFILE, random_state=42)


def _norm(value: str) -> str:
    return str(value).strip().lower()


def _align_columns(df: pd.DataFrame, names: list[str]) -> tuple[list[str], list[str]]:
    if not names:
        return [], []
    norm_map: dict[str, str] = {}
    for col in df.columns:
        norm_map.setdefault(_norm(col), col)
    aligned: list[str] = []
    missing: list[str] = []
    for name in names:
        norm = _norm(name)
        if norm in norm_map:
            aligned.append(norm_map[norm])
        else:
            missing.append(name)
    return aligned, missing
