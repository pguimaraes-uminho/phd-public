"""Deterministic CSV→ERD baseline (pure pandas, no LLM).

Extracts what is mechanically WITNESSED BY THE DATA: column types + optionality,
candidate keys (single + composite unique column combinations), cross-CSV foreign
keys (inclusion dependencies) with precision gates, relationship cardinalities, and
junction (M:N) detection. The LLM later COMPLEMENTS this — semantic names,
conceptual entities, ambiguous splits — but the baseline alone is a real, grounded
ERD that needs no LLM.

Scope (v1): keys + FKs + cardinalities + types. FD-based 3NF entity-splitting of a
single denormalized CSV is intentionally out of scope (deferred).

Robustness: must never crash on messy CSVs (non-string headers, unhashable cells,
mixed-timezone dates). High precision on FKs — a name-affinity gate keeps spurious
inclusion dependencies out; the LLM adds differently-named FKs.

Refs: Abedjan/Golab/Naumann 2015 (data-profiling survey); HyUCC (Papenbrock &
Naumann 2017) for candidate keys; BINDER/SPIDER (Papenbrock 2015 / Bauckmann 2006)
for inclusion dependencies; classic FK gates (Rostin 2009 / Zhang 2010); junction
detection + reverse-engineering line (Chiang/Barron/Storey 1994; Blaha 1997).
"""
from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd
import pandas.api.types as pdt

from app.models.erd import ERDModel
from app.services.erd_validator import _discover_fds  # reuse the sufficient-sample FD discovery

# For SPLITTING, the determinant must genuinely REPEAT (be witnessed as an entity
# dimension, not a near-unique/computed near-key that trivially determines columns).
_SPLIT_MAX_DETERMINANT_RATIO = 0.5

_KEY_MAX_COLS = 2         # candidate keys up to 2 columns (single + pairs)
_IND_COVERAGE = 1.0       # exact value containment required to promote an IND to an FK
_MIN_DISTINCT_FOR_FK = 2  # a constant column is a subset of everything → not an FK
# cardinality vocabulary the app validator accepts: {"1", "0..1", "0..N", "1..N"}


def _stringify_cols(df: pd.DataFrame) -> pd.DataFrame:
    """All downstream label indexing assumes string column labels; normalize once
    (deduping collisions) so header=None / integer-labeled frames don't crash."""
    new: list[str] = []
    seen: dict[str, int] = {}
    for c in df.columns:
        s = str(c)
        if s in seen:
            seen[s] += 1
            s = f"{s}.{seen[s]}"
        else:
            seen[s] = 0
        new.append(s)
    out = df.copy()
    out.columns = new
    return out


def _valset(series: pd.Series) -> set:
    """Distinct non-null values as a set, tolerant of unhashable cells (lists/dicts)."""
    nn = series.dropna()
    try:
        return set(nn.astype(object))
    except TypeError:
        return set(nn.astype(str))


def _clean_entity_name(raw: str) -> str:
    stem = re.sub(r"\.(csv|tsv|txt)$", "", str(raw or "").strip(), flags=re.I)
    parts = re.split(r"[^a-zA-Z0-9]+", stem)
    name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return name or "Table"


def _infer_type(series: pd.Series) -> str:
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_integer_dtype(series):
        return "integer"
    if pdt.is_float_dtype(series):
        return "float"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    nn = series.dropna()
    if nn.empty:
        return "string"
    coerced = pd.to_numeric(nn, errors="coerce")
    if coerced.notna().all():
        return "integer" if (coerced % 1 == 0).all() else "float"
    s = nn.astype(str).str.strip()
    # only probe datetime on date-LOOKING values (a digit + a separator) → avoids
    # short calendar tokens (Jan/Feb) being mis-typed as datetime.
    if s.str.contains(r"\d").mean() >= 0.5 and s.str.contains(r"[-/:.]").mean() >= 0.5:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce", utc=True)  # utc=True tolerates mixed offsets
            if dt.notna().mean() >= 0.95:
                return "datetime"
        except Exception:
            pass
    lowered = s.str.lower()
    if lowered.isin({"true", "false", "0", "1", "yes", "no"}).all() and lowered.nunique() <= 2:
        return "boolean"
    return "string"


def _profile(df: pd.DataFrame) -> dict[str, dict]:
    n = len(df)
    prof: dict[str, dict] = {}
    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        try:
            distinct = int(nn.nunique())
        except TypeError:  # unhashable cells (lists/dicts)
            distinct = int(nn.astype(str).nunique())
        prof[str(col)] = {
            "type": _infer_type(s),
            "null_ratio": round(1 - (len(nn) / n), 4) if n else 0.0,
            "distinct": distinct,
            "n": n,
        }
    return prof


def _is_unique_key(df: pd.DataFrame, cols: list[str]) -> bool:
    """A candidate key must be NON-NULL and UNIQUE over all rows."""
    if df.empty:
        return False
    sub = df[cols]
    if sub.isna().any(axis=None):
        return False
    try:
        return not sub.duplicated().any()
    except TypeError:  # unhashable cells → cast to string for the uniqueness test
        return not sub.astype(str).duplicated().any()


def _candidate_keys(df: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    """Return (single_col_keys, composite_pair_keys). Composite keys are computed
    for junction detection even when a single-column key exists."""
    cols = [str(c) for c in df.columns]
    singles = [[c] for c in cols if _is_unique_key(df, [c])]
    single_set = {c[0] for c in singles}
    pairs: list[list[str]] = []
    if _KEY_MAX_COLS >= 2:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if a in single_set or b in single_set:
                    continue  # not minimal (superset of a single-col key)
                if _is_unique_key(df, [a, b]):
                    pairs.append([a, b])
    return singles, pairs


def _pick_pk(singles: list[list[str]], pairs: list[list[str]], df: pd.DataFrame) -> list[str]:
    """Heuristic PK election among minimal keys: prefer an 'id'-like single column,
    else the leftmost single, else the smallest composite."""
    if singles:
        id_like = [k for k in singles if re.search(r"(^id$|_id$|_key$|^key$|_code$|^code$)", k[0], re.I)]
        chosen = id_like or singles
        order = {str(c): i for i, c in enumerate(df.columns)}
        return min(chosen, key=lambda k: order.get(k[0], 1_000_000))
    return pairs[0] if pairs else []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(s or "").lower()) if t]


def _singular(s: str) -> str:
    return s[:-1] if len(s) > 3 and s.endswith("s") else s


_GENERIC_KEYS = {"id", "key", "code", "pk", "uuid", "guid"}


def _name_affinity(child_col: str, parent_name: str, parent_key_col: str) -> bool:
    """High-precision FK gate: the child column NAME must reference the parent —
    matching the referenced key column (`fk == pk name`, the common convention) or
    embedding the parent entity name, on TOKEN boundaries (so 'paid'/'valid' do not
    match a key named 'id'). When the parent key is a GENERIC name (id/key/code),
    a bare last-token match is too weak (every `*_id` would match every parent), so
    entity-name affinity is required. Differently-named FKs are left for the LLM."""
    c, pk = _norm(child_col), _norm(parent_key_col)
    if c == pk:
        return True
    toks = _tokens(child_col)
    raw = str(child_col or "").lower()
    pn = _singular(_norm(parent_name))
    # entity-name affinity (strong): the child column embeds the parent entity name
    if pn and (pn in toks or raw.endswith("_" + pn)):
        return True
    # key-name affinity: only for a SPECIFIC (non-generic) key column
    if pk and pk not in _GENERIC_KEYS and (toks[-1:] == [pk] or raw.endswith("_" + pk)):
        return True
    return False


def _type_broad(t: str) -> str:
    return "numeric" if t in ("integer", "float") else t


def _type_compatible(a: str, b: str) -> bool:
    # Lenient: value containment is the strong gate. Only reject clearly-disjoint
    # classes; strings are compatible with anything (CSV columns often read as text).
    if a == "string" or b == "string":
        return True
    return _type_broad(a) == _type_broad(b)


def _entity_name_from_key(col: str) -> str:
    stem = re.sub(r"_?(id|key|code|license|ref|no|num|number|pk)$", "", str(col or ""), flags=re.I) or str(col or "")
    return _clean_entity_name(stem)


_KEY_SUFFIXES = {"code", "id", "key", "license", "ref", "no", "number", "num", "pk"}


def _name_pattern_groups(attrs: list[str], pk_set: set) -> list[dict]:
    """Group columns by a shared name STEM where one column is key-suffixed
    (X_code / X_id / X_license ...). Column-naming conventions strongly signal
    entities (airline_code + airline_name → an Airline); the caller confirms each
    group against the data. Longer (more specific) stems are tried first."""
    toks = {c: _tokens(c) for c in attrs}
    keys = [(toks[c][:-1], c) for c in attrs
            if c not in pk_set and len(toks[c]) >= 2 and toks[c][-1] in _KEY_SUFFIXES]
    groups = []
    for stem, key in keys:
        members = [key] + [c for c in attrs
                           if c not in pk_set and c != key and toks[c][:len(stem)] == stem]
        if len(members) >= 2:
            groups.append({"key": key, "attrs": members})
    groups.sort(key=lambda g: len(_tokens(g["key"])), reverse=True)
    return groups


def _determines(df: pd.DataFrame, key: str, col: str) -> bool:
    """Does `key` functionally determine `col` in the data (each key value → one col value)?"""
    try:
        sub = df[[key, col]].dropna(subset=[key])
        return bool(not sub.empty and (sub.groupby(key)[col].nunique(dropna=False) <= 1).all())
    except (TypeError, KeyError, ValueError):
        return False


def _extract_subentities(df: pd.DataFrame, attrs: list[str], pk: list[str]) -> list[dict]:
    """Decompose a denormalized table into sub-entities. Two complementary,
    HIGH-PRECISION signals, both DATA-CONFIRMED:
      1. NAME PATTERNS — columns sharing a stem with a key-suffixed column
         (airline_code + airline_name) → an entity keyed by that column.
      2. FD fallback — an id-like non-key column that functionally determines other
         columns (for denormalized data whose columns aren't consistently named).
    A group is kept only when the key genuinely REPEATS, is a valid non-null key,
    and actually determines its members in the data. Returns non-overlapping groups
    [{"key": X, "attrs": [X, Y...]}]."""
    if not pk:
        return []
    pk_set = set(pk)
    non_key = {a for a in attrs if a not in pk_set}
    if not non_key:
        return []

    selected: list[dict] = []
    used: set = set(pk_set)

    def _accept(key: str, candidates: list[str]) -> None:
        if key in used or key not in non_key:
            return
        knn = df[[key]].dropna()
        if knn.empty or knn[key].nunique() > len(knn) * _SPLIT_MAX_DETERMINANT_RATIO:
            return  # the determinant must genuinely repeat (not a base candidate key)
        deps = [c for c in candidates
                if c != key and c not in used and c in non_key and _determines(df, key, c)]
        if not deps:
            return
        if not _is_unique_key(df[[key] + deps].drop_duplicates(), [key]):
            return  # X must be a valid (non-null, unique) key of the sub-entity
        selected.append({"key": key, "attrs": [key] + deps})
        used.add(key)
        used.update(deps)

    # 1. name-pattern groups (suggested by column names, confirmed by data)
    for grp in _name_pattern_groups(attrs, pk_set):
        _accept(grp["key"], grp["attrs"])

    # 2. FD fallback for the remaining columns — id-like determinants only, so a
    #    non-key column can't coincidentally determine others on a small sample.
    try:
        fds, _w = _discover_fds(df, list(attrs))
    except Exception:
        fds = []
    fd_groups: dict[str, set] = {}
    for fd in fds:
        if len(fd.lhs) != 1:
            continue
        x, y = fd.lhs[0], fd.rhs
        if x in used or y in used or x == y or y not in non_key or x not in non_key:
            continue
        if not re.search(r"(^id$|_id$|_key$|^key$|_code$|^code$)", str(x), re.I):
            continue
        fd_groups.setdefault(x, set()).add(y)
    for x, ys in sorted(fd_groups.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True):
        _accept(x, [x] + list(ys))

    return selected


def build_baseline_erd(
    dfs: list[pd.DataFrame] | pd.DataFrame, table_names: list[str] | None = None
) -> dict[str, Any]:
    """Build a deterministic baseline ERD from one or more CSV DataFrames.

    Returns {"erd": <ERDModel dict>, "profile": {entity: {col: {...}}},
             "notes": [...]}  — all fields data-derived (provenance: deterministic).
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    dfs = [_stringify_cols(d) for d in dfs]  # consistent string labels everywhere
    names = list(table_names or [])
    while len(names) < len(dfs):
        names.append(f"Table{len(names) + 1}")

    # 1. profile + keys per table. A DENORMALIZED CSV is split into a base entity +
    #    sub-entities extracted from transitive dependencies (a CSV can be >1 entity).
    tables: list[dict] = []
    used_names: set[str] = set()
    notes: list[str] = []

    def _uniq(name: str) -> str:
        base, k, nm = name, 2, name
        while nm in used_names:
            nm, k = f"{base}{k}", k + 1
        used_names.add(nm)
        return nm

    def _spec(frame: pd.DataFrame, name: str, pk: list[str] | None = None) -> dict:
        singles, pairs = _candidate_keys(frame)
        return {"name": name, "df": frame, "prof": _profile(frame), "singles": singles,
                "pairs": pairs, "pk": pk if pk is not None else _pick_pk(singles, pairs, frame)}

    for df, raw in zip(dfs, names):
        cols = [str(c) for c in df.columns]
        subs = _extract_subentities(df, cols, _pick_pk(*_candidate_keys(df), df))
        if not subs:
            tables.append(_spec(df, _uniq(_clean_entity_name(raw))))
            continue
        moved = {c for s in subs for c in s["attrs"] if c != s["key"]}
        base_name = _uniq(_clean_entity_name(raw))
        tables.append(_spec(df[[c for c in cols if c not in moved]], base_name))
        for s in subs:
            sub_name = _uniq(_entity_name_from_key(s["key"]))
            tables.append(_spec(df[s["attrs"]].drop_duplicates(), sub_name, pk=[s["key"]]))
            notes.append(f"Split '{raw}': extracted {sub_name} (keyed by {s['key']}) as a sub-entity.")

    # 2. cross-table inclusion dependencies → foreign keys (unary), with gates
    relationships: list[dict] = []
    for child in tables:
        cdf = child["df"]
        for ccol in [str(c) for c in cdf.columns]:
            cvals = _valset(cdf[ccol])
            if len(cvals) < _MIN_DISTINCT_FOR_FK:
                continue
            best = None  # (parent_name, parent_key_col, coverage)
            for parent in tables:
                if parent is child:
                    continue
                for key in parent["singles"]:  # referenced side MUST be a key (UCC)
                    pcol = key[0]
                    if not _name_affinity(ccol, parent["name"], pcol):
                        continue  # high-precision gate: FK name must reference the parent
                    if not _type_compatible(child["prof"][ccol]["type"], parent["prof"][pcol]["type"]):
                        continue
                    pvals = _valset(parent["df"][pcol])
                    if len(pvals) < _MIN_DISTINCT_FOR_FK:
                        continue
                    coverage = len(cvals & pvals) / len(cvals) if cvals else 0.0
                    if coverage >= _IND_COVERAGE and (best is None or coverage > best[2]):
                        best = (parent["name"], pcol, coverage)
            if best is None:
                continue
            parent_name, pcol, _cov = best
            child_unique = _is_unique_key(cdf, [ccol])
            nn = child["prof"][ccol]["null_ratio"] == 0
            relationships.append({
                "from_entity": child["name"],
                "to_entity": parent_name,
                "from_cardinality": "0..1" if child_unique else "0..N",  # child (FK-holder) end
                "to_cardinality": "1" if nn else "0..1",                 # parent (referenced) end
                "fk_attribute": ccol,
                "pk_attribute": pcol,                                    # referenced parent key
            })

    # 2b. dedupe reciprocal FK edges (A→B and B→A on the SAME column): keep the
    # edge whose FK column is the PARENT's key but NOT the child's key; if it is a
    # key of both (pure 1:1) keep exactly one, deterministically.
    pk_of = {t["name"]: set(t["pk"]) for t in tables}
    by_pair: dict[tuple, list[dict]] = {}
    for r in relationships:
        by_pair.setdefault((frozenset((r["from_entity"], r["to_entity"])), r["fk_attribute"]), []).append(r)
    deduped: list[dict] = []
    for (_pair, col), edges in by_pair.items():
        if len(edges) == 1:
            deduped.append(edges[0])
            continue
        preferred = [e for e in edges
                     if col in pk_of.get(e["to_entity"], set()) and col not in pk_of.get(e["from_entity"], set())]
        chosen = preferred or sorted(edges, key=lambda e: (e["from_entity"], e["to_entity"]))
        deduped.append(chosen[0])
    relationships = deduped

    fk_cols: dict[str, set] = {t["name"]: set() for t in tables}
    for r in relationships:
        fk_cols[r["from_entity"]].add(r["fk_attribute"])

    # 3. assemble entities
    entities: list[dict] = []
    for t in tables:
        pk_set = set(t["pk"])
        attrs = []
        for col in [str(c) for c in t["df"].columns]:
            attrs.append({
                "name": col,
                "data_type": t["prof"][col]["type"],
                "is_primary_key": col in pk_set,
                "is_foreign_key": col in fk_cols[t["name"]],
            })
        entities.append({"name": t["name"], "attributes": attrs, "primary_key": t["pk"]})

    for t in tables:  # junction (M:N) detection: PK is exactly two FK columns
        if len(t["pk"]) == 2 and set(t["pk"]).issubset(fk_cols[t["name"]]):
            notes.append(f"{t['name']} looks like a junction (M:N) table: PK = two FKs.")

    erd = ERDModel.model_validate({"entities": entities, "relationships": relationships})
    return {"erd": erd.model_dump(), "profile": {t["name"]: t["prof"] for t in tables}, "notes": notes}
