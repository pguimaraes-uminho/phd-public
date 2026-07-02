"""Hybrid Step-1 ERD merge: LLM decomposition ⊕ deterministic hardening.

The LLM decomposes a denormalized CSV into entities far better than the data-only
baseline can on a sparse sample (it reads column-name semantics + world knowledge).
So the LLM's ENTITY DECOMPOSITION is the backbone. The deterministic baseline then
HARDENS it with facts that are witnessed by the data:
  - each attribute that is a real column gets the data-verified DATATYPE (⚙️);
  - every baseline column the LLM OMITTED is backfilled (⚙️) so the hybrid is a
    superset of what the baseline witnessed — it never loses a data column;
  - a primary key made of data-verified unique columns is marked ⚙️;
  - foreign keys the baseline verified in the data are remapped onto the LLM entities
    and added (⚙️) if the LLM missed them.

Per-field provenance: `_prov = "deterministic" | "llm"` on attributes and relationships,
`_pk_prov` on each entity's key, `_name_prov` on the entity name. With no LLM this
returns the pure deterministic baseline (all ⚙️).

Correctness guards (from adversarial review):
  - column→type hardening only uses NORMALIZED keys that are UNAMBIGUOUS in the baseline
    (distinct columns colliding under `_norm` with different types are NOT hardened, so a
    collision is never mislabeled "deterministic");
  - omitted baseline columns are backfilled into the best-matching entity (no data loss);
  - the FK child endpoint is disambiguated by overlap with the baseline FK's own source
    entity (and by preferring a non-key holder), not by arbitrary iteration order.
"""
from __future__ import annotations

import re
from typing import Any


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _rel_key(a: Any, b: Any, fk: Any) -> tuple:
    return (frozenset((_norm(a), _norm(b))), _norm(fk))


def _baseline_facts(baseline_erd: dict):
    """From the deterministic baseline:
      col_type      norm→datatype, ONLY for keys unambiguous in the baseline (a normalized
                    key mapping to >1 distinct type is dropped — a collision, not a fact);
      col_anytype   norm→first-seen datatype (fallback for backfilling an ambiguous column);
      col_orig      norm→original column name (for backfilling);
      verified_keys normalized baseline PK columns;
      base_fks      (orig_fk, norm_fk, norm_pk, from_card, to_card, from_entity_colset);
      base_colsets  per-baseline-entity set of normalized columns (for placement)."""
    norm_types: dict[str, set] = {}
    col_anytype: dict[str, str] = {}
    col_orig: dict[str, str] = {}
    verified_keys: set[str] = set()
    base_colsets: list[set] = []
    ent_by_name: dict[str, set] = {}
    for e in (baseline_erd.get("entities") or []):
        if not isinstance(e, dict):
            continue
        colset: set[str] = set()
        for a in (e.get("attributes") or []):
            if isinstance(a, dict) and a.get("name"):
                n = _norm(a["name"])
                colset.add(n)
                col_orig.setdefault(n, a["name"])
                if a.get("data_type"):
                    norm_types.setdefault(n, set()).add(a["data_type"])
                    col_anytype.setdefault(n, a["data_type"])
        for k in (e.get("primary_key") or []):
            verified_keys.add(_norm(k))
        base_colsets.append(colset)
        if e.get("name"):
            ent_by_name[_norm(e["name"])] = colset
    col_type = {k: next(iter(v)) for k, v in norm_types.items() if len(v) == 1}
    base_fks = []
    for r in (baseline_erd.get("relationships") or []):
        if isinstance(r, dict) and r.get("fk_attribute"):
            base_fks.append((r["fk_attribute"], _norm(r["fk_attribute"]), _norm(r.get("pk_attribute") or ""),
                             r.get("from_cardinality"), r.get("to_cardinality"),
                             ent_by_name.get(_norm(r.get("from_entity")), set())))
    return col_type, col_anytype, col_orig, verified_keys, base_fks, base_colsets


def _pure_baseline(baseline_erd: dict) -> dict:
    ents = []
    for e in (baseline_erd.get("entities") or []):
        if not isinstance(e, dict):
            continue
        attrs = [{**a, "_prov": "deterministic"} for a in (e.get("attributes") or []) if isinstance(a, dict)]
        ents.append({**e, "attributes": attrs, "_prov": "deterministic", "_name_prov": "deterministic", "_pk_prov": "deterministic"})
    rels = [{**r, "_prov": "deterministic"} for r in (baseline_erd.get("relationships") or []) if isinstance(r, dict)]
    return {"entities": ents, "relationships": rels}


def build_hybrid_erd(baseline_erd: dict[str, Any], llm_erd: dict[str, Any] | None) -> dict[str, Any]:
    baseline_erd = baseline_erd or {}
    l_entities = [e for e in ((llm_erd or {}).get("entities") or []) if isinstance(e, dict)]
    if not l_entities:
        return _pure_baseline(baseline_erd)

    col_type, col_anytype, col_orig, verified_keys, base_fks, base_colsets = _baseline_facts(baseline_erd)

    # 1. entities = LLM decomposition, hardened with baseline facts; names kept injective
    used_names: set[str] = set()

    def _uniq(name: str) -> str:
        base, k, nm = str(name or "Entity"), 2, str(name or "Entity")
        while _norm(nm) in used_names:
            nm, k = f"{base} {k}", k + 1
        used_names.add(_norm(nm))
        return nm

    llm_norm_count: dict[str, int] = {}
    for le in l_entities:
        if le.get("name"):
            llm_norm_count[_norm(le["name"])] = llm_norm_count.get(_norm(le["name"]), 0) + 1

    entities: list[dict] = []
    col_to_ents: dict[str, list[dict]] = {}
    pk_to_ent: dict[str, dict] = {}
    llm_name_map: dict[str, str] = {}
    for le in l_entities:
        name = _uniq(le.get("name") or "Entity")
        if le.get("name") and llm_norm_count[_norm(le["name"])] == 1:
            llm_name_map[_norm(le["name"])] = name
        attrs = []
        for a in (le.get("attributes") or []):
            if not isinstance(a, dict) or not a.get("name"):
                continue
            ncol = _norm(a["name"])
            aa = dict(a)
            bt = col_type.get(ncol)                 # unambiguous baseline column → data-verified type
            if bt:
                aa["data_type"] = bt
                aa["_prov"] = "deterministic"
            else:
                aa["_prov"] = "llm"                 # LLM-invented, or an ambiguous/colliding column
            attrs.append(aa)
        pk = [k for k in (le.get("primary_key") or []) if k]
        pk_prov = "deterministic" if pk and all(_norm(k) in verified_keys for k in pk) else "llm"
        ent = {"name": name, "attributes": attrs, "primary_key": pk,
               "_prov": "llm", "_name_prov": "llm", "_pk_prov": pk_prov}
        entities.append(ent)
        for a in attrs:
            col_to_ents.setdefault(_norm(a["name"]), []).append(ent)
        for k in pk:
            pk_to_ent.setdefault(_norm(k), ent)

    hybrid_colsets = {id(e): {_norm(a["name"]) for a in e["attributes"]} for e in entities}

    # 2. completeness: backfill every baseline column the LLM omitted so the hybrid is a
    #    superset of the data-witnessed columns (no silent data loss).
    placed = set().union(*hybrid_colsets.values()) if hybrid_colsets else set()
    for ncol, orig in col_orig.items():
        if ncol in placed:
            continue
        home = next((s for s in base_colsets if ncol in s), set())
        target = max(entities, key=lambda e: (len(hybrid_colsets[id(e)] & home), len(hybrid_colsets[id(e)])))
        dtype = col_type.get(ncol) or col_anytype.get(ncol) or "string"
        target["attributes"].append({"name": orig, "data_type": dtype,
                                     "is_primary_key": False, "is_foreign_key": False, "_prov": "deterministic"})
        hybrid_colsets[id(target)].add(ncol)
        col_to_ents.setdefault(ncol, []).append(target)
        placed.add(ncol)

    entity_pk_norm = {id(e): {_norm(k) for k in e["primary_key"]} for e in entities}
    hybrid_norm = {_norm(e["name"]) for e in entities}
    base_fk_norms = {nf for _o, nf, _p, _fc, _tc, _fc2 in base_fks}

    # 3. relationships: the LLM's (⚙️ if the FK is data-verified, else 🤖) + baseline
    #    FKs the LLM missed, remapped onto the LLM entities (⚙️).
    rels: list[dict] = []
    seen: set = set()
    for r in ((llm_erd or {}).get("relationships") or []):
        if not isinstance(r, dict):
            continue
        f = llm_name_map.get(_norm(r.get("from_entity")), r.get("from_entity"))
        t = llm_name_map.get(_norm(r.get("to_entity")), r.get("to_entity"))
        if _norm(f) not in hybrid_norm or _norm(t) not in hybrid_norm:
            continue
        fk = r.get("fk_attribute")
        if fk:
            k = _rel_key(f, t, fk)
            if k in seen:
                continue
            seen.add(k)
        rr = dict(r)
        rr["from_entity"], rr["to_entity"] = f, t
        rr["from_cardinality"] = r.get("from_cardinality") or "0..N"   # child side (has the FK)
        rr["to_cardinality"] = r.get("to_cardinality") or "1"          # parent side
        rr["_prov"] = "deterministic" if _norm(fk or "") in base_fk_norms else "llm"
        rels.append(rr)

    for orig_fk, nf, npk, fc, tc, from_colset in base_fks:
        # parent (referenced) side owns the PK column; child (FK) side is disambiguated by
        # overlap with the baseline FK's own source entity, preferring a non-key holder.
        parent = pk_to_ent.get(npk) or next(iter(col_to_ents.get(npk, [])), None)
        if parent is None:
            continue
        holders = [e for e in col_to_ents.get(nf, []) if e is not parent]
        if not holders:
            continue
        child = max(holders, key=lambda e: (nf not in entity_pk_norm[id(e)], len(hybrid_colsets[id(e)] & from_colset)))
        if child is parent:
            continue
        frm, to = child["name"], parent["name"]
        k = _rel_key(frm, to, orig_fk)
        if k in seen:
            continue
        seen.add(k)
        rels.append({"from_entity": frm, "to_entity": to, "fk_attribute": orig_fk,
                     "from_cardinality": fc or "0..N", "to_cardinality": tc or "1", "_prov": "deterministic"})

    return {"entities": entities, "relationships": rels}
