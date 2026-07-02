from __future__ import annotations

from collections import defaultdict
import datetime as dt
import json
import math
import re
from typing import Any

from app.core.config import settings
from app.models.erd import ERDModel
from app.models.vocab import EntityMapping, Mapping, Term, Vocabulary, OntologyProposal
from app.services.llm import LLMClient


def _resolve_json_schema_refs(schema: dict) -> dict:
    """Resolve all $ref references in a JSON schema inline by replacing them with definitions from $defs.
    This prevents the Google GenAI SDK's schema resolver from getting stuck in an infinite loop.
    """
    defs = schema.get("$defs", {})
    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    return resolve(dict(defs[def_name]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        elif isinstance(node, list):
            return [resolve(item) for item in node]
        return node
    return resolve(schema)


def build_vocab(
    erd: ERDModel,
    refine_prompt: str | None = None,
    previous_vocab: dict[str, Any] | None = None,
    existing_vocab: dict[str, Any] | None = None,
    sample_data: list[dict[str, Any]] | None = None,
    prompt_history: list[dict[str, Any]] | None = None,
    method: str = "llm",
    include_ontology: bool = True,
    custom_ontology: dict[str, Any] | None = None,
    is_compare: bool = False,
    sample_csv_text: str | None = None,
    expert_brief_text: str | None = None,
    temperature: float = 0.0,
) -> Vocabulary:
    if method == "non_llm":
        return build_vocab_non_llm(erd, sample_data, existing_vocab)

    if method == "rule_based":
        # Deterministic ERD -> ontology mapping (no LLM). Evaluated on the raw
        # output, like the compare path, so it is never post-processed.
        raw = _rule_based_ontology(erd)
        return Vocabulary.model_validate(convert_expert_ground_truth(raw, qualified=is_compare))

    client = LLMClient()
    unpacking_context = None
    if include_ontology:
        unpacking_context = custom_ontology if custom_ontology is not None else _build_unpacking_context(erd)
    llm_available = client.is_available()
    if not llm_available:
        new_vocab = _mock_vocab(erd, existing_vocab)
    else:
        prompt = build_vocab_prompt(
            erd,
            refine_prompt=refine_prompt,
            previous_vocab=previous_vocab,
            existing_vocab=existing_vocab,
            sample_data=sample_data,
            prompt_history=prompt_history,
            unpacking_context=unpacking_context,
            is_compare=is_compare,
            sample_csv_text=sample_csv_text,
            expert_brief_text=expert_brief_text,
        )
        raw = client.generate_json(prompt, temperature=temperature, schema=None)
        if is_compare:
            import os
            from app.helpers.validate_output import schema_check, coverage_check
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            schema_path = os.path.join(root_dir, "prompts", "output_schema.json")
            serr = schema_check(raw, schema_path)
            if serr:
                # Warn, don't abort: the run proceeds and downstream conversion
                # (e.g. splitting a combined composite-FK source) handles it.
                print("[warn] Schema validation issues in proposed mapping:")
                for e in (serr if isinstance(serr, list) else [serr]):
                    print(f"  - {e}")
            cerr = coverage_check(raw, erd.model_dump())
            if cerr:
                print(f"[warn] Coverage warnings for proposed mapping:")
                for e in cerr:
                    print(f"  - {e}")
        vocab_dict = convert_expert_ground_truth(raw, qualified=is_compare)
        new_vocab = Vocabulary.model_validate(vocab_dict)

        if is_compare:
            # Evaluation path: measure the LLM's raw mapping quality directly.
            # Bypass the production post-processing pipeline (term abstraction,
            # dedupe, coverage-filling, reification rules) which rewrites the
            # LLM's canonical terms/verb phrases and contaminates the metric.
            return new_vocab

    # Programmatic Merge: Ensure we don't lose existing terms unless explicitly replaced
    # Supports SYNONYM-AWARE matching: if a new term name matches an existing
    # term's synonym, fold the new term's data into the existing one.
    if llm_available and existing_vocab:
        existing = Vocabulary.model_validate(existing_vocab)
        existing_terms = {t.name.lower(): t for t in existing.terms}
        new_terms = {t.name.lower(): t for t in new_vocab.terms}

        # Build a reverse map: synonym (lower) -> canonical name (lower)
        synonym_to_canonical: dict[str, str] = {}
        for name, term in existing_terms.items():
            for syn in term.synonyms:
                synonym_to_canonical[syn.lower()] = name
            # Also map the canonical name to itself
            synonym_to_canonical[name] = name

        # Track which new term names were remapped to canonical names
        remap: dict[str, str] = {}  # new_name_lower -> canonical_name_lower

        merged_terms: dict[str, Term] = {}
        # Start with all existing terms
        for name, term in existing_terms.items():
            merged_terms[name] = term

        # Overlay new terms with synonym-aware matching
        for name, term in new_terms.items():
            canonical = synonym_to_canonical.get(name)
            if canonical and canonical in merged_terms:
                # Fold into the existing canonical term
                target = merged_terms[canonical]
                # Add the new term's name as a synonym if it's different
                all_synonyms = set(s.lower() for s in target.synonyms)
                if name != canonical and name not in all_synonyms:
                    target.synonyms.append(term.name)  # preserve original case
                # Also merge the new term's synonyms
                for syn in term.synonyms:
                    if syn.lower() not in all_synonyms and syn.lower() != canonical:
                        target.synonyms.append(syn)
                        all_synonyms.add(syn.lower())
                # Update description if the new one is longer
                if term.description and len(term.description) > len(target.description or ""):
                    target.description = term.description
                # Use new source_attributes (current ERD)
                target.source_attributes = list(term.source_attributes)
                if name != canonical:
                    remap[name] = canonical
            elif name in merged_terms:
                # Exact match: update in-place
                target = merged_terms[name]
                target.synonyms = list(set(target.synonyms + term.synonyms))
                if term.description and len(term.description) > len(target.description or ""):
                    target.description = term.description
                target.source_attributes = list(term.source_attributes)
            else:
                # Entirely new term
                merged_terms[name] = term

        new_vocab.terms = list(merged_terms.values())

        # Mappings: keep ONLY current-run mappings (for the current ERD).
        # Old mappings live in the global vocab snapshot and are used in later steps.
        # Just rewrite canonical_term references that were remapped during synonym merge.
        for m in new_vocab.mappings:
            key = m.canonical_term.lower()
            if key in remap:
                canonical_term_obj = merged_terms[remap[key]]
                m.canonical_term = canonical_term_obj.name

    # ── Post-processing: enforce ERD entity names in attributes ──
    # The LLM may use entity names from the existing vocab snapshot instead
    # of the current ERD. Rewrite any non-ERD entity prefixes.
    erd_entity_attrs = {}  # entity_name -> set of attr_names
    for ent in erd.entities:
        erd_entity_attrs[ent.name] = {a.name for a in ent.attributes}

    erd_entities = set(erd_entity_attrs.keys())

    # Build a reverse index: attr_name_lower -> list of ERD entity names
    attr_to_entities: dict[str, list[str]] = {}
    for ent_name, attrs in erd_entity_attrs.items():
        for a in attrs:
            attr_to_entities.setdefault(a.lower(), []).append(ent_name)

    def _fix_attribute(attr_str: str) -> str:
        """Rewrite 'ForeignEntity.Attr' -> 'ERDEntity.Attr' if needed."""
        if "." not in attr_str:
            return attr_str
        prefix, suffix = attr_str.split(".", 1)
        if prefix in erd_entities:
            return attr_str  # Already correct
        # Find an ERD entity that has this attribute
        candidates = attr_to_entities.get(suffix.lower(), [])
        if len(candidates) == 1:
            return f"{candidates[0]}.{suffix}"
        elif len(candidates) > 1:
            # Prefer exact case match
            for c in candidates:
                if suffix in erd_entity_attrs[c]:
                    return f"{c}.{suffix}"
            return f"{candidates[0]}.{suffix}"
        return attr_str  # Can't resolve, keep as-is

    # Fix mappings
    for m in new_vocab.mappings:
        m.attribute = _fix_attribute(m.attribute)

    # Fix source_attributes in terms
    for t in new_vocab.terms:
        t.source_attributes = [_fix_attribute(sa) for sa in t.source_attributes]

    # Build a set of valid ERD attributes for final filtering
    valid_attrs = set()
    for ent_name, attrs in erd_entity_attrs.items():
        for a in attrs:
            valid_attrs.add(f"{ent_name}.{a}")

    # Drop mappings for Entity.Attribute combos not in the ERD
    new_vocab.mappings = [
        m for m in new_vocab.mappings
        if m.attribute in valid_attrs
    ]

    # Drop stale source_attributes from terms
    for t in new_vocab.terms:
        t.source_attributes = [
            sa for sa in t.source_attributes
            if sa in valid_attrs
        ]

    # Canonicalization pass: abstract entity-specific term names
    # (e.g. "Menu Item Price" -> "Price") while preserving mappings/synonyms.
    _abstract_canonical_terms(new_vocab)
    _merge_existing_mappings(new_vocab, existing_vocab)
    _dedupe_mappings(new_vocab)
    _filter_mappings_to_valid_attrs(new_vocab, valid_attrs)
    _align_mappings_with_terms(new_vocab)
    _ensure_mapping_coverage(new_vocab, valid_attrs)
    _apply_unpacking_reification_post_rules(new_vocab, unpacking_context, valid_attrs)
    _dedupe_mappings(new_vocab)
    _filter_mappings_to_valid_attrs(new_vocab, valid_attrs)
    _align_mappings_with_terms(new_vocab)
    _enforce_unique_source_attributes(new_vocab)
    _enrich_mapping_semantics(new_vocab, erd, sample_data=sample_data, unpacking_context=unpacking_context)
    _finalize_entity_mappings(new_vocab, erd, existing_vocab=existing_vocab)
    archive_zero_item_terms(new_vocab)

    return new_vocab


def archive_zero_item_terms(vocab: Vocabulary) -> Vocabulary:
    """Archive terms with zero usage in the current vocabulary snapshot."""
    mapping_usage: dict[str, set[str]] = defaultdict(set)
    for m in vocab.mappings or []:
        term_key = _normalize_key_part(m.canonical_term or "")
        attr_key = _mapping_key(m.attribute or "")
        if term_key and attr_key:
            mapping_usage[term_key].add(attr_key)

    existing_archived = getattr(vocab, "archived_terms", None)
    archived_terms: list[dict[str, Any]] = []
    archived_by_key: dict[str, dict[str, Any]] = {}
    if isinstance(existing_archived, list):
        for item in existing_archived:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = _normalize_key_part(name)
            if not key:
                continue
            archived_by_key[key] = dict(item)
            archived_terms.append(dict(item))

    kept_terms: list[Term] = []
    for t in vocab.terms or []:
        term_key = _normalize_key_part(t.name or "")
        src_keys = {_mapping_key(sa) for sa in (t.source_attributes or []) if _mapping_key(sa)}
        mapping_keys = mapping_usage.get(term_key, set())
        usage_count = len(src_keys.union(mapping_keys))
        if usage_count > 0:
            kept_terms.append(t)
            continue

        if term_key and term_key not in archived_by_key:
            archived_payload = t.model_dump()
            archived_payload["archive_reason"] = "zero_items_after_step2"
            archived_payload["items"] = 0
            archived_terms.append(archived_payload)
            archived_by_key[term_key] = archived_payload

    vocab.terms = kept_terms
    setattr(vocab, "archived_terms", archived_terms)
    setattr(vocab, "archived_term_count", len(archived_terms))
    return vocab


def convert_vocab_to_ontology_proposal(vocab: dict[str, Any]) -> dict[str, Any]:
    """Convert old Vocabulary format (entity_mappings, terms, mappings) to new OntologyProposal format."""
    concepts = []
    
    # 1. Group mappings by entity name
    entity_to_mappings = defaultdict(list)
    for mapping in vocab.get("mappings", []):
        attr = mapping.get("attribute", "")
        if "." in attr:
            ent = attr.split(".")[0]
        else:
            ent = ""
        entity_to_mappings[ent].append(mapping)
        
    # 2. Build concepts from entity_mappings
    terms_dict = {t.get("name").lower(): t for t in vocab.get("terms", []) if t.get("name")}
    
    for em in vocab.get("entity_mappings", []):
        src_entity = em.get("source_entity")
        canonical_entity = em.get("canonical_entity")
        
        attributes = []
        for mapping in entity_to_mappings.get(src_entity, []):
            attr_name = mapping.get("attribute", "")
            canonical_term = mapping.get("canonical_term", "")
            
            # Find the term in Vocabulary terms to see if it maps to relation or property
            term_obj = terms_dict.get(canonical_term.lower())
            
            # Default to property mapping
            maps_to = {
                "type": "property",
                "canonical_name": canonical_term,
                "role": "descriptive"
            }
            attributes.append({
                "source": attr_name,
                "maps_to": maps_to,
                "confidence": mapping.get("confidence", 1.0)
            })
            
        concepts.append({
            "canonical_name": canonical_entity,
            "ontological_category": "kind",  # default category
            "source_entity": src_entity,
            "confidence": em.get("confidence", 1.0),
            "refinement": {
                "type": "rename",
                "note": em.get("rationale", "")
            },
            "attributes": attributes
        })
        
    return {
        "concepts": concepts,
        "relations": []
    }


def build_vocab_prompt(
    erd: ERDModel,
    refine_prompt: str | None = None,
    previous_vocab: dict[str, Any] | None = None,
    existing_vocab: dict[str, Any] | None = None,
    sample_data: list[dict[str, Any]] | None = None,
    prompt_history: list[dict[str, Any]] | None = None,
    unpacking_context: dict[str, Any] | None = None,
    is_compare: bool = False,
    sample_csv_text: str | None = None,
    expert_brief_text: str | None = None,
) -> str:
    # Build EVIDENCE_BLOCK dynamically based on active scenario (slots)
    evidence_parts = []
    
    # 1. ERD (always present)
    erd_payload = erd.model_dump()
    evidence_parts.append(
        "# ERD\n"
        "ERD:\n"
        f"{json.dumps(erd_payload, ensure_ascii=True, indent=2)}"
    )
    
    # 2. DATA SAMPLE (only in LLM-2 and LLM-3)
    if sample_data:
        evidence_parts.append(
            "# DATA SAMPLE (instances per table):\n"
            f"{json.dumps(sample_data, ensure_ascii=True, indent=2)}"
        )
        
    # 3. EXPERT DOMAIN BRIEF (only in LLM-3)
    if unpacking_context:
        evidence_parts.append(
            "# EXPERT DOMAIN BRIEF (general knowledge, not element-level answers):\n"
            f"{json.dumps(unpacking_context, ensure_ascii=True, indent=2)}"
        )
        
    evidence_block = "\n\n".join(evidence_parts)

    if is_compare:
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        prompt_file = os.path.join(root_dir, "prompts", "mapping_prompt.txt")
        with open(prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read()
            
        # Build evidence block for compare run
        erd_text = _erd_to_text(erd)
        if sample_csv_text:
            # Scenario 2/3 (ERD + Data): raw denormalized CSV is the most
            # token-efficient form and lets the LLM see the real instances.
            evidence_block = (
                f"ERD (entities, attributes, relationships):\n{erd_text}\n\n"
                "DATA SAMPLE (denormalized CSV, one row per passenger-flight record):\n"
                f"{sample_csv_text}"
            )
        elif sample_data:
            # Fallback: structured per-table JSON sample.
            sample_json_str = json.dumps(sample_data, ensure_ascii=True, indent=2)
            evidence_block = (
                f"ERD (entities, attributes, relationships):\n{erd_text}\n\n"
                f"DATA SAMPLE (instances per table):\n{sample_json_str}"
            )
        else:
            # Scenario 1 (ERD only)
            evidence_block = f"ERD (entities, attributes, relationships):\n{erd_text}"

        if expert_brief_text:
            # Scenario 3 (LLM-3): expert domain brief — general modelling
            # guidance, not element-level answers.
            evidence_block += (
                "\n\nEXPERT DOMAIN BRIEF (general modelling guidance):\n"
                f"{expert_brief_text}"
            )

        prompt = base_prompt.replace("{{EVIDENCE_BLOCK}}", evidence_block)
    else:
        # Full output contract for production use
        output_contract = (
            "OUTPUT CONTRACT:\n"
            "Return ONLY valid JSON matching this schema (no prose, no markdown):\n"
            "{\n"
            "  \"concepts\": [\n"
            "    { \"canonical_name\": \"\", \"ontological_category\": \"\",\n"
            "      \"source_entity\": \"\", \"confidence\": 0.0,\n"
            "      \"refinement\": { \"type\": \"\", \"note\": \"\" },\n"
            "      \"attributes\": [\n"
            "        { \"source\": \"\", \"maps_to\": { \"type\": \"property|relation_role\",\n"
            "          \"canonical_name\": \"\", \"role\": \"identifier|descriptive\",\n"
            "          \"relation\": \"\", \"target_concept\": \"\" },\n"
            "          \"confidence\": 0.0 } ] } ],\n"
            "  \"relations\": [\n"
            "    { \"canonical_name\": \"\", \"domain\": \"\", \"range\": \"\",\n"
            "      \"from_cardinality\": \"\", \"to_cardinality\": \"\", \"confidence\": 0.0 } ]\n"
            "}\n"
        )

        prompt = (
            "SYSTEM:\n"
            "You are a schema-to-ontology mapping assistant. You map the elements of a\n"
            "relational schema (entities, attributes, relationships) to canonical domain\n"
            "concepts. You GENERATE and EXPLAIN hypotheses; you never assert ground truth.\n"
            "A human curator makes all final decisions.\n\n"
            "METHOD (apply where the evidence supports it; do not force):\n"
            "- Direct mapping: entity \u2192 concept; non-key column \u2192 datatype property\n"
            "  (role: identifier | descriptive); foreign key \u2192 relation.\n"
            "- Naming Conventions (strictly enforce):\n"
            "  - Concepts/Entities: PascalCase nouns indicating the entity name.\n"
            "  - Datatype Properties/Attributes: camelCase nouns indicating the property/attribute name.\n"
            "  - Object Properties/Relations: camelCase active verb phrases describing the semantic role or connection\n"
            "    from the domain concept to the range concept (never use nouns or generic labels like \"Identifier\",\n"
            "    \"relation\", or the target entity name; instead, construct an active verb phrase explaining the direction).\n"
            "- Refinements you may apply, each recorded explicitly:\n"
            "  - rename: source name \u2192 canonical term (applying the naming conventions above)\n"
            "  - reification: a junction table that carries its own attribute(s) \u2192\n"
            "    a concept of ontological_category \"relator\"; a pure junction (no own\n"
            "    attributes) stays a plain relation\n"
            "  - role_disambiguation: two FKs to the same target are distinct relations\n"
            "  - interpretation: any other modelling judgment, with a note\n"
            "- ontological_category \u2208 { kind, relator, category }\n\n"
            f"{output_contract}\n"
            "INPUT:\n"
            f"{evidence_block}"
        )

    if previous_vocab:
        # Convert to OntologyProposal format if in the old Vocabulary format to ensure schema consistency
        if "entity_mappings" in previous_vocab or "mappings" in previous_vocab:
            previous_vocab = convert_vocab_to_ontology_proposal(previous_vocab)
        prompt += (
            "\n\nCURRENT_HYPOTHESIS_JSON:\n"
            f"{json.dumps(previous_vocab, ensure_ascii=True, indent=2)}\n"
            "Refine or correct the vocabulary hypothesis above based on the instructions below.\n"
        )
    if existing_vocab:
        # Convert to OntologyProposal format if in the old Vocabulary format to ensure schema consistency
        if "entity_mappings" in existing_vocab or "mappings" in existing_vocab:
            existing_vocab = convert_vocab_to_ontology_proposal(existing_vocab)
        prompt += (
            "\n\nGLOBAL_VOCAB_SNAPSHOT:\n"
            f"{json.dumps(existing_vocab, ensure_ascii=True, indent=2)}\n"
            "Use this as the existing controlled vocabulary term context. Only add terms required by the ERD.\n"
        )
    if prompt_history:
        prompt += (
            "\n\nPROMPT_HISTORY:\n"
            f"{json.dumps(prompt_history, ensure_ascii=True, indent=2)}\n"
        )
    if refine_prompt and refine_prompt.strip():
        prompt += (
            "\n\nUSER REFINEMENT REQUEST:\n"
            f"{refine_prompt.strip()}\n"
            "Ensure your response adheres to these instructions without introducing unrelated attributes or terms.\n"
        )

    return prompt


_CAMEL_SPLIT_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_ACRONYMS = {"id": "ID", "pk": "PK", "fk": "FK", "url": "URL", "uuid": "UUID"}

_TEMPORAL_TOKENS = {"date", "time", "timestamp", "datetime", "created", "updated", "start", "end", "expires"}
_STATUS_TOKENS = {"status", "state", "phase"}
_CATEGORY_TOKENS = {"category", "type", "class", "kind"}
_MEASURE_TOKENS = {"price", "cost", "amount", "total", "value", "score", "qty", "quantity", "rate", "weight"}
_IDENTIFIER_TOKENS = {"id", "identifier", "uuid", "pk", "fk", "code"}
_BOOLEAN_TOKENS = {"is", "has", "active", "enabled", "flag", "valid", "deleted", "archived"}
_CURRENCY_TOKENS = {"usd", "eur", "gbp", "brl", "cad", "aud", "jpy", "inr", "currency", "price", "cost", "amount"}
_UNIT_TOKENS = {"kg", "g", "lb", "m", "cm", "mm", "km", "s", "ms", "percent", "pct", "ratio"}


def _build_unpacking_context(erd: ERDModel) -> dict[str, Any]:
    """Derive ontological hints (unpacking + reification candidates) from ERD."""
    entity_profiles: list[dict[str, Any]] = []
    attribute_profiles: list[dict[str, Any]] = []
    reification_candidates: list[dict[str, Any]] = []

    entity_kind: dict[str, str] = {}
    for ent in erd.entities:
        fk_attrs = [a for a in ent.attributes if _is_fk_attr(a)]
        pk_attrs = [a for a in ent.attributes if a.is_primary_key]
        non_key_attrs = [a for a in ent.attributes if not a.is_primary_key and not _is_fk_attr(a)]

        referenced_entities: list[str] = []
        for a in fk_attrs:
            ref_ent = _reference_entity_name(a.references)
            if ref_ent:
                referenced_entities.append(ref_ent)
        referenced_entities = sorted(set(referenced_entities))

        if len(fk_attrs) >= 2 and len(non_key_attrs) >= 1:
            kind = "relator_candidate"
            reason = "Entity likely reifies a relationship because it has 2+ FK attributes plus qualifiers."
        elif len(fk_attrs) >= 2 and len(non_key_attrs) == 0:
            kind = "association_candidate"
            reason = "Entity likely represents pure association (join table) between other entities."
        else:
            kind = "entity_candidate"
            reason = "Entity appears to represent a standalone domain object."

        entity_kind[ent.name] = kind
        profile = {
            "entity": ent.name,
            "kind": kind,
            "pk_count": len(pk_attrs),
            "fk_count": len(fk_attrs),
            "non_key_count": len(non_key_attrs),
            "references": referenced_entities,
            "reasoning": reason,
        }
        entity_profiles.append(profile)

        if kind in {"relator_candidate", "association_candidate"}:
            reification_candidates.append(
                {
                    "entity": ent.name,
                    "kind": kind,
                    "mediates": referenced_entities,
                    "guidance": "Prefer attribute-centric canonical terms and avoid entity-prefixed labels.",
                }
            )

    for ent in erd.entities:
        kind = entity_kind.get(ent.name, "entity_candidate")
        for attr in ent.attributes:
            role = _infer_attribute_role(attr.name, is_pk=attr.is_primary_key, is_fk=_is_fk_attr(attr), entity_kind=kind)
            preferred_term = _preferred_term_for_attribute(ent.name, attr.name, role=role, entity_kind=kind)
            attribute_profiles.append(
                {
                    "attribute": f"{ent.name}.{attr.name}",
                    "entity": ent.name,
                    "entity_kind": kind,
                    "role": role,
                    "preferred_term": preferred_term,
                    "is_primary_key": attr.is_primary_key,
                    "is_foreign_key": _is_fk_attr(attr),
                    "references_entity": _reference_entity_name(attr.references),
                }
            )

    return {
        "entity_profiles": entity_profiles,
        "attribute_profiles": attribute_profiles,
        "reification_candidates": reification_candidates,
    }


def _erd_to_text(erd: ERDModel) -> str:
    lines = []
    for ent in erd.entities:
        pk_names = [a.name for a in ent.attributes if a.is_primary_key and not _is_fk_attr(a)]
        fk_pk_names = [a for a in ent.attributes if a.is_primary_key and _is_fk_attr(a)]
        lines.append(f"ENTITY {ent.name}")
        if pk_names:
            lines.append(f"  PK: {', '.join(pk_names)}")
        for attr in fk_pk_names:
            ref = getattr(attr, "references", "") or ""
            lines.append(f"  {attr.name}  ({attr.data_type})  FK(PK)→{ref}")
        for attr in ent.attributes:
            if attr.is_primary_key:
                continue
            if _is_fk_attr(attr):
                ref = getattr(attr, "references", "") or ""
                lines.append(f"  {attr.name}  ({attr.data_type})  FK→{ref}")
            else:
                lines.append(f"  {attr.name}  ({attr.data_type})")
    if erd.relationships:
        lines.append("")
        lines.append("RELATIONSHIPS")
        for r in erd.relationships:
            card = f"{getattr(r, 'from_cardinality', '') or ''}:{getattr(r, 'to_cardinality', '') or ''}"
            lines.append(f"  {r.from_entity} -{card}-> {r.to_entity}")
    return "\n".join(lines)


def _is_fk_attr(attr: Any) -> bool:
    return bool(getattr(attr, "is_foreign_key", False) or getattr(attr, "references", None))


def _reference_entity_name(reference: str | None) -> str | None:
    if not reference:
        return None
    if "." in reference:
        return reference.split(".", 1)[0].strip() or None
    return reference.strip() or None


def _reference_column_name(reference: str | None) -> str | None:
    if not reference or "." not in reference:
        return None
    return reference.split(".", 1)[1].strip() or None


def _rb_camel(s: str) -> str:
    parts = [p for p in re.split(r"[\s_\-]+", s.strip()) if p]
    if not parts:
        return s
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def _rb_pascal(s: str) -> str:
    parts = [p for p in re.split(r"[\s_\-]+", s.strip()) if p]
    if len(parts) == 1 and any(c.isupper() for c in parts[0][1:]):
        return parts[0][0].upper() + parts[0][1:]  # already PascalCase (e.g. BaggageBelt)
    return "".join(p.capitalize() for p in parts)


def _rule_based_ontology(erd: ERDModel) -> dict[str, Any]:
    """Deterministic ERD -> ontology mapping (W3C Direct Mapping style + junction/
    reification heuristics). Recovers the TYPE structure mechanically:
    - entity -> concept (relator if 2+ FKs with own attribute(s); kind otherwise)
    - pure junction (2+ FKs, no own attribute) -> plain relation, not a concept
    - non-FK column -> datatype property (role identifier if PK, else descriptive)
    - FK column(s) -> object property (relation_role); columns that jointly cover a
      target's composite PK collapse into ONE relation, otherwise each FK is its own
      relation (role disambiguation). Names are mechanical (hasTarget / column-based),
      with no world knowledge - that is precisely what an LLM is expected to add.
    """
    pk_by_entity = {e.name: list(getattr(e, "primary_key", []) or []) for e in erd.entities}
    concepts: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    for ent in erd.entities:
        fk_attrs = [a for a in ent.attributes if _is_fk_attr(a)]
        fk_by_name = {a.name: a for a in fk_attrs}
        own_attrs = [a for a in ent.attributes if not a.is_primary_key and not _is_fk_attr(a)]

        # An associative entity has its PRIMARY KEY built from 2+ FKs to distinct
        # entities (its identity IS the link). A natural-keyed entity that merely
        # references others (e.g. Flight, with its own flight_number PK) is a kind.
        pk_fk_targets = set()
        for col in pk_by_entity.get(ent.name, []):
            a = fk_by_name.get(col)
            if a:
                t = _reference_entity_name(getattr(a, "references", None))
                if t:
                    pk_fk_targets.add((col, t))
        is_associative = len(pk_fk_targets) >= 2 and len({t for _, t in pk_fk_targets}) >= 2
        is_pure_junction = is_associative and len(own_attrs) == 0
        is_relator = is_associative and len(own_attrs) >= 1

        # Group FK columns by their target entity.
        groups: dict[str, list] = {}
        for a in fk_attrs:
            tent = _reference_entity_name(getattr(a, "references", None))
            if tent:
                groups.setdefault(tent, []).append(a)

        # Build relation_role entries (composite FK -> one; else one per column).
        rel_entries = []  # (source, relation_name, target)
        for tent, members in groups.items():
            tpk = pk_by_entity.get(tent, [])
            ref_cols = [_reference_column_name(getattr(m, "references", None)) for m in members]
            is_composite = (
                len(members) >= 2
                and len(set(ref_cols)) == len(ref_cols)
                and set(ref_cols) == set(tpk)
            )
            if is_composite:
                src = ", ".join(f"{ent.name}.{m.name}" for m in members)
                rel_entries.append((src, f"has{_rb_pascal(tent)}", tent))
            else:
                multi = len(members) > 1
                for m in members:
                    rname = _rb_camel(m.name) if multi else f"has{_rb_pascal(tent)}"
                    rel_entries.append((f"{ent.name}.{m.name}", rname, tent))

        if is_pure_junction:
            tents = list(groups.keys())
            if len(tents) >= 2:
                relations.append({
                    "canonical_name": f"links{_rb_pascal(tents[0])}{_rb_pascal(tents[1])}",
                    "domain": tents[0],
                    "range": tents[1],
                })
            continue  # a pure junction is a relation, not a concept

        attrs = []
        for a in ent.attributes:
            if _is_fk_attr(a):
                continue
            attrs.append({
                "source": f"{ent.name}.{a.name}",
                "maps_to": {
                    "type": "property",
                    "canonical_name": _rb_camel(a.name),
                    "role": "identifier" if a.is_primary_key else "descriptive",
                },
            })
        for src, rname, tent in rel_entries:
            attrs.append({
                "source": src,
                "maps_to": {"type": "relation_role", "relation": rname, "target_concept": tent},
            })

        concepts.append({
            "canonical_name": _rb_pascal(ent.name),
            "ontological_category": "relator" if is_relator else "kind",
            "source_entity": ent.name,
            "attributes": attrs,
        })

    return {"concepts": concepts, "relations": relations}


def _infer_attribute_role(attr_name: str, is_pk: bool, is_fk: bool, entity_kind: str) -> str:
    tokens = set(_tokenize(attr_name))
    if is_pk or tokens.intersection(_IDENTIFIER_TOKENS):
        return "identifier"
    if is_fk:
        return "foreign_identifier"
    if entity_kind in {"relator_candidate", "association_candidate"}:
        if tokens.intersection(_TEMPORAL_TOKENS):
            return "relationship_temporal_qualifier"
        if tokens.intersection(_MEASURE_TOKENS):
            return "relationship_measure_qualifier"
        return "relationship_qualifier"
    if tokens.intersection(_TEMPORAL_TOKENS):
        return "temporal_quality"
    if tokens.intersection(_STATUS_TOKENS):
        return "status_quality"
    if tokens.intersection(_CATEGORY_TOKENS):
        return "categorical_quality"
    if tokens.intersection(_MEASURE_TOKENS):
        return "quantitative_quality"
    return "descriptive_quality"


def _preferred_term_for_attribute(entity_name: str, attr_name: str, role: str, entity_kind: str) -> str:
    attr_tokens = _tokenize(attr_name)
    entity_tokens = set(_tokenize(entity_name))
    core_tokens = [tok for tok in attr_tokens if tok not in entity_tokens]
    if not core_tokens:
        core_tokens = attr_tokens

    if role in {"identifier", "foreign_identifier"}:
        return "Identifier"
    if role in {"relationship_temporal_qualifier", "temporal_quality"}:
        if any(tok in {"timestamp", "datetime", "time"} for tok in core_tokens):
            return "Timestamp"
        return "Date"
    if role == "status_quality":
        return "Status"
    if role == "categorical_quality":
        if "type" in core_tokens:
            return "Type"
        return "Category"

    if entity_kind in {"relator_candidate", "association_candidate"} and core_tokens:
        # For reified relationships, prefer compact qualifier terms.
        return _format_term(core_tokens)

    return _format_term(core_tokens or attr_tokens) or _format_term(attr_tokens) or attr_name


def _abstract_canonical_terms(vocab: Vocabulary) -> None:
    """Rename over-specific canonical terms to more reusable abstractions."""
    source_index: dict[str, list[str]] = defaultdict(list)

    for t in vocab.terms:
        source_index[t.name.lower()].extend(t.source_attributes or [])
    for m in vocab.mappings:
        source_index[m.canonical_term.lower()].append(m.attribute)

    rename_map: dict[str, str] = {}
    for t in vocab.terms:
        key = t.name.lower()
        abstracted = _abstract_term_name(t.name, source_index.get(key, []))
        if abstracted and abstracted.lower() != key:
            rename_map[key] = abstracted

    if not rename_map:
        return

    for m in vocab.mappings:
        mapped = rename_map.get(m.canonical_term.lower())
        if mapped:
            m.canonical_term = mapped

    merged_terms: dict[str, Term] = {}
    for t in vocab.terms:
        target_name = rename_map.get(t.name.lower(), t.name)
        target_key = target_name.lower()
        if target_key not in merged_terms:
            merged_terms[target_key] = Term(
                name=target_name,
                description=t.description,
                synonyms=list(t.synonyms or []),
                source_attributes=list(t.source_attributes or []),
            )
        else:
            target = merged_terms[target_key]
            if t.description and len(t.description) > len(target.description or ""):
                target.description = t.description
            target.synonyms.extend(t.synonyms or [])
            target.source_attributes.extend(t.source_attributes or [])

        if target_name.lower() != t.name.lower():
            merged_terms[target_key].synonyms.append(t.name)

    vocab.terms = list(merged_terms.values())
    for t in vocab.terms:
        t.synonyms = _dedupe_case_insensitive(
            [s for s in (t.synonyms or []) if s and s.lower() != t.name.lower()]
        )
        t.source_attributes = _dedupe_preserve_order(t.source_attributes or [])


def _merge_existing_mappings(vocab: Vocabulary, existing_vocab: dict[str, Any] | None) -> None:
    """Carry forward approved historical mappings, then let current run override."""
    if not existing_vocab:
        return
    try:
        existing = Vocabulary.model_validate(existing_vocab)
    except Exception:
        return

    # Existing first, current run after. Dedup chooses the best candidate.
    vocab.mappings = list(existing.mappings) + list(vocab.mappings)


def _dedupe_mappings(vocab: Vocabulary) -> None:
    """Enforce one mapping per normalized Entity.Attribute key."""
    best_by_key: dict[str, Mapping] = {}

    for m in vocab.mappings:
        key = _mapping_key(m.attribute)
        if not key:
            continue

        candidate = Mapping.model_validate(m.model_dump())
        current = best_by_key.get(key)
        if not current:
            best_by_key[key] = candidate
            continue

        if _should_replace_mapping(current, candidate):
            best_by_key[key] = candidate

    vocab.mappings = list(best_by_key.values())


def _filter_mappings_to_valid_attrs(vocab: Vocabulary, valid_attrs: set[str]) -> None:
    """Keep only mappings that belong to current ERD and normalize display casing."""
    valid_by_key = {_mapping_key(attr): attr for attr in valid_attrs if _mapping_key(attr)}
    filtered: list[Mapping] = []
    for m in vocab.mappings:
        key = _mapping_key(m.attribute)
        if not key or key not in valid_by_key:
            continue
        m.attribute = valid_by_key[key]
        filtered.append(m)
    vocab.mappings = filtered


def _ensure_mapping_coverage(vocab: Vocabulary, valid_attrs: set[str]) -> None:
    """Guarantee every ERD attribute has exactly one mapping candidate."""
    valid_by_key = {_mapping_key(attr): attr for attr in valid_attrs if _mapping_key(attr)}
    mapped_by_key = {
        _mapping_key(m.attribute): m
        for m in vocab.mappings
        if _mapping_key(m.attribute)
    }

    term_by_norm, syn_to_term = _build_term_indexes(vocab)
    attr_part_to_term = _build_attr_part_index(vocab)

    for key, qualified_attr in valid_by_key.items():
        if key in mapped_by_key:
            continue

        canonical, confidence, rationale = _choose_canonical_for_missing_attr(
            qualified_attr,
            term_by_norm,
            syn_to_term,
            attr_part_to_term,
        )

        vocab.mappings.append(
            Mapping(
                attribute=qualified_attr,
                canonical_term=canonical,
                confidence=confidence,
                rationale=rationale,
            )
        )
        mapped_by_key[key] = vocab.mappings[-1]

        term = _get_or_create_term(vocab, canonical)
        if qualified_attr not in term.source_attributes:
            term.source_attributes.append(qualified_attr)

        # Refresh lightweight indexes so later missing attrs can reuse new decisions.
        term_by_norm[_normalize_key_part(term.name)] = term.name
        for syn in term.synonyms or []:
            syn_to_term.setdefault(_normalize_key_part(syn), term.name)
        part_key = _attribute_part_key(qualified_attr)
        if part_key:
            attr_part_to_term.setdefault(part_key, term.name)


def _build_term_indexes(vocab: Vocabulary) -> tuple[dict[str, str], dict[str, str]]:
    term_by_norm: dict[str, str] = {}
    syn_to_term: dict[str, str] = {}
    for t in vocab.terms:
        norm = _normalize_key_part(t.name)
        if norm and norm not in term_by_norm:
            term_by_norm[norm] = t.name
        for syn in t.synonyms or []:
            syn_norm = _normalize_key_part(syn)
            if syn_norm and syn_norm not in syn_to_term:
                syn_to_term[syn_norm] = t.name
    return term_by_norm, syn_to_term


def _build_attr_part_index(vocab: Vocabulary) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = defaultdict(dict)

    for t in vocab.terms:
        for sa in t.source_attributes or []:
            part_key = _attribute_part_key(sa)
            if not part_key:
                continue
            votes.setdefault(part_key, {})
            votes[part_key][t.name] = votes[part_key].get(t.name, 0) + 1

    for m in vocab.mappings:
        part_key = _attribute_part_key(m.attribute)
        if not part_key:
            continue
        votes.setdefault(part_key, {})
        votes[part_key][m.canonical_term] = votes[part_key].get(m.canonical_term, 0) + 1

    out: dict[str, str] = {}
    for part_key, term_votes in votes.items():
        if not term_votes:
            continue
        # Highest vote wins; lexical tie-break for determinism.
        winner = sorted(term_votes.items(), key=lambda x: (-x[1], x[0].lower()))[0][0]
        out[part_key] = winner
    return out


def _choose_canonical_for_missing_attr(
    qualified_attr: str,
    term_by_norm: dict[str, str],
    syn_to_term: dict[str, str],
    attr_part_to_term: dict[str, str],
) -> tuple[str, float, str]:
    _, attr_name = _split_qualified_attribute(qualified_attr)
    attr_name = attr_name or qualified_attr

    part_key = _attribute_part_key(qualified_attr)
    if part_key and part_key in attr_part_to_term:
        term = attr_part_to_term[part_key]
        return term, 0.75, "Coverage fallback: reused term by attribute-name evidence."

    attr_title = _format_term(_tokenize(attr_name)) or attr_name
    abstract_title = _abstract_term_name(attr_title, [qualified_attr]) or attr_title

    for candidate in (abstract_title, attr_title):
        norm = _normalize_key_part(candidate)
        if norm and norm in term_by_norm:
            term = term_by_norm[norm]
            return term, 0.7, "Coverage fallback: matched existing canonical term by normalized name."
        if norm and norm in syn_to_term:
            term = syn_to_term[norm]
            return term, 0.68, "Coverage fallback: matched existing term via synonym."

    return abstract_title, 0.55, "Coverage fallback: created canonical term from ERD attribute."


def _get_or_create_term(vocab: Vocabulary, term_name: str) -> Term:
    for t in vocab.terms:
        if t.name.lower() == term_name.lower():
            return t
    term = Term(name=term_name, description=None, synonyms=[], source_attributes=[])
    vocab.terms.append(term)
    return term


def _apply_unpacking_reification_post_rules(
    vocab: Vocabulary,
    unpacking_context: dict[str, Any] | None,
    valid_attrs: set[str],
) -> None:
    """Apply deterministic semantic adjustments using unpacking/reification hints."""
    if not unpacking_context:
        return

    profiles = unpacking_context.get("attribute_profiles") or []
    profile_by_attr: dict[str, dict[str, Any]] = {}
    profile_by_key: dict[str, dict[str, Any]] = {}
    for p in profiles:
        attr = p.get("attribute")
        if not attr:
            continue
        key = _mapping_key(attr)
        profile_by_key[key] = p
        if attr in valid_attrs:
            profile_by_attr[attr] = p

    for m in vocab.mappings:
        profile = profile_by_attr.get(m.attribute) or profile_by_key.get(_mapping_key(m.attribute))
        if not profile:
            continue

        preferred = str(profile.get("preferred_term") or "").strip()
        if not preferred:
            continue

        resolved = _resolve_existing_term_name(vocab, preferred) or preferred
        if not _should_reify_to_preferred(m.canonical_term, resolved, profile):
            continue

        if m.canonical_term.lower() != resolved.lower():
            _append_term_synonym(vocab, resolved, m.canonical_term)
            m.canonical_term = resolved
            if not m.rationale:
                m.rationale = "Adjusted by unpacking/reification post-rule."

    # Optional metadata for explainability in UI/history.
    relator_entities = {
        str(rc.get("entity"))
        for rc in (unpacking_context.get("reification_candidates") or [])
        if rc.get("entity")
    }
    for t in vocab.terms:
        roles: set[str] = set()
        contexts: set[str] = set()
        for sa in t.source_attributes or []:
            profile = profile_by_attr.get(sa) or profile_by_key.get(_mapping_key(sa))
            if not profile:
                continue
            role = profile.get("role")
            if role:
                roles.add(str(role))
            ent = profile.get("entity")
            if ent and str(ent) in relator_entities:
                contexts.add(str(ent))
        if roles:
            setattr(t, "unpacking_roles", sorted(roles))
        if contexts:
            setattr(t, "reification_contexts", sorted(contexts))


def _should_reify_to_preferred(current_term: str, preferred_term: str, profile: dict[str, Any]) -> bool:
    current_norm = _normalize_key_part(current_term)
    preferred_norm = _normalize_key_part(preferred_term)
    if not current_norm or not preferred_norm:
        return False
    if current_norm == preferred_norm:
        return True

    role = str(profile.get("role") or "")
    entity = str(profile.get("entity") or "")

    if role in {"identifier", "foreign_identifier"}:
        return True

    current_tokens = _tokenize(current_term)
    preferred_tokens = _tokenize(preferred_term)
    entity_tokens = set(_tokenize(entity))

    # If the current term is entity-prefixed and preferred is the compact suffix,
    # prioritize the preferred canonical term.
    if preferred_tokens and len(current_tokens) > len(preferred_tokens):
        suffix = current_tokens[-len(preferred_tokens):]
        prefix = current_tokens[:-len(preferred_tokens)]
        if suffix == preferred_tokens and prefix and all(tok in entity_tokens for tok in prefix):
            return True

    # For reified relationship qualifiers and identifiers, enforce compact terms.
    if role.startswith("relationship_") or role in {"identifier", "foreign_identifier"}:
        if preferred_norm in current_norm:
            return True

    return False


def _resolve_existing_term_name(vocab: Vocabulary, candidate: str) -> str | None:
    wanted = _normalize_key_part(candidate)
    if not wanted:
        return None
    for t in vocab.terms:
        if _normalize_key_part(t.name) == wanted:
            return t.name
    for t in vocab.terms:
        for syn in t.synonyms or []:
            if _normalize_key_part(syn) == wanted:
                return t.name
    return None


def _append_term_synonym(vocab: Vocabulary, canonical: str, synonym: str) -> None:
    if not canonical or not synonym:
        return
    if canonical.lower() == synonym.lower():
        return
    term = _get_or_create_term(vocab, canonical)
    term.synonyms = _dedupe_case_insensitive(list(term.synonyms or []) + [synonym])


def _enrich_mapping_semantics(
    vocab: Vocabulary,
    erd: ERDModel,
    sample_data: list[dict[str, Any]] | None = None,
    unpacking_context: dict[str, Any] | None = None,
) -> None:
    """Attach deterministic semantic hints to each mapping."""
    attr_type_index = _build_erd_attr_type_index(erd)
    role_index = _build_role_index(unpacking_context)
    sample_index = _index_sample_values(sample_data, erd)

    for m in vocab.mappings:
        key = _mapping_key(m.attribute)
        if not key:
            continue

        attr_meta = attr_type_index.get(key, {})
        samples = sample_index.get(key, [])
        role_hint = role_index.get(key)
        _, attr_name = _split_qualified_attribute(m.attribute)
        attr_name = attr_name or m.attribute

        logical_type = str(getattr(m, "logical_type", "") or "").strip().lower()
        if not logical_type:
            logical_type = _infer_logical_type(
                attr_name=attr_name,
                data_type=attr_meta.get("data_type"),
                role_hint=role_hint,
                sample_values=samples,
                is_pk=bool(attr_meta.get("is_pk")),
                is_fk=bool(attr_meta.get("is_fk")),
            )
        if logical_type:
            setattr(m, "logical_type", logical_type)

        semantic_role = str(getattr(m, "semantic_role", "") or "").strip()
        if not semantic_role:
            semantic_role = role_hint or _infer_semantic_role(attr_name, logical_type)
        if semantic_role:
            setattr(m, "semantic_role", semantic_role)

        value_domain = getattr(m, "value_domain", None)
        if not isinstance(value_domain, list) or not value_domain:
            inferred_domain = _infer_value_domain(logical_type, samples)
            if inferred_domain:
                setattr(m, "value_domain", inferred_domain)
                value_domain = inferred_domain

        normalization_profile = str(getattr(m, "normalization_profile", "") or "").strip().lower()
        if not normalization_profile:
            normalization_profile = _infer_normalization_profile(
                logical_type=logical_type,
                attr_name=attr_name,
                value_domain=value_domain if isinstance(value_domain, list) else None,
            )
        if normalization_profile:
            setattr(m, "normalization_profile", normalization_profile)

        unit = str(getattr(m, "unit", "") or "").strip()
        if not unit:
            inferred_unit = _infer_unit(attr_name, samples)
            if inferred_unit:
                setattr(m, "unit", inferred_unit)

        currency = str(getattr(m, "currency", "") or "").strip().upper()
        if not currency:
            inferred_currency = _infer_currency(attr_name, samples)
            if inferred_currency:
                setattr(m, "currency", inferred_currency)
                currency = inferred_currency
        elif currency:
            setattr(m, "currency", currency)

        existing_conf = getattr(m, "semantic_confidence", None)
        if existing_conf is None:
            setattr(
                m,
                "semantic_confidence",
                _semantic_confidence(
                    logical_type=logical_type,
                    data_type=attr_meta.get("data_type"),
                    sample_values=samples,
                    role_hint=role_hint,
                    value_domain=value_domain if isinstance(value_domain, list) else None,
                ),
            )


def _build_erd_attr_type_index(erd: ERDModel) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for ent in erd.entities:
        for attr in ent.attributes:
            key = _mapping_key(f"{ent.name}.{attr.name}")
            if not key:
                continue
            index[key] = {
                "data_type": (attr.data_type or "").strip().lower(),
                "is_pk": bool(attr.is_primary_key),
                "is_fk": bool(attr.is_foreign_key or attr.references),
                "nullable": attr.nullable,
            }
    return index


def _build_role_index(unpacking_context: dict[str, Any] | None) -> dict[str, str]:
    roles: dict[str, str] = {}
    if not unpacking_context:
        return roles
    for profile in unpacking_context.get("attribute_profiles") or []:
        attr = str(profile.get("attribute") or "").strip()
        role = str(profile.get("role") or "").strip()
        key = _mapping_key(attr)
        if key and role:
            roles[key] = role
    return roles


def _index_sample_values(
    sample_data: list[dict[str, Any]] | None,
    erd: ERDModel,
    max_values: int = 40,
) -> dict[str, list[Any]]:
    by_full_key: dict[str, list[Any]] = {}
    by_attr_part: dict[str, list[Any]] = {}

    if sample_data:
        for row in sample_data:
            if not isinstance(row, dict):
                continue
            for raw_col, raw_value in row.items():
                if raw_value is None:
                    continue
                col = str(raw_col or "").strip()
                if not col:
                    continue

                full_key = _mapping_key(col) if "." in col else ""
                part_key = _normalize_key_part(col.split(".", 1)[-1])

                if full_key:
                    by_full_key.setdefault(full_key, [])
                    if len(by_full_key[full_key]) < max_values:
                        by_full_key[full_key].append(raw_value)
                if part_key:
                    by_attr_part.setdefault(part_key, [])
                    if len(by_attr_part[part_key]) < max_values:
                        by_attr_part[part_key].append(raw_value)

    out: dict[str, list[Any]] = {}
    for ent in erd.entities:
        for attr in ent.attributes:
            qualified = f"{ent.name}.{attr.name}"
            key = _mapping_key(qualified)
            if not key:
                continue
            part_key = _normalize_key_part(attr.name)
            if key in by_full_key:
                out[key] = by_full_key[key]
            elif part_key in by_attr_part:
                out[key] = by_attr_part[part_key]
            else:
                out[key] = []
    return out


def _infer_logical_type(
    attr_name: str,
    data_type: str | None,
    role_hint: str | None,
    sample_values: list[Any],
    is_pk: bool,
    is_fk: bool,
) -> str:
    dtype = (data_type or "").strip().lower()
    tokens = set(_tokenize(attr_name))
    role = (role_hint or "").strip().lower()

    if is_pk or is_fk or role in {"identifier", "foreign_identifier"}:
        return "identifier"
    if tokens.intersection(_IDENTIFIER_TOKENS):
        return "identifier"

    bool_ratio = _sample_bool_ratio(sample_values)
    if "bool" in dtype or bool_ratio >= 0.9:
        return "boolean"
    if bool_ratio >= 0.6 and tokens.intersection(_BOOLEAN_TOKENS):
        return "boolean"

    date_ratio = _sample_date_ratio(sample_values)
    if any(tok in tokens for tok in {"timestamp", "datetime"}) or "timestamp" in dtype or "datetime" in dtype:
        return "datetime"
    if tokens.intersection(_TEMPORAL_TOKENS) or "date" in dtype or date_ratio >= 0.9:
        return "date"

    num_ratio = _sample_numeric_ratio(sample_values)
    if any(k in dtype for k in ("int", "float", "double", "decimal", "numeric", "number")) or num_ratio >= 0.9:
        if tokens.intersection(_CURRENCY_TOKENS):
            return "currency"
        if any(tok in tokens for tok in {"percent", "pct", "ratio"}):
            return "ratio"
        return "number"
    if num_ratio >= 0.65 and tokens.intersection(_MEASURE_TOKENS):
        if tokens.intersection(_CURRENCY_TOKENS):
            return "currency"
        return "number"

    if tokens.intersection(_STATUS_TOKENS) or tokens.intersection(_CATEGORY_TOKENS):
        return "categorical"
    if _looks_like_category(sample_values):
        return "categorical"

    return "text"


def _infer_normalization_profile(
    logical_type: str,
    attr_name: str,
    value_domain: list[str] | None,
) -> str:
    lt = (logical_type or "").strip().lower()
    tokens = set(_tokenize(attr_name))

    if lt == "identifier":
        return "id_normalized"
    if lt == "boolean":
        return "boolean_cast"
    if lt == "date":
        return "date_iso"
    if lt == "datetime":
        return "datetime_iso"
    if lt == "currency":
        return "currency_amount"
    if lt in {"number", "ratio"}:
        return "decimal_number"
    if lt == "categorical":
        return "enum_casefold"

    if value_domain:
        return "enum_casefold"
    if tokens.intersection(_IDENTIFIER_TOKENS):
        return "id_normalized"
    return "text_trim"


def _infer_semantic_role(attr_name: str, logical_type: str) -> str:
    tokens = set(_tokenize(attr_name))
    lt = (logical_type or "").strip().lower()
    if lt == "identifier" or tokens.intersection(_IDENTIFIER_TOKENS):
        return "identifier"
    if lt in {"date", "datetime"} or tokens.intersection(_TEMPORAL_TOKENS):
        return "temporal"
    if tokens.intersection(_STATUS_TOKENS):
        return "status"
    if tokens.intersection(_CATEGORY_TOKENS):
        return "category"
    if lt in {"number", "currency", "ratio"} or tokens.intersection(_MEASURE_TOKENS):
        return "measure"
    return "descriptor"


def _infer_value_domain(logical_type: str, sample_values: list[Any], max_domain: int = 12) -> list[str] | None:
    lt = (logical_type or "").strip().lower()
    if lt == "boolean":
        return ["false", "true"]
    if lt not in {"categorical", "text"}:
        return None
    if not sample_values:
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in sample_values:
        if value is None:
            continue
        s = str(value).strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(s)
        if len(cleaned) > max_domain:
            return None
    if not cleaned:
        return None

    if _looks_like_category(sample_values):
        return sorted(cleaned, key=lambda x: x.casefold())
    return None


def _infer_unit(attr_name: str, sample_values: list[Any]) -> str | None:
    tokens = _tokenize(attr_name)
    for tok in tokens:
        if tok in _UNIT_TOKENS:
            return tok

    unit_match = re.compile(r"\b(kg|g|lb|m|cm|mm|km|ms|s|%)\b", re.IGNORECASE)
    for value in sample_values[:20]:
        if value is None:
            continue
        s = str(value)
        found = unit_match.search(s)
        if found:
            return found.group(1).lower()
    return None


def _infer_currency(attr_name: str, sample_values: list[Any]) -> str | None:
    tokens = set(_tokenize(attr_name))
    for token in tokens:
        upper = token.upper()
        if upper in {"USD", "EUR", "GBP", "BRL", "CAD", "AUD", "JPY", "INR"}:
            return upper
    if tokens.intersection({"currency"}):
        return None

    symbol_to_iso = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
    for value in sample_values[:20]:
        if value is None:
            continue
        s = str(value)
        for symbol, iso in symbol_to_iso.items():
            if symbol in s:
                return iso
        upper = s.upper()
        for iso in ("USD", "EUR", "GBP", "BRL", "CAD", "AUD", "JPY", "INR"):
            if iso in upper:
                return iso
    return None


def _sample_numeric_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    total = 0
    ok = 0
    for value in values:
        if value is None:
            continue
        total += 1
        if _is_numeric(value):
            ok += 1
    return (ok / total) if total else 0.0


def _sample_bool_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    total = 0
    ok = 0
    for value in values:
        if value is None:
            continue
        total += 1
        if _is_boolean(value):
            ok += 1
    return (ok / total) if total else 0.0


def _sample_date_ratio(values: list[Any]) -> float:
    if not values:
        return 0.0
    total = 0
    ok = 0
    for value in values:
        if value is None:
            continue
        total += 1
        if _is_date_like(value):
            ok += 1
    return (ok / total) if total else 0.0


def _looks_like_category(values: list[Any]) -> bool:
    if not values:
        return False
    cleaned = [str(v).strip().casefold() for v in values if v is not None and str(v).strip()]
    if not cleaned:
        return False
    unique = set(cleaned)
    if len(unique) <= 1:
        return False
    if len(unique) <= 12 and len(unique) <= max(2, int(len(cleaned) * 0.4)):
        return True
    return False


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        s = s.replace(",", "")
        s = re.sub(r"^[\$€£¥]\s*", "", s)
        return bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", s))
    return False


def _is_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return value in {0, 1}
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
    return False


def _is_date_like(value: Any) -> bool:
    if isinstance(value, (dt.date, dt.datetime)):
        return True
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False

    s_norm = s.replace("Z", "+00:00")
    try:
        dt.datetime.fromisoformat(s_norm)
        return True
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            dt.datetime.strptime(s, fmt)
            return True
        except ValueError:
            continue
    return False


def _semantic_confidence(
    logical_type: str,
    data_type: str | None,
    sample_values: list[Any],
    role_hint: str | None,
    value_domain: list[str] | None,
) -> float:
    score = 0.45
    if data_type:
        score += 0.12
    if sample_values:
        score += 0.12
    if role_hint:
        score += 0.08
    if value_domain:
        score += 0.05
    if logical_type in {"identifier", "boolean", "date", "datetime", "currency", "number", "ratio", "categorical"}:
        score += 0.06

    sample_support = max(
        _sample_numeric_ratio(sample_values),
        _sample_bool_ratio(sample_values),
        _sample_date_ratio(sample_values),
    )
    score += min(0.12, sample_support * 0.12)
    return round(min(score, 0.99), 2)


def _finalize_entity_mappings(
    vocab: Vocabulary,
    erd: ERDModel,
    existing_vocab: dict[str, Any] | None,
) -> None:
    valid_entities = {ent.name for ent in erd.entities}
    _merge_existing_entity_mappings(vocab, existing_vocab)
    _dedupe_entity_mappings(vocab)
    _filter_entity_mappings_to_valid_entities(vocab, valid_entities)
    _ensure_entity_mapping_coverage(vocab, erd, existing_vocab)
    _dedupe_entity_mappings(vocab)
    _filter_entity_mappings_to_valid_entities(vocab, valid_entities)


def _merge_existing_entity_mappings(vocab: Vocabulary, existing_vocab: dict[str, Any] | None) -> None:
    if not existing_vocab:
        return
    try:
        existing = Vocabulary.model_validate(existing_vocab)
    except Exception:
        return
    vocab.entity_mappings = list(existing.entity_mappings or []) + list(vocab.entity_mappings or [])


def _dedupe_entity_mappings(vocab: Vocabulary) -> None:
    best_by_source: dict[str, EntityMapping] = {}
    for mapping in vocab.entity_mappings or []:
        source_key = _entity_mapping_key(mapping.source_entity)
        if not source_key:
            continue
        candidate = EntityMapping.model_validate(mapping.model_dump())
        current = best_by_source.get(source_key)
        if not current:
            best_by_source[source_key] = candidate
            continue
        if _should_replace_entity_mapping(current, candidate):
            best_by_source[source_key] = candidate
    vocab.entity_mappings = list(best_by_source.values())


def _should_replace_entity_mapping(current: EntityMapping, candidate: EntityMapping) -> bool:
    curr_conf = float(current.confidence) if current.confidence is not None else -1.0
    cand_conf = float(candidate.confidence) if candidate.confidence is not None else -1.0

    if cand_conf > curr_conf:
        return True
    if cand_conf < curr_conf:
        return False

    curr_rat = bool((current.rationale or "").strip())
    cand_rat = bool((candidate.rationale or "").strip())
    if cand_rat and not curr_rat:
        return True
    if curr_rat and not cand_rat:
        return False
    return True


def _filter_entity_mappings_to_valid_entities(vocab: Vocabulary, valid_entities: set[str]) -> None:
    valid_by_key = {_entity_mapping_key(name): name for name in valid_entities if _entity_mapping_key(name)}
    filtered: list[EntityMapping] = []

    for mapping in vocab.entity_mappings or []:
        source_key = _entity_mapping_key(mapping.source_entity)
        if not source_key or source_key not in valid_by_key:
            continue
        mapping.source_entity = valid_by_key[source_key]
        mapping.canonical_entity = _normalize_entity_label(mapping.canonical_entity or mapping.source_entity)
        filtered.append(mapping)

    vocab.entity_mappings = filtered


def _ensure_entity_mapping_coverage(
    vocab: Vocabulary,
    erd: ERDModel,
    existing_vocab: dict[str, Any] | None,
) -> None:
    existing_source_to_canonical, canonical_pool = _build_existing_entity_indexes(existing_vocab)
    for mapping in vocab.entity_mappings or []:
        source_key = _entity_mapping_key(mapping.source_entity)
        canonical = _normalize_entity_label(mapping.canonical_entity)
        if source_key and canonical:
            existing_source_to_canonical.setdefault(source_key, canonical)
            canonical_pool.add(canonical)

    mapped_by_key = {
        _entity_mapping_key(m.source_entity): m
        for m in (vocab.entity_mappings or [])
        if _entity_mapping_key(m.source_entity)
    }

    for ent in erd.entities:
        source_name = ent.name
        source_key = _entity_mapping_key(source_name)
        if source_key in mapped_by_key:
            continue

        canonical, confidence, rationale = _choose_canonical_for_entity(
            source_name,
            existing_source_to_canonical=existing_source_to_canonical,
            canonical_pool=canonical_pool,
        )

        new_mapping = EntityMapping(
            source_entity=source_name,
            canonical_entity=canonical,
            confidence=confidence,
            rationale=rationale,
        )
        vocab.entity_mappings.append(new_mapping)
        mapped_by_key[source_key] = new_mapping
        existing_source_to_canonical[source_key] = canonical
        canonical_pool.add(canonical)


def _build_existing_entity_indexes(existing_vocab: dict[str, Any] | None) -> tuple[dict[str, str], set[str]]:
    source_to_canonical: dict[str, str] = {}
    canonical_pool: set[str] = set()

    if not existing_vocab:
        return source_to_canonical, canonical_pool

    try:
        existing = Vocabulary.model_validate(existing_vocab)
    except Exception:
        return source_to_canonical, canonical_pool

    for mapping in existing.entity_mappings or []:
        source_key = _entity_mapping_key(mapping.source_entity)
        canonical = _normalize_entity_label(mapping.canonical_entity)
        if not source_key or not canonical:
            continue
        source_to_canonical[source_key] = canonical
        canonical_pool.add(canonical)

    return source_to_canonical, canonical_pool


def _choose_canonical_for_entity(
    source_entity: str,
    existing_source_to_canonical: dict[str, str],
    canonical_pool: set[str],
) -> tuple[str, float, str]:
    source_key = _entity_mapping_key(source_entity)
    if source_key and source_key in existing_source_to_canonical:
        canonical = existing_source_to_canonical[source_key]
        return canonical, 0.86, "Entity coverage fallback: reused canonical entity from approved vocabulary."

    source_norm = _normalize_entity_label(source_entity)
    source_norm_key = _entity_mapping_key(source_norm)

    for canonical in sorted(canonical_pool, key=lambda v: v.lower()):
        if _entity_mapping_key(canonical) == source_norm_key:
            return canonical, 0.74, "Entity coverage fallback: matched existing canonical entity by normalized name."

    source_tokens = set(_tokenize(source_entity))
    best_candidate = None
    best_score = 0.0
    for canonical in canonical_pool:
        canonical_tokens = set(_tokenize(canonical))
        if not source_tokens or not canonical_tokens:
            continue
        overlap = len(source_tokens.intersection(canonical_tokens))
        union = len(source_tokens.union(canonical_tokens))
        score = overlap / union if union else 0.0
        if score > best_score:
            best_score = score
            best_candidate = canonical

    if best_candidate and best_score >= 0.6:
        return best_candidate, 0.68, "Entity coverage fallback: matched existing canonical entity by lexical overlap."

    return source_norm, 0.58, "Entity coverage fallback: created canonical entity from ERD entity name."


def _entity_mapping_key(value: str) -> str:
    return _normalize_key_part(value or "")


def _normalize_entity_label(value: str) -> str:
    tokens = _tokenize(value)
    if not tokens:
        return str(value or "").strip()
    return _format_term(tokens)


def _should_replace_mapping(current: Mapping, candidate: Mapping) -> bool:
    curr_conf = float(current.confidence) if current.confidence is not None else -1.0
    cand_conf = float(candidate.confidence) if candidate.confidence is not None else -1.0

    if cand_conf > curr_conf:
        return True
    if cand_conf < curr_conf:
        return False

    curr_rat = bool((current.rationale or "").strip())
    cand_rat = bool((candidate.rationale or "").strip())
    if cand_rat and not curr_rat:
        return True
    if curr_rat and not cand_rat:
        return False

    # Tie-break: prefer later candidate (usually current run).
    return True


def _align_mappings_with_terms(vocab: Vocabulary) -> None:
    """Ensure mappings reference an existing canonical term object."""
    term_by_lower: dict[str, Term] = {t.name.lower(): t for t in vocab.terms}
    synonym_to_term: dict[str, str] = {}
    for t in vocab.terms:
        synonym_to_term[t.name.lower()] = t.name.lower()
        for syn in t.synonyms or []:
            synonym_to_term[syn.lower()] = t.name.lower()

    for m in vocab.mappings:
        key = m.canonical_term.lower()
        if key not in term_by_lower and key in synonym_to_term:
            key = synonym_to_term[key]
        if key in term_by_lower:
            canonical = term_by_lower[key]
            m.canonical_term = canonical.name
            if m.attribute and m.attribute not in canonical.source_attributes:
                canonical.source_attributes.append(m.attribute)
            continue

        new_term = Term(
            name=m.canonical_term,
            description=None,
            synonyms=[],
            source_attributes=[m.attribute] if m.attribute else [],
        )
        vocab.terms.append(new_term)
        term_by_lower[key] = new_term

    for t in vocab.terms:
        t.source_attributes = _dedupe_preserve_order(t.source_attributes or [])
        t.synonyms = _dedupe_case_insensitive(
            [s for s in (t.synonyms or []) if s and s.lower() != t.name.lower()]
        )


def _enforce_unique_source_attributes(vocab: Vocabulary) -> None:
    """Prevent duplicate source attributes within/across terms."""
    term_by_lower = {t.name.lower(): t for t in vocab.terms}
    owner_by_attr: dict[str, str] = {}
    display_by_attr: dict[str, str] = {}

    for m in vocab.mappings:
        key = _mapping_key(m.attribute)
        if not key:
            continue
        owner_by_attr[key] = m.canonical_term.lower()
        display_by_attr[key] = m.attribute

    # Remove attributes from terms that map to another canonical term.
    for t in vocab.terms:
        cleaned: list[str] = []
        seen_keys: set[str] = set()
        for sa in t.source_attributes or []:
            key = _mapping_key(sa)
            if not key or key in seen_keys:
                continue
            owner = owner_by_attr.get(key)
            if owner and owner != t.name.lower():
                continue
            seen_keys.add(key)
            cleaned.append(sa)
        t.source_attributes = cleaned

    # Ensure each mapped attribute exists exactly once, owned by the mapped term.
    for key, owner in owner_by_attr.items():
        term = term_by_lower.get(owner)
        if not term:
            continue
        attr = display_by_attr[key]
        if _mapping_key(attr) not in {_mapping_key(v) for v in term.source_attributes}:
            term.source_attributes.append(attr)

    for t in vocab.terms:
        t.source_attributes = _dedupe_by_key(t.source_attributes or [], _mapping_key)
        t.synonyms = _dedupe_case_insensitive(
            [s for s in (t.synonyms or []) if s and s.lower() != t.name.lower()]
        )


def _abstract_term_name(term_name: str, source_attributes: list[str]) -> str:
    term_tokens = _tokenize(term_name)
    if not term_tokens:
        return term_name
    if not source_attributes:
        return _format_term(term_tokens)

    entity_token_sets: list[set[str]] = []
    salient_attr_tokens: set[str] = set()

    for attr in source_attributes:
        entity, attribute = _split_qualified_attribute(attr)
        if not entity or not attribute:
            continue
        entity_tokens = set(_tokenize(entity))
        attribute_tokens = _tokenize(attribute)
        if not attribute_tokens:
            continue

        # Remove duplicated entity prefixes from the attribute when present
        # (e.g. menu_item_id under MenuItems -> id).
        core_tokens = [tok for tok in attribute_tokens if tok not in entity_tokens]
        chosen_tokens = core_tokens or attribute_tokens

        entity_token_sets.append(entity_tokens)
        salient_attr_tokens.update(chosen_tokens)

    if not entity_token_sets:
        return _format_term(term_tokens)

    removable_tokens = set(entity_token_sets[0])
    for tok_set in entity_token_sets[1:]:
        removable_tokens &= tok_set

    if not removable_tokens:
        return _format_term(term_tokens)

    removable_prefix = 0
    for tok in term_tokens:
        if tok in removable_tokens and tok not in salient_attr_tokens:
            removable_prefix += 1
        else:
            break

    # Conservative trim: require at least 2 leading entity tokens, so
    # "Order Date" doesn't collapse to "Date" by default.
    if removable_prefix >= 2 and removable_prefix < len(term_tokens):
        term_tokens = term_tokens[removable_prefix:]

    return _format_term(term_tokens)


def _split_qualified_attribute(value: str) -> tuple[str | None, str | None]:
    if "." not in value:
        return None, None
    entity, attr = value.split(".", 1)
    return entity.strip(), attr.strip()


def _tokenize(value: str) -> list[str]:
    if not value:
        return []
    value = _CAMEL_SPLIT_RE.sub(r"\1 \2", value)
    value = _NON_ALNUM_RE.sub(" ", value).strip().lower()
    if not value:
        return []
    return [_singularize(tok) for tok in value.split() if tok]


def _singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _format_term(tokens: list[str]) -> str:
    if not tokens:
        return ""
    parts = []
    for tok in tokens:
        parts.append(_ACRONYMS.get(tok, tok.capitalize()))
    return " ".join(parts)


def _mapping_key(value: str) -> str:
    if not value:
        return ""
    if "." in value:
        entity, attr = value.split(".", 1)
        return f"{_normalize_key_part(entity)}.{_normalize_key_part(attr)}"
    return _normalize_key_part(value)


def _attribute_part_key(value: str) -> str:
    if not value:
        return ""
    if "." in value:
        _, attr = value.split(".", 1)
        return _normalize_key_part(attr)
    return _normalize_key_part(value)


def _normalize_key_part(value: str) -> str:
    if not value:
        return ""
    value = _CAMEL_SPLIT_RE.sub(r"\1 \2", value).lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _dedupe_case_insensitive(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_by_key(values: list[str], key_fn) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = key_fn(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _mock_vocab(erd: ERDModel, existing_vocab: dict[str, Any] | None = None) -> Vocabulary:
    terms: dict[str, Term] = {}
    mappings: list[Mapping] = []
    entity_mappings: list[EntityMapping] = []

    if existing_vocab:
        try:
            existing = Vocabulary.model_validate(existing_vocab)
            for term in existing.terms:
                terms[term.name] = Term(
                    name=term.name,
                    description=term.description,
                    synonyms=list(term.synonyms),
                    source_attributes=list(term.source_attributes),
                )
            mappings.extend(existing.mappings)
            entity_mappings.extend(existing.entity_mappings)
        except Exception:
            pass

    seen_entity_keys = {_entity_mapping_key(m.source_entity) for m in entity_mappings}
    for ent in erd.entities:
        ent_key = _entity_mapping_key(ent.name)
        if ent_key and ent_key not in seen_entity_keys:
            entity_mappings.append(
                EntityMapping(
                    source_entity=ent.name,
                    canonical_entity=_normalize_entity_label(ent.name),
                    confidence=0.5,
                    rationale="Mock entity mapping",
                )
            )
            seen_entity_keys.add(ent_key)
        for attr in ent.attributes:
            canonical = attr.name.strip().lower().replace(" ", "_")
            if canonical not in terms:
                terms[canonical] = Term(
                    name=canonical,
                    description=None,
                    synonyms=[attr.name],
                    source_attributes=[f"{ent.name}.{attr.name}"],
                )
            else:
                terms[canonical].synonyms.append(attr.name)
                terms[canonical].source_attributes.append(f"{ent.name}.{attr.name}")

            mappings.append(
                Mapping(
                    attribute=f"{ent.name}.{attr.name}",
                    canonical_term=canonical,
                    confidence=0.5,
                    rationale="Mock mapping",
                )
            )

    return Vocabulary(terms=list(terms.values()), mappings=mappings, entity_mappings=entity_mappings)


# ---------------------------------------------------------------------------
# STEP2 B: Algorithmic (Non-LLM) Schema Mapping
# ---------------------------------------------------------------------------

def build_vocab_non_llm(
    erd: ERDModel,
    sample_data: list[dict[str, Any]] | None = None,
    existing_vocab_data: dict[str, Any] | None = None,
) -> Vocabulary:
    existing_vocab = None
    if existing_vocab_data:
        try:
            existing_vocab = Vocabulary.model_validate(existing_vocab_data)
        except Exception:
            pass

    # 1. Normalização Estrutural (handled by helper normalizers & _normalize_entity_label)
    # Extract entities and attributes as candidates
    terms_dict: dict[str, Term] = {}
    mappings: list[Mapping] = []
    entity_mappings: list[EntityMapping] = []

    # Map Entities to Canonical Entities (3. Entity alignment / 4. Concept consolidation)
    seen_entity_keys = set()
    for ent in erd.entities:
        ent_key = _entity_mapping_key(ent.name)
        if not ent_key or ent_key in seen_entity_keys:
            continue
        seen_entity_keys.add(ent_key)

        best_entity_canonical = None
        best_entity_score = 0.0
        best_entity_details = None

        # Scan existing canonical entity targets from baseline
        existing_canonical_entities = set()
        if existing_vocab:
            for em in existing_vocab.entity_mappings or []:
                if em.canonical_entity:
                    existing_canonical_entities.add(em.canonical_entity)

        for candidate in sorted(existing_canonical_entities):
            score, details = _calculate_entity_match_score(ent, candidate, erd, existing_vocab)
            if score > best_entity_score:
                best_entity_score = score
                best_entity_canonical = candidate
                best_entity_details = details

        # If best match is strong enough, consolidate. Otherwise create new canonical abstraction.
        if best_entity_canonical and best_entity_score >= 0.6:
            canonical_entity = best_entity_canonical
            rationale = f"Algorithmic alignment (score={best_entity_score:.2f}) to baseline concept."
            entity_details = best_entity_details
        else:
            canonical_entity = _normalize_entity_label(ent.name)
            rationale = "Generated new canonical entity concept via structural normalization."
            best_entity_score = 0.8  # high confidence for clean auto-generated term
            entity_details = {
                "syntactic": 0.8,
                "structural": 0.0,
                "connectivity": 1.0,
                "weights": {
                    "syntactic": 0.5,
                    "structural": 0.3,
                    "connectivity": 0.2
                }
            }

        entity_mappings.append(
            EntityMapping(
                source_entity=ent.name,
                canonical_entity=canonical_entity,
                confidence=round(best_entity_score, 2),
                rationale=rationale,
                match_details=entity_details,
            )
        )

    # Initialize terms dictionary from existing baseline to keep definitions/synonyms
    if existing_vocab:
        for term in existing_vocab.terms:
            terms_dict[term.name.lower()] = Term(
                name=term.name,
                description=term.description,
                synonyms=list(term.synonyms),
                source_attributes=list(term.source_attributes),
            )

    # Map Attributes to Canonical Terms
    for ent in erd.entities:
        for attr in ent.attributes:
            qualified_attr = f"{ent.name}.{attr.name}"
            best_term_name = None
            best_term_score = 0.0
            best_term_details = None

            # Match against existing baseline terms
            for term_lower, term_obj in terms_dict.items():
                score, details = _calculate_attribute_match_score(
                    ent.name, attr.name, term_obj, sample_data, mappings
                )
                if score > best_term_score:
                    best_term_score = score
                    best_term_name = term_obj.name
                    best_term_details = details

            # Consolidate or create new canonical term
            if best_term_name and best_term_score >= 0.65:
                canonical_term = best_term_name
                rationale = f"Algorithmic alignment (score={best_term_score:.2f}) to baseline term."
                # Add source attribute to consolidated term
                term_key = canonical_term.lower()
                if qualified_attr not in terms_dict[term_key].source_attributes:
                    terms_dict[term_key].source_attributes.append(qualified_attr)
                attr_details = best_term_details
            else:
                # Clean candidate concept name
                canonical_term = _format_term(_tokenize(attr.name)) or attr.name
                term_key = canonical_term.lower()
                best_term_score = 0.75
                rationale = "Generated new abstract canonical term concept."
                attr_details = {
                    "syntactic": 0.75,
                    "instance": 0.0,
                    "structural": 0.0,
                    "weights": {
                        "syntactic": 0.4,
                        "instance": 0.4,
                        "structural": 0.2
                    }
                }

                if term_key not in terms_dict:
                    terms_dict[term_key] = Term(
                        name=canonical_term,
                        description=f"Canonical abstraction generated from {qualified_attr}.",
                        synonyms=[attr.name],
                        source_attributes=[qualified_attr],
                    )
                else:
                    if qualified_attr not in terms_dict[term_key].source_attributes:
                        terms_dict[term_key].source_attributes.append(qualified_attr)
                    if attr.name not in terms_dict[term_key].synonyms:
                        terms_dict[term_key].synonyms.append(attr.name)

            mappings.append(
                Mapping(
                    attribute=qualified_attr,
                    canonical_term=canonical_term,
                    confidence=round(best_term_score, 2),
                    rationale=rationale,
                    match_details=attr_details,
                )
            )

    # Wrap in Vocabulary
    return Vocabulary(
        terms=list(terms_dict.values()),
        mappings=mappings,
        entity_mappings=entity_mappings
    )


# ── Helper functions for STEP2 B Matching Signals ──

def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _syntactic_similarity(s1: str, s2: str) -> float:
    s1_clean = _normalize_key_part(s1)
    s2_clean = _normalize_key_part(s2)
    if not s1_clean or not s2_clean:
        return 0.0
    dist = _levenshtein_distance(s1_clean, s2_clean)
    max_len = max(len(s1_clean), len(s2_clean))
    return 1.0 - (dist / max_len)


def _token_jaccard_similarity(s1: str, s2: str) -> float:
    t1 = set(_tokenize(s1))
    t2 = set(_tokenize(s2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _hybrid_syntactic_similarity(s1: str, s2: str) -> float:
    lev = _syntactic_similarity(s1, s2)
    jac = _token_jaccard_similarity(s1, s2)
    return max(lev, jac)


def _attribute_overlap_similarity(set1: set[str], set2: set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _get_sample_values(attr_name: str, sample_data: list[dict[str, Any]] | None) -> list[Any]:
    if not sample_data:
        return []
    norm_attr = _normalize_key_part(attr_name)
    values = []
    for row in sample_data:
        for k, v in row.items():
            if _normalize_key_part(k) == norm_attr:
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    values.append(v)
                break
    return values


def _value_overlap_ratio(vals1: list[Any], vals2: list[Any]) -> float:
    if not vals1 or not vals2:
        return 0.0
    set1 = set(str(v).strip().lower() for v in vals1)
    set2 = set(str(v).strip().lower() for v in vals2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _is_date_string(val: str) -> bool:
    if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", val):
        return True
    if re.match(r"^\d{2}[-/]\d{2}[-/]\d{4}", val):
        return True
    return False


def _detect_value_type(vals: list[Any]) -> str:
    if not vals:
        return "unknown"
    all_numeric = True
    all_dates = True
    for v in vals:
        v_str = str(v).strip()
        if not v_str:
            continue
        try:
            float(v_str)
        except ValueError:
            all_numeric = False
        if not _is_date_string(v_str):
            all_dates = False

    if all_dates:
        return "date"
    if all_numeric:
        return "numeric"
    return "text"


def _calculate_attribute_match_score(
    ent_name: str,
    attr_name: str,
    term: Term,
    sample_data: list[dict[str, Any]] | None,
    existing_mappings: list[Mapping]
) -> tuple[float, dict[str, Any]]:
    # 1. Syntactic similarity (0.4 weight)
    best_syn_sim = 0.0
    for syn in term.synonyms:
        best_syn_sim = max(best_syn_sim, _hybrid_syntactic_similarity(attr_name, syn))
    name_sim = _hybrid_syntactic_similarity(attr_name, term.name)
    syntactic_score = max(name_sim, best_syn_sim)

    # 2. Instance-based similarity (0.4 weight)
    instance_score = 0.0
    vals_curr = _get_sample_values(attr_name, sample_data)
    type_curr = _detect_value_type(vals_curr)

    mapped_attrs = []
    for m in existing_mappings:
        if m.canonical_term.lower() == term.name.lower():
            mapped_attrs.append(m.attribute)
    for sa in term.source_attributes:
        mapped_attrs.append(sa)
    mapped_attrs = list(set(mapped_attrs))

    if mapped_attrs:
        best_overlap = 0.0
        type_matches = 0
        for sa in mapped_attrs:
            sa_ent, sa_attr = _split_qualified_attribute(sa)
            sa_attr_name = sa_attr or sa
            vals_other = _get_sample_values(sa_attr_name, sample_data)
            type_other = _detect_value_type(vals_other)
            if type_curr == type_other and type_curr != "unknown":
                type_matches += 1
            best_overlap = max(best_overlap, _value_overlap_ratio(vals_curr, vals_other))

        type_compat = type_matches / len(mapped_attrs) if mapped_attrs else 0.0
        instance_score = (best_overlap * 0.7) + (type_compat * 0.3)
    else:
        term_tokens = set(_tokenize(term.name))
        if type_curr == "date" and term_tokens.intersection(_TEMPORAL_TOKENS):
            instance_score = 0.8
        elif type_curr == "numeric" and term_tokens.intersection(_MEASURE_TOKENS):
            instance_score = 0.8
        elif type_curr == "text" and not term_tokens.intersection(_TEMPORAL_TOKENS):
            instance_score = 0.5

    # 3. Structural Context score (0.2 weight)
    structural_score = 0.0
    for sa in term.source_attributes:
        sa_ent, _ = _split_qualified_attribute(sa)
        if sa_ent and ent_name:
            structural_score = max(structural_score, _hybrid_syntactic_similarity(ent_name, sa_ent))

    combined = (syntactic_score * 0.4) + (instance_score * 0.4) + (structural_score * 0.2)
    details = {
        "syntactic": round(syntactic_score, 4),
        "instance": round(instance_score, 4),
        "structural": round(structural_score, 4),
        "weights": {
            "syntactic": 0.4,
            "instance": 0.4,
            "structural": 0.2
        }
    }
    return combined, details


def _calculate_entity_match_score(
    ent: Entity,
    canonical_entity_name: str,
    erd: ERDModel,
    existing_vocab: Vocabulary | None
) -> tuple[float, dict[str, Any]]:
    # 1. Syntactic similarity (0.5 weight)
    syntactic_score = _hybrid_syntactic_similarity(ent.name, canonical_entity_name)

    # 2. Structural similarity - attribute overlap (0.3 weight)
    ent_attrs = set(_normalize_key_part(a.name) for a in ent.attributes)
    canonical_attrs = set()
    if existing_vocab:
        mapped_src_entities = []
        for em in existing_vocab.entity_mappings or []:
            if em.canonical_entity.lower() == canonical_entity_name.lower():
                mapped_src_entities.append(em.source_entity.lower())
        for m in existing_vocab.mappings or []:
            m_ent, m_attr = _split_qualified_attribute(m.attribute)
            if m_ent and m_ent.lower() in mapped_src_entities:
                canonical_attrs.add(_normalize_key_part(m_attr or m.attribute))

    attribute_score = _attribute_overlap_similarity(ent_attrs, canonical_attrs) if canonical_attrs else 0.0

    # 3. Connectivity similarity (0.2 weight)
    src_deg_out = len([r for r in erd.relationships if r.from_entity == ent.name])
    src_deg_in = len([r for r in erd.relationships if r.to_entity == ent.name])
    connectivity_score = 1.0 if (src_deg_out + src_deg_in) > 0 else 0.5

    combined = (syntactic_score * 0.5) + (attribute_score * 0.3) + (connectivity_score * 0.2)
    details = {
        "syntactic": round(syntactic_score, 4),
        "structural": round(attribute_score, 4),
        "connectivity": round(connectivity_score, 4),
        "weights": {
            "syntactic": 0.5,
            "structural": 0.3,
            "connectivity": 0.2
        }
    }
    return combined, details


def convert_expert_ground_truth(custom_format: dict[str, Any], qualified: bool = False) -> dict[str, Any]:
    """Convert expert/LLM concept format to the internal Vocabulary shape.

    qualified=False (production): attribute stored as bare column name ("passport").
    qualified=True (evaluation): attribute kept as "Entity.col" so the metric/audit
    stay entity-aware and don't conflate same-named columns across entities.
    """
    entity_mappings = []
    mappings = []
    terms_dict = {}

    for concept in custom_format.get("concepts", []):
        src_entity = concept.get("source_entity")
        canonical_entity = concept.get("canonical_name")
        if src_entity and canonical_entity:
            entity_mappings.append({
                "source_entity": src_entity,
                "canonical_entity": canonical_entity,
                "confidence": 1.0,
                "rationale": "Expert ground truth mapping",
                "ontological_category": concept.get("ontological_category"),
            })

        for attr in concept.get("attributes", []):
            source = attr.get("source")
            maps_to = attr.get("maps_to", {})
            canonical_term = maps_to.get("canonical_name") or maps_to.get("relation") or attr.get("canonical_name")
            if source and canonical_term:
                # A composite FK may be expressed as one combined source, e.g.
                # "Seat.flight_number, Seat.flight_date". Expand into one mapping
                # per column, all sharing the same canonical_term (mirrors gold).
                for part in source.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    bare = part.split(".")[-1].strip() if "." in part else part
                    if qualified:
                        # Keep the entity qualifier; if a part omits it (e.g.
                        # "Seat.a, b"), borrow the concept's source_entity.
                        if "." in part:
                            attr_key = f"{part.split('.', 1)[0].strip()}.{bare}"
                        else:
                            attr_key = f"{src_entity}.{bare}" if src_entity else bare
                    else:
                        attr_key = bare
                    mappings.append({
                        "attribute": attr_key,
                        "canonical_term": canonical_term,
                        "confidence": 1.0,
                        "rationale": f"Expert ground truth mapping ({maps_to.get('type', 'property')})",
                        "mapping_type": maps_to.get("type", "property"),
                        "accepted_aliases": list(maps_to.get("accepted", []) or []),
                    })
                    if canonical_term not in terms_dict:
                        terms_dict[canonical_term] = {
                            "name": canonical_term,
                            "description": f"Expert ground truth concept: {canonical_entity}",
                            "synonyms": [],
                            "source_attributes": []
                        }
                    if attr_key not in terms_dict[canonical_term]["source_attributes"]:
                        terms_dict[canonical_term]["source_attributes"].append(attr_key)

    return {
        "entity_mappings": entity_mappings,
        "terms": list(terms_dict.values()),
        "mappings": mappings
    }


def calculate_evaluation_metrics(
    proposed_vocab: dict[str, Any] | Vocabulary,
    gold_vocab: dict[str, Any] | Vocabulary,
) -> dict[str, Any]:
    # Ensure they are Vocabulary objects
    if isinstance(proposed_vocab, dict):
        if "concepts" in proposed_vocab and "mappings" not in proposed_vocab:
            proposed_vocab = convert_expert_ground_truth(proposed_vocab, qualified=True)
        proposed_vocab = Vocabulary.model_validate(proposed_vocab)
    if isinstance(gold_vocab, dict):
        if "concepts" in gold_vocab and "mappings" not in gold_vocab:
            gold_vocab = convert_expert_ground_truth(gold_vocab, qualified=True)
        gold_vocab = Vocabulary.model_validate(gold_vocab)

    # 1. Alignment Precision, Recall, F1  — scored by CONCEPT IDENTITY (PDF §8.5)
    #
    # "a proposed mapping is correct when it assigns a source element to the same
    #  domain concept as the ground truth, whatever name it carries."
    #
    # Operationally: a mapping is CORRECT if its SOURCE element appears in gold,
    # regardless of the canonical label used. This gives the baseline full credit for
    # the structure it recovers; refinement quality (label correctness) is captured
    # separately by refinement_accuracy.

    proposed_entities = {
        em.source_entity.strip().lower()
        for em in proposed_vocab.entity_mappings or []
    }
    gold_entities = {
        em.source_entity.strip().lower()
        for em in gold_vocab.entity_mappings or []
    }

    # Proposed attrs may be qualified ("Customer.passport") or bare ("passport").
    # Gold attrs (after convert_expert_ground_truth) are always bare.
    # Normalise both to bare names for a fair concept-identity comparison.
    def _bare(name: str) -> str:
        return name.split(".")[-1] if "." in name else name

    proposed_attributes = {
        _bare(m.attribute.strip().lower())
        for m in proposed_vocab.mappings or []
    }
    gold_attributes = {
        _bare(m.attribute.strip().lower())
        for m in gold_vocab.mappings or []
    }

    correct_entities = proposed_entities & gold_entities
    correct_attributes = proposed_attributes & gold_attributes

    correct_matches = len(correct_entities) + len(correct_attributes)
    all_proposed = len(proposed_entities) + len(proposed_attributes)
    all_gold = len(gold_entities) + len(gold_attributes)

    precision = correct_matches / all_proposed if all_proposed > 0 else 1.0
    recall = correct_matches / all_gold if all_gold > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # FP: source elements proposed but not in gold (hallucinations or spurious mappings)
    # FN: gold source elements not covered by the proposal (misses)
    false_positives = len(
        (proposed_entities - gold_entities) | (proposed_attributes - gold_attributes)
    )
    false_negatives = len(
        (gold_entities - proposed_entities) | (gold_attributes - proposed_attributes)
    )



    # 2. Curation Effort
    # _norm: normalises a label for comparison by removing whitespace/punctuation formatting.
    # "baggage belt" == "baggagebelt", "flight_number" == "flightnumber" etc.
    # This avoids penalising the baseline for trivial formatting differences vs. genuine renames.
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[\s\-_]+", "", s.strip().lower())

    _VERB_PREFIXES = ("is", "has", "get")

    def _strip_prefix(s: str) -> str:
        for p in _VERB_PREFIXES:
            if s.startswith(p) and len(s) > len(p):
                return s[len(p):]
        return s

    def _relation_match(gold: str, proposed_set: set) -> bool:
        """Exact match OR gold is contained in proposed after stripping is/has/get prefix."""
        if gold in proposed_set:
            return True
        for p in proposed_set:
            if gold in p or _strip_prefix(p) == gold:
                return True
        return False

    num_gold_concepts = len(gold_vocab.terms or [])
    entity_ops = 0
    prop_ent_map = {em.source_entity.strip().lower(): _norm(em.canonical_entity) for em in proposed_vocab.entity_mappings or []}
    for em in gold_vocab.entity_mappings or []:
        src = em.source_entity.strip().lower()
        target = _norm(em.canonical_entity)
        if src not in prop_ent_map or prop_ent_map[src] != target:
            entity_ops += 1

    attribute_ops = 0
    prop_attr_map = {_bare(m.attribute.strip().lower()): _norm(m.canonical_term) for m in proposed_vocab.mappings or []}
    for m in gold_vocab.mappings or []:
        attr = _bare(m.attribute.strip().lower())
        target = _norm(m.canonical_term)
        if attr not in prop_attr_map or prop_attr_map[attr] != target:
            attribute_ops += 1

    # Term ops — use _norm keys so "Baggage Belt" matches "BaggageBelt"
    gold_terms_by_name = {_norm(t.name): t for t in gold_vocab.terms or []}
    prop_terms_by_name = {_norm(t.name): t for t in proposed_vocab.terms or []}

    term_ops = 0
    added_terms = set(gold_terms_by_name.keys()) - set(prop_terms_by_name.keys())
    term_ops += len(added_terms)

    deleted_terms = set(prop_terms_by_name.keys()) - set(gold_terms_by_name.keys())
    term_ops += len(deleted_terms)

    common_terms = set(gold_terms_by_name.keys()).intersection(set(prop_terms_by_name.keys()))
    for name in common_terms:
        g_term = gold_terms_by_name[name]
        p_term = prop_terms_by_name[name]
        if (g_term.description or "").strip().lower() != (p_term.description or "").strip().lower():
            term_ops += 1
        g_syns = {_norm(s) for s in g_term.synonyms or []}
        p_syns = {_norm(s) for s in p_term.synonyms or []}
        term_ops += len(g_syns.symmetric_difference(p_syns))

    total_ops = entity_ops + attribute_ops + term_ops
    curation_effort = total_ops / num_gold_concepts if num_gold_concepts > 0 else 0.0


    # 3. Over-merge rate
    over_merge_collapses = 0
    prop_term_attrs = defaultdict(list)
    for m in proposed_vocab.mappings or []:
        prop_term_attrs[m.canonical_term.strip().lower()].append(m.attribute.strip().lower())
        
    gold_attr_term = {m.attribute.strip().lower(): m.canonical_term.strip().lower() for m in gold_vocab.mappings or []}
    for p_term, attrs in prop_term_attrs.items():
        g_terms = {gold_attr_term[attr] for attr in attrs if attr in gold_attr_term}
        if len(g_terms) > 1:
            over_merge_collapses += (len(g_terms) - 1)
            
    num_proposals = len(proposed_vocab.mappings or [])
    over_merge_rate = over_merge_collapses / num_proposals if num_proposals > 0 else 0.0

    # 4. Under-merge rate
    under_merge_separations = 0
    gold_term_attrs = defaultdict(list)
    for m in gold_vocab.mappings or []:
        gold_term_attrs[m.canonical_term.strip().lower()].append(m.attribute.strip().lower())
        
    for g_term, attrs in gold_term_attrs.items():
        p_terms = {prop_attr_map[attr] for attr in attrs if attr in prop_attr_map}
        if len(p_terms) > 1:
            under_merge_separations += (len(p_terms) - 1)
            
    under_merge_rate = under_merge_separations / num_proposals if num_proposals > 0 else 0.0

    # 5. Hallucination rate
    # A proposed term is hallucinated only when NONE of its source_attributes
    # appear in the gold grounding (i.e., the term has no support in the schema).
    # Naming mismatches (right source, wrong label) are NOT hallucinations — they
    # are already captured by refinement_accuracy.
    gold_grounded_attrs = {
        _bare(m.attribute.strip().lower()) for m in gold_vocab.mappings or []
    }
    gold_grounded_attrs |= {
        em.source_entity.strip().lower() for em in gold_vocab.entity_mappings or []
    }

    hallucinated_count = 0
    for p_term in proposed_vocab.terms or []:
        p_sources = {_bare(s.strip().lower()) for s in (p_term.source_attributes or [])}
        if p_sources and p_sources.isdisjoint(gold_grounded_attrs):
            hallucinated_count += 1

    num_prop_terms = len(proposed_vocab.terms or [])
    hallucination_rate = hallucinated_count / num_prop_terms if num_prop_terms > 0 else 0.0

    # 6. Provenance completeness
    prop_prov_count = sum(1 for t in proposed_vocab.terms or [] if len(t.source_attributes or []) > 0)
    prop_prov_completeness = prop_prov_count / num_prop_terms if num_prop_terms > 0 else 1.0
    
    gold_prov_count = sum(1 for t in gold_vocab.terms or [] if len(t.source_attributes or []) > 0)
    gold_prov_completeness = gold_prov_count / num_gold_concepts if num_gold_concepts > 0 else 1.0

    # 7. Refinement Accuracy  (PDF §8.5)
    # "correct refinements / refined ground-truth mappings"
    # A "refined" ground-truth element is one where the gold canonical label differs
    # from the source name (rename, reification, role_disambiguation, etc.).
    # The proposed vocab is correct on that element if its canonical label matches gold.
    #
    # We work directly from the gold_vocab already passed in (already converted via
    # convert_expert_ground_truth), so no file I/O or heuristic ERD detection is needed.
    #
    # Default 0.0 — if there are no refined elements, we report None (N/A).

    refined_total = 0
    refined_correct = 0

    prop_ent_canonical = {
        em.source_entity.strip().lower(): _norm(em.canonical_entity)
        for em in proposed_vocab.entity_mappings or []
    }
    # Key by the FULL qualified attribute ("Entity.col") so the same column name
    # in different entities (e.g. "name" in Customer/Airline/Airport) does not
    # collide. A set per key tolerates rare duplicate sources.
    prop_attr_canonical: dict[str, set] = defaultdict(set)
    for m in proposed_vocab.mappings or []:
        key = m.attribute.strip().lower()
        prop_attr_canonical[key].add(_norm(m.canonical_term))

    # Check entity-level refinements
    for em in gold_vocab.entity_mappings or []:
        src = em.source_entity.strip().lower()
        gold_canonical = _norm(em.canonical_entity)
        if _norm(src) != gold_canonical:
            refined_total += 1
            if prop_ent_canonical.get(src) == gold_canonical:
                refined_correct += 1

    # Check attribute-level refinements
    for m in gold_vocab.mappings or []:
        attr = m.attribute.strip().lower()
        attr_bare = attr.split(".")[-1] if "." in attr else attr
        gold_canonical = _norm(m.canonical_term)
        if _norm(attr_bare) != gold_canonical:
            refined_total += 1
            proposed_canonicals = prop_attr_canonical.get(attr, set())
            # The expert may list semantically-equivalent alternatives (e.g. a
            # relation verb is a free lexical choice). Accept canonical OR any alias.
            gold_accepted = {_norm(a) for a in (getattr(m, "accepted_aliases", None) or [])}
            if _relation_match(gold_canonical, proposed_canonicals) or any(
                _relation_match(a, proposed_canonicals) for a in gold_accepted
            ):
                refined_correct += 1

    refinement_accuracy = (refined_correct / refined_total) if refined_total > 0 else None

    # 8. Structural correctness (separates TYPE/CATEGORY from naming, which Ref.Acc
    #    measures). A flat vocabulary carries no type/category info, so it implicitly
    #    commits everything to property/kind -> default missing values accordingly.
    prop_type_by_attr = {
        m.attribute.strip().lower(): getattr(m, "mapping_type", None) or "property"
        for m in proposed_vocab.mappings or []
    }
    type_total = 0
    type_correct = 0
    for m in gold_vocab.mappings or []:
        gold_type = getattr(m, "mapping_type", None)
        if gold_type is None:
            continue
        type_total += 1
        if prop_type_by_attr.get(m.attribute.strip().lower(), "property") == gold_type:
            type_correct += 1
    type_accuracy = (type_correct / type_total) if type_total > 0 else None

    prop_cat_by_entity = {
        em.source_entity.strip().lower(): getattr(em, "ontological_category", None) or "kind"
        for em in proposed_vocab.entity_mappings or []
    }
    cat_total = 0
    cat_correct = 0
    for em in gold_vocab.entity_mappings or []:
        gold_cat = getattr(em, "ontological_category", None)
        if gold_cat is None:
            continue
        cat_total += 1
        if prop_cat_by_entity.get(em.source_entity.strip().lower(), "kind") == gold_cat:
            cat_correct += 1
    category_accuracy = (cat_correct / cat_total) if cat_total > 0 else None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "curation_effort": round(curation_effort, 4),
        "over_merge_rate": round(over_merge_rate, 4),
        "under_merge_rate": round(under_merge_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "provenance_completeness": round(gold_prov_completeness, 4),
        "proposed_provenance_completeness": round(prop_prov_completeness, 4),
        "refinement_accuracy": round(refinement_accuracy, 4) if refinement_accuracy is not None else None,
        "type_accuracy": round(type_accuracy, 4) if type_accuracy is not None else None,
        "category_accuracy": round(category_accuracy, 4) if category_accuracy is not None else None,
    }

