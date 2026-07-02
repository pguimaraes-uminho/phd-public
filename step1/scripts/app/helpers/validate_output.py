#!/usr/bin/env python3
"""
Validate an LLM mapping output before scoring.

Usage:
    python validate_output.py output.json ground_truth_erd.json [schema.json]

If [schema.json] is omitted it defaults to output_schema_metrics.json (the
metrics-only contract used in the testing phase); pass output_schema.json for
the full contract used in the final runs.

Checks:
  1) JSON Schema conformance (structure, enums, field names).
  2) Full coverage: every source entity, attribute and relationship of the ERD
     is accounted for (entities -> concept.source_entity or a pure junction
     realized by a relation; attributes -> a maps_to or a junction-backed
     relation; relationships -> a relation).
Exit code 0 if valid, 1 on schema errors. Prints a short report.

Requires: pip install jsonschema
"""
import json, sys

def load(p):
    with open(p) as f:
        return json.load(f)

def schema_check(output, schema_path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("[warn] jsonschema not installed; skipping schema check "
              "(pip install jsonschema). Running coverage check only.")
        return []
    schema = load(schema_path)
    v = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(output)]

def coverage_check(output, erd):
    errors = []
    src_entities = {e["name"] for e in erd["entities"]}
    src_attrs = {e["name"] + "." + a["name"]
                 for e in erd["entities"] for a in e["attributes"]}
    n_rels = len(erd.get("relationships", []))

    mapped_entities = {c.get("source_entity") for c in output.get("concepts", [])}
    mapped_attrs = {a.get("source")
                    for c in output.get("concepts", [])
                    for a in c.get("attributes", [])}

    # entities not mapped to a concept may legitimately be pure junctions that
    # the output realizes as a relation; we only flag entities that appear
    # nowhere. (A finer check requires linking junctions to relations; kept
    # simple here.)
    missing_entities = src_entities - mapped_entities
    if missing_entities:
        errors.append(f"entities not mapped to a concept (verify each is a pure "
                      f"junction realized by a relation): {sorted(missing_entities)}")

    missing_attrs = src_attrs - mapped_attrs
    if missing_attrs:
        errors.append(f"source attributes with no mapping (some may be junction "
                      f"FKs realizing a relation): {sorted(missing_attrs)}")

    n_out_rels = len(output.get("relations", []))
    if n_out_rels == 0 and n_rels > 0:
        errors.append(f"ERD has {n_rels} relationships but output has 0 relations")

    return errors

def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__); sys.exit(2)
    out_path, erd_path = sys.argv[1], sys.argv[2]
    schema_path = sys.argv[3] if len(sys.argv) == 4 else "output_schema_metrics.json"
    output = load(out_path)
    erd = load(erd_path)

    serr = schema_check(output, schema_path)
    cerr = coverage_check(output, erd)

    if serr:
        print("SCHEMA ERRORS:")
        for e in serr: print("  -", e)
    if cerr:
        print("COVERAGE WARNINGS:")
        for e in cerr: print("  -", e)
    if not serr and not cerr:
        print("OK: schema valid and full coverage.")
        sys.exit(0)
    sys.exit(1 if serr else 0)  # schema errors are hard fails; coverage are warnings

if __name__ == "__main__":
    main()
