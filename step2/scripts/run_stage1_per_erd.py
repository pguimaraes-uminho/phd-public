import json
import sys
import os
import time
import argparse
from collections import defaultdict

# Insert backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app.models.erd import ERDModel
    from app.services.vocab_builder import (
        build_vocab,
        calculate_evaluation_metrics,
        convert_expert_ground_truth,
        _rule_based_ontology,
    )
    from app.core.config import settings
    from app.models.vocab import Vocabulary
    from app.db.sample_loader import load_csv_text, CSV_A, CSV_B
except ImportError:
    print("Error: could not import dependencies. From the package root run:")
    print("  pip install -r scripts/requirements.txt")
    sys.exit(1)


def log(msg):
    print(msg)


def _aggregate_runs(run_metrics: list) -> tuple:
    """Mean and population std-dev per numeric metric across repeated runs."""
    keys = set()
    for m in run_metrics:
        keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    mean, std = {}, {}
    for k in keys:
        vals = [m[k] for m in run_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            mean[k], std[k] = None, None
            continue
        mu = sum(vals) / len(vals)
        sigma = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
        mean[k], std[k] = round(mu, 4), round(sigma, 4)
    return mean, std
    sys.stdout.flush()


def generate_detailed_evaluation_report(proposed_vocab: dict | Vocabulary, gold_vocab: dict | Vocabulary) -> dict:
    """
    Generates a detailed, itemized report comparing the proposed alignment vocabulary
    against the expert ground truth (Gold Vocabulary). This isolates TPs, FPs, FNs 
    and lists correct/incorrect canonical label alignments.
    """
    # Ensure they are Vocabulary objects
    if isinstance(proposed_vocab, dict):
        if "concepts" in proposed_vocab and "mappings" not in proposed_vocab:
            proposed_vocab = convert_expert_ground_truth(proposed_vocab, qualified=True)
        proposed_vocab = Vocabulary.model_validate(proposed_vocab)
    if isinstance(gold_vocab, dict):
        if "concepts" in gold_vocab and "mappings" not in gold_vocab:
            gold_vocab = convert_expert_ground_truth(gold_vocab, qualified=True)
        gold_vocab = Vocabulary.model_validate(gold_vocab)

    def _bare(name: str) -> str:
        return name.split(".")[-1] if "." in name else name

    # 1. Prepare raw maps
    proposed_ent_map = {em.source_entity.strip().lower(): em.source_entity for em in proposed_vocab.entity_mappings or []}
    gold_ent_map = {em.source_entity.strip().lower(): em.source_entity for em in gold_vocab.entity_mappings or []}
    
    proposed_attr_map = {m.attribute.strip().lower(): m.attribute for m in proposed_vocab.mappings or []}
    gold_attr_map = {m.attribute.strip().lower(): m.attribute for m in gold_vocab.mappings or []}

    # Convert to lowercase sets for matching keys
    proposed_ents_set = set(proposed_ent_map.keys())
    gold_ents_set = set(gold_ent_map.keys())
    
    proposed_attrs_bare_map = {_bare(k): v for k, v in proposed_attr_map.items()}
    gold_attrs_bare_map = {_bare(k): v for k, v in gold_attr_map.items()}
    
    proposed_attrs_set = set(proposed_attrs_bare_map.keys())
    gold_attrs_set = set(gold_attrs_bare_map.keys())

    # TP (True Positives) - source element is mapped in both
    tp_entities = [proposed_ent_map[e] for e in (proposed_ents_set & gold_ents_set)]
    tp_attributes = [proposed_attrs_bare_map[a] for a in (proposed_attrs_set & gold_attrs_set)]

    # FP (False Positives) - source element is mapped in proposed but not in gold
    fp_entities = [proposed_ent_map[e] for e in (proposed_ents_set - gold_ents_set)]
    fp_attributes = [proposed_attrs_bare_map[a] for a in (proposed_attrs_set - gold_attrs_set)]

    # FN (False Negatives) - source element is mapped in gold but not in proposed
    fn_entities = [gold_ent_map[e] for e in (gold_ents_set - proposed_ents_set)]
    fn_attributes = [gold_attrs_bare_map[a] for a in (gold_attrs_set - proposed_attrs_set)]

    # 2. Refinements (Canonical name matching)
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[\s\-_]+", "", s.strip().lower())

    refinement_matches = []
    refinement_mismatches = []

    # Check Entity Refinement Alignments
    prop_ent_canonical = {em.source_entity.strip().lower(): em.canonical_entity for em in proposed_vocab.entity_mappings or []}
    for em in gold_vocab.entity_mappings or []:
        src = em.source_entity.strip().lower()
        if src in prop_ent_canonical:
            gold_val = em.canonical_entity
            prop_val = prop_ent_canonical[src]
            is_correct = _norm(gold_val) == _norm(prop_val)
            ref_info = {
                "source_entity": em.source_entity,
                "gold_canonical": gold_val,
                "proposed_canonical": prop_val
            }
            if is_correct:
                refinement_matches.append(ref_info)
            else:
                refinement_mismatches.append(ref_info)

    # Check Attribute Refinement Alignments — keyed by the FULL qualified
    # attribute ("Entity.col") so same-named columns in different entities are
    # not conflated. Mirror the metric's match logic (is/has/get tolerance).
    def _strip_prefix(s):
        for p in ("is", "has", "get"):
            if s.startswith(p) and len(s) > len(p):
                return s[len(p):]
        return s

    def _rel_match(gold, proposed_set):
        if gold in proposed_set:
            return True
        return any(gold in p or _strip_prefix(p) == gold for p in proposed_set)

    from collections import defaultdict as _dd
    prop_attr_canonical = _dd(set)
    prop_attr_display = _dd(list)
    for m in proposed_vocab.mappings or []:
        key = m.attribute.strip().lower()
        prop_attr_canonical[key].add(_norm(m.canonical_term))
        prop_attr_display[key].append(m.canonical_term)
    for m in gold_vocab.mappings or []:
        key = m.attribute.strip().lower()
        if key in prop_attr_canonical:
            gold_val = m.canonical_term
            gold_accepted = {_norm(a) for a in (getattr(m, "accepted_aliases", None) or [])}
            is_correct = _rel_match(_norm(gold_val), prop_attr_canonical[key]) or any(
                _rel_match(a, prop_attr_canonical[key]) for a in gold_accepted
            )
            ref_info = {
                "source_attribute": m.attribute,
                "gold_canonical": gold_val,
                "proposed_canonical": ", ".join(prop_attr_display[key])
            }
            if is_correct:
                refinement_matches.append(ref_info)
            else:
                refinement_mismatches.append(ref_info)

    return {
        "true_positives": {
            "entities": sorted(tp_entities),
            "attributes": sorted(tp_attributes),
            "count": len(tp_entities) + len(tp_attributes)
        },
        "false_positives": {
            "entities": sorted(fp_entities),
            "attributes": sorted(fp_attributes),
            "count": len(fp_entities) + len(fp_attributes)
        },
        "false_negatives": {
            "entities": sorted(fn_entities),
            "attributes": sorted(fn_attributes),
            "count": len(fn_entities) + len(fn_attributes)
        },
        "refinements": {
            "correct_matches": refinement_matches,
            "incorrect_matches": refinement_mismatches,
            "correct_count": len(refinement_matches),
            "incorrect_count": len(refinement_mismatches)
        }
    }


def run_evaluation():
    parser = argparse.ArgumentParser(description="Ontology Mapping Scenarios Evaluation Runner")
    parser.add_argument(
        "--targets", 
        nargs="+", 
        choices=["ERD-A", "ERD-B"], 
        default=["ERD-A", "ERD-B"],
        help="Select specific ERD targets to evaluate (default: both)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["0", "1", "2", "3", "4", "5", "6",
                 "rule_based", "llm_erd", "llm_erd_data", "llm_erd_data_expert_brief",
                 "hybrid_erd", "hybrid_erd_data", "hybrid_erd_data_expert_brief"],
        help="Select specific scenarios to run (e.g., --scenarios 0 1 2 3 4 5 6). "
             "4/5/6 are the hybrid arms (rules structure + LLM semantics)."
    )
    parser.add_argument(
        "--save-vocabs", 
        action="store_true", 
        help="Save the fully generated vocabulary JSON for each scenario to disk"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override default Gemini model configured in .env (e.g. gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=None,
        help="Temperature(s) for LLM runs (default: settings.gemini_temps). "
             "Fixed temp: --temps 0.2 ; sweep: --temps 0.2 0.6 0.9"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per temperature, for mean+/-std (e.g. --temps 0.2 --repeats 5)"
    )

    args = parser.parse_args()

    # Override model if requested
    if args.model:
        settings.gemini_model = args.model

    # Effective temperatures for LLM runs: base temps (CLI --temps or config),
    # each repeated --repeats times -> one run per entry (for mean+/-std).
    base_temps = args.temps if args.temps is not None else settings.temperature_list
    llm_temps = [t for t in base_temps for _ in range(max(1, args.repeats))]

    log("======================================================================")
    log("            Ontology Mapping Scenarios Evaluation Runner")
    log("======================================================================")
    log(f"Active Gemini Model: {settings.gemini_model}")
    log(f"API Key Configured:  {bool(settings.gemini_api_key)}")
    log(f"Targets to Run:      {', '.join(args.targets)}")
    log(f"LLM runs/scenario:   {len(llm_temps)}  (temps={base_temps} x repeats={max(1, args.repeats)})")
    log(f"Save Raw Vocabs:     {args.save_vocabs}")
    log("======================================================================")

    # 1. Paths — read inputs from the top-level folders; each run is written to
    #    its own timestamped folder under results/stage1_per_erd/.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> repo root
    erd_dir = os.path.join(root_dir, "erd")
    gt_dir = os.path.join(root_dir, "ground-truth")
    db_dir = os.path.join(root_dir, "prompts")  # refinement briefs live here
    run_dir = os.path.join(root_dir, "results", "stage1_per_erd", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "evaluation_results.json")
    log(f"Output folder:       results/stage1_per_erd/{os.path.basename(run_dir)}")

    # 2. Load ERDs and Ground Truth alignments
    try:
        with open(os.path.join(erd_dir, "erd_a.json"), "r", encoding="utf-8") as f:
            erd_a_data = json.load(f)
        with open(os.path.join(erd_dir, "erd_b.json"), "r", encoding="utf-8") as f:
            erd_b_data = json.load(f)

        with open(os.path.join(gt_dir, "ground_truth_mapping_erd_a.json"), "r", encoding="utf-8") as f:
            gold_a_raw = json.load(f)
        with open(os.path.join(gt_dir, "ground_truth_mapping_erd_b.json"), "r", encoding="utf-8") as f:
            gold_b_raw = json.load(f)
    except FileNotFoundError as e:
        log(f"CRITICAL ERROR: Required input file not found: {e}")
        sys.exit(1)

    erd_a = ERDModel.model_validate(erd_a_data)
    erd_b = ERDModel.model_validate(erd_b_data)
    
    gold_a_vocab = convert_expert_ground_truth(gold_a_raw, qualified=True)
    gold_b_vocab = convert_expert_ground_truth(gold_b_raw, qualified=True)

    # 3. Sample data — the CSVs in app/db are the single source of truth.
    #    LLM scenarios get the raw CSV text (token-efficient).
    erd_a_csv = load_csv_text(CSV_A)
    erd_b_csv = load_csv_text(CSV_B)

    # Expert domain briefs for Scenario 3 (LLM-3), one per ERD.
    def _load_brief(fname):
        with open(os.path.join(db_dir, fname), "r", encoding="utf-8") as f:
            items = json.load(f)
        return "\n".join(f"- {s}" for s in items)

    erd_a_brief = _load_brief("refinement_brief_erd_a.json")
    erd_b_brief = _load_brief("refinement_brief_erd_b.json")

    # Deterministic structural scaffold per ERD, reused by the hybrid arms
    # (Scenarios 4/5/6). This is the SAME structure the rule-based baseline
    # (Scenario 0) is scored on — the LLM keeps it and only renames.
    scaffold_a = _rule_based_ontology(erd_a)
    scaffold_b = _rule_based_ontology(erd_b)

    all_targets = [
        {"name": "ERD-A", "key": "erd_a", "erd": erd_a, "gold_vocab": gold_a_vocab,
         "sample_csv": erd_a_csv, "expert_brief": erd_a_brief, "scaffold": scaffold_a},
        {"name": "ERD-B", "key": "erd_b", "erd": erd_b, "gold_vocab": gold_b_vocab,
         "sample_csv": erd_b_csv, "expert_brief": erd_b_brief, "scaffold": scaffold_b}
    ]

    # Filter targets based on arguments
    targets = [t for t in all_targets if t["name"] in args.targets]

    all_scenarios = [
        {"id": "rule_based", "name": "Scenario 0: Rule-based ERD->ontology (deterministic)", "method": "rule_based", "include_expert": False, "use_samples": False, "use_scaffold": False, "num_id": "0"},
        {"id": "llm_erd", "name": "Scenario 1: LLM-1 (ERD only)", "method": "llm", "include_expert": False, "use_samples": False, "use_scaffold": False, "num_id": "1"},
        {"id": "llm_erd_data", "name": "Scenario 2: LLM-2 (ERD + Data)", "method": "llm", "include_expert": False, "use_samples": True, "use_scaffold": False, "num_id": "2"},
        {"id": "llm_erd_data_expert_brief", "name": "Scenario 3: LLM-3 (ERD + Data + Expert Brief)", "method": "llm", "include_expert": True, "use_samples": True, "use_scaffold": False, "num_id": "3"},
        # Hybrid arms: rules recover the structure, the LLM only refines the
        # semantics on top of it. Each mirrors the LLM arm with the same context
        # (ERD / +Data / +Brief) PLUS the deterministic structural scaffold.
        {"id": "hybrid_erd", "name": "Scenario 4: Hybrid (rules structure + LLM semantics, ERD)", "method": "llm", "include_expert": False, "use_samples": False, "use_scaffold": True, "num_id": "4"},
        {"id": "hybrid_erd_data", "name": "Scenario 5: Hybrid (rules structure + LLM semantics, ERD + Data)", "method": "llm", "include_expert": False, "use_samples": True, "use_scaffold": True, "num_id": "5"},
        {"id": "hybrid_erd_data_expert_brief", "name": "Scenario 6: Hybrid (rules structure + LLM semantics, ERD + Data + Expert Brief)", "method": "llm", "include_expert": True, "use_samples": True, "use_scaffold": True, "num_id": "6"}
    ]

    # Filter scenarios if requested
    if args.scenarios:
        scenario_filters = set(args.scenarios)
        scenarios = []
        for s in all_scenarios:
            if s["id"] in scenario_filters or s["num_id"] in scenario_filters:
                scenarios.append(s)
    else:
        scenarios = all_scenarios

    # Load existing metrics from disk if available to progressive merging
    results = {"erd_a": {}, "erd_b": {}}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if isinstance(old_data, dict):
                    if "erd_a" in old_data:
                        results["erd_a"].update(old_data["erd_a"])
                    if "erd_b" in old_data:
                        results["erd_b"].update(old_data["erd_b"])
        except Exception:
            pass

    for target in targets:
        log(f"\nEvaluating target: {target['name']}")
        log("-" * 50)
        
        for sc in scenarios:
            log(f"Running {sc['name']}...")
            # non-LLM baseline consumes structured per-entity tables; LLM
            # scenarios get the raw CSV text (more faithful and token-cheaper).
            active_csv = target["sample_csv"] if sc["use_samples"] else None
            # Scenario 3 (LLM-3) adds the expert domain brief.
            active_brief = target["expert_brief"] if sc["include_expert"] else None
            # Hybrid scenarios (4/5/6) add the deterministic structural scaffold.
            active_scaffold = target["scaffold"] if sc.get("use_scaffold") else None

            # LLM scenarios run once per entry in llm_temps (mean+/-std);
            # the rule-based baseline is deterministic (one run).
            temps = llm_temps if sc["method"] == "llm" else [0.0]

            runs = []          # [{temperature, time_seconds, metrics}]
            first_audit = None
            last_error = None
            t_start = time.time()
            for ti, temp in enumerate(temps):
                tr0 = time.time()
                try:
                    vocab = build_vocab(
                        target["erd"],
                        refine_prompt=None,
                        previous_vocab=None,
                        existing_vocab=None,
                        sample_data=None,
                        prompt_history=None,
                        method=sc["method"],
                        include_ontology=False,
                        is_compare=True,
                        sample_csv_text=active_csv,
                        expert_brief_text=active_brief,
                        structural_scaffold=active_scaffold,
                        temperature=temp,
                    )
                    rdt = time.time() - tr0
                    metrics = calculate_evaluation_metrics(vocab, target["gold_vocab"])
                    if first_audit is None:
                        first_audit = generate_detailed_evaluation_report(vocab, target["gold_vocab"])
                    runs.append({"temperature": temp, "time_seconds": round(rdt, 2), "metrics": metrics})
                    ra = metrics.get("refinement_accuracy")
                    ra_str = f"{ra:.4f}" if ra is not None else "N/A"
                    log(f"  [run t={temp}] {rdt:.1f}s | F1: {metrics.get('f1'):.4f} | "
                        f"Ref.Acc: {ra_str}")
                except Exception as e:
                    last_error = str(e)
                    log(f"  [run t={temp}] FAILED | {last_error}")
                if sc["method"] == "llm" and ti < len(temps) - 1:
                    time.sleep(10)

            dt = time.time() - t_start

            if runs:
                mean, std = _aggregate_runs([r["metrics"] for r in runs])
                results[target["key"]][sc["id"]] = {
                    "status": "success",
                    "time_seconds": round(dt, 2),
                    "n_runs": len(runs),
                    "temperatures": [r["temperature"] for r in runs],
                    "metrics": mean,           # mean is the headline metric set
                    "metrics_mean": mean,
                    "metrics_std": std,
                    "runs": runs,
                }
                ra_m, ra_s = mean.get("refinement_accuracy"), std.get("refinement_accuracy")
                ra_str = f"{ra_m:.4f}+/-{ra_s:.4f}" if ra_m is not None else "N/A"
                log(f"  [Success] {len(runs)} runs | {dt:.1f}s | "
                    f"F1: {mean.get('f1'):.4f}+/-{std.get('f1'):.4f} | "
                    f"Effort: {mean.get('curation_effort'):.2f} | Ref.Acc: {ra_str}")

                report_file_name = f"evaluation_report_{target['key']}_{sc['id']}.json"
                report_path = os.path.join(run_dir, report_file_name)
                with open(report_path, "w", encoding="utf-8") as rf:
                    json.dump({
                        "scenario_name": sc["name"],
                        "target_model": target["name"],
                        "n_runs": len(runs),
                        "metrics_mean": mean,
                        "metrics_std": std,
                        "runs": runs,
                        "matching_audit": first_audit,
                    }, rf, indent=2)
                log(f"  [Saved Report] -> {report_file_name}")
            else:
                results[target["key"]][sc["id"]] = {
                    "status": "failed",
                    "time_seconds": round(dt, 2),
                    "error": last_error or "all runs failed",
                }
                log(f"  [Failed] {dt:.1f}s | Error: {last_error}")

            # Save metrics progressively to disk
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    log("\n======================================================================")
    log("                         EVALUATION SUMMARY")
    log("======================================================================")
    for target in targets:
        log(f"\nTarget: {target['name']}")
        log(f" {'Scenario':<22} | {'n':<3} | {'Type':<7} | {'Categ':<7} | {'Ref.Acc (mean+/-std)':<20} | {'Effort':<6}")
        log("-" * 92)
        for sc in scenarios:
            res = results[target["key"]].get(sc["id"], {})
            status = res.get("status", "pending")
            if status == "success":
                m = res.get("metrics_mean", res.get("metrics", {}))
                s = res.get("metrics_std", {})
                n = res.get("n_runs", 1)

                def fmt(key, pct=False):
                    v = m.get(key)
                    if v is None:
                        return "N/A"
                    return f"{v*100:.0f}%" if pct else f"{v:.4f}"

                ra = m.get("refinement_accuracy")
                ra_str = f"{ra:.4f}+/-{s.get('refinement_accuracy', 0) or 0:.4f}" if ra is not None else "N/A"
                log(f" {sc['id']:<22} | {n:<3} | {fmt('type_accuracy', True):<7} | {fmt('category_accuracy', True):<7} | {ra_str:<20} | {m.get('curation_effort'):.2f}")
            else:
                log(f" {sc['id']:<22} | {'-':<3} | {'N/A':<7} | {'N/A':<7} | {'N/A':<20} | {'N/A':<6}")
    
    log(f"\nDetailed metrics report successfully written to:\n  {output_path}")
    log("======================================================================")


if __name__ == "__main__":
    run_evaluation()
