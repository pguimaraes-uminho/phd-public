"""
run_cross_erd_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standalone runner for the Cross-ERD integration task.

Usage examples:
    # All 3 scenarios, using the best per-ERD result from evaluation_results.json
    cd backend && .venv/bin/python run_cross_erd_evaluation.py

    # Override source scenario and model
    cd backend && .venv/bin/python run_cross_erd_evaluation.py \
        --source-scenario llm_erd_data_expert_brief \
        --scenarios cross_rule_based cross_llm \
        --model gemini-2.5-flash
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from app.services.cross_erd_builder import merge_rule_based, merge_llm, merge_llm_brief
    from app.services.integration_evaluator import calculate_integration_evaluation_metrics
    from app.core.config import settings
    from app.models.erd import ERDModel
    from app.db.sample_loader import csv_to_tables, CSV_A, CSV_B, COLMAP_A, COLMAP_B
except ImportError as e:
    print(f"Error: could not import dependencies: {e}")
    print("From the package root run:  pip install -r scripts/requirements.txt")
    sys.exit(1)


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def _aggregate_runs(run_metrics: list) -> tuple[dict, dict]:
    keys = set()
    for m in run_metrics:
        keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    mean, std = {}, {}
    for k in keys:
        vals = [m[k] for m in run_metrics if isinstance(m.get(k), (int, float))]
        if not vals:
            mean[k] = std[k] = None
            continue
        mu = sum(vals) / len(vals)
        sigma = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
        mean[k], std[k] = round(mu, 4), round(sigma, 4)
    return mean, std


def _load_model(path: str) -> dict:
    """Load a raw ground-truth model JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cross_erd_evaluation() -> None:
    parser = argparse.ArgumentParser(description="Cross-ERD Integration Evaluation Runner")
    parser.add_argument(
        "--source-scenario",
        type=str,
        default=None,
        help="Which per-ERD scenario's output to use as input "
             "(default: highest refinement_accuracy in evaluation_results.json, "
             "or gold vocab if no results available)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["cross_rule_based", "cross_llm", "cross_llm_brief"],
        default=["cross_rule_based", "cross_llm", "cross_llm_brief"],
        help="Cross-ERD scenarios to run (default: all three)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Gemini model (e.g. gemini-2.5-flash)",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=None,
        help="Temperature(s) for LLM runs (default: settings.gemini_temps)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per temperature for mean+/-std",
    )

    args = parser.parse_args()

    if args.model:
        settings.gemini_model = args.model

    base_temps = args.temps if args.temps is not None else settings.temperature_list
    llm_temps = [t for t in base_temps for _ in range(max(1, args.repeats))]

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> repo root
    erd_dir  = os.path.join(root_dir, "erd")
    gt_dir   = os.path.join(root_dir, "ground-truth")
    prompts_dir = os.path.join(root_dir, "prompts")
    run_dir  = os.path.join(root_dir, "results", "stage2_cross_erd", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "cross_evaluation_results.json")
    log(f"Output folder:       results/stage2_cross_erd/{os.path.basename(run_dir)}")

    log("======================================================================")
    log("              Cross-ERD Integration Evaluation Runner")
    log("======================================================================")
    log(f"Active Gemini Model: {settings.gemini_model}")
    log(f"Scenarios:           {', '.join(args.scenarios)}")
    log(f"LLM runs/scenario:   {len(llm_temps)}  (temps={base_temps} x repeats={max(1, args.repeats)})")
    log("======================================================================")

    # ------------------------------------------------------------------
    # Load the raw ground-truth aligned models for each ERD.
    # These are the two inputs for the cross-ERD integration task.
    # ------------------------------------------------------------------
    try:
        model_a = _load_model(os.path.join(gt_dir, "ground_truth_mapping_erd_a.json"))
        model_b = _load_model(os.path.join(gt_dir, "ground_truth_mapping_erd_b.json"))
    except FileNotFoundError as e:
        log(f"CRITICAL: Missing ground truth model file: {e}")
        sys.exit(1)

    log(f"Input MODEL-A ({model_a.get('source_erd','?')}): "
        f"{len(model_a.get('concepts', []))} concepts")
    log(f"Input MODEL-B ({model_b.get('source_erd','?')}): "
        f"{len(model_b.get('concepts', []))} concepts")
    log("")

    # Instance value sets per source_entity, for the matcher's instance-overlap
    # signal (entity name -> set of all cell values in that entity's CSV rows).
    def _instance_values(csv_path, colmap, erd_path):
        erd = ERDModel.model_validate(_load_model(erd_path))
        out: dict[str, set] = {}
        for t in csv_to_tables(csv_path, colmap, erd):
            out[t["table"].lower()] = {
                str(v).strip().lower() for r in t["rows"] for v in r.values() if str(v).strip()
            }
        return out

    try:
        instances_a = _instance_values(CSV_A, COLMAP_A, os.path.join(erd_dir, "erd_a.json"))
        instances_b = _instance_values(CSV_B, COLMAP_B, os.path.join(erd_dir, "erd_b.json"))
    except Exception as e:
        log(f"[warn] instance signal unavailable ({e}); matcher will use lexical+structural only")
        instances_a, instances_b = {}, {}

    # Expert domain brief for cross_llm_brief
    cross_brief_path = os.path.join(prompts_dir, "integration_brief.json")
    try:
        with open(cross_brief_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        cross_expert_brief = "\n".join(f"- {s}" for s in items)
    except FileNotFoundError:
        cross_expert_brief = ""

    all_scenarios = [
        {
            "id":     "cross_rule_based",
            "name":   "Cross-ERD Scenario 0: Deterministic baseline",
            "method": "rule_based",
        },
        {
            "id":     "cross_llm",
            "name":   "Cross-ERD Scenario 1: LLM (2 vocab inputs)",
            "method": "llm",
        },
        {
            "id":     "cross_llm_brief",
            "name":   "Cross-ERD Scenario 2: LLM (2 vocab inputs + expert brief)",
            "method": "llm_brief",
        },
    ]

    scenarios = [s for s in all_scenarios if s["id"] in args.scenarios]

    # Load existing results for progressive merge
    results: dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass

    for sc in scenarios:
        log(f"Running {sc['name']}...")

        temps = llm_temps if sc["method"] in ("llm", "llm_brief") else [0.0]
        runs: list[dict] = []
        last_error: str | None = None
        t_start = time.time()

        for ti, temp in enumerate(temps):
            tr0 = time.time()
            try:
                if sc["method"] == "rule_based":
                    proposed = merge_rule_based(model_a, model_b, instances_a, instances_b)
                elif sc["method"] == "llm":
                    proposed = merge_llm(model_a, model_b, temperature=temp)
                else:  # llm_brief
                    proposed = merge_llm_brief(model_a, model_b, cross_expert_brief, temperature=temp)

                rdt = time.time() - tr0
                metrics = calculate_integration_evaluation_metrics(proposed)
                runs.append({
                    "temperature": temp,
                    "time_seconds": round(rdt, 2),
                    "metrics": metrics,
                    "proposed_plan": proposed,
                })
                log(f"  [run t={temp}] {rdt:.1f}s | "
                    f"F1: {metrics.get('match_f1', 0):.4f} | "
                    f"Conflict: {metrics.get('conflict_resolution_accuracy', 0):.4f} | "
                    f"UniqueA: {metrics.get('unique_a_coverage', 0):.4f} | "
                    f"UniqueB: {metrics.get('unique_b_coverage', 0):.4f}")
            except Exception as e:
                last_error = str(e)
                log(f"  [run t={temp}] FAILED | {last_error}")

            if sc["method"] in ("llm", "llm_brief") and ti < len(temps) - 1:
                time.sleep(10)

        dt = time.time() - t_start

        if runs:
            mean, std = _aggregate_runs([r["metrics"] for r in runs])
            results[sc["id"]] = {
                "status": "success",
                "time_seconds": round(dt, 2),
                "n_runs": len(runs),
                "temperatures": [r["temperature"] for r in runs],
                "metrics": mean,
                "metrics_mean": mean,
                "metrics_std": std,
                "runs": runs,
            }
            log(f"  [Success] {len(runs)} runs | {dt:.1f}s | "
                f"F1: {mean.get('match_f1'):.4f}+/-{std.get('match_f1', 0) or 0:.4f} | "
                f"Conflict: {mean.get('conflict_resolution_accuracy'):.4f}")
        else:
            results[sc["id"]] = {
                "status": "failed",
                "time_seconds": round(dt, 2),
                "error": last_error or "all runs failed",
            }
            log(f"  [Failed] {dt:.1f}s | Error: {last_error}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    log("\n======================================================================")
    log("                         CROSS-ERD SUMMARY")
    log("======================================================================")
    log(f" {'Scenario':<22} | {'F1':<6} | {'Conflict%':<10} | {'Cov-A':<6} | {'Cov-B':<6}")
    log("-" * 70)
    for sc in scenarios:
        res = results.get(sc["id"], {})
        if res.get("status") == "success":
            m = res.get("metrics_mean", res.get("metrics", {}))
            log(f" {sc['id']:<22} | {m.get('match_f1', 0):.4f} | "
                f"{m.get('conflict_resolution_accuracy', 0):.4f}      | "
                f"{m.get('unique_a_coverage', 0):.4f} | "
                f"{m.get('unique_b_coverage', 0):.4f}")
        else:
            log(f" {sc['id']:<22} | {'N/A':<6} | {'N/A':<10} | {'N/A':<6} | {'N/A':<6}")

    log(f"\nResults written to: {output_path}")
    log("======================================================================")


if __name__ == "__main__":
    run_cross_erd_evaluation()
