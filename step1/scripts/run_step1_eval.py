#!/usr/bin/env python3
"""Step-1 replication: CSV → ERD reconstruction, scored against the expert ERD.

Deterministic baseline vs LLM (with PROMPT information levels, simplest → most
complete) vs hybrid. Unlike Step 2, the LLM is NOT given the ERD (that would be the
answer) — the scenarios vary how much information about the CSV the prompt carries.

Scenarios
  S0 baseline               deterministic (keys, FKs, types, denormalized split) — no LLM
  S1 llm_data_noheaders     LLM, data values WITHOUT column names
  S2 llm_data_headers       LLM, columns + sample rows (headers)
  S3 llm_data_headers_brief LLM, + expert domain brief
  S4 hybrid (post-hoc)      merge the baseline AFTER generation: hybrid_from_s2, hybrid_from_s3
  S5 grounded               LLM with the baseline injected INTO the prompt (llm_grounded);
                            plus that grounded output hardened post-hoc (hybrid_from_grounded)

Each LLM scenario is repeated (mean ± std). The baseline is deterministic (1 run).
Run offline first to sanity-check (only S0 needs no key):
  GEMINI_MOCK=true python3 run_step1_eval.py --scenarios 0
Full run (your live key; costs tokens):
  python3 run_step1_eval.py --repeats 3
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from app.models.erd import ERDModel
    from app.services.erd_baseline import build_baseline_erd
    from app.services.erd_hybrid import build_hybrid_erd
    from app.services.llm import LLMClient
    from app.core.config import settings
    from erd_eval import evaluate_erd_vs_truth
except ImportError as exc:
    print(f"Import error ({exc}). From the package root: pip install -r scripts/requirements.txt")
    sys.exit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = [
    ("erd_a", "instances_erd_a.csv", "ground_truth_erd_a.json", "brief_erd_a.txt"),
    ("erd_b", "instances_erd_b.csv", "ground_truth_erd_b.json", "brief_erd_b.txt"),
]

def _prompt(df: pd.DataFrame, level: int, brief: str | None = None) -> str:
    # The base prompt is a file (prompts/erd_prompt.txt) with a {{DATA_BLOCK}}
    # placeholder — same convention as step2/prompts/mapping_prompt.txt.
    template = open(os.path.join(ROOT, "prompts", "erd_prompt.txt")).read()
    rows = df.head(10)
    if level == 1:
        # data WITHOUT headers: only value rows — infer attributes from the values
        block = ("DATA_ROWS (no column names given — infer the attributes from the values):\n"
                 + json.dumps(rows.values.tolist(), default=str))
    else:
        block = ("COLUMNS: " + json.dumps([str(c) for c in df.columns])
                 + "\nSAMPLE_ROWS: " + json.dumps(rows.to_dict(orient="records"), default=str))
        if level >= 3 and brief:
            block += "\n\nEXPERT_BRIEF:\n" + brief
    return template.replace("{{DATA_BLOCK}}", block)


def _llm_erd(df: pd.DataFrame, level: int, temperature: float, brief: str | None = None) -> dict:
    raw = LLMClient().generate_json(_prompt(df, level, brief), temperature=temperature)
    return ERDModel.model_validate(raw).model_dump()


def _grounded_prompt(df: pd.DataFrame, baseline_erd: dict) -> str:
    """S5: the deterministic baseline is injected INTO the prompt, so the LLM BUILDS ON
    the data-verified structure instead of the baseline being merged post-hoc. This is
    the app's real design (erd_generator.build_erd_prompt with baseline_erd)."""
    template = open(os.path.join(ROOT, "prompts", "erd_prompt.txt")).read()
    rows = df.head(10)
    block = (
        "COLUMNS: " + json.dumps([str(c) for c in df.columns])
        + "\nSAMPLE_ROWS: " + json.dumps(rows.to_dict(orient="records"), default=str)
        + "\n\nDETERMINISTIC_BASELINE (data-verified: its column data_types, primary keys and "
          "foreign keys are GROUND TRUTH — keep them, and keep EVERY column). COMPLEMENT it: "
          "decompose into the right conceptual entities, give meaningful names, and add the "
          "relationships it could not detect:\n"
        + json.dumps(baseline_erd, default=str)
    )
    return template.replace("{{DATA_BLOCK}}", block)


def _llm_erd_grounded(df: pd.DataFrame, baseline_erd: dict, temperature: float) -> dict:
    raw = LLMClient().generate_json(_grounded_prompt(df, baseline_erd), temperature=temperature)
    return ERDModel.model_validate(raw).model_dump()


def _aggregate(runs: list[dict]) -> dict:
    if not runs:
        return {}
    keys = {k for r in runs for k, v in r.items() if isinstance(v, (int, float))}
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        mu = sum(vals) / len(vals)
        sd = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
        out[k] = round(mu, 4)
        out[k + "_std"] = round(sd, 4)
    out["_runs"] = len(runs)
    return out


LLM_LEVELS = [(1, "llm_data_noheaders"), (2, "llm_data_headers"), (3, "llm_data_headers_brief")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="subset: 0 baseline, 1/2/3 LLM levels, 4 hybrid (default: all)")
    ap.add_argument("--repeats", type=int, default=3, help="LLM repetitions per temperature")
    ap.add_argument("--temps", type=float, nargs="*", default=[0.2])
    args = ap.parse_args()
    want = set(args.scenarios) if args.scenarios else None

    def _on(*ids):
        return want is None or any(i in want for i in ids)

    llm_ok = LLMClient().is_available()
    print(f"LLM available: {llm_ok}  ·  repeats={args.repeats}  ·  temps={args.temps}")
    if not llm_ok:
        print("  (no LLM → only the deterministic baseline (S0) is scored)")

    results: dict = {}
    for key, csv_file, gt_file, brief_file in DATASETS:
        df = pd.read_csv(os.path.join(ROOT, "data", csv_file))
        truth = json.load(open(os.path.join(ROOT, "ground-truth", gt_file)))
        brief = open(os.path.join(ROOT, "prompts", brief_file)).read()
        print(f"\n=== {key}: {csv_file} ({len(df)} rows, {len(df.columns)} cols) "
              f"vs truth ({len(truth.get('entities', []))} entities) ===")
        res: dict = {}

        baseline = build_baseline_erd(df, [csv_file])
        if _on("0"):
            res["baseline"] = evaluate_erd_vs_truth(baseline["erd"], truth)

        if llm_ok:
            hyb_s2, hyb_s3 = [], []
            for level, sid in LLM_LEVELS:
                if not _on(str(level)):
                    continue
                runs = []
                for _rep in range(max(1, args.repeats)):
                    for temp in args.temps:
                        try:
                            llm = _llm_erd(df, level, temp, brief)
                        except Exception as exc:  # keep going on transient/parse failures
                            print(f"  [{sid}] run failed: {exc}")
                            continue
                        runs.append(evaluate_erd_vs_truth(llm, truth))
                        if _on("4"):  # post-hoc hybrid: merge the baseline AFTER generation
                            if level == 2:
                                hyb_s2.append(evaluate_erd_vs_truth(build_hybrid_erd(baseline["erd"], llm), truth))
                            elif level == 3:
                                hyb_s3.append(evaluate_erd_vs_truth(build_hybrid_erd(baseline["erd"], llm), truth))
                if runs:
                    res[sid] = _aggregate(runs)
            if hyb_s2:
                res["hybrid_from_s2"] = _aggregate(hyb_s2)
            if hyb_s3:
                res["hybrid_from_s3"] = _aggregate(hyb_s3)

            # S5: grounded generation — the baseline is injected INTO the prompt (the LLM
            # BUILDS ON the data-verified structure), plus that grounded output hardened
            # post-hoc. Tests whether "the LLM starts from the correct base" wins.
            if _on("5"):
                g_runs, g_hyb = [], []
                for _rep in range(max(1, args.repeats)):
                    for temp in args.temps:
                        try:
                            g = _llm_erd_grounded(df, baseline["erd"], temp)
                        except Exception as exc:
                            print(f"  [llm_grounded] run failed: {exc}")
                            continue
                        g_runs.append(evaluate_erd_vs_truth(g, truth))
                        g_hyb.append(evaluate_erd_vs_truth(build_hybrid_erd(baseline["erd"], g), truth))
                if g_runs:
                    res["llm_grounded"] = _aggregate(g_runs)
                if g_hyb:
                    res["hybrid_from_grounded"] = _aggregate(g_hyb)

        results[key] = res
        _print_table(key, res)

    out_dir = os.path.join(ROOT, "results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "step1_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: results/{os.path.basename(out_dir)}/step1_evaluation.json")


_COLS = ["entity_f1", "attribute_coverage", "attribute_placement", "pk_accuracy",
         "type_accuracy", "relationship_f1"]


def _print_table(key: str, res: dict):
    print(f"  {'scenario':<24}" + "".join(f"{c.replace('_',' ')[:14]:>15}" for c in _COLS))
    for name, m in res.items():
        row = f"  {name:<24}"
        for c in _COLS:
            v = m.get(c)
            row += f"{(f'{v:.3f}' if isinstance(v, (int, float)) else '-'):>15}"
        print(row)


if __name__ == "__main__":
    main()
