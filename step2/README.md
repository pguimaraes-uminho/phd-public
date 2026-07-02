# Schema-to-Ontology Mapping with LLMs

** THIS READ.ME WAS GENEREATED BY AI **

Replication package for the two-stage evaluation of LLM-assisted relational-schema → domain-ontology
mapping, and cross-ERD ontology integration, against expert ground truth.

> Parts of this package (scaffolding, scripts and documentation) were prepared with AI assistance;
> all ground-truth models and design decisions are the author's.

## Runs reported here

| Item | Value |
|------|-------|
| **Model** | Google **Gemini 2.5 Flash** — model id `gemini-2.5-flash` |
| **Run date** | **Stage 1: 2026-07-01** (revised business-rules briefs, code-enforced hybrid freeze) · **Stage 2: 2026-06-29** |
| **Temperature** | **0.2** |
| **Repetitions** | **3 runs per LLM scenario** (results reported as mean ± std) |

The exact published outputs of these runs are in `results/` (see the reproducibility note below).

## What is evaluated

- **Stage 1 — ERD → ontology mapping** (`scripts/run_stage1_per_erd.py`): for each ERD, a
  deterministic rule-based baseline (Scenario 0), three LLM scenarios and three **hybrid**
  scenarios are scored against the expert ground-truth ontology.
  - LLM-1 (ERD only) · LLM-2 (ERD + data) · LLM-3 (ERD + data + expert brief). The expert brief is
    genuine domain/business rules (not element-level answers): it deliberately does **not** contain
    the ground-truth canonical names or how to model any specific element.
  - **Hybrid-1/2/3** (Scenarios 4/5/6): the deterministic pass recovers the structure and the LLM
    only refines the *names* on top of it — the same three context levels (ERD · +data · +brief)
    but with the rule-based structural scaffold supplied to the model. The freeze is **enforced in
    code** (`_enforce_scaffold_structure`): after the LLM responds, its structural fields
    (ontological category, property-vs-relation type, target concept, column grouping) are
    overwritten by the scaffold's and only its names are kept, so the LLM is physically unable to
    change the deterministic structure. This measures the *combined* system argued for in RQ3
    (rules for structure, LLM for semantics) as its own arm, and isolates whether giving the LLM a
    correct structure lets it name better than it does from the raw ERD.
- **Stage 2 — cross-ERD integration** (`scripts/run_stage2_cross_erd.py`): the two ground-truth
  ontologies are integrated (a deterministic classic matcher + two LLM scenarios) and scored
  against the cross-ERD ground truth.

## Headline results (mean ± std over 3 runs)

**Stage 1 — refinement accuracy (semantic naming)**

| Scenario | ERD-A | ERD-B |
|----------|-------|-------|
| Rule-based (deterministic) | 0.035 | 0.000 |
| LLM-1 (ERD) | 0.678 ± .033 | 0.333 ± .039 |
| LLM-2 (+data) | 0.667 ± .016 | 0.302 ± .090 |
| LLM-3 (+brief) | 0.655 ± .098 | 0.365 ± .045 |
| Hybrid-1 (rules structure + LLM naming, ERD) | 0.701 ± .016 | 0.397 ± .045 |
| Hybrid-2 (+data) | 0.586 ± .123 | 0.333 ± .067 |
| Hybrid-3 (+brief) | 0.483 ± .074 | 0.508 ± .045 |

Type accuracy is 100% for every scenario. The hybrid arms **hard-freeze** the deterministic
structure (the LLM may only rename, never re-type — enforced in code), so they inherit the
baseline's ontological-category accuracy (91% ERD-A / 86% ERD-B), below the free LLM's 95–100% —
the deliberate cost of not letting the LLM touch structure.

They test the *interaction* effect: does the LLM name better when handed the already-recovered
structure than when it must recover structure and semantics together from the raw ERD? **The effect
is schema-dependent.** On the harder schema (ERD-B) the scaffold **helps** at every context level
(Hybrid−LLM = +0.064 / +0.032 / +0.143); the Hybrid-3 vs LLM-3 gap (0.508 vs 0.365) is the only
comparison with clearly separated, tight bands (both std ≈ .045, Welch t ≈ 3.18) and is the cleanest
evidence. On the easier schema (ERD-A) the LLM already recovers structure alone, so the hybrid is
neutral-to-worse (+0.023 / −0.080 / −0.172); with n = 3 these ERD-A deltas are **directional, not
statistically tight** (the k=3 LLM band is wide, std ≈ .098; Welch t ≈ −1.99, n.s.). Treat all
Stage-1 results as indicative (2 ERDs, 3 runs).

Stage-1 numbers are from a **2026-07-01 same-snapshot re-run** with the revised business-rules
briefs and code-enforced hard-freeze; rule-based is deterministic. (The earlier 2026-06-29 outputs
used richer briefs, since trimmed to genuine domain rules so they do not telegraph the answer.)

**Stage 2 — integration**

| Scenario | Match F1 | Conflict-resolution |
|----------|----------|---------------------|
| `cross_rule_based` (deterministic matcher) | 1.000 | 0.615 |
| `cross_llm` | 0.939 ± .04 | 0.641 |
| `cross_llm_brief` | 0.970 ± .04 | 0.667 |

(The classic matcher solves concept matching; the LLM's contribution is conflict resolution.)

## Folder map

```
erd/            the two source ERD schemas               (erd_a.json, erd_b.json)
ground-truth/   expert ground-truth ontologies + cross-ERD ground truth
                (ground_truth_mapping_erd_a.json, ground_truth_mapping_erd_b.json, ground_truth_cross_erd.json)
data/           instance data, one CSV per ERD           (instances_erd_a.csv, instances_erd_b.csv)
prompts/        mapping_prompt.txt, output_schema.json,
                refinement_brief_erd_{a,b}.json, integration_brief.json
results/        published raw + aggregate outputs per stage, in dated run folders
                  stage1_per_erd/20260629_085849/   (per-scenario reports + evaluation_results.json)
                  stage2_cross_erd/20260629_091202/ (cross_evaluation_results.json)
scripts/        runnable Python (both stages) + app/ package + requirements.txt
```

Licensing and citation live at the **repository root** (shared with `step1/`):
`../LICENSE` (MIT — code), `../LICENSE-DATA` (CC-BY-4.0 — data/ground truth/prompts/docs),
`../CITATION.cff` (how to cite).

The data/ground-truth/prompt files keep human-readable names here; inside `scripts/app/` the code
reads them from these top-level folders (single source of truth, no duplication).

## Reproduce

Requires **Python 3.10+** and (for the LLM scenarios only) a **Google Gemini API key**.

```bash
cd step2
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r scripts/requirements.txt
cp .env.example .env                       # then edit .env and paste your GEMINI_API_KEY
```

**Stage 1** (all scenarios, both ERDs, mean ± std over 3 runs):
```bash
python scripts/run_stage1_per_erd.py --targets ERD-A ERD-B --scenarios 0 1 2 3 4 5 6 --temps 0.2 --repeats 3
```

Scenarios `4 5 6` are the hybrid arms (rules structure + LLM semantics), mirroring `1 2 3` with
the deterministic structural scaffold supplied to the model. Run just the hybrid arms with
`--scenarios 4 5 6`.

**Stage 2**:
```bash
python scripts/run_stage2_cross_erd.py --scenarios cross_rule_based cross_llm cross_llm_brief --temps 0.2 --repeats 3
```

**Deterministic only** (no API key, instant): use `--scenarios 0` (Stage 1) / `--scenarios cross_rule_based` (Stage 2).

Each run writes a new dated folder under `results/stage1_per_erd/` or `results/stage2_cross_erd/`.

## ⚠️ Reproducibility note

A hosted LLM such as Gemini can **change over time**, and even at **temperature 0.2 (not 0)** the
model gives **run-to-run variation**. That is exactly why results are reported as **mean ± standard
deviation over 3 runs** rather than a single number.

To keep the work reproducible despite this:

1. We **pin and record the exact model identifier** (`gemini-2.5-flash`) and the **run date**
   (2026-06-29).
2. We **publish the actual outputs of the reported runs** under `results/`.

Therefore, **all reported analysis reproduces from the published outputs in `results/`**, even if
re-calling the model later yields slightly different numbers. The qualitative conclusions
(structure is deterministic; semantic naming and conflict resolution are where the LLM adds value;
more context → better mapping) are stable across runs.

## Metrics (brief)

- **type_accuracy** — is each column mapped to the right ontological type (datatype property vs object-property/relation)?
- **category_accuracy** — is each entity given the right ontological category (kind / relator / category)?
- **refinement_accuracy** — semantic naming correctness; relation verbs accept the expert's curated
  synonym sets (the `accepted` lists inside the gold ontologies).
- **Stage 2** — match precision/recall/**F1** (concept alignment) and **conflict-resolution accuracy**.

## License & citation

- **Code** — MIT (`LICENSE`).
- **Data, ground truth, prompts, documentation** — CC-BY-4.0 (`LICENSE-DATA`).
- **How to cite** — see `CITATION.cff`.
