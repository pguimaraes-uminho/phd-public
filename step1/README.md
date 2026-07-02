# Step 1 — CSV → ERD: deterministic baseline vs LLM (replication)

Companion to [`../step2/`](../step2/) (which covers Step 2, ERD → ontology). This
package answers a Step-1 question: **when modeling an ERD from raw CSV data,
how much does the deterministic method recover on its own, and how necessary is the
LLM?**

It uses the **same datasets** as the Step-2 package. The difference from Step 2: the
LLM is **not** given the ERD (that would be the answer) — here the input is only the
CSV, and the LLM scenarios vary the **prompt information level**, simplest → most
complete. Each candidate ERD is scored against the **expert ground-truth ERD**.

> The modeling is hard on purpose: each dataset is ONE flat, denormalized CSV
> (`erd_a`: 25 cols / 19 rows → 12 entities; `erd_b`: 29 cols / 14 rows → 7 entities),
> so the method must decompose a single table into many entities.

## Scenarios

| id | scenario | input to the method |
|----|----------|---------------------|
| S0 | `baseline` | deterministic only — keys, FKs, types, **name-pattern + FD split**. **No LLM.** |
| S1 | `llm_data_noheaders` | LLM, prompt = data rows **without column names** (infer attributes from the values) |
| S2 | `llm_data_headers` | LLM, prompt = columns + sample rows (**with headers**) |
| S3 | `llm_data_headers_brief` | LLM, prompt = columns + sample rows + an **expert domain brief** |
| S4 | `hybrid` | **LLM decomposition ⊕ deterministic hardening** — the S3 LLM's entities, with each column's data-verified type + verified keys/FKs from the baseline |

The LLM scenarios isolate the value of each information layer: S1→S2 = the value of the
**column names (headers)**; S2→S3 = the value of an **expert brief**.

## Metrics (predicted ERD vs expert ERD)

Entities are matched by attribute-set overlap (names differ across methods), then:

- `entity_precision/recall/f1` — recovered the right entities (attribute groups)?
- `attribute_coverage` — of the truth's attributes, how many appear anywhere?
- `attribute_placement` — how many are in the RIGHT (matched) entity?
- `pk_accuracy` — matched entities whose PK equals the truth PK
- `type_accuracy` — datatype agreement on shared attributes (broad classes)
- `relationship_precision/recall/f1` — FK edges recovered (endpoints matched + same fk)

## Reproduce

```bash
pip install -r scripts/requirements.txt
cd scripts

# offline sanity — only the deterministic baseline (no key, no cost):
GEMINI_MOCK=true python3 run_step1_eval.py --scenarios 0

# full run (your live Gemini key — costs tokens; scenarios 1-4 call the LLM):
python3 run_step1_eval.py --repeats 3
```
Results are written to `results/<timestamp>/step1_evaluation.json`.

## Headline — full 8-condition matrix (`--repeats 3`, gemini-2.5-flash @ 0.2)

Mean over 3 generations. Run of record: `results/20260701_214947/`.
Full write-up: [EXPERIMENT_REPORT.txt](EXPERIMENT_REPORT.txt). Groups: **no LLM** (S0)
· **LLM alone** (S1–S3) · **hybrid post-hoc** (merge after) · **hybrid grounded**
(baseline in the prompt).

**erd_a** (25 cols / 19 rows → 12 entities)

| condition | entity_f1 | attr_cov | attr_place | pk | type | rel_f1 |
|-----------|-----------|----------|------------|-----|------|--------|
| S0 baseline | 0.526 | 0.622 | 0.243 | 0.600 | 1.00 | 0.000 |
| S1 llm_noheaders | 0.407 | 0.405 | 0.216 | 0.200 | 1.00 | 0.000 |
| S2 llm_headers | 0.805 | 0.739 | 0.550 | 0.659 | 1.00 | 0.293 |
| S3 llm_headers+brief | 0.783 | 0.667 | 0.495 | 0.627 | 0.94 | 0.271 |
| hybrid_from_s2 (post-hoc) | 0.779 | 0.757 | 0.531 | 0.648 | 1.00 | 0.263 |
| hybrid_from_s3 (post-hoc) | 0.696 | 0.703 | 0.441 | 0.579 | 1.00 | 0.190 |
| **llm_grounded** | **0.853** | **0.757** | **0.577** | 0.652 | 1.00 | **0.309** |
| **hybrid_from_grounded** | **0.853** | **0.757** | **0.577** | 0.652 | 1.00 | **0.309** |

**erd_b** (29 cols / 14 rows → 7 entities)

| condition | entity_f1 | attr_cov | attr_place | pk | type | rel_f1 |
|-----------|-----------|----------|------------|-----|------|--------|
| S0 baseline | 0.545 | 0.812 | 0.375 | 1.000 | 1.00 | 0.200 |
| S1 llm_noheaders | 0.267 | 0.281 | 0.156 | 0.500 | 1.00 | 0.000 |
| **S2 llm_headers** | **1.000** | 0.906 | **0.865** | 1.000 | 1.00 | **0.667** |
| S3 llm_headers+brief | 0.762 | 0.750 | 0.604 | 0.833 | 1.00 | 0.238 |
| **hybrid_from_s2 (post-hoc)** | **1.000** | **0.969** | **0.865** | 1.000 | 1.00 | **0.667** |
| hybrid_from_s3 (post-hoc) | 0.762 | 0.969 | 0.635 | 0.833 | 1.00 | 0.238 |
| llm_grounded | 0.949 | 0.927 | 0.833 | 0.889 | 1.00 | 0.648 |
| hybrid_from_grounded | 0.949 | 0.969 | 0.833 | 0.889 | 1.00 | 0.648 |

**Reading it:**
1. **Headers carry the signal.** Without them (S1) the LLM is *below* the baseline
   (0.41 / 0.27); with them (S2) it jumps to 0.81 / 1.00.
2. **HOW you combine matters more than THAT you combine.** **Grounded generation**
   (baseline injected into the prompt → the LLM builds on it) is the **best** condition
   on erd_a (0.853 > S2 0.805) and near-best on erd_b (0.949). **Post-hoc merging** is
   the weakest hybrid — `hybrid_from_s2` even drops *below* S2 on erd_a (0.779),
   because backfilling omitted columns perturbs the match. Generate-then-merge can
   hurt; grounding does not.
3. **The two datasets disagree on the single winner** (grounded wins erd_a, S2 wins
   erd_b 1.000); on entity_f1 averaged they tie (0.901 vs 0.903), but **grounded is
   more robust** — best coverage on both, wins the harder case, type_acc 1.000.
4. **Types are deterministic** (1.000); the brief even erodes them (S3). **The brief
   HURTS** everywhere (S3 < S2), robust at n=3 — feed clean headers, not prose.
5. **Grounding subsumes post-hoc hardening** (`hybrid_from_grounded ≈ llm_grounded`).

Net: the deterministic method wins types/keys and recovers the name-patterned
entities; the LLM wins the semantic decomposition/naming/roles (all riding on the
headers); the brief and the current hybrid need more work. Mirrors the Step-2 finding
("structure deterministic, semantics LLM"), now measured for CSV → ERD.

## Layout

```
data/                 instances_erd_a.csv, instances_erd_b.csv     (same datasets as Step 2)
ground-truth/         ground_truth_erd_a.json, ground_truth_erd_b.json   (the expert ERDs)
prompts/erd_prompt.txt      the base prompt (a {{DATA_BLOCK}} template, like Step 2's mapping_prompt.txt)
prompts/brief_erd_*.txt     the expert domain briefs (the author's Step-2 orientation briefs, not an answer key)
scripts/run_step1_eval.py   the runner (scenarios + prompt levels)
scripts/erd_eval.py         predicted-ERD vs ground-truth-ERD scorer
scripts/app/                the Step-1 pipeline modules (baseline, hybrid, generator, validator)
results/              per-run output (timestamped)
```
