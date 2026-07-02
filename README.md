# PhD replication packages: CSV → ERD → ontology

Two self-contained replication packages, one per step of the pipeline. Each folder has its own
README, data, ground truth, prompts, scripts and results.

| Folder | Step | Question |
|--------|------|----------|
| [`step1/`](step1/) | **CSV → ERD** | Modeling an ERD from raw CSV: how much does the deterministic method recover on its own, and how necessary is the LLM? (The LLM is *not* given the ERD; the input is only the CSV, and the LLM scenarios vary the prompt information level.) |
| [`step2/`](step2/) | **ERD → ontology** (+ cross-ERD integration) | Mapping a relational schema to a domain ontology, and integrating two ontologies, against expert ground truth. Includes the deterministic baseline, the LLM arms, and the **hybrid** arms (rules recover the structure, the LLM refines only the names). |

Both steps use the **same datasets** and are scored against expert ground truth.

## Licensing & citation (whole repository)

- [`LICENSE`](LICENSE) — MIT, for the code.
- [`LICENSE-DATA`](LICENSE-DATA) — CC-BY-4.0, for data, ground truth, prompts and documentation.
- [`CITATION.cff`](CITATION.cff) — how to cite this work.

To reproduce either step, see its own README ([`step1/README.md`](step1/README.md) ·
[`step2/README.md`](step2/README.md)).
