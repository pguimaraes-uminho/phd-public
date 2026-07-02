from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.core.config import settings
from app.models.erd import ERDModel
from app.services.llm import LLMClient
from app.services.erd_baseline import build_baseline_erd
from app.services.erd_hybrid import build_hybrid_erd


def generate_erd_candidates(
    dfs: list[pd.DataFrame] | pd.DataFrame,
    refine_prompt: str | None = None,
    previous_erd: dict[str, Any] | None = None,
    prompt_history: list[dict[str, Any]] | None = None,
    table_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    temps = settings.temperature_list

    # Deterministic baseline FIRST: a real, grounded ERD (data-verified keys, FKs,
    # cardinalities, types). It works with zero LLM AND grounds the LLM when present.
    baseline = build_baseline_erd(dfs, table_names)
    baseline_cand = {"temperature": 0.0, "erd": baseline["erd"], "source": "baseline"}

    client = LLMClient()
    if not client.is_available():
        return [baseline_cand]  # real ERD, not a mock

    prompt = build_erd_prompt(
        dfs,
        refine_prompt=refine_prompt,
        previous_erd=previous_erd,
        prompt_history=prompt_history,
        baseline_erd=baseline["erd"],
    )
    # Each LLM candidate is MERGED with the baseline: the baseline's data-verified
    # structure (keys/FKs/types) is locked, the LLM contributes names + relationships
    # it missed, with per-field provenance (⚙️ deterministic / 🤖 llm).
    candidates: list[dict[str, Any]] = []
    for temp in temps:
        try:
            raw = client.generate_json(prompt, temperature=temp)
            llm_erd = ERDModel.model_validate(raw).model_dump()
            hybrid = ERDModel.model_validate(build_hybrid_erd(baseline["erd"], llm_erd)).model_dump()
            candidates.append({"temperature": temp, "erd": hybrid, "source": "hybrid"})
        except Exception:
            # Keep going even if one temperature fails (transient provider load).
            continue

    candidates.append(baseline_cand)  # pure deterministic fallback always available
    return candidates


def build_erd_prompt(
    dfs: list[pd.DataFrame],
    refine_prompt: str | None = None,
    previous_erd: dict[str, Any] | None = None,
    prompt_history: list[dict[str, Any]] | None = None,
    baseline_erd: dict[str, Any] | None = None,
) -> str:
    all_contexts = []
    
    for i, df in enumerate(dfs):
        columns = [str(c) for c in df.columns.tolist()]
        
        # [SMART SAMPLING START]
        valid_rows = df.dropna(how="all")
        total_rows = len(valid_rows)
        sample_size = settings.csv_sample_rows
        
        if total_rows <= sample_size:
            sample_df = valid_rows
        else:
            head_rows = valid_rows.head(5)
            tail_rows = valid_rows.tail(5)
            middle_indices = valid_rows.index[5:-5]
            if len(middle_indices) > 0:
                random_count = max(0, sample_size - 10)
                if random_count > len(middle_indices):
                    random_count = len(middle_indices)
                random_rows = valid_rows.loc[middle_indices].sample(n=random_count, random_state=42)
            else:
                random_rows = pd.DataFrame()
            sample_df = pd.concat([head_rows, random_rows, tail_rows]).drop_duplicates()
        
        sample_rows = sample_df.to_dict(orient="records")
        col_stats = _analyze_csv_statistics(df)
        
        all_contexts.append({
            "file_index": i,
            "columns": columns,
            "statistics": col_stats,
            "sample_rows": sample_rows
        })

    prompt = (
        "You are a database normalization expert.\n"
        "Given the schemas, statistical profiles, and sample rows from MULTIPLE related CSV files below, infer a single, UNIFIED 3NF-compliant ERD that integrates all this data.\n"
        "Rules:\n"
        "- Use attributes that appear in the provided CSV columns.\n"
        "- Do NOT invent attributes not present in the data.\n"
        "- Identify relationships and foreign keys across the files.\n"
        "- Output ONLY valid JSON (no markdown, no commentary).\n"
        "- Include entities, attributes, primary keys, foreign keys, and relationships with cardinalities.\n"
        "- Ensure 3NF: no partial or transitive dependencies in any entity.\n"
        "JSON schema:\n"
        "{\n"
        "  \"entities\": [\n"
        "    {\n"
        "      \"name\": \"...\",\n"
        "      \"attributes\": [\n"
        "        {\"name\": \"...\", \"data_type\": \"...\", \"is_primary_key\": true|false, \"is_foreign_key\": true|false, \"references\": \"Entity.attribute\"}\n"
        "      ],\n"
        "      \"primary_key\": [\"attr1\", \"attr2\"]\n"
        "    }\n"
        "  ],\n"
        "  \"relationships\": [\n"
        "    {\"name\": \"...\", \"from_entity\": \"...\", \"to_entity\": \"...\", \"from_cardinality\": \"1\", \"to_cardinality\": \"0..N\", \"fk_attribute\": \"...\", \"pk_attribute\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "INPUT_DATA_CONTEXTS:\n"
        f"{json.dumps(all_contexts, ensure_ascii=True)}\n"
    )
    if baseline_erd:
        prompt += (
            "\nDETERMINISTIC_BASELINE_ERD (extracted mechanically from the data — its primary keys,\n"
            "foreign keys, cardinalities and datatypes are DATA-VERIFIED; keep them unless clearly wrong):\n"
            f"{json.dumps(baseline_erd, ensure_ascii=True)}\n"
            "COMPLEMENT this baseline: give entities/attributes/relationships meaningful semantic names,\n"
            "add relationships or entities the baseline could NOT detect (foreign keys whose column names\n"
            "don't match the referenced key, conceptual entities), and split any denormalized table by\n"
            "meaning. Do NOT drop data-verified keys/FKs without a clear reason.\n"
        )
    if previous_erd:
        prompt += (
            "\nCURRENT_ERD_JSON:\n"
            f"{json.dumps(previous_erd, ensure_ascii=True)}\n"
            "Refine or correct the ERD above without introducing new attributes.\n"
        )
    if prompt_history:
        prompt += (
            "\nPROMPT_HISTORY:\n"
            f"{json.dumps(prompt_history, ensure_ascii=True)}\n"
        )
    if refine_prompt and refine_prompt.strip():
        prompt += (
            "\nADDITIONAL_EXPERT_INSTRUCTIONS:\n"
            f"{refine_prompt.strip()}\n"
            "Ensure these instructions do not introduce attributes or entities not present in the CSV.\n"
        )
    return prompt


def _analyze_csv_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generates a statistical profile of the dataframe columns to aid 
    the LLM in inferring cardinality and data types.
    """
    stats = {}
    for col in df.columns:
        try:
            series = df[col]
            n_unique = series.nunique()
            n_total = len(series)
            n_null = series.isna().sum()
            
            # Simple heuristic for uniqueness
            is_unique = (n_unique == n_total) and (n_total > 0)
            
            # Sample values for context (top 3 frequent)
            top_vals = series.value_counts().head(3).index.tolist()
            # Convert to string to avoid serialization issues
            top_vals = [str(v) for v in top_vals]
            
            stats[str(col)] = {
                "dtype": str(series.dtype),
                "unique_count": int(n_unique),
                "null_count": int(n_null),
                "is_unique_candidate": bool(is_unique),
                "sample_values": top_vals
            }
        except Exception:
            # Fallback for complex types
            stats[str(col)] = {"error": "Could not analyze"}
    return stats


def _mock_erd(df: pd.DataFrame) -> ERDModel:
    columns = [str(c) for c in df.columns.tolist()]
    attributes = []
    for idx, col in enumerate(columns):
        attributes.append(
            {
                "name": col,
                "data_type": "string",
                "is_primary_key": idx == 0,
                "is_foreign_key": False,
            }
        )
    entity = {
        "name": "Record",
        "attributes": attributes,
        "primary_key": [columns[0]] if columns else [],
    }
    return ERDModel.model_validate({"entities": [entity], "relationships": []})
