import json
import re
import pandas as pd

from pathlib import Path

from ...utils.llm_client import llm_chat, run_parallel
from ...utils.persist import append_dedup_csv


# Load only the partition containing the requested batch.
def select_chunk_batch(
    batch_manifest: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    partition_manifest: pd.DataFrame,
) -> pd.DataFrame:

    # Locate the target batch in the global batch manifest
    batch_rows = batch_manifest[
        batch_manifest["batch_id"] == batch_id
    ]

    # Validate existence of requested batch
    if batch_rows.empty:
        raise ValueError(
            f"Batch {batch_id} does not exist."
        )

    # Calculate the corresponding partition identifier
    partition_id = batch_id // batch_group_size

    # Locate partition metadata in partition manifest
    partition_rows = partition_manifest[
        partition_manifest["part_id"] == partition_id
    ]

    # Validate existence of calculated partition
    if partition_rows.empty:
        raise ValueError(
            f"Partition {partition_id} does not exist."
        )

    # Extract target partition record and file path
    partition_info = partition_rows.iloc[0]
    # Manifests created on Windows contain backslashes: normalize them so the path also works on Linux
    partition_path = Path(
        str(partition_info["file_path"]).replace("\\", "/")
    )

    # Ensure target partition CSV exists on disk
    if not partition_path.exists():
        raise FileNotFoundError(
            f"Partition file not found: {partition_path}"
        )

    # Load partition DataFrame from disk
    partition = pd.read_csv(
        partition_path,
        encoding="utf-8",
    )

    # Retrieve row boundary markers for global batch
    batch_info = batch_rows.iloc[0]
    start_row = int(batch_info["start_row"])
    end_row = int(batch_info["end_row"])

    # Retrieve global starting row index of current partition
    partition_start_row = int(
        partition_info["start_row"]
    )

    # Convert global row boundaries into relative local indices for current partition
    local_start = start_row - partition_start_row
    local_end = end_row - partition_start_row

    # Slice target batch chunks locally from partition
    selected_chunks = partition.iloc[
        local_start:local_end + 1
    ].copy()

    # Validate that sliced count matches expected batch size
    expected_count = int(
        batch_info["chunk_count"]
    )

    if len(selected_chunks) != expected_count:
        raise ValueError(
            f"Batch {batch_id} contains "
            f"{len(selected_chunks)} chunks, "
            f"expected {expected_count}."
        )

    # Return selected batch chunks with clean index
    return selected_chunks.reset_index(drop=True)


# Calculate the persistent group identifier for a batch.
def get_batch_group_id(
    batch_id: int,
    batch_group_size: int,
) -> int:    

    # Validate boundary conditions
    if batch_id < 0:
        raise ValueError("batch_id must be greater than or equal to zero.")

    if batch_group_size <= 0:
        raise ValueError("batch_group_size must be greater than zero.")

    # Compute zero-based partition group index via integer division
    return batch_id // batch_group_size


# Generates partial ontologies for each text chunk using the local LLM in parallel
def generate_partial_ontologies(
    chunks: pd.DataFrame,
    batch_id: int,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
    llm_backend: dict,
) -> pd.DataFrame:

    # Helper closure to process a single chunk row through the LLM pipeline
    def _generate(row) -> dict:
        user_prompt = user_prompt_template.format(text=row["chunk_text"])

        # Query local LLM via helper utility
        response = llm_chat(
            llm_backend, model, system_prompt, user_prompt, temperature,
            json_mode=json_mode,
        )
        ontology_json = extract_json(response.content)
        return {
            "batch_id": batch_id,
            "chunk_id": row["chunk_id"],
            "CELEX": row["CELEX"],
            "ontology": json.dumps(ontology_json, ensure_ascii=False),
        }

    rows = [row for _, row in chunks.iterrows()]
    results = run_parallel(_generate, rows, llm_backend.get("max_workers", 1))
    
    return pd.DataFrame(results)


# Append the current batch results to the corresponding persistent CSV file.    
def persist_ontology_candidates(
    ontology_candidates: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
) -> str:    

    # Identify target partition file group
    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    output_path = Path(output_directory) / f"part_{group_id:03d}.csv"

    return append_dedup_csv(output_path, ontology_candidates)


# Helper to safely extract JSON from LLM output
def extract_json(text: str) -> dict:

    try:
        return json.loads(text)     # Direct parsing

    except json.JSONDecodeError:
        # Regex fallback: find anything between the first '{' and last '}'
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Return error payload if all parsing attempts fail
    return {
        "error": "Invalid JSON generated",
        "raw_output": text
    }