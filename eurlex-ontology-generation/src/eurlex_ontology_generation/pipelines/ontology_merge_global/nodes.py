from pathlib import Path
import json

import pandas as pd

from eurlex_ontology_generation.utils.ontology_merge_common.assembly import (
    assemble_merged_ontology,
    validate_merged_ontology,
)
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    canonicalize_clusters,
    cluster_entities,
    extract_entities,
)


# Load every partition-level merged ontology produced so far by the
# ontology_merge_partition pipeline. Runs on whatever subset of
# partitions is available - you do NOT need to have merged all 178066
# batches to test this end to end: merging just part_000 with itself
# is a valid (trivial) smoke test.
def load_all_partition_merges(partition_merges_directory: str) -> pd.DataFrame:
    paths = sorted(Path(partition_merges_directory).glob("part_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No partition merges found in {partition_merges_directory}. "
            "Run the ontology_merge_partition pipeline for at least one "
            "part_id first."
        )

    rows = []
    for path in paths:
        onto = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                # Reuse the same (chunk_id, CELEX, ontology) column shape
                # the shared assembly/entity_resolution modules expect,
                # so this pipeline can call the exact same functions as
                # the partition-level one.
                "chunk_id": path.stem,
                "CELEX": path.stem,
                "ontology": json.dumps(onto, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)


def merge_global(
    partition_merges: pd.DataFrame,
    embedding_model: str,
    similarity_threshold: float,
) -> tuple[dict, dict, pd.DataFrame]:
    entities = extract_entities(partition_merges)
    entities_with_clusters = cluster_entities(entities, embedding_model, similarity_threshold)
    canonical_map, cluster_report = canonicalize_clusters(entities_with_clusters)

    merged, provenance = assemble_merged_ontology(
        partition_merges, canonical_map, source_label="global"
    )
    provenance["n_partitions_included"] = len(partition_merges)
    provenance["validation_issues"] = validate_merged_ontology(merged)

    return merged, provenance, cluster_report


def persist_global_merge(
    merged_ontology: dict,
    provenance: dict,
    cluster_report: pd.DataFrame,
    output_path: str,
    provenance_path: str,
    cluster_report_path: str,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(merged_ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    Path(provenance_path).parent.mkdir(parents=True, exist_ok=True)
    Path(provenance_path).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    Path(cluster_report_path).parent.mkdir(parents=True, exist_ok=True)
    cluster_report.to_csv(cluster_report_path, index=False, encoding="utf-8")

    return output_path