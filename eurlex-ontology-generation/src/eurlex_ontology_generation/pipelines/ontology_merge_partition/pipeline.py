from kedro.pipeline import Pipeline, node

from .nodes import (
    load_partition_revisions,
    merge_partition,
    persist_partition_merge,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_partition_revisions,
                inputs=[
                    "params:ontology_merge_partition.part_id",
                    "params:ontology_merge_partition.revisions_directory",
                ],
                outputs="partition_revisions",
                name="load_partition_revisions_node",
            ),
            node(
                func=merge_partition,
                inputs=[
                    "partition_revisions",
                    "params:ontology_merge_partition.part_id",
                    "params:ontology_merge_partition.embedding_model",
                    "params:ontology_merge_partition.similarity_threshold",
                ],
                outputs=[
                    "partition_merged_ontology",
                    "partition_merge_provenance",
                    "partition_cluster_report",
                ],
                name="merge_partition_node",
            ),
            node(
                func=persist_partition_merge,
                inputs=[
                    "partition_merged_ontology",
                    "partition_merge_provenance",
                    "partition_cluster_report",
                    "params:ontology_merge_partition.part_id",
                    "params:ontology_merge_partition.output_directory",
                    "params:ontology_merge_partition.provenance_directory",
                    "params:ontology_merge_partition.cluster_report_directory",
                ],
                outputs="partition_merge_path",
                name="persist_partition_merge_node",
            ),
        ]
    )