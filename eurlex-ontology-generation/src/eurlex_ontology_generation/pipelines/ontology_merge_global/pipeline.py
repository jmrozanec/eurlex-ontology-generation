from kedro.pipeline import Pipeline, node

from .nodes import (
    load_all_partition_merges,
    merge_global,
    persist_global_merge,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_all_partition_merges,
                inputs="params:ontology_merge_global.partition_merges_directory",
                outputs="partition_merges",
                name="load_all_partition_merges_node",
            ),
            node(
                func=merge_global,
                inputs=[
                    "partition_merges",
                    "params:ontology_merge_global.embedding_model",
                    "params:ontology_merge_global.similarity_threshold",
                ],
                outputs=[
                    "global_merged_ontology",
                    "global_merge_provenance",
                    "global_cluster_report",
                ],
                name="merge_global_node",
            ),
            node(
                func=persist_global_merge,
                inputs=[
                    "global_merged_ontology",
                    "global_merge_provenance",
                    "global_cluster_report",
                    "params:ontology_merge_global.output_path",
                    "params:ontology_merge_global.provenance_path",
                    "params:ontology_merge_global.cluster_report_path",
                ],
                outputs="global_merge_path",
                name="persist_global_merge_node",
            ),
        ]
    )