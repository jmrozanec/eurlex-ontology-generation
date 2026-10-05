from kedro.pipeline import Pipeline, node

from .nodes import partition_chunks


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=partition_chunks,
                inputs=[
                    "eurlex_full_chunks",
                    "params:chunk_batching.batch_size",
                    "params:ontology_generation.batch_group_size",
                    "params:chunk_storage.output_directory",
                ],
                outputs="chunk_partition_manifest",
                name="partition_full_chunks_node",
            ),
        ]
    )