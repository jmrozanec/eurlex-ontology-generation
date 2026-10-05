from kedro.pipeline import Pipeline, node

from .nodes import create_chunk_batch_manifest


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_chunk_batch_manifest,
                inputs=[
                    "eurlex_full_chunks",
                    "params:chunk_batching.batch_size",
                ],
                outputs="chunk_batch_manifest",
                name="create_chunk_batch_manifest_node",
            ),
        ]
    )