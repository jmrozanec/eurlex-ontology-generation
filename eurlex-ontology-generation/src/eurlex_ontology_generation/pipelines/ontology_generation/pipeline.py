from kedro.pipeline import Pipeline, node

from .nodes import (
    generate_partial_ontologies,
    persist_ontology_candidates,
    select_chunk_batch,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_chunk_batch,
                inputs=[
                    "chunk_batch_manifest",
                    "params:ontology_generation.batch_id",
                    "params:ontology_generation.batch_group_size",
                    "chunk_partition_manifest",
                ],
                outputs="current_chunk_batch",
                name="select_chunk_batch_node",
            ),
            node(
                func=generate_partial_ontologies,
                inputs=[
                    "current_chunk_batch",
                    "params:ontology_generation.batch_id",
                    "params:ontology_generation.model",
                    "params:ontology_generation.temperature",
                    "params:ontology_generation.system_prompt",
                    "params:ontology_generation.user_prompt_template",
                    "params:ontology_generation.json_mode",
                    "params:ontology_generation.llm_backend",
                ],
                outputs="ontology_candidates_batch",
                name="generate_partial_ontologies_node",
            ),
            node(
                func=persist_ontology_candidates,
                inputs=[
                    "ontology_candidates_batch",
                    "params:ontology_generation.batch_id",
                    "params:ontology_generation.batch_group_size",
                    "params:ontology_generation.output_directory",
                ],
                outputs="ontology_candidates_path",
                name="persist_ontology_candidates_node",
            ),
        ]
    )