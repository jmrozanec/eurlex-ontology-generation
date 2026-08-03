from kedro.pipeline import Pipeline, node
from .nodes import generate_partial_ontologies


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
                func=generate_partial_ontologies,
                inputs=[
                    "eurlex_chunks",
                    "params:ontology_generation.max_chunks",
                    "params:ontology_generation.model",
                    "params:ontology_generation.temperature",
                    "params:ontology_generation.system_prompt",
                    "params:ontology_generation.user_prompt_template",
                    "params:ontology_generation.json_mode",
                ],
                outputs="ontology_candidates",
                name="generate_partial_ontologies_node",
            ),
        ]
    )