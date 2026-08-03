from kedro.pipeline import Pipeline, node
from .nodes import review_ontologies


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
                func=review_ontologies,
                inputs=[
                    "ontology_candidates",
                    "eurlex_chunks",
                    "params:ontology_review.max_candidates",
                    "params:ontology_review.model",
                    "params:ontology_review.temperature",
                    "params:ontology_review.system_prompt",
                    "params:ontology_review.user_prompt_template",
                    "params:ontology_review.json_mode",
                ],
                outputs="ontology_reviews",
                name="review_ontologies_node",
            ),
        ]
    )