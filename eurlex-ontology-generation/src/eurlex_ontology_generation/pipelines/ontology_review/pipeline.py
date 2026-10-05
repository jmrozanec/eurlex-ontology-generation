from kedro.pipeline import Pipeline, node

from .nodes import (
    persist_ontology_reviews,
    review_ontologies,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=review_ontologies,
            inputs=[
                "params:ontology_review.batch_id",
                "params:ontology_review.batch_group_size",
                "params:ontology_review.candidates_directory",
                "chunk_partition_manifest",
                "params:ontology_review.model",
                "params:ontology_review.temperature",
                "params:ontology_review.seed",
                "params:ontology_review.num_ctx",
                "params:ontology_review.num_predict",
                "params:ontology_review.max_retries",
                "params:ontology_review.system_prompt",
                "params:ontology_review.user_prompt_template",
                "params:ontology_review.approval_thresholds",
                "params:ontology_review.llm_backend",
            ],
            outputs="ontology_reviews_batch",
            name="review_ontologies_node",
        ),
        node(
            func=persist_ontology_reviews,
            inputs=[
                "ontology_reviews_batch",
                "params:ontology_review.batch_id",
                "params:ontology_review.batch_group_size",
                "params:ontology_review.output_directory",
            ],
            outputs="ontology_reviews_path",
            name="persist_ontology_reviews_node",
        ),
    ])