from kedro.pipeline import Pipeline, node
from .nodes import (
    profile_concepts,
    generate_merge_rules,
    validate_merge_rules,
    apply_merge_rules_iteratively,
)


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
                func=profile_concepts,
                inputs=[
                    "reviewed_ontologies",
                    "params:ontology_merge_rules.model",
                    "params:ontology_merge_rules.temperature",
                    "params:ontology_merge_rules.concept_profiling_system_prompt",
                    "params:ontology_merge_rules.concept_profiling_user_prompt_template",
                    "params:ontology_merge_rules.json_mode",
                ],
                outputs="concept_profile",
                name="profile_concepts_node",
            ),
            node(
                func=generate_merge_rules,
                inputs=[
                    "concept_profile",
                    "params:ontology_merge_rules.model",
                    "params:ontology_merge_rules.temperature",
                    "params:ontology_merge_rules.rule_generation_system_prompt",
                    "params:ontology_merge_rules.rule_generation_user_prompt_template",
                    "params:ontology_merge_rules.json_mode",
                ],
                outputs="merge_rules_raw",
                name="generate_merge_rules_node",
            ),
            node(
                func=validate_merge_rules,
                inputs="merge_rules_raw",
                outputs="merge_rules",
                name="validate_merge_rules_node",
            ),
            node(
                func=apply_merge_rules_iteratively,
                inputs=[
                    "reviewed_ontologies",
                    "concept_profile",
                    "merge_rules",
                    "params:ontology_merge_rules.name_similarity_threshold",
                ],
                outputs=["final_merged_ontology_v2", "rule_application_trace"],
                name="apply_merge_rules_node",
            ),
        ]
    )