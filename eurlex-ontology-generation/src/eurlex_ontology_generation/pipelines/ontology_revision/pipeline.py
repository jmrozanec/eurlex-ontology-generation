from kedro.pipeline import Pipeline, node

from .nodes import (
    persist_ontology_revisions,
    persist_validation_reports,
    revise_ontologies,
    validate_revised_ontologies,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=revise_ontologies,
            inputs=[
                "params:ontology_revision.batch_id",
                "params:ontology_revision.batch_group_size",
                "params:ontology_revision.reviews_directory",
                "params:ontology_revision.model",
                "params:ontology_revision.temperature",
                "params:ontology_revision.system_prompt",
                "params:ontology_revision.user_prompt_template",
                "params:ontology_revision.json_mode",
                "params:ontology_revision.expected_ontology_keys",
                "params:ontology_revision.num_ctx",
                "params:ontology_revision.num_predict",
                "params:ontology_revision.llm_backend",
            ],
            outputs="ontology_revised",
            name="revise_ontologies_node",
        ),
        node(
            func=persist_ontology_revisions,
            inputs=[
                "ontology_revised",
                "params:ontology_revision.batch_id",
                "params:ontology_revision.batch_group_size",
                "params:ontology_revision.output_directory",
            ],
            outputs="ontology_revisions_path",
            name="persist_ontology_revisions_node",
        ),
        node(
            func=validate_revised_ontologies,
            inputs=[
                "ontology_revised",
                "params:ontology_revision.expected_ontology_keys",
                "params:ontology_revision.disallowed_currency_terms",
                "params:ontology_revision.valid_odrl_core_types",
                "params:ontology_revision.currency_check_exclusions",
            ],
            outputs="ontology_revision_report",
            name="validate_revised_ontologies_node",
        ),
        node(
            func=persist_validation_reports,
            inputs=[
                "ontology_revision_report",
                "params:ontology_revision.batch_id",
                "params:ontology_revision.batch_group_size",
                "params:ontology_revision.validation_output_directory",
            ],
            outputs="ontology_revision_reports_path",
            name="persist_validation_reports_node",
        ),
    ])