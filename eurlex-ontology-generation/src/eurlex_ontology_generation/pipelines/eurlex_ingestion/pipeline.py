from kedro.pipeline import Pipeline, node
from .nodes import prepare_corpus


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_corpus,
                inputs=[
                    "eurlex_raw",
                    "params:eurlex_ingestion.selected_columns",
                    "params:eurlex_ingestion.text_column",
                    "params:eurlex_ingestion.drop_missing_text",
                ],
                outputs="eurlex_corpus",
                name="prepare_corpus_node",
            ),
        ]
    )