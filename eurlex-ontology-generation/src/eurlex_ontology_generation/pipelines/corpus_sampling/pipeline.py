from kedro.pipeline import Pipeline, node
from .nodes import sample_corpus, chunk_corpus


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            # Sample the dataset to reduce size
            node(
                func=sample_corpus,
                inputs=[
                    "eurlex_corpus",
                    "params:corpus_sampling.sample_size",
                    "params:corpus_sampling.random_seed",
                ],
                outputs="eurlex_sample",
                name="sample_corpus_node",
            ),

            # Split the sample into smaller chunk for LLM processing
            node(
                func=chunk_corpus,
                inputs=[
                    "eurlex_sample",
                    "params:corpus_sampling.text_column",
                    "params:corpus_sampling.chunk_size",
                    "params:corpus_sampling.chunk_overlap",
                ],
                outputs="eurlex_chunks",
                name="chunk_corpus_node",
            ),
        ]
    )