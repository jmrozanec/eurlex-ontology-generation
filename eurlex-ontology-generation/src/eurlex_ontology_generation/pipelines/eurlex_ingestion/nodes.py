import pandas as pd


# Prepare the EurLex corpus for the ontology extraction
def prepare_corpus(
    eurlex_raw: pd.DataFrame,       # Raw EurLex dataframe
    selected_columns: list[str],    # Columns to retain
    text_column: str,               # Name of the text column
    drop_missing_text: bool,        # flag used to remove documents without text
) -> pd.DataFrame:

    # .copy() to avoid modifying the original dataframe
    corpus = eurlex_raw[selected_columns].copy()

    # If the flag is True, remove rows where the text is missing, clean whitespace and empty strings
    if drop_missing_text:
        corpus = corpus.dropna(subset=[text_column])
        corpus[text_column] = corpus[text_column].str.strip()
        corpus = corpus[corpus[text_column] != ""]

    # Reset the index numbers because, since we dropped rows they will have gaps, so it becomes sequential again
    corpus = corpus.reset_index(drop=True)

    return corpus