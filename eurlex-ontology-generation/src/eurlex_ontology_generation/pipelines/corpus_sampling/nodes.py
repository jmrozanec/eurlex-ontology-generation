import pandas as pd


# Select a reproducible sample of the EurLex corpus
def sample_corpus(
    corpus: pd.DataFrame,   # Cleaned EurLex corpus
    sample_size: int,       # Number of documents to sample
    random_seed: int,       # Seed for reproducibility
) -> pd.DataFrame:

    # Ensure the sample size does not exceed the total number of available documents
    sample_size = min(sample_size, len(corpus))

    # Randomly sample the specified number of documents using the seed for reproducibility
    sampled = corpus.sample(
        n=sample_size,
        random_state=random_seed,
    )

    # Reset the DataFrame index
    sampled = sampled.reset_index(drop=True)

    return sampled


# Splits the dataset sample into smaller text chunks
def chunk_corpus(
    sample: pd.DataFrame,
    text_column: str,
    chunk_size: int,
    chunk_overlap: int,
) -> pd.DataFrame:

    # List to store the generated chunk dictionaries
    chunks = []

    # Iterate through each row of the input sample DataFrame
    for _, row in sample.iterrows():

        # Extract the text from the specified column
        text = row[text_column]

        # Skip rows where the content is not a valid string
        if not isinstance(text, str):
            continue

        start = 0
        chunk_number = 0

        # Loop through the text until the start pointer reaches the end of the string
        while start < len(text):

            end = start + chunk_size    # Calculate the end index for the current chunk            
            chunk = text[start:end]     # Extract the current text slice

            if chunk.strip():

                # Append a new record with metadata and the chunk content
                chunks.append(
                    {
                        "CELEX": row["CELEX"],
                        "chunk_id": f"{row['CELEX']}_{chunk_number}",
                        "chunk_number": chunk_number,
                        "chunk_text": chunk,
                    }
                )

            start += chunk_size - chunk_overlap     # Move the start pointer forward while preserving the configured overlap
            chunk_number += 1                       # Increment the counter for the next chunk identifier

    # Return all generated chunks as a DataFrame
    return pd.DataFrame(chunks)