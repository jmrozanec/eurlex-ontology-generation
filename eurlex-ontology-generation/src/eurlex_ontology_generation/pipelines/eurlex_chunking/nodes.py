import pandas as pd


"""
Splits each EUR-Lex document into chunks while keeping the document's source information
"""
def chunk_corpus(
    corpus: pd.DataFrame,
    text_column: str,
    chunk_size: int,
    chunk_overlap: int,
) -> pd.DataFrame:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be less than chunk_size."
        )

    chunks = []

    step = chunk_size - chunk_overlap

    for _, row in corpus.iterrows():
        text = row[text_column]

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        start = 0
        chunk_number = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "CELEX": row["CELEX"],
                        "Act_name": row["Act_name"],
                        "Act_type": row["Act_type"],
                        "Subject_matter": row["Subject_matter"],
                        "Date_document": row["Date_document"],
                        "chunk_id": f"{row['CELEX']}_{chunk_number}",
                        "chunk_number": chunk_number,
                        "chunk_text": chunk_text,
                    }
                )

            start += step
            chunk_number += 1

    return pd.DataFrame(chunks)
