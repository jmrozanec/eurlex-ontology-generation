import pandas as pd


"""
Create a manifest describing the batches of chunks to process.
"""
def create_chunk_batch_manifest(
    chunks: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:

    # Validate the batch size configuration
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    # Calculate total chunks and the total number of batches required
    number_of_chunks = len(chunks)
    number_of_batches = (
        number_of_chunks + batch_size - 1
    ) // batch_size

    # Initialize the manifest DataFrame with sequential batch IDs
    manifest = pd.DataFrame(
        {
            "batch_id": range(number_of_batches),
        }
    )

    # Calculate the zero-based starting row index for each batch
    manifest["start_row"] = (
        manifest["batch_id"] * batch_size
    )

    # Calculate the ending row index, capping it at the total number of chunks - 1
    manifest["end_row"] = (
        manifest["start_row"] + batch_size - 1
    ).clip(upper=number_of_chunks - 1)

    # Compute the actual number of chunks contained in each batch
    manifest["chunk_count"] = (
        manifest["end_row"]
        - manifest["start_row"]
        + 1
    )

    # Set the initial execution status for all batches. Every batch starts as pending
    manifest["status"] = "pending"

    return manifest