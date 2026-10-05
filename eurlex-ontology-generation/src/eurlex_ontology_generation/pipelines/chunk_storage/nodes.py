from pathlib import Path
import pandas as pd


"""
Partition the full chunk dataset into smaller CSV files.

Each partition contains the chunks belonging to a fixed number of consecutive processing batches.
"""
def partition_chunks(
    chunks: pd.DataFrame,
    batch_group_size: int,
    batch_size: int,
    output_directory: str,
) -> pd.DataFrame:    

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if batch_group_size <= 0:
        raise ValueError("batch_group_size must be greater than zero.")

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_per_part = batch_size * batch_group_size

    partition_manifest = []

    for part_id, start_row in enumerate(
        range(0, len(chunks), rows_per_part)
    ):
        end_row = min(
            start_row + rows_per_part,
            len(chunks),
        )

        partition = chunks.iloc[start_row:end_row].copy()

        file_path = (
            output_path
            / f"part_{part_id:03d}.csv"
        )

        partition.to_csv(
            file_path,
            index=False,
            encoding="utf-8",
        )

        partition_manifest.append(
            {
                "part_id": part_id,
                "start_row": start_row,
                "end_row": end_row - 1,
                "chunk_count": len(partition),
                "file_path": str(file_path),
            }
        )

    return pd.DataFrame(partition_manifest)
