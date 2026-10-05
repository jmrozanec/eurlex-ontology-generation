import os
from pathlib import Path

import pandas as pd
from filelock import FileLock


# Append rows to a shared CSV, deduplicating by key. Safe with concurrent processes.
def append_dedup_csv(output_path, new_rows: pd.DataFrame, key: str = "chunk_id") -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(output_path) + ".lock"):
        if output_path.exists():
            existing = pd.read_csv(output_path, encoding="utf-8")
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=[key], keep="last")
        else:
            combined = new_rows.copy()

        # Atomic write: readers never see a half-written file
        tmp_path = output_path.with_suffix(".csv.tmp")
        combined.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, output_path)

    return str(output_path)