"""Collect the results of several Colab notebooks into the main data folder.

Each notebook saves its results in its own folder (runs/nb_A, runs/nb_B, ...)
using the --sync-dir option of run_ontology_batches.py. This script merges
those folders back into the main data folder.
"""
import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

# Persistent output folders (inside data/04_feature) that contain part_XXX.csv files
SUBDIRS = [
    "ontology_candidates",
    "ontology_reviews",
    "ontology_revisions",
    "ontology_revision_reports",
]

# Higher rank wins when the same batch has a different status in different runs.
# Any other status (pending, running) has rank 0.
STATUS_RANK = {"completed": 2, "failed": 1}

MANIFEST_NAME = "chunk_batch_manifest.csv"


# Write a CSV atomically, so the file is never left half-written
def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, path)


# Keep only run folders that contain a manifest, ordered from the oldest to the newest.
# With drop_duplicates(keep="last"), the most recent run wins on duplicated chunk_id.
def find_run_dirs(runs_dir: Path) -> list[Path]:
    run_dirs = []

    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if (path / MANIFEST_NAME).exists():
            run_dirs.append(path)
        else:
            print(f"[warning] {path.name}: no manifest found, run skipped")

    return sorted(run_dirs, key=lambda p: (p / MANIFEST_NAME).stat().st_mtime)


# Copy the current data folder to a timestamped backup before touching anything
def backup_data(data_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir.parent / f"data_backup_{stamp}"
    backup_dir.mkdir(parents=True)

    shutil.copytree(
        data_dir / "04_feature",
        backup_dir / "04_feature",
        dirs_exist_ok=True,
    )
    shutil.copy2(data_dir / "03_primary" / MANIFEST_NAME, backup_dir)

    print(f"Backup saved in {backup_dir}")


# Merge every part_XXX.csv of the main folder with the ones found in the runs
def merge_part_files(data_dir: Path, run_dirs: list[Path]) -> None:
    for sub in SUBDIRS:
        # The main folder comes first, then the runs from the oldest to the newest
        sources = [data_dir / "04_feature" / sub] + [
            run / "04_feature" / sub for run in run_dirs
        ]

        # Collect the names of all partition files found in any source
        part_names = set()
        for source in sources:
            if source.exists():
                part_names.update(p.name for p in source.glob("part_*.csv"))

        for name in sorted(part_names):
            frames = [
                pd.read_csv(source / name, encoding="utf-8")
                for source in sources
                if (source / name).exists()
            ]

            combined = pd.concat(frames, ignore_index=True)
            rows_before = len(combined)

            # Same rule used by the persist_* functions of the pipelines
            combined = combined.drop_duplicates(subset=["chunk_id"], keep="last")

            atomic_to_csv(combined, data_dir / "04_feature" / sub / name)
            print(f"{sub}/{name}: {rows_before} rows -> {len(combined)} after dedup")


# Update the status of each batch in the main manifest using the manifests of the runs
def merge_manifests(data_dir: Path, run_dirs: list[Path]) -> None:
    manifest_path = data_dir / "03_primary" / MANIFEST_NAME
    base = pd.read_csv(manifest_path, encoding="utf-8")
    status = dict(zip(base["batch_id"], base["status"]))

    for run in run_dirs:
        run_manifest = pd.read_csv(run / MANIFEST_NAME, encoding="utf-8")

        for batch_id, run_status in zip(run_manifest["batch_id"], run_manifest["status"]):
            # A better status never gets overwritten by a worse one
            if STATUS_RANK.get(run_status, 0) > STATUS_RANK.get(status.get(batch_id), 0):
                status[batch_id] = run_status

    base["status"] = base["batch_id"].map(status)
    atomic_to_csv(base, manifest_path)

    print("Manifest status counts:", base["status"].value_counts().to_dict())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the results of several Colab runs into the main data folder."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Main data folder (contains 03_primary and 04_feature).",
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Folder containing the run folders (nb_A, nb_B, ...).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the backup of the main data folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    data_dir = Path(args.data_dir)
    runs_dir = Path(args.runs_dir)

    if not (data_dir / "03_primary" / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"Main manifest not found in {data_dir / '03_primary'}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs folder not found: {runs_dir}")

    run_dirs = find_run_dirs(runs_dir)
    if not run_dirs:
        print("No valid run folders found. Nothing to merge.")
        return

    print("Runs to merge (oldest to newest):", [r.name for r in run_dirs])

    if not args.no_backup:
        backup_data(data_dir)

    merge_part_files(data_dir, run_dirs)
    merge_manifests(data_dir, run_dirs)

    print("Done.")


if __name__ == "__main__":
    main()