"""Orchestrates the merge pipelines end to end:

1. Scans data/04_feature/ontology_revisions/ for every part_nnn.csv
   that already exists (i.e. every partition for which at least one
   chunk has completed generation -> review -> revision).
2. Runs the ontology_merge_partition pipeline for every part_id that
   does not yet have a corresponding merged JSON in
   data/05_model_input/ontology_merges_partition/ (skips it otherwise,
   unless --force is passed).
3. Optionally runs ontology_merge_global once at the end, so the
   final ontology reflects every partition merged so far.

This mirrors the checkpoint/resume pattern already used by
scripts/run_ontology_batches.py, but at the partition level instead
of the batch level.
"""
from pathlib import Path
import argparse
import re
import subprocess
import sys

REVISIONS_DIR = Path("data/04_feature/ontology_revisions")
PARTITION_MERGES_DIR = Path("data/05_model_input/ontology_merges_partition")

PART_FILE_PATTERN = re.compile(r"part_(\d{3})\.csv$")


# List every part_id for which a revisions file exists, sorted.
def discover_available_part_ids() -> list[int]:
    if not REVISIONS_DIR.exists():
        raise FileNotFoundError(f"Revisions directory not found: {REVISIONS_DIR}")

    part_ids = []
    for path in REVISIONS_DIR.glob("part_*.csv"):
        match = PART_FILE_PATTERN.search(path.name)
        if match:
            part_ids.append(int(match.group(1)))

    return sorted(part_ids)


# True if a merged ontology already exists for this part_id.
def is_partition_already_merged(part_id: int) -> bool:
    return (PARTITION_MERGES_DIR / f"part_{part_id:03d}.json").exists()


# Run the ontology_merge_partition pipeline for a single part_id.
def run_partition_merge(part_id: int) -> int:
    command = [
        "uv", "run", "kedro", "run",
        "--pipelines", "ontology_merge_partition",
        "--params", f"ontology_merge_partition.part_id={part_id}",
    ]

    print()
    print("-" * 70)
    print(f"Merging partition {part_id:03d}")
    print("-" * 70)

    result = subprocess.run(command, check=False)
    return result.returncode


# Run the ontology_merge_global pipeline once.
def run_global_merge() -> int:
    command = ["uv", "run", "kedro", "run", "--pipelines", "ontology_merge_global"]

    print()
    print("=" * 70)
    print("Running global merge")
    print("=" * 70)

    result = subprocess.run(command, check=False)
    return result.returncode


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge every available ontology_revisions partition, then (optionally) the global ontology."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the partition merge even if a merged JSON already exists for it.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Do not run ontology_merge_global at the end (only merge partitions).",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Process at most this many partitions in this run (useful for incremental testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    part_ids = discover_available_part_ids()
    if not part_ids:
        print("No revisions partitions found yet. Nothing to merge.")
        return

    to_process = [
        pid for pid in part_ids
        if args.force or not is_partition_already_merged(pid)
    ]

    if args.max_partitions is not None:
        to_process = to_process[: args.max_partitions]

    if not to_process:
        print("Every available partition is already merged. Use --force to redo them.")
    else:
        print(f"Found {len(part_ids)} revisions partition(s) on disk; "
              f"{len(to_process)} to merge in this run.")

    for part_id in to_process:
        return_code = run_partition_merge(part_id)
        if return_code != 0:
            print(f"Partition merge failed for part_id={part_id}. Stopping.")
            sys.exit(1)

    print()
    print(f"Finished. Merged {len(to_process)} partition(s).")

    if not args.skip_global:
        return_code = run_global_merge()
        if return_code != 0:
            print("Global merge failed.")
            sys.exit(1)
        print("Global merge completed.")


if __name__ == "__main__":
    main()