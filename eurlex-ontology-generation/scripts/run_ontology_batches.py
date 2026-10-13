from pathlib import Path
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


MANIFEST_PATH = Path(
    "data/03_primary/chunk_batch_manifest.csv"
)

# Protects the manifest when several batches run at the same time
manifest_lock = threading.Lock()

# Makes sure only one thread at a time copies results to Drive (to run batches in different notebook in Colab)
sync_lock = threading.Lock()


# Load the persistent batch manifest
def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Batch manifest not found: {MANIFEST_PATH}"
        )

    return pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8",
    )


# Atomic write, so the file is never left half-written. Must be called with manifest_lock held
def save_manifest(manifest: pd.DataFrame) -> None:

    tmp_path = MANIFEST_PATH.with_suffix(".csv.tmp")
    manifest.to_csv(
        tmp_path, 
        index=False, 
        encoding="utf-8"
    )
    os.replace(tmp_path, MANIFEST_PATH)


# Takes the lock before touching the manifest
def update_status(
    manifest: pd.DataFrame,
    batch_id: int,
    status: str,
) -> None:

    with manifest_lock:
        manifest.loc[
            manifest["batch_id"] == batch_id,
            "status",
        ] = status

        save_manifest(manifest)


# Atomically pick the next batch and mark it as running, so two workers never take the same batch
def claim_next_batch(
    manifest: pd.DataFrame,
    state: dict,
    start_batch: int | None,
    end_batch: int | None,
):

    with manifest_lock:
        if state["left"] <= 0:
            return None

        mask = (
            manifest["status"].isin(["pending", "failed"])
            & ~manifest["batch_id"].isin(state["attempted"])
        )
        if start_batch is not None:
            mask &= manifest["batch_id"] >= start_batch
        if end_batch is not None:
            mask &= manifest["batch_id"] <= end_batch

        candidates = manifest[mask]

        if candidates.empty:
            return None

        row = candidates.iloc[0]
        batch_id = int(row["batch_id"])

        state["left"] -= 1
        # a batch that failed in this run is not retried in the same run
        state["attempted"].add(batch_id)

        manifest.loc[
            manifest["batch_id"] == batch_id,
            "status",
        ] = "running"
        save_manifest(manifest)

        return batch_id, int(row["chunk_count"])


# Copy the results of this session to a Drive folder (to run batches in different notebook in Colab).
# Called after every batch, so a Colab disconnection loses very little work.
def sync_to_drive(sync_dir: str | None) -> None:
    # Nothing to do if --sync-dir was not given
    if not sync_dir:
        return

    dest = Path(sync_dir)
    (dest / "04_feature").mkdir(parents=True, exist_ok=True)
    (dest / "logs").mkdir(parents=True, exist_ok=True)

    # Do not copy lock files and temporary files
    excludes = ["--exclude", "*.lock", "--exclude", "*.tmp"]
    commands = [
        ["rsync", "-a", *excludes, "data/04_feature/", f"{dest}/04_feature/"],
        ["rsync", "-a", "data/03_primary/chunk_batch_manifest.csv", f"{dest}/"],
        ["rsync", "-a", "logs/", f"{dest}/logs/"],
    ]

    with sync_lock:
        for cmd in commands:
            try:
                subprocess.run(cmd, check=False)
            except Exception as exc:
                print(f"[sync] warning: {exc}", flush=True)


# Execute one Kedro pipeline. The output goes to a log file per batch/pipeline
def run_pipeline(
    pipeline_name: str,
    parameter_namespace: str,
    batch_id: int,
) -> int:

    command = [
        "uv",
        "run",
        "--no-sync",
        "kedro",
        "run",
        "--pipelines",
        pipeline_name,
        "--params",
        f"{parameter_namespace}.batch_id={batch_id}",
    ]

    log_path = (
        Path("logs/batches")
        / f"batch_{batch_id:05d}_{pipeline_name}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[batch {batch_id}] start {pipeline_name}",
        flush=True,
    )

    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(
        f"[batch {batch_id}] end {pipeline_name} "
        f"(exit {result.returncode})",
        flush=True,
    )

    return result.returncode


"""
Check that ontology review produced a valid result for the batch.

The review pipeline can finish with exit code 0 even when the LLM
review itself failed. Therefore the persistent output must also
be inspected.
"""
def validate_review_output(
    batch_id: int,
    expected_count: int,
) -> bool:

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_reviews")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Review output not found: {output_path}"
        )
        return False

    reviews = pd.read_csv(
        output_path,
        encoding="utf-8",
    )

    batch_reviews = reviews[
        reviews["batch_id"] == batch_id
    ]

    if len(batch_reviews) != expected_count:
        print(
            f"Review output contains {len(batch_reviews)} rows, "
            f"expected {expected_count}."
        )
        return False

    failed_reviews = batch_reviews[
        batch_reviews["status"].isin(
            ["error"]
        )
    ]

    if not failed_reviews.empty:
        print(
            f"Review failed for "
            f"{len(failed_reviews)} chunk(s)."
        )
        return False

    return True


"""
Check that ontology revision produced a valid result for the batch.

A revision fallback means that the LLM revision failed, so the
batch must not be marked as completed.
"""
def validate_revision_output(
    batch_id: int,
    expected_count: int,
) -> bool:    

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_revisions")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Revision output not found: {output_path}"
        )
        return False

    revisions = pd.read_csv(
        output_path,
        encoding="utf-8",
    )

    batch_revisions = revisions[
        revisions["batch_id"] == batch_id
    ]

    if len(batch_revisions) != expected_count:
        print(
            f"Revision output contains "
            f"{len(batch_revisions)} rows, "
            f"expected {expected_count}."
        )
        return False

    failed_revisions = batch_revisions[
        batch_revisions["revision_status"].astype(str).str.startswith(
            "failed_"
        )
    ]

    if not failed_revisions.empty:
        print(
            f"Revision failed for "
            f"{len(failed_revisions)} chunk(s)."
        )
        return False

    return True


# Execute the complete ontology-processing pipeline for one batch
def run_batch(
    batch_id: int,
    expected_count: int,
) -> bool:   

    # Step 1: Generate candidate ontologies.
    return_code = run_pipeline(
        pipeline_name="ontology_generation",
        parameter_namespace="ontology_generation",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology generation failed for batch {batch_id}."
        )
        return False

    # Step 2: Review generated ontologies.
    return_code = run_pipeline(
        pipeline_name="ontology_review",
        parameter_namespace="ontology_review",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology review pipeline failed for batch {batch_id}."
        )
        return False

    # The review pipeline may exit successfully even if the LLM judge
    # returned an error for one or more chunks.
    if not validate_review_output(
        batch_id=batch_id,
        expected_count=expected_count,
    ):
        print(
            f"Ontology review validation failed for batch {batch_id}."
        )
        return False

    # Step 3: Revise ontologies that require revision.
    return_code = run_pipeline(
        pipeline_name="ontology_revision",
        parameter_namespace="ontology_revision",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology revision pipeline failed for batch {batch_id}."
        )
        return False

    # The revision pipeline can keep the original ontology when
    # the LLM output is truncated or invalid. Detect that explicitly.
    if not validate_revision_output(
        batch_id=batch_id,
        expected_count=expected_count,
    ):
        print(
            f"Ontology revision validation failed for batch {batch_id}."
        )
        return False

    return True


"""
Convert interrupted 'running' batches back to 'pending'.

This allows the script to resume after an unexpected interruption.
"""
def recover_interrupted_batches(
    manifest: pd.DataFrame,
) -> pd.DataFrame:    

    running_mask = manifest["status"] == "running"

    if running_mask.any():
        recovered_count = int(running_mask.sum())

        print(
            f"Recovering {recovered_count} interrupted batch(es)."
        )

        manifest.loc[
            running_mask,
            "status",
        ] = "pending"

        save_manifest(manifest)

    return manifest


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run pending ontology-processing batches "
            "with checkpoint support."
        )
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum number of batches to process in this execution.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of batches processed at the same time.",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=None,
        help="Only process batches with batch_id >= this value.",
    )
    parser.add_argument(
        "--end-batch",
        type=int,
        default=None,
        help="Only process batches with batch_id <= this value.",
    )
    parser.add_argument(
        "--sync-dir", 
        type=str, 
        default=None,
        help="Drive folder where results are copied after every batch.",
    )

    return parser.parse_args()


# Process pending ontology batches sequentially or in parallel
def main() -> None:

    args = parse_arguments()

    if args.max_batches <= 0 or args.parallel <= 0:
        raise ValueError(
            "--max-batches and --parallel must be greater than zero."
        )

    manifest = load_manifest()
    manifest = recover_interrupted_batches(manifest)

    state = {
        "left": args.max_batches,
        "attempted": set(),
        "done": [],
        "failed": [],
    }

    def worker() -> None:
        while True:
            claimed = claim_next_batch(
                manifest, 
                state, 
                args.start_batch, 
                args.end_batch
            )

            if claimed is None:
                return

            batch_id, expected_count = claimed

            try:
                ok = run_batch(
                    batch_id=batch_id,
                    expected_count=expected_count,
                )
            except Exception as exc:
                # An unexpected error in one batch must not kill the worker
                print(f"[batch {batch_id}] exception: {exc}", flush=True)
                ok = False

            update_status(
                manifest=manifest,
                batch_id=batch_id,
                status="completed" if ok else "failed",
            )

            with manifest_lock:
                state["done" if ok else "failed"].append(batch_id)

            print(
                f"[batch {batch_id}] "
                f"{'completed' if ok else 'FAILED'}",
                flush=True,
            )

            # Copy results to Drive right after each batch
            sync_to_drive(args.sync_dir)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(worker) for _ in range(args.parallel)]
        for future in futures:
            future.result()

    # Final sync, in case something changed after the last batch
    sync_to_drive(args.sync_dir)

    print()
    print(f"Completed: {sorted(state['done'])}")
    print(f"Failed:    {sorted(state['failed'])}")

    sys.exit(1 if state["failed"] else 0)


if __name__ == "__main__":
    main()