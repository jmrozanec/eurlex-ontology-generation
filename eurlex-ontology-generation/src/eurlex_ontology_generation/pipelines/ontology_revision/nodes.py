import json
import logging
import re
from pathlib import Path
import pandas as pd

from ...utils.llm_client import llm_chat, run_parallel
from ...utils.ontology_normalize import normalize_chunk_ontology
from ...utils.ontology_schema import build_ontology_model
from ...utils.persist import append_dedup_csv
from ...utils.vocabulary import load_vocabulary, render_prompt

logger = logging.getLogger(__name__)


# Save the raw text of a failed LLM attempt, to see where a runaway generation starts
def _dump_failed_response(chunk_id, attempt: int, content) -> None:
    try:
        directory = Path("logs/batches/failed_revisions")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{chunk_id}_attempt{attempt + 1}.txt").write_text(
            content or "", encoding="utf-8"
        )
    except OSError:
        pass


# The revision output is validated against the same vocabulary-driven schema
# as generation (utils/ontology_schema.py): one definition for both steps.



# Calculate the persistent group identifier for a batch
def get_batch_group_id(
    batch_id: int,
    batch_group_size: int,
) -> int:    

    if batch_id < 0:
        raise ValueError(
            "batch_id must be greater than or equal to zero."
        )

    if batch_group_size <= 0:
        raise ValueError(
            "batch_group_size must be greater than zero."
        )

    return batch_id // batch_group_size


# Load only the ontology reviews belonging to one batch
def load_ontology_reviews(
    batch_id: int,
    batch_group_size: int,
    reviews_directory: str,
    review_round: int = 0,
) -> pd.DataFrame:    

    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    reviews_path = (
        Path(reviews_directory)
        / f"part_{group_id:03d}.csv"
    )

    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Ontology reviews file not found: "
            f"{reviews_path}"
        )

    reviews = pd.read_csv(
        reviews_path,
        encoding="utf-8",
    )

    batch_reviews = reviews[
        reviews["batch_id"] == batch_id
    ].copy()

    # The reviews file keeps every round, including rows left by earlier runs of the
    # same batch. A revision of round N must use the review of round N-1 and nothing
    # else: taking "the latest round" silently picks up stale rows from a previous run.
    if "round" not in batch_reviews.columns:
        batch_reviews["round"] = 0
    batch_reviews["round"] = batch_reviews["round"].fillna(0).astype(int)
    batch_reviews = (
        batch_reviews[batch_reviews["round"] == review_round]
        .drop_duplicates(subset=["chunk_id"], keep="last")
    )

    batch_reviews["revision_required"] = (
        batch_reviews["revision_required"]
        .astype(str).str.strip().str.lower().eq("true")
    )

    if batch_reviews.empty:
        raise ValueError(
            f"No ontology reviews found for batch "
            f"{batch_id}, review round {review_round}, in {reviews_path}."
        )

    return batch_reviews.reset_index(drop=True)


# Applies feedback-driven LLM revisions to target candidate ontologies in parallel
def revise_ontologies(
    batch_id: int,
    batch_group_size: int,
    reviews_directory: str,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
    expected_ontology_keys: list,
    num_ctx: int,
    num_predict: int,
    llm_backend: dict,
    revision_round: int = 1,
    normalize_after_revision: bool = True,
    vocabulary_path: str | None = None,
    max_retries: int = 2,
) -> pd.DataFrame:

    rendered_system_prompt = render_prompt(system_prompt, vocabulary_path)
    schema_model = build_ontology_model(vocabulary_path) if json_mode else None

    def _normalize(ontology: dict, text: str | None = None,
                   original: dict | None = None) -> tuple[dict, list]:
        if not normalize_after_revision:
            return ontology, []
        return normalize_chunk_ontology(
            ontology, load_vocabulary(vocabulary_path), source_text=text, original=original
        )

    ontology_reviews = load_ontology_reviews(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
        reviews_directory=reviews_directory,
        review_round=revision_round - 1,
    )

    # Helper closure to process and revise a single ontology row
    def _revise_row(row) -> dict:
        base = {
            "batch_id": batch_id,
            "chunk_id": row["chunk_id"],
            "round": revision_round,
            "CELEX": row["CELEX"],
            "text": row["text"],
        }

        if not row["revision_required"]:
            try:
                kept, changes = _normalize(json.loads(row["ontology"]), row["text"])
                ontology = json.dumps(kept, ensure_ascii=False)
            except (TypeError, json.JSONDecodeError):
                ontology, changes = row["ontology"], []
            return {**base, "ontology": ontology, "revised": False,
                    "revision_status": "skipped_not_required",
                    "approved_in_round": int(row.get("round", 0) or 0),
                    "normalization_changes": json.dumps(changes, ensure_ascii=False)}

        user_prompt = render_prompt(
            user_prompt_template, vocabulary_path,
            ontology=row["ontology"], text=row["text"], review=row["review"],
        )

        revised_ontology_json, used_fallback, response = None, True, None
        for attempt in range(max_retries + 1):
            # at temperature 0 a generation loop repeats itself identically:
            # warm up a little on every retry
            attempt_temperature = (
                temperature if attempt == 0 else min(temperature + 0.3 * attempt, 0.9)
            )
            response = llm_chat(
                llm_backend, model, rendered_system_prompt, user_prompt, attempt_temperature,
                schema_model=schema_model,
                num_ctx=num_ctx,
                max_tokens=num_predict,
            )
            parsed, failed = extract_json(response.content, row["ontology"])
            if not failed and parsed.get("classes"):
                revised_ontology_json, used_fallback = parsed, False
                break
            logger.warning(
                "chunk %s: revision attempt %d/%d failed (done_reason=%s, %d chars, head=%r, tail=%r)",
                row["chunk_id"], attempt + 1, max_retries + 1, response.done_reason,
                len(response.content or ""), (response.content or "")[:300],
                (response.content or "")[-200:],
            )
            _dump_failed_response(row["chunk_id"], attempt, response.content)
            # A runaway generation (hit num_predict) did not recover with a higher
            # temperature (chunks 14/15 of batch 46 failed 3/3): retrying only burns minutes.
            if response.done_reason == "length":
                break
        if used_fallback:
            revised_ontology_json = _build_fallback_json(row["ontology"])
        done_reason = response.done_reason

        revised_ontology_json, had_extra_keys = _strip_unexpected_keys(
            revised_ontology_json, expected_ontology_keys
        )
        revised_ontology_json = _sync_extensions_with_classes(revised_ontology_json)

        # Deterministic fixes the LLM is unreliable at (typing, leakage,
        # vocabulary, referential closure). Recorded per chunk.
        # Normalise also the fallback (original) ontology: a failed LLM revision must
        # still drop values that have no support in the source text.
        # `original` (the ontology that was reviewed) is used to restore fields the LLM dropped, e.g. `provision`.
        revised_ontology_json, changes = _normalize(
            revised_ontology_json, row["text"], _safe_load(row["ontology"])
        )

        if used_fallback:
            status = ("failed_truncated_output" if done_reason == "length"
                      else "failed_fallback_original_kept")
        elif had_extra_keys:
            status = "revised_with_extra_keys_stripped"
        else:
            status = "revised"

        return {
            **base,
            "ontology": json.dumps(revised_ontology_json, ensure_ascii=False),
            "revised": not used_fallback,
            "revision_status": status,
            "approved_in_round": None,
            "normalization_changes": json.dumps(changes, ensure_ascii=False),
        }

    rows = [row for _, row in ontology_reviews.iterrows()]
    results = run_parallel(_revise_row, rows, llm_backend.get("max_workers", 1))
    
    return pd.DataFrame(results)


# Append the current batch revisions to the persistent revision file
def persist_ontology_revisions(
    ontology_revised: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
) -> str:    

    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    output_path = Path(output_directory) / f"part_{group_id:03d}.csv"

    return append_dedup_csv(output_path, ontology_revised)


# Node 2: validate revised ontologies (schema + grounding checks)
"""Run lightweight structural and grounding checks on the revised
ontologies and attach a validation report to each row.

Checks performed:
- schema_valid: top-level keys match expected_ontology_keys exactly.
- invalid_class_type: classes whose "type" is neither a valid ODRL
    core type nor exactly "Legal_extension".
- extensions_without_class: values in odrl_alignment.extensions that
    have no matching "name" in "classes".
- classes_without_extension: classes of type "Legal_extension" not
    declared in odrl_alignment.extensions.
- possible_hallucination: currency terms found in the ontology that
    are not present in the source text, using word-boundary matching
    AND excluding known false-positive phrases (e.g. "EUR-Lex") before
    the check runs.
"""
def validate_revised_ontologies(
    ontology_revised: pd.DataFrame,
    expected_ontology_keys: list,
    disallowed_currency_terms: list,
    valid_odrl_core_types: list,
    currency_check_exclusions: list,
) -> pd.DataFrame:
    
    valid_types = set(valid_odrl_core_types) | {"Legal_extension"}

    reports = []

    for _, row in ontology_revised.iterrows():
        ontology = _safe_load(row["ontology"])
        source_text = row.get("text", "") or ""

        issues = []

        if ontology is None:
            issues.append("ontology_not_valid_json")
            reports.append(_build_report(row, issues, schema_valid=False))
            continue

        # Schema check
        actual_keys = set(ontology.keys())
        expected_keys = set(expected_ontology_keys)
        schema_valid = actual_keys == expected_keys
        if not schema_valid:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            if missing:
                issues.append(f"missing_keys:{sorted(missing)}")
            if extra:
                issues.append(f"unexpected_keys:{sorted(extra)}")

        classes = ontology.get("classes", [])
        if not isinstance(classes, list):
            classes = []

        # Class "type" validity check
        invalid_type_classes = [
            c.get("name")
            for c in classes
            if isinstance(c, dict) and c.get("type") not in valid_types
        ]
        if invalid_type_classes:
            issues.append(f"invalid_class_type:{sorted(n for n in invalid_type_classes if n)}")

        # Controlled-vocabulary checks (only meaningful after normalisation)
        vocab = load_vocabulary()
        leaked = sorted(c.get("name") for c in classes
                        if isinstance(c, dict) and vocab.metamodel_leakage(c.get("name") or ""))
        if leaked:
            issues.append(f"metamodel_leakage:{leaked}")
        bad_is_a = sorted(c.get("name") for c in classes
                          if isinstance(c, dict) and c.get("is_a") not in vocab.upper)
        if bad_is_a:
            issues.append(f"missing_or_invalid_is_a:{bad_is_a}")

        # Extensions <-> classes consistency check
        extensions = set(
            ontology.get("odrl_alignment", {}).get("extensions", []) or []
        )
        class_names = {c.get("name") for c in classes if isinstance(c, dict)}
        legal_extension_class_names = {
            c.get("name")
            for c in classes
            if isinstance(c, dict) and c.get("type") == "Legal_extension"
        }

        extensions_without_class = extensions - class_names
        classes_without_extension = legal_extension_class_names - extensions

        if extensions_without_class:
            issues.append(f"extensions_without_class:{sorted(extensions_without_class)}")
        if classes_without_extension:
            issues.append(f"classes_without_extension:{sorted(classes_without_extension)}")
        
        # Strip known false-positive phrases (e.g. "EUR-Lex", the EU legal
        # database name, which is NOT the Euro currency) from BOTH the
        # ontology text and the source text before matching, so mentions
        # like "EUR-Lex text provided" no longer trigger a false alarm
        # on the currency term "EUR".
        ontology_str = json.dumps(ontology, ensure_ascii=False)
        cleaned_ontology_str = _strip_exclusions(ontology_str, currency_check_exclusions)
        cleaned_source_text = _strip_exclusions(source_text, currency_check_exclusions)

        for term in disallowed_currency_terms:
            term_pattern = r"\b" + re.escape(term) + r"\b"
            term_in_ontology = re.search(term_pattern, cleaned_ontology_str) is not None
            term_in_source = re.search(term_pattern, cleaned_source_text) is not None
            if term_in_ontology and not term_in_source:
                issues.append(f"possible_hallucination:{term}_not_in_source_text")

        reports.append(_build_report(row, issues, schema_valid=schema_valid))

    return pd.DataFrame(reports)


def _build_report(row: pd.Series, issues: list, schema_valid: bool) -> dict:
    return {
        "batch_id": row.get("batch_id"),
        "chunk_id": row["chunk_id"],
        "CELEX": row["CELEX"],
        "revision_status": row.get("revision_status"),
        "schema_valid": schema_valid,
        "validation_passed": len(issues) == 0,
        "issues": json.dumps(issues, ensure_ascii=False),
    }


# Append the current batch validation reports to the persistent report file
def persist_validation_reports(
    validation_reports: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
) -> str:    

    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    output_path = Path(output_directory) / f"part_{group_id:03d}.csv"

    return append_dedup_csv(output_path, validation_reports)


# Helpers
def _safe_load(ontology_field) -> dict | None:
    """Load an ontology that may already be a dict or a JSON string."""
    if isinstance(ontology_field, dict):
        return ontology_field
    try:
        return json.loads(ontology_field)
    except (json.JSONDecodeError, TypeError):
        return None


def _strip_unexpected_keys(ontology: dict, expected_keys: list) -> tuple[dict, bool]:
    # Remove any top-level key not in expected_keys
    expected = set(expected_keys)
    extra_keys = [k for k in ontology.keys() if k not in expected]
    if not extra_keys:
        return ontology, False
    cleaned = {k: v for k, v in ontology.items() if k in expected}
    return cleaned, True


def _sync_extensions_with_classes(ontology: dict) -> dict:
    """Deterministically rebuild odrl_alignment.extensions from the NAMES
    of every class of type 'Legal_extension'. Rebuilding (rather than only
    adding) also drops stray/incorrect values the LLM may have put there
    (e.g. a class "type" used instead of its "name").
    """
    if not isinstance(ontology, dict):
        return ontology

    classes = ontology.get("classes", [])
    if not isinstance(classes, list):
        return ontology

    legal_ext_names = {
        c.get("name")
        for c in classes
        if isinstance(c, dict) and c.get("type") == "Legal_extension" and c.get("name")
    }

    odrl_alignment = ontology.get("odrl_alignment")
    if not isinstance(odrl_alignment, dict):
        odrl_alignment = {}
        ontology["odrl_alignment"] = odrl_alignment

    odrl_alignment["extensions"] = sorted(legal_ext_names)

    return ontology


def _strip_exclusions(text: str, exclusions: list) -> str:
    """Remove known false-positive phrases (case-insensitive) from a text
    before running substring/word-boundary checks on it. Used to prevent
    e.g. 'EUR-Lex' from being mistaken for the currency term 'EUR'.
    """
    cleaned = text
    for phrase in exclusions:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    return cleaned


# Helper to safely extract JSON from LLM output.
def extract_json(text: str, original_ontology: str) -> tuple[dict, bool]:

    parsed_data = None

    if not text or text.strip() == "":
        return _build_fallback_json(original_ontology), True

    cleaned_text = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"```$", "", cleaned_text, flags=re.MULTILINE).strip()

    try:
        parsed_data = json.loads(cleaned_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match:
            try:
                parsed_data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if not parsed_data or not isinstance(parsed_data, dict) or len(parsed_data) == 0:
        return _build_fallback_json(original_ontology), True

    return parsed_data, False


def _build_fallback_json(original_ontology: str) -> dict:

    try:
        fallback = json.loads(original_ontology)
        if isinstance(fallback, dict) and len(fallback) > 0:
            return fallback
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "ontology_name": "Unknown",
        "description": "Revision failed: LLM output was empty/invalid and original ontology could not be parsed.",
        "odrl_alignment": {
            "core_entities": ["Asset", "Party", "Permission", "Duty", "Constraint"],
            "extensions": [],
        },
        "classes": [],
        "relationships": [],
        "rights_model": {"permissions": [], "duties": []},
        "legal_patterns": [],
        "example_instances": [],
    }