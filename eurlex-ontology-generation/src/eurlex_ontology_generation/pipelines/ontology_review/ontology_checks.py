"""Deterministic checks and post-processing for the ontology review node.

Everything that can be verified mechanically is verified here, so the LLM
judge only has to handle the semantic part of the review.
"""

import json
import math
import re
from typing import Any

from ...utils.entity_linking import load_gazetteer
from ...utils.ontology_cleanup import action_ungrounded_reason, example_problem, party_far_from_action, rule_key
from ...utils.ontology_normalize import type_entity
from ...utils.vocabulary import OTHER_ACTION, load_vocabulary

# Keep this list aligned with the convention used in the ontology-generation prompt (see the system prompt of the judge)
ODRL_CORE_NAMES = {
    "Policy",
    "Rule",
    "Asset",
    "Party",
    "Action",
    "Permission",
    "Prohibition",
    "Duty",
    "Constraint",
}

# Generic party names: the generator filled a slot without finding a real party in the text
PLACEHOLDER_PARTIES = {
    "targetparty",
    "responsibleparty",
    "party",
    "unknownparty",
    "genericparty",
}

# Fields that hold modelling vocabulary, not statements taken from the text.
VOCABULARY_KEYS = {"relation", "pattern_name", "ontology_name", "description"}

# Finding codes that count against ODRL compliance.
ODRL_CODES = {
    "misclassified_core_class",   # legacy "ODRL_core" type
    "invalid_class_type",
    "metamodel_leakage",
    "gazetteer_type_mismatch",
    "rule_slot_type",
    "rights_model_empty",
    "placeholder_party",
    "undefined_entities",
}

# Severity 'info' findings are reported (and feed the metrics) but never
# affect scores or approval: they describe vocabulary drift that the
# deterministic post-processor fixes on its own.
INFO = "info"

# Words that mark a name as an actor (Party) or as a domain concept
# (Legal_extension). Domain words win when both appear
# (e.g. EuropeanUnionSolidarityFund is a fund, not an actor).
_ACTOR_WORDS = {
    "parliament", "council", "commission", "union", "state", "states",
    "authority", "authorities", "agency", "bank", "court", "committee",
    "institution", "institutions", "government", "member",
}
_DOMAIN_WORDS = {
    "fund", "instrument", "decision", "regulation", "programme", "program",
    "measures", "year", "period", "budget", "appropriations", "heading",
}

# Pattern matching claims of empty structures
_EMPTY_CLAIM = re.compile(
    r"\bempty\b|\blacks? any\b|\bwithout any\b",
    re.IGNORECASE,
)
# Claim phrasing typical of what the automatic checks already say. Used to
# tell "this weakness just restates an automatic finding" (discard it) apart
# from "this weakness happens to quote the same name but makes a different
# point" (keep it) - e.g. "X is undefined" vs "X is assigned the wrong role".
_UNDEFINED_ENTITY_CLAIM = re.compile(
    r"not defined|undefined|not declared|no.{0,15}class|ambigu",
    re.IGNORECASE,
)
_MISCLASSIFIED_CLAIM = re.compile(
    r"odrl_core|legal_extension|incorrectly typed|should be (a |an )?(party|legal_extension)",
    re.IGNORECASE,
)
_PLACEHOLDER_CLAIM = re.compile(
    r"placeholder|generic party|not (a party )?named|not referenced in the (source )?text",
    re.IGNORECASE,
)
_UNGROUNDED_ACTION_CLAIM = re.compile(
    r"unsupported action|hallucinat|no support|does not appear|not (mentioned|present)",
    re.IGNORECASE,
)
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_WORD = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")



# Normalize strings for fuzzy string comparison by lowercasing and stripping whitespace/separators
def _norm(value: Any) -> str:
    """Lowercase and drop whitespace, commas and underscores.

    'EUR 178,715,475' matches 'EUR 178 715 475' and
    '1_January_2019' matches '1 January 2019'.
    """
    return re.sub(r"[\s,_]+", "", str(value).lower())


# Helper returning a list filtered exclusively for dictionary elements
def _dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# Determine candidate entity type (Party vs Legal_extension) using keyword heuristics
def _guess_entity_type(name: str) -> str:
    # Heuristic: 'Party' for actors, 'Legal_extension' for everything else
    words = {w.lower() for w in _WORD.findall(str(name))}
    if words & _DOMAIN_WORDS:
        return "Legal_extension"
    if words & _ACTOR_WORDS:
        return "Party"
    return "Legal_extension"


# Construct a finding dictionary containing error code, human message, and optional fix information
def _finding(
    code: str,
    severity: str,
    message: str,
    fix: str | None = None,
    data: dict | None = None,
) -> dict:
    """'fix' is a human-readable amendment, 'data' its machine-readable form
    (for the deterministic correction step)."""
    finding = {"code": code, "severity": severity, "message": message}
    if fix:
        finding["fix"] = fix
    if data:
        finding["data"] = data
    return finding


_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


"""
True if the value appears in the text, also as an ISO date
(2019-01-01 matches '1 January 2019' / 'January 1, 2019')
"""
def _value_in_text(value: Any, text_norm: str) -> bool:
    
    raw = str(value).strip()
    if _norm(raw) in text_norm:
        return True
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        year, month, day = match.groups()
        if 1 <= int(month) <= 12:
            name = _MONTHS[int(month) - 1]
            variants = [f"{int(day)} {name} {year}", f"{name} {int(day)} {year}"]
            return any(_norm(v) in text_norm for v in variants)
    return False


# Yield all permissions, prohibitions, and duties present in the rights model
def _iter_rules(onto: dict):
    rights_model = onto.get("rights_model")
    if not isinstance(rights_model, dict):
        return
    for kind in ("permissions", "prohibitions", "duties"):
        for rule in _dicts(rights_model.get(kind)):
            yield kind, rule


# Collect all entity names referenced across relationships, rules, patterns, and examples
def _collect_entity_references(onto: dict) -> set[str]:
    refs: set[Any] = set()

    for rel in _dicts(onto.get("relationships")):
        refs.update([rel.get("source"), rel.get("target")])

    for _, rule in _iter_rules(onto):
        refs.update([rule.get("asset"), rule.get("party"), rule.get("assigner")])

    for pattern in _dicts(onto.get("legal_patterns")):
        involved = pattern.get("entities_involved")
        if isinstance(involved, list):
            refs.update(involved)

    for instance in _dicts(onto.get("example_instances")):
        refs.add(instance.get("entity"))

    return {ref for ref in refs if isinstance(ref, str) and ref}


"""Parse JSON and perform deterministic checks on ODRL schema structure and text grounding.

Returns:
    Tuple of (parsed dictionary or None on JSON error, list of finding dicts).
"""
def run_deterministic_checks(
    ontology_json: str,
    text: str,
) -> tuple[dict | None, list[dict]]:

    try:
        onto = json.loads(ontology_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, [
            _finding(
                "ontology_json_invalid",
                "error",
                f"Candidate ontology is not valid JSON: {exc}",
            )
        ]

    if not isinstance(onto, dict):
        return None, [
            _finding(
                "ontology_json_invalid",
                "error",
                "Candidate ontology is not a JSON object.",
            )
        ]

    findings: list[dict] = []
    text = text or ""
    text_norm = _norm(text)
    text_lower = text.lower()

    classes = _dicts(onto.get("classes"))

    # Names that count as "defined": classes + ODRL alignment entries
    defined = {c.get("name") for c in classes if c.get("name")}
    alignment = onto.get("odrl_alignment")
    if isinstance(alignment, dict):
        for key in ("core_entities", "extensions"):
            values = alignment.get(key)
            if isinstance(values, list):
                defined.update(v for v in values if isinstance(v, str))

    # 1. Empty rights model
    if not list(_iter_rules(onto)):
        findings.append(
            _finding(
                "rights_model_empty",
                "warning",
                "rights_model contains no permissions, prohibitions or duties.",
                "Add the permissions, prohibitions or duties expressed in the text to rights_model.",
            )
        )

    # 2. Entities referenced but never defined (placeholder parties are
    #    reported by their own finding, so they are not listed here)
    refs = _collect_entity_references(onto)
    placeholders = {r for r in refs if r.lower().replace("_", "") in PLACEHOLDER_PARTIES}
    undefined = sorted(refs - defined - placeholders)
    if undefined:
        typed = [{"name": n, "type": _guess_entity_type(n)} for n in undefined]
        findings.append(
            _finding(
                "undefined_entities",
                "warning",
                "Entities used in relationships/rights_model/patterns/examples "
                "but not defined in classes or odrl_alignment: "
                + ", ".join(undefined),
                "Define these classes: "
                + ", ".join(f"{t['name']} (type {t['type']})" for t in typed)
                + ".",
                data={"classes_to_add": typed},
            )
        )

    # 3. Class-level checks against the controlled vocabulary and gazetteer
    vocab = load_vocabulary()
    gazetteer = load_gazetteer()
    class_type = {}
    for cls in classes:
        name = cls.get("name")
        if not isinstance(name, str) or not name:
            continue
        declared = cls.get("type")
        class_type[name] = declared

        leak = vocab.metamodel_leakage(name)
        if leak:
            findings.append(_finding(
                "metamodel_leakage", "warning",
                f"Class {leak}: ODRL structure must be expressed as a rule, not as a class.",
                f"Remove class '{name}' and express it as a permission, prohibition or duty in rights_model.",
                data={"class": name},
            ))
            continue

        expected = type_entity(name, str(cls.get("description") or ""), vocab, gazetteer,
                               declared, cls.get("is_a"))
        if declared == "ODRL_core" and name not in ODRL_CORE_NAMES:
            findings.append(_finding(
                "misclassified_core_class", "warning",
                f"Class '{name}' is typed ODRL_core but is not an ODRL "
                f"core concept (expected {expected['type']}).",
                f"Change the type of class '{name}' to {expected['type']}.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"]},
            ))
        elif declared not in vocab.core_types:
            findings.append(_finding(
                "invalid_class_type", "warning",
                f"Class '{name}' has type '{declared}', which is not one of {vocab.core_types} "
                f"(expected {expected['type']}).",
                f"Change the type of class '{name}' to {expected['type']}.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"]},
            ))
        elif expected.get("type_source") == "gazetteer" and (
                declared != expected["type"] or cls.get("kind") == "class"):
            findings.append(_finding(
                "gazetteer_type_mismatch", "warning",
                f"'{name}' is a known {expected['is_a']} individual (type {expected['type']}), "
                f"but the ontology declares type {declared}, kind {cls.get('kind')}.",
                f"Set '{name}' to type {expected['type']}, is_a {expected['is_a']}, kind individual.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"],
                      "kind": "individual"},
            ))
        if cls.get("is_a") is None or cls.get("kind") is None:
            findings.append(_finding(
                "missing_is_a_or_kind", INFO,
                f"Class '{name}' has no is_a/kind (legacy schema).",
            ))

    # 4. Rule-level checks: placeholders, ungrounded actions and values
    for kind, rule in _iter_rules(onto):
        party = rule.get("party")
        if isinstance(party, str) and party.lower().replace("_", "") in PLACEHOLDER_PARTIES:
            findings.append(
                _finding(
                    "placeholder_party",
                    "error",
                    f"{kind}: party '{party}' is a generic placeholder, "
                    "not a party named in the text.",
                    f"Replace placeholder party '{party}' with a party named "
                    "in the text, or remove the rule.",
                    data={"rule_kind": kind, "party": party, "action": rule.get("action")},
                )
            )

        asset = rule.get("asset")
        if isinstance(party, str) and class_type.get(party) not in (None, "Party", "RightsHolder", "ODRL_core"):
            findings.append(_finding(
                "rule_slot_type", "warning",
                f"{kind}: party '{party}' is typed {class_type.get(party)}, not Party.",
                f"Use a Party as the party of this rule, or retype '{party}'.",
                data={"rule_kind": kind, "party": party},
            ))
        if isinstance(asset, str) and class_type.get(asset) == "Party":
            findings.append(_finding(
                "rule_slot_type", "warning",
                f"{kind}: asset '{asset}' is typed Party; an asset is what the action applies to.",
                f"Use an Asset or Legal_extension as the asset of this rule, or retype '{asset}'.",
                data={"rule_kind": kind, "asset": asset},
            ))
        if rule.get("provision") == "recital" and kind in ("duties", "prohibitions"):
            findings.append(_finding(
                "rule_from_recital", "warning",
                f"{kind}: rule '{rule.get('action_label') or rule.get('action')}' comes from a recital, "
                "which is not binding.",
                "Remove duties and prohibitions taken from recitals, or ground them in an article.",
                data={"rule_kind": kind, "action": rule.get("action")},
            ))

        action = rule.get("action")
        if isinstance(action, str) and action not in vocab.actions:
            norm = vocab.normalize_action(action, rule.get("action_label"))
            if not norm["controlled"] or action == OTHER_ACTION:
                findings.append(_finding(
                    "uncontrolled_action", INFO,
                    f"{kind}: action '{rule.get('action_label') or action}' is not in the controlled action vocabulary.",
                ))
        # ground the wording the generator used, not the canonical verb: the verb and
        # every word of the object must appear in the text (a single shared word such as
        # "financial" is not enough)
        grounded_on = rule.get("action_label") or action
        if isinstance(grounded_on, str) and grounded_on != OTHER_ACTION:
            why = action_ungrounded_reason(rule, text)
            if why:
                findings.append(
                    _finding(
                        "ungrounded_action",
                        "error",
                        f"{kind}: action '{grounded_on}' has no support in the text ({why}).",
                        f"Remove action '{action}' or replace it with an "
                        "action expressed in the text.",
                        data={"rule_kind": kind, "action": action, "party": rule.get("party")},
                    )
                )

        for constraint in _dicts(rule.get("constraints")):
            if "left_operand" not in constraint:
                findings.append(_finding(
                    "unstructured_constraint", INFO,
                    f"{kind}: constraint '{constraint.get('name')}' is free text, not "
                    "left_operand/operator/right_operand.",
                ))
            value = constraint.get("right_operand", constraint.get("value"))
            if (
                isinstance(value, (str, int, float))
                and str(value).strip()
                and not _value_in_text(value, text_norm)
            ):
                findings.append(
                    _finding(
                        "ungrounded_value",
                        "warning",
                        f"{kind}: constraint '{constraint.get('name') or constraint.get('left_operand')}' has "
                        f"value '{value}' which does not appear in the text.",
                        f"Check constraint '{constraint.get('name') or constraint.get('left_operand')}': its "
                        "value must be taken from the text.",
                    )
                )

    # 5. Relations outside the controlled set
    for rel in _dicts(onto.get("relationships")):
        _, _, _, controlled = vocab.normalize_relation(rel.get("relation"), rel.get("source"), rel.get("target"))
        if not controlled:
            findings.append(_finding(
                "uncontrolled_relation", INFO,
                f"Relation '{rel.get('relation')}' is not in the controlled relation vocabulary.",
            ))

    # 6. Example instances that just copy their own context
    for instance in _dicts(onto.get("example_instances")):
        example = instance.get("example_value")
        context = instance.get("source_context")
        if example and context and _norm(example) == _norm(context):
            findings.append(
                _finding(
                    "example_equals_context",
                    "warning",
                    f"Example for entity '{instance.get('entity')}' has "
                    "example_value identical to source_context.",
                    f"For entity '{instance.get('entity')}' use a short, "
                    "specific example_value and keep the surrounding passage "
                    "in source_context.",
                )
            )

    # 7. Examples that do not quote the text (placeholder, paraphrase, value not in the text)
    for instance in _dicts(onto.get("example_instances")):
        problem = example_problem(instance, text)
        if problem:
            findings.append(
                _finding(
                    "example_not_in_text",
                    "warning",
                    f"Example for entity '{instance.get('entity')}' does not quote the text ({problem}).",
                    f"For entity '{instance.get('entity')}' copy example_value and source_context "
                    "verbatim from the text, or remove the example.",
                )
            )

    # 8. Identical rules repeated
    seen_rules: set[str] = set()
    for kind, rule in _iter_rules(onto):
        key = rule_key(rule)
        if key in seen_rules:
            findings.append(
                _finding(
                    "duplicate_rule",
                    "warning",
                    f"{kind}: rule '{rule.get('action_label') or rule.get('action')}' is repeated.",
                    "Remove the repeated rules.",
                )
            )
        seen_rules.add(key)

    # 9. Party mentioned nowhere near the action in the text (heuristic, warning only)
    for kind, rule in _iter_rules(onto):
        if party_far_from_action(rule, text):
            findings.append(
                _finding(
                    "party_far_from_action",
                    "warning",
                    f"{kind}: party '{rule.get('party')}' is not mentioned near "
                    f"'{rule.get('action_label') or rule.get('action')}' in the text.",
                    "Check who actually holds this rule in the text.",
                )
            )

    # Remove exact duplicates, keep order
    seen = set()
    unique = []
    for finding in findings:
        key = (finding["code"], finding["message"])
        if key not in seen:
            seen.add(key)
            unique.append(finding)

    return onto, unique


# Render findings for the judge prompt
def format_findings(findings: list[dict]) -> str:
    
    # info findings are fixed in code later; keep them out of the judge prompt
    shown = [f for f in findings if f["severity"] != INFO]
    if not shown:
        return "(none)"
    return "\n".join(f"- [{f['severity']}] {f['message']}" for f in shown)


# Resolve 'a.b[0].c' or 'a.b.0.c' inside a parsed JSON object
def resolve_path(obj: Any, path: str) -> tuple[bool, Any]:
    
    current = obj
    for name, index in _PATH_TOKEN.findall(path):
        try:
            if index:
                current = current[int(index)]
            elif name.isdigit() and isinstance(current, list):
                current = current[int(name)]
            else:
                current = current[name]
        except (KeyError, IndexError, TypeError):
            return False, None
    return True, current


"""
Split the judge's weaknesses into (grounded, discarded).

A weakness is kept only if:
    - its json_path exists in the ontology ('a.0.b' and 'a[0].b' accepted),
    - the last key of the path is not a modelling-vocabulary field,
    - it does not claim that a non-empty element is empty,
    - its quote appears in the value found at json_path (or, when json_path
    is empty, in the source text),
    - it is not a duplicate of an automatic finding.
"""
def verify_weaknesses(
    weaknesses: list[dict],
    onto: dict | None,
    ontology_json: str,
    text: str,
    findings: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    
    text_norm = _norm(text or "")
    findings_norm = _norm(" ".join(f["message"] for f in (findings or [])))
    kept, discarded = [], []

    for weakness in weaknesses:
        reasons = []
        claim = weakness.get("claim", "")
        path = (weakness.get("json_path") or "").strip()
        quote = (weakness.get("quote") or "").strip()
        quote_norm = _norm(quote)
        path_found = True
        value_norm = ""

        if path:
            path_found, value = resolve_path(onto or {}, path)
            if not path_found:
                reasons.append(f"json_path not found in ontology: {path}")
            else:
                keys = [k for k in re.findall(r"[^.\[\]]+", path) if not k.isdigit()]
                if keys and keys[-1] in VOCABULARY_KEYS:
                    reasons.append(
                        "modelling vocabulary field (not checked against the text)"
                    )
                value_norm = _norm(json.dumps(value, ensure_ascii=False))
                if value not in (None, "", [], {}) and _EMPTY_CLAIM.search(claim):
                    reasons.append(f"claims '{path}' is empty but it is not")

        if path_found:
            if not quote:
                reasons.append("no supporting quote")
            elif path and quote_norm not in value_norm:
                reasons.append("quote not found in the value at json_path")
            elif not path and quote_norm not in text_norm:
                reasons.append("quote not found in the source text")

        # A quote matching automatic-finding text is only a real duplicate
        # if the claim itself restates that finding's point, not just because
        # the same entity/class name is mentioned in some other automatic
        # finding (e.g. a party-misassignment claim naming an entity that
        # undefined_entities also lists, for an unrelated reason).
        codes_hit = {
            f["code"]
            for f in (findings or [])
            if quote_norm and quote_norm in _norm(f["message"])
        }
        restates = (
            ("undefined_entities" in codes_hit and _UNDEFINED_ENTITY_CLAIM.search(claim))
            or ("misclassified_core_class" in codes_hit and _MISCLASSIFIED_CLAIM.search(claim))
            or ("placeholder_party" in codes_hit and _PLACEHOLDER_CLAIM.search(claim))
            or ("ungrounded_action" in codes_hit and _UNGROUNDED_ACTION_CLAIM.search(claim))
        )
        if not reasons and len(quote_norm) >= 4 and restates:
            reasons.append("duplicate of an automatic finding")
        if (
            not reasons
            and path.endswith("example_value")
            and any(f["code"] == "example_equals_context" for f in (findings or []))
        ):
            reasons.append("duplicate of an automatic finding")

        if reasons:
            discarded.append({**weakness, "discard_reasons": reasons})
        else:
            kept.append(weakness)

    return kept, discarded


"""
Drop 'missing' concepts that are already defined (classes or
odrl_alignment), are just part of an existing class name, or do not
appear in the source text
"""
def filter_missing_concepts(
    missing: list[str],
    onto: dict | None,
    text: str = "",
) -> list[str]:
    
    if not onto:
        return list(missing)

    class_names = {_norm(c.get("name", "")) for c in _dicts(onto.get("classes"))}
    defined = set(class_names)
    alignment = onto.get("odrl_alignment")
    if isinstance(alignment, dict):
        for key in ("core_entities", "extensions"):
            values = alignment.get(key)
            if isinstance(values, list):
                defined.update(_norm(v) for v in values if isinstance(v, str))

    text_norm = _norm(text)
    return [
        concept
        for concept in missing
        if _norm(concept) not in defined
        and not any(_norm(concept) in name for name in class_names)
        and (not text_norm or _norm(concept) in text_norm)
    ]


"""
For 'unsupported_concepts' / 'hallucinations' reported by the judge:
keep only terms that really are in the ontology AND have no support in
the text (none of their words appear)
"""
def filter_ungrounded_terms(
    terms: list[str],
    ontology_json: str,
    text: str,
) -> list[str]:
    
    onto_norm = _norm(ontology_json or "")
    text_lower = (text or "").lower()
    kept = []
    for term in terms:
        if _norm(term) not in onto_norm:
            continue
        stems = [w.lower()[:5] for w in _WORD.findall(term) if len(w) >= 4]
        if stems and any(stem in text_lower for stem in stems):
            continue
        kept.append(term)
    return kept


"""
Scores derived from evidence, not from the LLM's opinion.

overall: 5 minus up to 3 for errors and up to 2 for (warnings + grounded
weaknesses, two per point). odrl_compliance: 5 minus one per ODRL-related
finding (max 4). Provisional: tune together with approval_thresholds.
"""
def count_warnings(findings: list[dict], count_by: str = "code") -> int:
    """Warnings counted per distinct code (one root cause = one penalty) or per finding."""
    warnings = [f for f in findings if f["severity"] == "warning"]
    return len({f["code"] for f in warnings}) if count_by == "code" else len(warnings)


def compute_scores(findings: list[dict], n_grounded_weaknesses: int,
                   count_by: str = "code") -> tuple[int, int]:
    """count_by='code': ten mistyped classes are one problem (the generator
    misunderstood typing), not ten. 'finding' reproduces the original scoring."""
    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = count_warnings(findings, count_by)
    overall = 5 - min(3, errors) - min(2, math.ceil((warnings + n_grounded_weaknesses) / 2))
    odrl_findings = [f for f in findings if f["code"] in ODRL_CODES and f["severity"] != INFO]
    n_odrl = len({f["code"] for f in odrl_findings}) if count_by == "code" else len(odrl_findings)
    odrl = 5 - min(4, n_odrl)
    return max(1, overall), max(1, odrl)