"""Deterministic assembly of one merged ontology from many partial ontologies"""

import json
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    build_resolver,
    celex_of,
    clean_constraint,
    is_leaky_action,
    is_placeholder,
    is_rejected_class,
    normalize_action,
    normalize_relation,
    split_entities,
    group_key,
)

# Core entity types expected by the ontology model
VALID_CORE_TYPES = {"Asset", "Party", "RightsHolder", "Permission", "Prohibition", "Duty", "Constraint"}

# Limits used to keep the merged ontology compact and deterministic
MAX_MERGED_DESCRIPTIONS_PER_CLASS = 3
MAX_CONSTRAINTS_PER_RULE = 10
MAX_EXAMPLES_PER_ENTITY = 3
MIN_PATTERN_ENTITIES = 2
RULE_KINDS = ("permissions", "duties", "prohibitions")
GENERIC_RELATIONS = {"applies_to", "related_to", "relates_to", "involves", "involved_in",
                     "associated_with", "has"}
MIN_GENERIC_RELATION_SUPPORT = 2   # generic relations seen only once are noise
MAX_PATTERN_ENTITIES = 8
MAX_EXAMPLE_LEN = 120

# Regular expression to extract source counts from partition descriptions
_SOURCE_COUNT_RE = re.compile(r"from (\d+) source ontologies covering (\d+) EUR-Lex")


# Safely return a list of dictionaries from an untyped JSON value
def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


# Extract (chunk_count, document_count) support metrics from an entity dictionary
def _support(obj: dict) -> tuple[int, int]:
    s = obj.get("support")
    if isinstance(s, dict):
        try:
            return int(s.get("chunks", 1)), int(s.get("documents", 1))
        except (TypeError, ValueError):
            pass
    return 1, 1


# Merges multiple partial JSON ontologies into a single consolidated ontology
def assemble_merged_ontology(
    ontologies: pd.DataFrame,
    canonical_map: dict[str, dict[str, str]],
    source_label: str,
) -> tuple[dict, dict]:

    # Build entity resolver to map raw strings to canonical names
    _resolve = build_resolver(canonical_map)
    declared = set(canonical_map.get("class", {}).values())

    # Resolve a raw name to its canonical class name; drop unmapped entities
    def resolve(name):
        # Only preserve names that were explicitly declared as canonical classes
        r = _resolve(name)
        return r if r in declared else None
    action_map = canonical_map.get("action", {})

    # Data structures for accumulating resolved entities and support metrics
    classes: dict[str, dict] = {}
    type_votes: dict[str, Counter] = defaultdict(Counter)
    class_chunks: Counter = Counter()
    class_docs: dict[str, set] = defaultdict(set)
    class_docs_n: Counter = Counter()          # documents carried over from lower-level support
    raw_class_names: set[str] = set()
    relationships: dict[tuple, dict] = {}
    rules: dict[str, dict[tuple, dict]] = {k: {} for k in RULE_KINDS}
    patterns: dict[tuple, dict] = {}
    examples: dict[tuple, dict] = {}
    source_ids: set[str] = set()
    source_celex: set[str] = set()
    dropped: Counter = Counter()
    carried_sources = carried_docs = 0     # counts declared by lower-level (partition) ontologies

    # Phase 1: Ingest and aggregate raw input ontologies
    for _, row in ontologies.iterrows():
        try:
            onto = json.loads(row["ontology"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(onto, dict):
            continue

        # Extract source lineage and document CELEX identifier
        source_id = str(row["chunk_id"])
        raw_celex = str(row.get("CELEX", "") or "")
        celex = raw_celex if raw_celex and raw_celex != source_id else celex_of(source_id)
        source_ids.add(source_id)
        source_celex.add(celex)

        # Parse inherited source/document metadata if available
        m = _SOURCE_COUNT_RE.search(str(onto.get("description", "")))
        if m:
            carried_sources += int(m.group(1))
            carried_docs += int(m.group(2))
        else:
            carried_sources += 1
            carried_docs += 0

        # 1a: Process classes
        for cls in _dicts(onto.get("classes")):
            raw = cls.get("name")
            if isinstance(raw, str):
                raw_class_names.add(raw)
            description = (cls.get("description") or "").strip()

            # Filter out invalid, noisy, or unresolvable classes
            if is_rejected_class(raw or "", description):
                dropped["class_rejected"] += 1
                continue
            name = resolve(raw)
            if not name:
                dropped["class_unresolved"] += 1
                continue

            # Update class statistics and type voting
            chunks, docs = _support(cls)
            class_chunks[name] += chunks
            class_docs[name].add(celex)
            class_docs_n[name] += max(docs - 1, 0)
            if cls.get("type"):
                type_votes[name][cls["type"]] += chunks

            # Merge class descriptions
            existing = classes.setdefault(name, {"description": ""})
            if description:
                pieces = [p for p in existing["description"].split(" | ") if p]
                if description not in pieces and len(pieces) < MAX_MERGED_DESCRIPTIONS_PER_CLASS:
                    pieces.append(description)
                    existing["description"] = " | ".join(pieces)

        # 1b: Process relationships
        for rel in _dicts(onto.get("relationships")):
            src, tgt = resolve(rel.get("source")), resolve(rel.get("target"))
            relation = normalize_relation(rel.get("relation"))

            # Filter unresolved entities or invalid self-loops
            if not (src and tgt and relation):
                dropped["relationship_unresolved"] += 1
                continue
            if src == tgt:
                dropped["relationship_self_loop"] += 1
                continue

            # Deduplicate by triple (source, relation, target) and increment support
            entry = relationships.setdefault(
                (src, relation, tgt),
                {"source": src, "relation": relation, "target": tgt,
                 "description": rel.get("description", "") or "", "support": 0},
            )
            entry["support"] += _support(rel)[0]

        # 1c: Process rights model
        rm = onto.get("rights_model")
        if isinstance(rm, dict):
            for kind in RULE_KINDS:
                for rule in _dicts(rm.get(kind)):

                    # Sanitize and resolve rule action
                    raw_action = rule.get("action")
                    if not isinstance(raw_action, str) or not raw_action.strip():
                        continue
                    action = normalize_action(raw_action)
                    if not action or is_leaky_action(action):
                        dropped["rule_leaky_action"] += 1
                        continue
                    action = action_map.get(action, action)

                    # Resolve parties and assets
                    raw_party = rule.get("party")
                    parties = [resolve(p) for p in split_entities(raw_party)] if isinstance(raw_party, str) else []
                    parties = [p for p in dict.fromkeys(parties) if p]
                    if not parties:
                        dropped["rule_unresolved_party"] += 1
                        continue
                    asset = resolve(rule.get("asset")) if rule.get("asset") else None
                    if rule.get("asset") and not asset:
                        dropped["rule_unresolved_asset"] += 1
                        continue
                    if kind == "permissions" and not asset:
                        dropped["permission_without_asset"] += 1
                        continue

                    # Clean and collect constraints
                    constraints = [c for c in (clean_constraint(c) for c in _dicts(rule.get("constraints"))) if c]
                    for party in parties:
                        key = (action, party, asset)
                        entry = rules[kind].setdefault(
                            key, {"action": action, "party": party, "asset": asset,
                                  "constraints": [], "support": 0})
                        entry["support"] += _support(rule)[0]

                        # Unique constraint aggregation per rule
                        seen = {(c["name"], c["value"]) for c in entry["constraints"]}
                        for c in constraints:
                            if (c["name"], c["value"]) not in seen and len(entry["constraints"]) < MAX_CONSTRAINTS_PER_RULE:
                                entry["constraints"].append(c)
                                seen.add((c["name"], c["value"]))

        # 1d: Process legal patterns
        for pat in _dicts(onto.get("legal_patterns")):
            involved = pat.get("entities_involved")
            if not isinstance(involved, list):
                continue
            ents = list(dict.fromkeys(e for e in (resolve(x) for x in involved if isinstance(x, str)) if e))
            if len(ents) < MIN_PATTERN_ENTITIES or not pat.get("pattern_name"):
                dropped["pattern_dropped"] += 1
                continue

            # Group equivalent pattern names
            key = group_key("class", str(pat["pattern_name"]))      # EGF_Mobilization_Pattern == EGF_Mobilisation_Pattern
            entry = patterns.setdefault(key, {"pattern_name": pat["pattern_name"],
                                              "description": pat.get("description", "") or "",
                                              "entities_involved": [], "support": 0})
            for e in ents:
                if e not in entry["entities_involved"] and len(entry["entities_involved"]) < MAX_PATTERN_ENTITIES:
                    entry["entities_involved"].append(e)
            entry["support"] += _support(pat)[0]

        # 1e: Process example instances
        for inst in _dicts(onto.get("example_instances")):
            entity = resolve(inst.get("entity"))
            value = inst.get("example_value")
            ctx = inst.get("source_context") or ""

            # Filter out noisy or generic example values
            if not entity or not isinstance(value, str) or is_placeholder(value) or is_placeholder(ctx) \
                    or "text fragment" in ctx.lower() or "throughout" in ctx.lower() \
                    or len(value) > MAX_EXAMPLE_LEN or value[:1].islower() \
                    or re.match(r"\s*article\s*\d", ctx, re.I) \
                    or re.search(r"\((EU|EC|EEC)\)\s*\d|\d{4}/\d+", value):
                dropped["example_dropped"] += 1
                continue
            examples.setdefault((entity, value.strip()), {**inst, "entity": entity})


    # Phase 2: Finalize schemas, infer missing types, and format output
    final_classes = []
    for name, cls in classes.items():
        # Assign entity type via majority vote; default to 'Legal_extension'
        votes = type_votes.get(name)
        declared = votes.most_common(1)[0][0] if votes else "Legal_extension"
        
        if declared not in (VALID_CORE_TYPES | {"Legal_extension"}):
            declared = "Legal_extension"
        final_classes.append({
            "name": name, "type": declared, "description": cls["description"],
            "support": {"chunks": class_chunks[name],
                        "documents": len(class_docs[name]) + class_docs_n[name]},
        })

    # a name that is used as party/asset in the rights model but typed Legal_extension stays as is:
    # we do not guess types, but we report the mismatch in provenance
    party_names = {r["party"] for k in RULE_KINDS for r in rules[k].values()}
    type_by_name = {c["name"]: c["type"] for c in final_classes}

    # a class used as party / asset in the rights model but typed Legal_extension gets that core type
    asset_names = {r["asset"] for k in RULE_KINDS for r in rules[k].values() if r.get("asset")}
    type_inferred = []
    for c in final_classes:
        if c["type"] == "Legal_extension":
            if c["name"] in party_names:
                c["type"] = "Party"
            elif c["name"] in asset_names:
                c["type"] = "Asset"
            else:
                continue
            type_inferred.append(c["name"])

    # Track unresolved type mismatches for audit trails/provenance
    type_by_name = {c["name"]: c["type"] for c in final_classes}
    party_type_mismatch = sorted(n for n in party_names if type_by_name.get(n) not in ("Party", "RightsHolder"))

    # Filter out low-support generic relationships
    final_relationships = []
    for r in relationships.values():
        if r["relation"] in GENERIC_RELATIONS and r["support"] < MIN_GENERIC_RELATION_SUPPORT:
            dropped["relationship_generic_low_support"] += 1
            continue
        final_relationships.append(r)

    n_sources = carried_sources or len(source_ids)
    n_docs = carried_docs or len(source_celex)

    # Limit example instances to MAX_EXAMPLES_PER_ENTITY per entity type
    per_entity: Counter = Counter()
    final_examples = []
    for (entity, _), inst in examples.items():
        if per_entity[entity] < MAX_EXAMPLES_PER_ENTITY:
            per_entity[entity] += 1
            final_examples.append(inst)

    # Build consolidated JSON schema structure
    merged = {
        "ontology_name": f"EurLex ODRL Merged Ontology ({source_label})",
        "description": (f"Ontology merged from {n_sources} source ontologies "
                        f"covering {n_docs} EUR-Lex document(s)."),
        "odrl_alignment": {
            "core_entities": sorted(VALID_CORE_TYPES),
            "extensions": sorted(c["name"] for c in final_classes if c["type"] == "Legal_extension"),
        },
        "classes": final_classes,
        "relationships": final_relationships,
        "rights_model": {k: [_public(r, k) for r in rules[k].values()] for k in RULE_KINDS},
        "legal_patterns": list(patterns.values()),
        "example_instances": final_examples,
    }

    # Build metadata tracking execution, filtering stats, and type conversions
    provenance = {
        "source_label": source_label,
        "n_source_ontologies": n_sources,
        "n_source_documents": n_docs,
        "type_inferred": type_inferred,
        "source_ids": sorted(source_ids),
        "source_celex": sorted(source_celex),
        "n_classes_before_dedup": len(raw_class_names),
        "n_classes_after_dedup": len(final_classes),
        "dropped": dict(dropped),
        "party_type_mismatch": party_type_mismatch,
    }
    return merged, provenance


# Format an internal rule entry into a standardized public JSON representation
def _public(rule: dict, kind: str) -> dict:
    out = {"action": rule["action"], "party": rule["party"], "constraints": rule["constraints"],
           "support": {"chunks": rule["support"]}}
    if rule.get("asset") or kind == "permissions":
        out["asset"] = rule.get("asset")
    return out


# Validates the structure and referential integrity of a merged ontology dictionary
def validate_merged_ontology(ontology: dict) -> list[str]:
    issues: list[str] = []

    # 1: Verify schema top-level keys match expected signature exactly
    expected = {"ontology_name", "description", "odrl_alignment", "classes", "relationships",
                "rights_model", "legal_patterns", "example_instances"}
    if set(ontology) != expected:
        issues.append(f"schema_mismatch: missing={sorted(expected - set(ontology))}, extra={sorted(set(ontology) - expected)}")

    # 2: Check for unrecognized class types
    valid_types = VALID_CORE_TYPES | {"Legal_extension"}
    classes = _dicts(ontology.get("classes"))
    names = {c.get("name") for c in classes}
    for c in classes:
        if c.get("type") not in valid_types:
            issues.append(f"invalid_class_type:{c.get('name')}")

    # 3: Ensure 'extensions' in odrl_alignment matches classes typed as Legal_extension
    ext = set(ontology.get("odrl_alignment", {}).get("extensions", []) or [])
    legal = {c["name"] for c in classes if c.get("type") == "Legal_extension"}
    if ext != legal:
        issues.append(f"extensions_classes_mismatch: extensions_only={sorted(ext - legal)}, classes_only={sorted(legal - ext)}")

    # 4: Collect referenced entity names to check for broken cross-references or placeholder constraints
    refs: set[str] = set()
    for r in _dicts(ontology.get("relationships")):
        refs.update(v for v in (r.get("source"), r.get("target")) if isinstance(v, str))
        if r.get("source") == r.get("target"):
            issues.append(f"self_loop:{r.get('source')}")
    rm = ontology.get("rights_model") or {}
    for kind in RULE_KINDS:
        for rule in _dicts(rm.get(kind)):
            refs.update(v for v in (rule.get("party"), rule.get("asset")) if isinstance(v, str))
            for c in _dicts(rule.get("constraints")):
                if is_placeholder(c.get("value")):
                    issues.append(f"placeholder_constraint:{rule.get('action')}")
    for p in _dicts(ontology.get("legal_patterns")):
        refs.update(v for v in (p.get("entities_involved") or []) if isinstance(v, str))
    refs.update(i["entity"] for i in _dicts(ontology.get("example_instances")) if isinstance(i.get("entity"), str))

    # 5: Flag any referenced entity that is not defined in the classes list
    undefined = sorted(refs - names)
    if undefined:
        issues.append(f"undefined_entities:{undefined}")
    return issues