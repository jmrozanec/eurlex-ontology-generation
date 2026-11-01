"""Deterministic normalisation of a chunk-level ontology.

Applied after the LLM revision step (and usable anywhere else) so that what
reaches the merge is always:
  - free of ODRL meta-model leakage ("AuditDuty" classes are removed),
  - typed consistently (gazetteer first, then the upper ontology),
  - split into classes vs individuals (`kind`),
  - using canonical actions, structured constraints and canonical relations,
  - referentially closed (every referenced name is defined),
  - with odrl_alignment rebuilt from the classes.

Every change is recorded so the revision report can show what code fixed.
"""
from __future__ import annotations

from typing import Any

from .entity_linking import Gazetteer, detect_kind, load_gazetteer
from .ontology_schema import EXPECTED_TOP_LEVEL_KEYS, KINDS, RULE_KINDS
from .vocabulary import Vocabulary, load_vocabulary, split_identifier
from .ontology_cleanup import cleanup_ontology
from .ontology_grounding import constraint_is_ungrounded

CORE_ENTITIES = ["Asset", "Constraint", "Duty", "Party", "Permission", "Prohibition"]
PLACEHOLDER_NAMES = {"targetparty", "responsibleparty", "party", "unknownparty", "genericparty",
                     "targetasset", "asset", "unknown", "none", "null", "na"}


def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def is_placeholder_name(name: Any) -> bool:
    return not isinstance(name, str) or "".join(split_identifier(name)) in PLACEHOLDER_NAMES


def split_composite(name: str, gazetteer: Gazetteer) -> list[str] | None:
    """'EuropeanParliamentAndCouncil' -> ['European_Parliament', 'Council'] when every
    part is a known individual; None otherwise."""
    import re

    parts = [p for p in re.split(r"\s*[,;&]\s*|[\s_]+and(?:[\s_]+the)?[\s_]+", name.strip(), flags=re.I) if p]
    if len(parts) < 2:
        parts = re.split(r"(?<=[a-z])And(?:The)?(?=[A-Z])", name.strip())
    if len(parts) < 2:
        return None
    hits = [gazetteer.lookup(p) for p in parts]
    return [h.name for h in hits] if all(hits) else None


def type_entity(name: str, description: str, vocab: Vocabulary, gazetteer: Gazetteer,
                declared_type: str | None = None, declared_is_a: str | None = None,
                declared_kind: str | None = None) -> dict:
    """Resolve type, is_a and kind for one entity. Gazetteer > declared is_a > keyword guess."""
    hit = gazetteer.lookup(name)
    if hit:
        out = hit.as_dict()
        out.update({"name": name, "type_source": "gazetteer", "kind_confidence": "high"})
        return out
    if declared_is_a in vocab.upper:
        is_a, source = declared_is_a, "declared"
    else:
        is_a, evidence = vocab.guess_is_a(name, description)
        source = f"keyword ({evidence})"
        # A head-noun match ('...Fund', '...Committee') is strong evidence and wins
        # over the generator's declared type, which is exactly what is inconsistent.
        # Weaker keyword evidence loses to an explicit, valid declared type.
        if (declared_type in vocab.core_types and vocab.core_type_of(is_a) != declared_type
                and not evidence.startswith("head noun")):
            fallback = {"Party": "PartyRole", "Asset": "Resource", "Legal_extension": "Concept"}
            is_a, source = fallback[declared_type], "declared type"
    kind = detect_kind(name, description, gazetteer, vocab, declared_kind)
    out = {"name": name, "type": vocab.core_type_of(is_a), "is_a": is_a, "type_source": source,
           "kind": kind["kind"], "kind_confidence": kind["confidence"]}
    if kind.get("related_individual"):
        out["related_individual"] = kind["related_individual"]
    return out


def normalize_chunk_ontology(onto: dict, vocab: Vocabulary | None = None,
                             gazetteer: Gazetteer | None = None,
                             source_text: str | None = None,
                             original: dict | None = None) -> tuple[dict, list[dict]]:
    vocab = vocab or load_vocabulary()
    gazetteer = gazetteer or load_gazetteer()
    changes: list[dict] = []

    # Checks against the source text (repeated rules, actions and examples that are not in the
    # text, descriptions cut by the length cap, lost `provision`). `original` is the version
    # before the revision: it is used to restore `provision`.
    if source_text:
        onto, cleanup_notes = cleanup_ontology(onto, source_text, original=original)
        changes.extend(cleanup_notes)

    def note(code: str, detail: str) -> None:
        changes.append({"code": code, "detail": detail})

    out = {k: onto.get(k) for k in EXPECTED_TOP_LEVEL_KEYS}
    out["ontology_name"] = str(out.get("ontology_name") or "UnnamedOntology")
    out["description"] = str(out.get("description") or "")

    # ---- classes
    removed: set[str] = set()
    composites: dict[str, list[str]] = {}
    classes: dict[str, dict] = {}
    for cls in _dicts(onto.get("classes")):
        name = cls.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if split_composite(name, gazetteer):
            composites[name] = split_composite(name, gazetteer)
            note("composite_class_split", f"{name} -> {composites[name]}")
            continue
        leak = vocab.metamodel_leakage(name)
        if leak:
            removed.add(name)
            note("metamodel_leakage_removed", leak)
            continue
        typed = type_entity(name, str(cls.get("description") or ""), vocab, gazetteer,
                            cls.get("type"), cls.get("is_a"),
                            cls.get("kind") if cls.get("kind") in KINDS else None)
        if cls.get("type") != typed["type"]:
            note("class_retyped", f"{name}: {cls.get('type')} -> {typed['type']} ({typed['type_source']})")
        if cls.get("is_a") and cls.get("is_a") != typed["is_a"]:
            note("class_is_a_changed", f"{name}: {cls.get('is_a')} -> {typed['is_a']}")
        if cls.get("kind") != typed["kind"]:
            note("class_kind_set", f"{name}: {cls.get('kind')} -> {typed['kind']} ({typed['kind_confidence']})")
        entry = {"name": name, "type": typed["type"], "is_a": typed["is_a"], "kind": typed["kind"],
                 "description": str(cls.get("description") or "")}
        for extra in ("iri", "wikidata", "member_state", "related_individual"):
            if typed.get(extra) is not None:
                entry[extra] = typed[extra]
        if name in classes:
            note("duplicate_class_removed", name)
            continue
        classes[name] = entry

    def ok_ref(name: Any) -> bool:
        if not isinstance(name, str) or not name.strip() or name in removed:
            return False
        if vocab.metamodel_leakage(name):
            removed.add(name)
            note("metamodel_leakage_removed", f"reference '{name}'")
            return False
        return True

    # ---- rights model
    rm_in = onto.get("rights_model") if isinstance(onto.get("rights_model"), dict) else {}
    rights_model: dict[str, list] = {}
    for kind in RULE_KINDS:
        rules = []
        expanded = []
        for rule in _dicts(rm_in.get(kind)):
            party = rule.get("party")
            parts = composites.get(party) or (split_composite(party, gazetteer) if isinstance(party, str) else None)
            if parts:
                expanded += [{**rule, "party": p} for p in parts]
            else:
                expanded.append(rule)
        for rule in expanded:
            party = rule.get("party")
            if not ok_ref(party):
                note("rule_dropped", f"{kind}: party '{party}' missing or removed")
                continue
            norm = vocab.normalize_action(rule.get("action"), rule.get("action_label"))
            if norm["action"] != rule.get("action"):
                note("action_normalized", f"{rule.get('action')} -> {norm['action']}"
                     + (f" + object {norm['action_object']}" if norm["action_object"] else ""))
            asset = rule.get("asset") or ""
            if asset and not ok_ref(asset):
                note("asset_cleared", f"{kind}: asset '{asset}' was removed as leakage")
                asset = ""
            constraints = []
            for c in _dicts(rule.get("constraints")):
                sc = vocab.structure_constraint(c)
                if sc is None:
                    note("constraint_dropped", str(c)[:80])
                    continue
                # drop values that have no trace in the source text (invented dates, amounts, currencies)
                if source_text:
                    reason = constraint_is_ungrounded(sc, source_text)
                    if reason:
                        note("constraint_ungrounded_dropped",
                             f"{sc.get('left_operand')} {sc.get('operator')} "
                             f"{sc.get('right_operand')}: {reason}")
                        continue
                if "name" in c and "left_operand" not in c:
                    note("constraint_structured", f"{c.get('name')} -> {sc['left_operand']} {sc['operator']}")
                constraints.append(sc)
            new_rule = {"action": norm["action"], "action_label": norm["action_label"],
                        "action_object": norm["action_object"], "action_controlled": norm["controlled"],
                        "party": party, "asset": asset, "constraints": constraints}
            if rule.get("assigner"):
                new_rule["assigner"] = rule["assigner"]
            if rule.get("provision"):
                new_rule["provision"] = rule["provision"]
            rules.append(new_rule)
        rights_model[kind] = rules
    out["rights_model"] = rights_model

    # ---- relationships
    relationships, seen = [], set()
    rels_in = []
    for rel in _dicts(onto.get("relationships")):
        srcs = composites.get(rel.get("source")) or [rel.get("source")]
        tgts = composites.get(rel.get("target")) or [rel.get("target")]
        rels_in += [{**rel, "source": s_, "target": t_} for s_ in srcs for t_ in tgts]
    for rel in rels_in:
        relation, src, tgt, controlled = vocab.normalize_relation(rel.get("relation"), rel.get("source"), rel.get("target"))
        if not (relation and ok_ref(src) and ok_ref(tgt)) or src == tgt:
            note("relationship_dropped", f"{rel.get('source')} {rel.get('relation')} {rel.get('target')}")
            continue
        if relation != rel.get("relation") or src != rel.get("source"):
            note("relation_normalized", f"{rel.get('source')} {rel.get('relation')} {rel.get('target')}"
                 f" -> {src} {relation} {tgt}")
        key = (src, relation, tgt)
        if key in seen:
            continue
        seen.add(key)
        relationships.append({"source": src, "relation": relation, "target": tgt,
                              "description": str(rel.get("description") or ""),
                              "relation_controlled": controlled})
    out["relationships"] = relationships

    # ---- patterns and examples
    patterns = []
    for pat in _dicts(onto.get("legal_patterns")):
        ents = []
        for e in pat.get("entities_involved") or []:
            ents += composites.get(e, [e])
        ents = [e for e in dict.fromkeys(ents) if ok_ref(e)]
        patterns.append({"pattern_name": str(pat.get("pattern_name") or ""),
                         "description": str(pat.get("description") or ""),
                         "entities_involved": ents})
    out["legal_patterns"] = patterns
    out["example_instances"] = [i for i in _dicts(onto.get("example_instances")) if ok_ref(i.get("entity"))]

    # ---- referential closure: define every referenced name
    refs: list[str] = []
    for r in relationships:
        refs += [r["source"], r["target"]]
    for rules in rights_model.values():
        for r in rules:
            refs += [r["party"], r.get("asset") or "", r.get("assigner") or ""]
    for p in patterns:
        refs += p["entities_involved"]
    refs += [i.get("entity") for i in out["example_instances"]]
    for ref in dict.fromkeys(refs):
        if not isinstance(ref, str) or not ref.strip() or ref in classes or is_placeholder_name(ref):
            continue
        typed = type_entity(ref, "", vocab, gazetteer)
        classes[ref] = {"name": ref, "type": typed["type"], "is_a": typed["is_a"],
                        "kind": typed["kind"], "description": ""}
        for extra in ("iri", "wikidata"):
            if typed.get(extra):
                classes[ref][extra] = typed[extra]
        note("undefined_entity_defined", f"{ref} as {typed['type']}/{typed['is_a']}")

    out["classes"] = list(classes.values())
    out["odrl_alignment"] = {
        "core_entities": CORE_ENTITIES,
        "extensions": sorted(c["name"] for c in out["classes"] if c["type"] == "Legal_extension"),
    }
    return out, changes