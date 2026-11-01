"""Deterministic clean-up of a chunk-level ontology against its source text.

No LLM call and no dependency on the rest of the project, so it can run right
after the generation, inside the normalizer after every revision, and inside the
review checks (`action_ungrounded_reason`).

It removes what a small model produces when it fills the output template instead
of reading the text:
  - rules repeated many times (also a fact stated both as permission and duty);
  - rules whose action (verb + object) has no trace in the text;
  - examples with a placeholder or a paraphrased `source_context`, or a value that
    is not in the text (the context is repaired when the value is in the text);
  - descriptions cut by the schema length cap in the middle of a sentence;
  - `provision` lost by the revision.
It also tells when a chunk is only a list of footnotes ("(7) Council Regulation ... (OJ L ...)"),
which has no rights to extract, and when a list reached its schema cap.

Every change is returned as {"code", "detail"} so it can be stored with the chunk.
"""
from __future__ import annotations

import copy
import json
import re

RULE_KINDS = ("permissions", "prohibitions", "duties")

# Same limits as utils/ontology_schema.py; used only to flag lists that hit the cap
# (the symptom of a generation loop).
DEFAULT_CAPS = {
    "permissions": 20, "prohibitions": 20, "duties": 20,
    "classes": 40, "example_instances": 10, "legal_patterns": 10, "relationships": 30,
}

# Text copied from the generation prompt template.
PLACEHOLDER_MARKERS = ("the sentence it comes from", "copied from the text")

_DANGLING = {
    "the", "of", "in", "a", "an", "and", "to", "for", "by", "as", "with", "defined",
    "that", "which", "is", "are", "or", "on", "at", "from",
}


# ------------------------------------------------------------------ helpers
def _norm(s) -> str:
    s = str(s or "").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()


def _tokens(s) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z(0-9])", text or "")
    return [p.strip() for p in parts if p.strip()]


def _stem(word: str) -> str:
    return word[: max(4, len(word) - 2)]


def _rule_label(rule: dict) -> str:
    return str(rule.get("action_label") or rule.get("action") or "")


def _rule_key(rule: dict) -> str:
    cons = sorted(json.dumps(c, sort_keys=True, ensure_ascii=False)
                  for c in rule.get("constraints") or [])
    return json.dumps([_rule_label(rule), rule.get("party"), rule.get("asset"),
                       rule.get("assigner") or "", cons], ensure_ascii=False)


rule_key = _rule_key      # public name, also used by the review checks


# ---------------------------------------------------------- footnote-only chunks
_FOOTNOTE = re.compile(r"\(\d{1,3}\)\s(?:(?!\(\d{1,3}\)\s).)*?\(OJ [LC][^)]*\)\.?", re.S)


def footnote_share(text: str) -> float:
    """Share of the text covered by footnotes like '(7) Council Regulation ... (OJ L ...)'."""
    if not text:
        return 0.0
    return sum(len(m.group(0)) for m in _FOOTNOTE.finditer(text)) / len(text)


def is_footnote_chunk(text: str, threshold: float = 0.6) -> bool:
    return footnote_share(text) >= threshold


def empty_ontology(name: str = "NoRulesInFragment",
                   description: str = "Fragment without rights, prohibitions or duties.") -> dict:
    """A valid, empty ontology with the expected top-level keys."""
    return {
        "ontology_name": name,
        "description": description,
        "odrl_alignment": {"core_entities": ["Asset", "Constraint", "Duty", "Party", "Permission", "Prohibition"],
                           "extensions": []},
        "classes": [],
        "relationships": [],
        "rights_model": {"permissions": [], "prohibitions": [], "duties": []},
        "legal_patterns": [],
        "example_instances": [],
    }


# --------------------------------------------------------------- action grounding
def action_ungrounded_reason(rule: dict, text: str) -> str | None:
    """Why the action of `rule` has no support in `text`, or None if it has.

    The wording is the snake_case label ('submit_application'). The first word is
    the verb; the other words are the object. The verb and every object word must
    appear in the text, compared on their stem (the word without its last two letters,
    at least 4: 'mobilise' matches 'mobilisation', 'contribution' does not match 'control').
    """
    words = [w for w in _rule_label(rule).lower().split("_") if w]
    if not words:
        return None
    haystack = text.lower() if text else ""
    verb = words[0]
    if len(verb) >= 3 and _stem(verb) not in haystack:
        return f"verb '{verb}' not in text"
    for obj in words[1:]:
        if len(obj) > 3 and _stem(obj) not in haystack:
            return f"word '{obj}' not in text"
    return None


# ----------------------------------------------------------------- clean-up steps
def warn_lists_at_cap(onto: dict, caps: dict, notes: list) -> None:
    rm = onto.get("rights_model") or {}
    for k in RULE_KINDS:
        n = len(rm.get(k) or [])
        if k in caps and n >= caps[k]:
            notes.append({"code": "list_at_cap", "detail": f"rights_model.{k}={n}"})
    for k in ("classes", "example_instances", "legal_patterns", "relationships"):
        n = len(onto.get(k) or [])
        if k in caps and n >= caps[k]:
            notes.append({"code": "list_at_cap", "detail": f"{k}={n}"})


def drop_duplicate_rules(onto: dict, notes: list) -> None:
    rm = onto.get("rights_model")
    if not isinstance(rm, dict):
        return
    seen: set[str] = set()
    for k in RULE_KINDS:
        kept = []
        for r in rm.get(k) or []:
            key = _rule_key(r)
            if key in seen:
                notes.append({"code": "rule_duplicate_dropped", "detail": f"{k}:{_rule_label(r)}"})
                continue
            seen.add(key)
            kept.append(r)
        if k in rm:
            rm[k] = kept
    # the same fact as permission and as duty: the duty ("shall") wins
    duties = {(_rule_label(r), r.get("party"), r.get("asset")) for r in rm.get("duties") or []}
    if duties and rm.get("permissions"):
        kept = []
        for r in rm["permissions"]:
            if (_rule_label(r), r.get("party"), r.get("asset")) in duties:
                notes.append({"code": "permission_duplicates_duty", "detail": _rule_label(r)})
                continue
            kept.append(r)
        rm["permissions"] = kept


def drop_ungrounded_actions(onto: dict, text: str, notes: list) -> None:
    rm = onto.get("rights_model")
    if not isinstance(rm, dict) or not text:
        return
    for k in RULE_KINDS:
        kept = []
        for r in rm.get(k) or []:
            why = action_ungrounded_reason(r, text)
            if why:
                notes.append({"code": "rule_ungrounded_dropped", "detail": f"{k}:{_rule_label(r)} ({why})"})
                continue
            kept.append(r)
        if k in rm:
            rm[k] = kept


def example_problem(example: dict, text: str) -> str | None:
    """'placeholder', 'value_not_in_text', 'context_not_in_text' or None."""
    ctx = _norm(example.get("source_context"))
    if any(m in ctx for m in PLACEHOLDER_MARKERS):
        return "placeholder"
    value = _norm(example.get("example_value"))
    if not value or value not in _norm(text):          # the whole string, not its words
        return "value_not_in_text"
    if ctx not in _norm(text):
        return "context_not_in_text"
    if value not in ctx:
        return "context_without_value"
    return None


def fix_examples(onto: dict, text: str, notes: list, max_context: int = 600) -> None:
    """Keep only examples that quote the text; repair a bad context when the value is in the text."""
    if "example_instances" not in onto or not text:
        return
    sentences = _sentences(text)
    kept = []
    for e in onto.get("example_instances") or []:
        if not isinstance(e, dict):
            continue
        problem = example_problem(e, text)
        if problem == "value_not_in_text":
            notes.append({"code": "example_dropped", "detail": f"{e.get('entity')}: value not in text"})
            continue
        if problem in ("placeholder", "context_not_in_text", "context_without_value"):
            value = _norm(e.get("example_value"))
            hits = [s for s in sentences if value in _norm(s)]
            if not hits:
                notes.append({"code": "example_dropped", "detail": f"{e.get('entity')}: {problem}"})
                continue
            hit = min(hits, key=len)
            if len(hit) > max_context:
                at = max(0, hit.lower().find(str(e.get("example_value")).lower()) - 150)
                window = hit[at:at + max_context]
                if at > 0:                      # start and end on whole words
                    window = window.split(" ", 1)[-1]
                hit = window.rsplit(" ", 1)[0]
            e = {**e, "source_context": hit}
            notes.append({"code": "example_context_repaired", "detail": f"{e.get('entity')}"})
        kept.append(e)
    onto["example_instances"] = kept


def trim_description(desc: str, cap: int = 300) -> str:
    """Cut to the last complete sentence a description that was cut by the length cap
    or ends on a dangling word ('... as defined in the')."""
    d = re.sub(r"\s+", " ", desc or "").strip()
    if not d:
        return d
    cut = len(d) >= cap - 5
    last_word = d.split()[-1]
    # short names such as 'Eesti Energia AS' are not sentences cut in the middle
    dangling = (len(d) >= 60 and not last_word.isupper()
                and last_word.lower().strip(".,;:") in _DANGLING)
    if not (cut or dangling):
        return d
    last = max(d.rfind(". "), d.rfind("! "), d.rfind("? "))
    if last >= 40:
        return d[: last + 1]
    words = d.rstrip(".,;: ").split()
    while words and words[-1].lower().strip(".,;:") in _DANGLING:
        words.pop()
    return (" ".join(words) + ".") if words else d


def trim_descriptions(onto: dict, notes: list, cap: int = 300) -> None:
    def fix(obj: dict, label: str) -> None:
        old = obj.get("description")
        if not isinstance(old, str):
            return
        new = trim_description(old, cap)
        if new != old:
            if re.sub(r"\s+", " ", old).strip() != new:    # more than a whitespace clean-up
                notes.append({"code": "description_trimmed", "detail": label})
            obj["description"] = new

    fix(onto, "ontology")
    for key, name in (("classes", "name"), ("relationships", "source"), ("legal_patterns", "pattern_name")):
        for item in onto.get(key) or []:
            if isinstance(item, dict):
                fix(item, str(item.get(name, "?")))


def fill_provision(onto: dict, original: dict | None, notes: list, default: str | None = "article") -> None:
    """Rules that lost `provision` get it back from the identical rule of the previous
    version; otherwise `default` (the schema default) is used."""
    known = {}
    if original:
        for k in RULE_KINDS:
            for r in (original.get("rights_model") or {}).get(k) or []:
                if r.get("provision"):
                    known[(_rule_label(r), r.get("party"), r.get("asset"))] = r["provision"]
    rm = onto.get("rights_model")
    if not isinstance(rm, dict):
        return
    for k in RULE_KINDS:
        for r in rm.get(k) or []:
            if r.get("provision"):
                continue
            p = known.get((_rule_label(r), r.get("party"), r.get("asset")))
            if p:
                r["provision"] = p
                notes.append({"code": "provision_restored", "detail": _rule_label(r)})
            elif default:
                r["provision"] = default
                notes.append({"code": "provision_defaulted", "detail": _rule_label(r)})


# ------------------------------------------------- provision: recital or article?
_HEADING = re.compile(r"(?<![\w(])Article\s+\d+[a-z]?(?=\s+(?:\d+\.|[A-Z(]))")
_WHEREAS = re.compile(r"\bWhereas\b")


def _region(text: str, pos: int) -> str | None:
    """'recital' after a 'Whereas' with no article heading in between, 'article' after an
    article heading, None when the text gives no evidence (e.g. the chunk starts mid-article)."""
    last_w = max((m.start() for m in _WHEREAS.finditer(text, 0, pos)), default=-1)
    last_h = max((m.start() for m in _HEADING.finditer(text, 0, pos)), default=-1)
    if last_w >= 0 and last_h < last_w:
        return "recital"
    if last_h >= 0:
        return "article"
    return None


def provision_evidence(rule: dict, text: str) -> str | None:
    """'recital' / 'article' when the places where the rule's verb occurs in the text say so,
    otherwise None. An occurrence in an article wins (a rule is kept when in doubt)."""
    words = [w for w in _rule_label(rule).lower().split("_") if w]
    if not words or len(words[0]) < 3 or not text:
        return None
    stem = _stem(words[0])
    regions = {_region(text, m.start()) for m in re.finditer(r"\b" + re.escape(stem), text.lower())}
    regions.discard(None)
    if not regions:
        return None
    return "article" if "article" in regions else "recital"


def apply_provision_evidence(onto: dict, text: str, notes: list, drop_recital: bool = True) -> None:
    """Correct `provision` where the text gives evidence, and drop duties/prohibitions that
    only occur in recitals (recitals explain, they do not bind)."""
    rm = onto.get("rights_model")
    if not isinstance(rm, dict):
        return
    for k in RULE_KINDS:
        kept = []
        for r in rm.get(k) or []:
            ev = provision_evidence(r, text)
            if ev and r.get("provision") != ev:
                notes.append({"code": "provision_corrected",
                              "detail": f"{_rule_label(r)}: {r.get('provision')} -> {ev}"})
                r["provision"] = ev
            if drop_recital and ev == "recital" and k in ("duties", "prohibitions"):
                notes.append({"code": "rule_from_recital_dropped", "detail": f"{k}:{_rule_label(r)}"})
                continue
            kept.append(r)
        rm[k] = kept


# ------------------------------------------- party close to the action in the text
def _words(s) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s or ""))
    return [w.lower() for w in re.findall(r"[A-Za-z]+", s)]


def party_far_from_action(rule: dict, text: str, max_gap: int = 150) -> bool:
    """True when the verb occurs in the text but no word of the party name is within
    `max_gap` characters of it. Heuristic: used as a warning, never to delete."""
    words = [w for w in _rule_label(rule).lower().split("_") if w]
    party = [w for w in _words(rule.get("party")) if len(w) > 3]
    if not words or not party or not text:
        return False
    low = text.lower()
    verbs = [m.start() for m in re.finditer(r"\b" + re.escape(_stem(words[0])), low)]
    if not verbs:
        return False
    names = [m.start() for w in party for m in re.finditer(r"\b" + re.escape(_stem(w)), low)]
    if not names:
        return False                      # party not in the text at all: other checks cover it
    return min(abs(v - n) for v in verbs for n in names) > max_gap


# ------------------------------------------------------------------- entry point
def cleanup_ontology(onto: dict, text: str, original: dict | None = None,
                     caps: dict | None = None, description_cap: int = 300,
                     default_provision: str | None = "article",
                     drop_recital_rules: bool = True) -> tuple[dict, list]:
    """Apply every step. Returns (clean_ontology, notes); the input is not modified."""
    onto = copy.deepcopy(onto)
    notes: list = []
    warn_lists_at_cap(onto, caps or DEFAULT_CAPS, notes)
    drop_duplicate_rules(onto, notes)
    drop_ungrounded_actions(onto, text, notes)
    fix_examples(onto, text, notes)
    trim_descriptions(onto, notes, description_cap)
    fill_provision(onto, original, notes, default_provision)
    apply_provision_evidence(onto, text, notes, drop_recital=drop_recital_rules)
    return onto, notes