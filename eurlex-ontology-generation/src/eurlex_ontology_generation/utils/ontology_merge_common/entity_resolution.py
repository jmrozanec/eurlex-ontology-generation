"""Entity resolution shared by ontology_merge_partition and ontology_merge_global"""
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
CLASS_EMBEDDINGS = True          # embeddings are used for class names only
HEAD_NOUN_BYPASS_SIM = 0.97      # different head noun is tolerated only above this similarity
NO_SHARED_TOKEN_MIN_SIM = 0.95   # no shared token at all -> need at least this similarity

# Names of the core ODRL-like types: they are structural, so they can never be domain classes
CORE_TYPE_KEYS = {"asset", "party", "permission", "prohibition", "duty", "constraint", "rightsholder"}

# names that are prompt placeholders, never real entities
GENERIC_REFS = {"targetparty", "targetasset", "party", "asset", "permission", "duty", "constraint",
                "prohibition", "rightsholder", "unknown", "none", "null"}

# class names that describe a single instance rather than a concept
INSTANCE_NAME_PATTERNS = [
    re.compile(r"article[\s_]?\d", re.I),
    re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)"),
    re.compile(r"\d+\s*/\s*\d+"),            
    re.compile(r"\(.+\)"),                   
    re.compile(r"\bEUR\b|\bECU\b"),
    re.compile(r"\d{3,}"),                   
    re.compile(r"^\d"),                      
    re.compile(r"(?<![A-Za-z])MHz", re.I),
    re.compile(r"(?<![A-Za-z])(GmbH|Oy|Ltd|Inc|S\.?A\.?|AG|SpA|BV|NV)(?![a-z])"),
    re.compile(r"^(Council_|Commission_)?(Regulation|Directive|Decision)[\s_]+(EC|EU|EEC|Euratom)", re.I),
]
INSTANCE_DESC_PATTERN = re.compile(r"\b(company|airline|enterprise|suppliers?)\b", re.I)

# leakage from the generation prompt (import/export duty, cereal, 1985 ...)
LEAKY_ACTION_PATTERNS = [re.compile(p, re.I) for p in (r"import_duty", r"export_duty", r"levy")]
LEAKY_ACTION_EXACT = {"no_action", "none", "n_a", "null", "unknown"}
LEAKY_CONSTRAINT_TEXT = ("cereal",)

# constraint values that describe one concrete case (amounts, dates, years) are not ontology knowledge.
# True  -> such constraints are dropped from the merged ontology
# False -> they are kept
DROP_INSTANCE_CONSTRAINT_VALUES = True
INSTANCE_VALUE_PATTERNS = [
    re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)"),
    re.compile(r"\bEUR\b|\bECU\b|€"),
    re.compile(r"\d[\d\s.,]{3,}"),
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I),
]
GENERIC_SUFFIX_TOKENS = {"party", "asset"}   # CommissionParty / PRIMAISAsset -> Commission / PRIMAIS
LEAKY_CONSTRAINT_VALUES = {"1985"}

# values that mean "nothing" and must be treated as missing
PLACEHOLDER_VALUES = {"", "null", "none", "unknown", "n/a", "na", "tbd", "nan", "undefined"}
PLACEHOLDER_PREFIXES = ("to be determined", "not specified")

# Pairs of tokens with opposite meaning: names that differ by such a pair must never be merged
OPPOSITE_TOKENS = [
    {"provide", "receive"}, {"suspend", "terminate"}, {"promote", "protect"},
    {"import", "export"}, {"increase", "reduce"}, {"allow", "prohibit"},
    {"contribute", "receive"}, {"grant", "revoke"},
]

# Hand-curated aliases: canonical display name -> known variants.
# Used to force-merge well-known institutions that embeddings might miss
ALIAS_GROUPS: dict[str, list[str]] = {
    "European_Union": ["Union", "EU", "European Union", "EuropeanUnion"],
    "European_Commission": ["Commission", "EuropeanCommission", "European Commission"],
    "European_Parliament": ["Parliament", "EuropeanParliament", "European Parliament"],
    "Council": ["Council of the European Union", "Council of the EU"],
    "European_Council": [],
    "Member_State": ["MemberState", "Member States", "MemberStateParty"],
    "European_External_Action_Service": ["EEAS", "European External Action Service"],
    "OLAF": ["European Anti-Fraud Office", "EuropeanAntiFraudOffice"],
    "Court_of_Auditors": ["CourtOfAuditors", "European Court of Auditors"],
    "IMF": ["International Monetary Fund"],
    "EGF": ["European Globalisation Adjustment Fund", "Globalisation Adjustment Fund",
            "EuropeanGlobalizationAdjustmentFund", "GlobalizationAdjustmentFund"],
    "Republic_of_Moldova": ["Moldova", "RepublicOfMoldova"],
}

# Name normalisation
_SPELLING = [("globalis", "globaliz"), ("mobilis", "mobiliz"), ("authoris", "authoriz"),
             ("organis", "organiz"), ("recognis", "recogniz"), ("harmonis", "harmoniz"),
             ("utilis", "utiliz"), ("minimis", "minimiz"), ("finalis", "finaliz")]


def _split_identifier(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _singular(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")) and len(tok) > 3:
        return tok[:-1]
    
    return tok


def norm_tokens(name: str) -> list[str]:
    toks = []
    
    for t in _split_identifier(name):
        for a, b in _SPELLING:
            if t.startswith(a):
                t = b + t[len(a):]
                break
        toks.append(t)
    if toks:
        toks[-1] = _singular(toks[-1])
    
    return toks


def _plain_key(name: str) -> str:
    return "".join(norm_tokens(name))


# Lookup tables built once at import time from ALIAS_GROUPS:
#   ALIAS_KEY:     plain key of any variant (or display name) -> plain key of the display name
#   ALIAS_DISPLAY: plain key of the display name -> display name
ALIAS_KEY: dict[str, str] = {}
ALIAS_DISPLAY: dict[str, str] = {}
for _display, _variants in ALIAS_GROUPS.items():
    _dk = _plain_key(_display)
    ALIAS_DISPLAY[_dk] = _display
    ALIAS_KEY[_dk] = _dk
    
    for _v in _variants:
        ALIAS_KEY[_plain_key(_v)] = _dk


# Exact-merge key: two names with the same key are the same entity
def group_key(entity_type: str, name: str) -> str:
    if entity_type == "action":
        return "_".join(norm_tokens(name))
    
    toks = norm_tokens(name)
    
    if len(toks) > 1 and toks[-1] in GENERIC_SUFFIX_TOKENS:
        toks = toks[:-1]
    k = "".join(toks)
    
    return ALIAS_KEY.get(k, k)


def normalize_action(name: str) -> str:
    return "_".join(_split_identifier(name))

# Same normalisation as for actions, but tolerant of non-string input (returns None)
def normalize_relation(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    out = "_".join(_split_identifier(name))
    return out or None

# Chunk ids look like "<CELEX>_<n>": keep only the CELEX (document) part when it matches the format
def celex_of(source_id: str) -> str:
    return source_id.split("_")[0] if re.match(r"^\d{5}[A-Z]\d{4}", source_id) else source_id


# Filters
def is_placeholder(value: Any) -> bool:
    v = str(value).strip().lower()
    return (v in PLACEHOLDER_VALUES or v.startswith(PLACEHOLDER_PREFIXES)
            or (v.startswith("<") and v.endswith(">")))


# True for names that must never become a class (core-type names, instances, placeholders)
def is_rejected_class(name: str, description: str = "") -> bool:
    if not isinstance(name, str) or not name.strip():
        return True
    if _plain_key(name) in CORE_TYPE_KEYS or _plain_key(name) in GENERIC_REFS:
        return True
    if any(p.search(name) for p in INSTANCE_NAME_PATTERNS):
        return True
    if description and INSTANCE_DESC_PATTERN.search(description):
        return True
    
    return False


def is_leaky_action(action: str) -> bool:
    return action in LEAKY_ACTION_EXACT or any(p.search(action) for p in LEAKY_ACTION_PATTERNS)

# Returns a cleaned {"name", "value"} dict, or None if the constraint must be discarded
def clean_constraint(c: dict) -> dict | None:
    name, value = c.get("name"), c.get("value")
    
    if not name or value is None or is_placeholder(value) or is_placeholder(name):
        return None
    if str(name).strip().lower().endswith("constraint"):
        return None
    if DROP_INSTANCE_CONSTRAINT_VALUES and any(p.search(str(value)) for p in INSTANCE_VALUE_PATTERNS):
        return None
    text = f"{name} {value}".lower()
    if any(t in text for t in LEAKY_CONSTRAINT_TEXT) or str(value).strip() in LEAKY_CONSTRAINT_VALUES:
        return None
    
    return {"name": str(name), "value": str(value)}

# separators for compound names: comma, semicolon, ampersand, or the word "and" (optionally "and the")
_SEP = re.compile(r"\s*[,;&]\s*|(?:^|[\s_])and(?:[\s_]the)?(?:[\s_]|$)", re.I)


def split_entities(name: str) -> list[str]:
    parts = [p.strip() for p in _SEP.split(name.strip()) if p and p.strip()]
    if len(parts) > 1:
        return parts
    camel = re.split(r"(?<=[a-z])And(?:The)?(?=[A-Z])", name.strip())
    return camel if len(camel) > 1 else [name.strip()]


def is_composite_class(name: str) -> bool:
    # 'EuropeanParliamentAndCouncil' -> True (all parts are known institutions)
    parts = split_entities(name)
    return len(parts) > 1 and all(group_key("class", p) in ALIAS_DISPLAY for p in parts)


# Extraction

def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def _weight(obj: dict) -> int:
    s = obj.get("support")
    try:
        return int(s.get("chunks", 1)) if isinstance(s, dict) else 1
    except (TypeError, ValueError):
        return 1


def extract_entities(ontologies: pd.DataFrame) -> pd.DataFrame:
    """One row per mention of a class name or an action name (parties/assets are NOT mentions:
    they are resolved against the class namespace afterwards)"""
    rows = []
    for _, row in ontologies.iterrows():
        try:
            onto = json.loads(row["ontology"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(onto, dict):
            continue
        source_id = str(row["chunk_id"])
        celex = str(row.get("CELEX", "") or "")

        for cls in _dicts(onto.get("classes")):
            name = cls.get("name")
            if is_rejected_class(name, cls.get("description") or "") or is_composite_class(name):
                continue
            rows.append({"entity_type": "class", "name": name.strip(), "declared_type": cls.get("type"),
                         "weight": _weight(cls), "source_id": source_id, "CELEX": celex})

        rm = onto.get("rights_model")
        if isinstance(rm, dict):
            for kind in ("permissions", "duties", "prohibitions"):
                for rule in _dicts(rm.get(kind)):
                    action = rule.get("action")
                    
                    if not isinstance(action, str) or not action.strip():
                        continue
                    action = normalize_action(action)
                    if not action or is_leaky_action(action):
                        continue
                    rows.append({"entity_type": "action", "name": action, "declared_type": "Action",
                                 "weight": _weight(rule), "source_id": source_id, "CELEX": celex})
    
    return pd.DataFrame(rows)


# Embeddings + clustering
def compute_embeddings(texts: list[str], embedding_model: str, batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    
    from ollama import embed

    vectors: list[list[float]] = []
    
    for start in range(0, len(texts), batch_size):
        response = embed(model=embedding_model, input=texts[start:start + batch_size])
        vectors.extend(response["embeddings"])
    
    return np.array(vectors, dtype=np.float32)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _mergeable(a: list[str], b: list[str], sim: float) -> bool:
    # Veto rules applied on top of the embedding similarity
    if {t for t in a if t.isdigit()} != {t for t in b if t.isdigit()}:
        return False                                   
    
    for pair in OPPOSITE_TOKENS:
        if (set(a) & pair) and (set(b) & pair) and (set(a) & pair) != (set(b) & pair):
            return False
    if a and b and a[-1] != b[-1] and sim < HEAD_NOUN_BYPASS_SIM:
        return False                                   # ...Site vs ...Asset, ...Committee vs ...Decision
    if not (set(a) & set(b)) and sim < NO_SHARED_TOKEN_MIN_SIM:
        return False
    
    return True


# keys: unique exact-merge keys of the classes; reps: key -> most frequent surface name
# Returns a union-find over the indices of `keys`
def _cluster_class_keys(keys: list[str], reps: dict[str, str], model: str, threshold: float) -> _UnionFind:
    uf = _UnionFind(len(keys))
    
    if not CLASS_EMBEDDINGS or len(keys) < 2:
        return uf
    try:
        vecs = compute_embeddings([" ".join(_split_identifier(reps[k])) for k in keys], model)
    except Exception as exc:  # ollama not running etc.
        logger.warning("Embeddings unavailable (%s): class merge limited to exact keys.", exc)
        return uf
    
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = vecs / norms
    toks = [norm_tokens(reps[k]) for k in keys]
    block = 512
    
    for start in range(0, len(keys), block):
        sims = unit[start:start + block] @ unit.T
        ii, jj = np.where(sims >= threshold)
        for i, j in zip(ii + start, jj):
            if j > i and _mergeable(toks[i], toks[j], float(sims[i - start, j])):
                uf.union(int(i), int(j))
    
    return uf


# Adds `key` and `cluster_id` columns; cluster ids are unique across entity types
def cluster_entities(entities: pd.DataFrame, embedding_model: str, similarity_threshold: float) -> pd.DataFrame:
    entities = entities.copy()
    
    if entities.empty:
        entities["cluster_id"] = pd.Series(dtype=int)
        return entities

    entities = entities.reset_index(drop=True)
    entities["key"] = [group_key(t, n) for t, n in zip(entities["entity_type"], entities["name"])]
    entities["cluster_id"] = -1

    next_id = 0
    for etype, group in entities.groupby("entity_type"):
        keys = sorted(group["key"].unique())
        
        if etype == "class":
            reps = group.groupby("key")["name"].agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
            uf = _cluster_class_keys(keys, reps, embedding_model, similarity_threshold)
            root_of = {k: uf.find(i) for i, k in enumerate(keys)}
        else:                                           # actions: exact normalised key only
            root_of = {k: i for i, k in enumerate(keys)}
        
        roots = {r: next_id + n for n, r in enumerate(sorted(set(root_of.values())))}
        next_id += len(roots)
        entities.loc[group.index, "cluster_id"] = [roots[root_of[k]] for k in group["key"]]
    
    return entities


# Returns:
# canonical_map: {entity_type: {variant name: canonical name}}
# report:        one row per cluster, for inspection / auditing the merge
def canonicalize_clusters(entities_with_clusters: pd.DataFrame) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    canonical_map: dict[str, dict[str, str]] = {}
    report_rows = []
    if entities_with_clusters.empty:
        return canonical_map, pd.DataFrame(report_rows)

    for etype, group in entities_with_clusters.groupby("entity_type"):
        canonical_map[etype] = {}
        
        for _, cg in group.groupby("cluster_id"):
            weights: Counter = Counter()
            for n, w in zip(cg["name"], cg["weight"]):
                weights[n] += int(w)
            alias = ALIAS_DISPLAY.get(cg["key"].iloc[0]) if etype == "class" else None
            canonical = alias or max(
                weights, key=lambda n: (not re.search(r"(?<=[a-z])(Party|Asset)$", n), weights[n], -len(n)))
            
            if not alias:
                canonical = re.sub(r"(?<=[a-z])(Party|Asset)$", "", canonical)   # TrusteesParty -> Trustees
                canonical = re.sub(r"\s+", "_", canonical.strip())
            
            variants = sorted(weights)
            
            for v in variants:
                canonical_map[etype][v] = canonical
            types = Counter(t for t in cg["declared_type"] if t)
            report_rows.append({
                "entity_type": etype,
                "canonical_name": canonical,
                "majority_declared_type": types.most_common(1)[0][0] if types else None,
                "variant_count": len(variants),
                "variants": json.dumps(variants, ensure_ascii=False),
                "mention_count": int(sum(weights.values())),
                "n_documents": len({celex_of(str(s)) for s in cg["source_id"]}),
            })
    
    return canonical_map, pd.DataFrame(report_rows)


# Single resolver for class / party / asset names. Returns the canonical class name or None
# Index by exact-merge key so that small spelling/format differences still resolve
def build_resolver(canonical_map: dict[str, dict[str, str]]) -> Callable[[Any], str | None]:
    # Single resolver for class / party / asset names. Returns the canonical class name or None
    index = {group_key("class", v): c for v, c in canonical_map.get("class", {}).items()}

    def resolve(name: Any) -> str | None:
        if not isinstance(name, str) or not name.strip() or is_rejected_class(name):
            return None
        k = group_key("class", name)
        
        return index.get(k) or ALIAS_DISPLAY.get(k)

    return resolve