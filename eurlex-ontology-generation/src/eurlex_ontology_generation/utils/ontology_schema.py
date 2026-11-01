"""Pydantic schema for chunk-level ontologies, built from the controlled vocabulary.

Passed to Ollama as `format=<JSON schema>` so the model can only emit the
allowed class types, upper classes, actions, constraint operands, operators and
relations (grammar-constrained decoding). The same model validates revisions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, create_model

from .vocabulary import OTHER_ACTION, Vocabulary, load_vocabulary

PROVISIONS = ("article", "recital", "annex", "other")
KINDS = ("class", "individual")

# Upper bounds that make the grammar-constrained output finite. Without them a small
# model can stay inside one string (or one list) until num_predict runs out.
Name = Annotated[str, StringConstraints(max_length=100)]    # identifiers, labels
Text = Annotated[str, StringConstraints(max_length=300)]    # descriptions
Quote = Annotated[str, StringConstraints(max_length=600)]   # sentence copied from the text
Unit = Annotated[str, StringConstraints(max_length=30)]


def _lit(values) -> object:
    return Literal[tuple(values)]  # type: ignore[valid-type]


class ODRLAlignment(BaseModel):
    core_entities: list[Name] = Field(max_length=10)
    extensions: list[Name] = Field(max_length=40)


class LegalPattern(BaseModel):
    pattern_name: Name
    description: Text
    entities_involved: list[Name] = Field(max_length=20)


class ExampleInstance(BaseModel):
    entity: Name
    example_value: Text = Field(description="Short value copied from the text.")
    source_context: Quote = Field(description="The sentence the value comes from, copied from the text.")


@lru_cache(maxsize=4)
def build_ontology_model(vocabulary_path: str | None = None) -> type[BaseModel]:
    """Return the OntologyExtraction model for the given vocabulary file."""
    vocab: Vocabulary = load_vocabulary(vocabulary_path)

    OntologyClass = create_model(
        "OntologyClass",
        name=(Name, Field(description="CamelCase or snake_case name, no spaces.")),
        type=(_lit(vocab.core_types), ...),
        is_a=(_lit(vocab.upper_names), Field(description="Upper-ontology parent.")),
        kind=(_lit(KINDS), Field(description="'individual' for one named thing (a country, an institution, a named fund); 'class' for a kind of thing.")),
        description=(Text, ...),
    )
    Relationship = create_model(
        "Relationship",
        source=(Name, ...),
        relation=(_lit(vocab.relation_names), ...),
        target=(Name, ...),
        description=(Text, ""),
    )
    Constraint = create_model(
        "Constraint",
        left_operand=(_lit(vocab.left_operand_names), ...),
        operator=(_lit(vocab.operators), ...),
        right_operand=(Name, Field(description="Value copied from the text; dates as YYYY-MM-DD.")),
        unit=(Unit, ""),
    )
    rule_fields = dict(
        action=(_lit([*vocab.action_names, OTHER_ACTION]), Field(description="Canonical verb.")),
        action_label=(Name, Field(description="The action as worded in the text, snake_case, e.g. provide_financial_contribution.")),
        party=(Name, Field(description="Class or individual name of the party that holds the permission, prohibition or duty (ODRL assignee).")),
        assigner=(Name, Field("", description="Party that grants the permission or imposes the duty, if stated.")),
        asset=(Name, Field("", description="Class or individual name the action applies to.")),
        provision=(_lit(PROVISIONS), Field("article", description="Where the rule is stated. Recitals are not binding.")),
        constraints=(list[Constraint], Field(default_factory=list, max_length=6)),
    )
    Rule = create_model("Rule", **rule_fields)
    RightsModel = create_model(
        "RightsModel",
        permissions=(list[Rule], Field(default_factory=list, max_length=20)),
        prohibitions=(list[Rule], Field(default_factory=list, max_length=20)),
        duties=(list[Rule], Field(default_factory=list, max_length=20)),
    )
    return create_model(
        "OntologyExtraction",
        ontology_name=(Name, ...),
        description=(Text, ...),
        odrl_alignment=(ODRLAlignment, ...),
        classes=(list[OntologyClass], Field(..., max_length=40)),
        relationships=(list[Relationship], Field(default_factory=list, max_length=30)),
        rights_model=(RightsModel, ...),
        legal_patterns=(list[LegalPattern], Field(default_factory=list, max_length=10)),
        example_instances=(list[ExampleInstance], Field(default_factory=list, max_length=10)),
    )


EXPECTED_TOP_LEVEL_KEYS = [
    "ontology_name", "description", "odrl_alignment", "classes", "relationships",
    "rights_model", "legal_patterns", "example_instances",
]
RULE_KINDS = ("permissions", "prohibitions", "duties")