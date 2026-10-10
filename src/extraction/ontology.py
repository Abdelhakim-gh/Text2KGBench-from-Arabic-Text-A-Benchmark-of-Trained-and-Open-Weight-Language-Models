"""Loads the closed ontology (configs/ontology.yaml): the 20 allowed
relations and the 10 entity-type buckets, built by
scripts/01_build_ontology_and_testset.py (the paper, Sect. 3.1, explains the choice).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Relation:
    id: int
    name_en: str
    gloss_ar: str
    count_sample: int


@dataclass(frozen=True)
class EntityType:
    id: str
    raw_tags: tuple[str, ...]
    literal: bool  # True for date/time/number: an object of this type marks an attribute-style triple


@dataclass(frozen=True)
class Ontology:
    relations: tuple[Relation, ...]
    entity_types: tuple[EntityType, ...]

    def __post_init__(self):
        names = [r.name_en for r in self.relations]
        assert len(names) == len(set(names)), "duplicate relation name in ontology"

    @property
    def relation_names(self) -> frozenset[str]:
        return frozenset(r.name_en for r in self.relations)

    def raw_tag_to_type_id(self, raw_tag: str) -> str:
        """Maps one of ArSRED's 13 raw entity-type tags onto one of the 10
        ontology buckets. Unknown tags fall back to 'other' rather than raising,
        since the raw tag set is fixed by ArSRED and any future tag is by
        definition unclassified."""
        for et in self.entity_types:
            if raw_tag in et.raw_tags:
                return et.id
        return "other"

    def is_literal_type(self, type_id: str) -> bool:
        for et in self.entity_types:
            if et.id == type_id:
                return et.literal
        return False

    def prompt_relation_list(self) -> str:
        """Formats the relation list for the Arabic extraction prompt:
        one line per relation, English canonical name plus Arabic gloss."""
        return "\n".join(f"{r.id}. {r.name_en} - {r.gloss_ar}" for r in self.relations)


def load_ontology(path: Path | str) -> Ontology:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    relations = tuple(
        Relation(id=r["id"], name_en=r["name_en"], gloss_ar=r["gloss_ar"], count_sample=r["count_sample"])
        for r in raw["relations"]
    )
    entity_types = tuple(
        EntityType(id=et["id"], raw_tags=tuple(et["raw_tags"]), literal=bool(et.get("literal", False)))
        for et in raw["entity_types"]
    )
    return Ontology(relations=relations, entity_types=entity_types)
