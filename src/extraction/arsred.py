"""Parses ArSRED's (U4RASD/ArSRED) linearized `relations` string into Triples.

The format, confirmed by hand against real rows:

    (<bor> SUBJECT <SUBJECT_TYPE> (OBJECT <OBJECT_TYPE> RELATION <rt> EVIDENCE <e>)+ )*

Each `<bor>` starts a new subject. Every subject can be followed by *more than
one* fact before the next `<bor>`, all sharing that subject: `<bor>` marks a
new subject, not a new fact. Missing this produced wrong subject/object
splits in an earlier throwaway analysis pass; this version is verified
against known rows before being trusted for the real pipeline.
"""

import re

from .triple import Triple

_TAG = re.compile(r"<(\w+)>")


def parse_arsred_relations(relations_str: str) -> list[Triple]:
    triples: list[Triple] = []
    for block in relations_str.split("<bor>")[1:]:
        parts = [p.strip() for p in _TAG.split(block)]
        # parts alternates: [subject, subject_type, obj, obj_type, relation, "rt", evidence, "e", obj, obj_type, ...]
        if len(parts) < 8:
            continue  # too short to hold subject+type plus one full fact; skip rather than guess
        subject, subject_type = parts[0], parts[1]
        if not subject:
            continue
        rest = parts[2:]
        for i in range(0, len(rest) - 5, 6):
            obj, obj_type, relation, rt_tag, _evidence, e_tag = rest[i : i + 6]
            if rt_tag != "rt" or e_tag != "e" or not obj or not relation:
                continue  # malformed fact, skip rather than emit a wrong triple
            triples.append(
                Triple(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    subject_type=subject_type or None,
                    object_type=obj_type or None,
                )
            )
    return triples
