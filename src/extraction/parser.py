"""Parses model output into Triples. Two sources, two formats:

- LLM chat models: asked for a JSON array of {subject, relation, object}
  (prompts/study2_ontology_extraction_ar.txt). 13 different local models
  will not all follow the format equally well, so extraction is tolerant of
  code fences and surrounding text, not a bare `json.loads`.
- mREBEL: a seq2seq model, output format confirmed by direct inspection
  (checked by running the model, not read off documentation) to be
  `<triplet> head <head_type> tail <tail_type> relation` repeated per fact,
  wrapped in `<s>...</s>`.
"""

import json
import re

from .triple import Triple

_CODE_FENCE = re.compile(r"```(?:json)?")
_MREBEL_STRUCTURAL_TOKENS = ("<s>", "</s>", "<pad>")
_MREBEL_TAG = re.compile(r"^<\w+>$")


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced, likely truncated by max_tokens


def _complete_objects(text: str) -> list[str]:
    """Returns the source of every balanced `{...}` block in `text`, in order.

    Brace depth is tracked outside string literals only, so a brace inside a
    quoted Arabic span cannot open or close a block. Used to salvage the
    triples a truncated response finished writing before it was cut.
    """
    blocks: list[str] = []
    depth = 0
    start = None
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : i + 1])
                start = None
    return blocks


def _triples_from_items(data: list) -> list[Triple]:
    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject, relation, obj = item.get("subject"), item.get("relation"), item.get("object")
        if not (isinstance(subject, str) and isinstance(relation, str) and isinstance(obj, str)):
            continue
        subject, relation, obj = subject.strip(), relation.strip(), obj.strip()
        if not subject or not relation or not obj:
            continue
        triples.append(Triple(subject=subject, relation=relation, object=obj))
    return triples


def parse_llm_output(raw_text: str) -> tuple[list[Triple], bool]:
    """Returns (triples, valid).

    `valid` is False when no complete JSON array could be located and decoded,
    per the output-validity metric: a schema-shaped
    response with a malformed item inside it still counts as valid output, just
    as a partial extraction.

    A response cut off by `max_tokens` mid-array is the one case where those
    two things come apart. Its array never closes, so it is not valid output,
    but the objects it finished writing before the cut are complete and
    well-formed, and the model did produce them. Discarding them charged the
    model for a harness setting: for six of the ten chat models most invalid
    responses were truncations, and Llama3.1 lost 537 well-formed triples this
    way, about 13% of its output. Those objects are now recovered and scored;
    `valid` still reports False, so output validity keeps measuring whether a
    response closed properly and the truncation rate stays visible in
    scripts/06_diagnose_validity.py.

    Salvage runs only when an opening `[` was found but never closed, which is
    the truncation signature. A response with no array at all is still invalid
    and contributes nothing, so a model that ignores the schema entirely gains
    nothing here.
    """
    text = _CODE_FENCE.sub("", raw_text)
    candidate = _extract_json_array(text)
    if candidate is None:
        if "[" not in text:
            return [], False
        salvaged = []
        for block in _complete_objects(text[text.find("[") :]):
            try:
                item = json.loads(block)
            except json.JSONDecodeError:
                continue
            salvaged.append(item)
        return _triples_from_items(salvaged), False
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return [], False
    if not isinstance(data, list):
        return [], False
    return _triples_from_items(data), True


def parse_mrebel_output(raw_text: str) -> list[Triple]:
    text = raw_text
    for tok in _MREBEL_STRUCTURAL_TOKENS:
        text = text.replace(tok, " ")

    triples: list[Triple] = []
    state = None  # "subject" | "object" | "relation"
    subject = obj = relation = ""
    subject_type = obj_type = ""

    def flush():
        if subject and relation and obj:
            triples.append(
                Triple(
                    subject=subject.strip(),
                    relation=relation.strip(),
                    object=obj.strip(),
                    subject_type=subject_type or None,
                    object_type=obj_type or None,
                )
            )

    for token in text.split():
        if token in ("<triplet>", "<relation>"):
            flush()
            state = "subject"
            subject = obj = relation = ""
            subject_type = obj_type = ""
        elif _MREBEL_TAG.match(token):
            tag = token[1:-1]
            if state == "subject":
                subject_type = tag
                state = "object"
            elif state == "object":
                obj_type = tag
                state = "relation"
            elif state == "relation":
                # A tag while a relation is being read marks the end of that
                # fact and the start of another one sharing the same subject.
                # mREBEL emits, for one subject with two facts:
                #
                #   <triplet> S <loc> O1 <loc> rel1 <loc> O2 <loc> rel2
                #
                # Treating this tag as a decoding artifact and ignoring it,
                # as this branch previously did, kept appending tokens to the
                # relation, producing names like
                # "country الولايات المتحدة country": the second object was
                # swallowed into the first fact's relation, so one triple was
                # emitted with a corrupt relation and the other was lost
                # entirely. That deflated the baseline's ontology conformance
                # (58.8%) and its recall, and it accounted for the majority of
                # its apparent relation errors.
                flush()
                obj = relation = ""
                obj_type = ""
                state = "object"
        elif state == "subject":
            subject = f"{subject} {token}".strip()
        elif state == "object":
            obj = f"{obj} {token}".strip()
        elif state == "relation":
            relation = f"{relation} {token}".strip()

    flush()
    return triples
