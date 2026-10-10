"""Triple-level scoring against ArSRED gold: precision/recall/F1 (overall
and per relation), ontology conformance, hallucination counts (SH/RH/OH),
and Arabic-normalization-aware span matching. Every score is computed
twice, raw and normalised (scripts/07_diagnose_normalisation.py shows why the
two agree on this task).
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarabic.araby as araby
import yaml
from rapidfuzz import fuzz

from .triple import Triple

DEFAULT_FUZZY_THRESHOLD = 86.0  # rapidfuzz.fuzz.ratio, 0-100
# Frozen 2026-08-13, after sweeping 60-100 over all
# sanity_100 extraction files (10-model roster plus mREBEL) and reading every
# distinct near-miss span pair between 80 and 96 by hand against its source
# sentence, rather than picking the value that maximises F1 (F1 rises
# monotonically as the threshold drops, so "maximise it" has no answer).
#
# 86 sits inside the only gap in the observed ratio distribution: nothing in
# sanity_100 scores between 85.7 and 87.5, so 86 and 87 give identical results
# on this data and the choice is not a knife-edge one. It accepts genuine
# near-misses, e.g. predicted "دبليو. ويزلي بيترسون" against gold "و. ويزلي
# بيترسون" (ratio 88.9, the sentence itself uses the fuller form), and it
# still rejects the one confirmed confusion in the set: predicted "نهائي كوبا
# أمريكا 2001" against gold "كوبا أمريكا 2001" (ratio 84.2) are two different
# things, per that sentence's own gold "part of" triple relating one to the
# other.
#
# This uses fuzz.ratio, not fuzz.partial_ratio. An earlier design named
# partial-ratio, but partial_ratio scores any substring as a perfect
# match (e.g. predicted "2 سبتمبر 1986" against gold "1986", or predicted
# "الولايات المتحدة الأمريكية" against gold "أمريكي", would both count as
# exact), which inflates pooled F1 from 0.255 to 0.342 at this threshold on
# substring accidents rather than real near-misses.
#
# A country name against its demonym used to score inconsistently, because
# rapidfuzz has no notion of Arabic morphology and each pair happens to share
# a different number of letters: "إيطاليا" against "إيطالي" scores 92.3 and is
# accepted, while "السويد" against "سويدية" scores 66.7 and is rejected,
# though the two pairs stand in exactly the same relation. Which of the two a
# model was charged for depended on the length of the country's name, which is
# not a property of the model. `_nisba_stem` below removes that accident.


_AR_NISBA_SUFFIXES = ("يون", "يين", "ية", "يه", "ي", "ة", "ه", "ا")
_MIN_STEM_LEN = 3


def _nisba_stem(token: str) -> str:
    """Strips the definite article and the nisba (relational-adjective)
    ending from one token, so that a place name and the adjective formed from
    it reduce to the same stem: إيطاليا and إيطالي both to إيطال, السويد and
    سويدية both to سويد.

    Why span matching should accept the pair at all: the prompt requires the
    model to copy spans out of the source sentence, and ArSRED's gold spans
    are cut from that same sentence. Where the sentence carries the nisba form
    (`رياضياتي مجري`, "a Hungarian mathematician") the model copying it has
    copied correctly, and gold's citizenship object for the same triple is the
    country. This is morphology on one entity, not two different entities, and
    not the near-synonym relation substitution that stays out of scope.

    Deliberately blunt, and it has to be: an Arabic demonym lexicon is what
    would do this properly, and we do not have one. Stripping is capped at two
    passes (إيطاليا loses ا then ي) and a stem shorter than three characters
    is rejected by `_stems_equal`, so short tokens cannot collapse into each
    other.
    """
    if token.startswith("ال") and len(token) > 4:
        token = token[2:]
    for _ in range(2):
        for suffix in _AR_NISBA_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
                token = token[: -len(suffix)]
                break
        else:
            break
    return token


def _stems_equal(a: str, b: str) -> bool:
    """Token-by-token stem comparison, used only after exact and fuzzy
    matching have both failed. Requires the same token count and at least one
    stem of three characters or more, so that neither a length mismatch nor a
    pair of two-letter tokens can produce a match."""
    a_tokens, b_tokens = a.split(), b.split()
    if not a_tokens or len(a_tokens) != len(b_tokens):
        return False
    a_stems = [_nisba_stem(t) for t in a_tokens]
    b_stems = [_nisba_stem(t) for t in b_tokens]
    if a_stems != b_stems:
        return False
    return max(len(s) for s in a_stems) >= _MIN_STEM_LEN


ATTRIBUTE_OBJECT_TYPES = frozenset({"date", "time", "number"})
# An object typed Date, Time or Number marks an attribute-style triple (a
# literal value attached to the subject); every other object type marks an
# entity-style triple (an edge to another named thing). P/R/F1 are also
# reported split along that line.


def derive_attribute_relations(gold_triples: list[Triple]) -> frozenset[str]:
    """Returns the normalised names of the relations whose gold objects are
    literal values rather than named things, read off the gold data instead of
    hardcoded.

    Data-driven because the split has to stay correct if script 01 is re-run
    against a different corpus (the pipeline is
    dataset-agnostic on purpose). On main_1000 the answer is not a judgement
    call: four relations are 100% date-valued (`date of birth`, `point in
    time`, `date of death`, `inception`, 1,000 of 5,772 gold triples) and the
    other sixteen are 0%, so no relation sits near the boundary and a majority
    rule and an any-match rule would return the same set. The majority rule is
    used anyway, so that a future corpus with a genuinely mixed relation lands
    on whichever side it mostly belongs to rather than flipping the whole
    relation on one stray row.

    Classifying by *relation* rather than per triple is what makes the
    precision half of the split computable at all: a false positive is a
    predicted triple that matched no gold triple, and LLM output carries no
    `object_type` (see Triple's own field comment), so its style can only be
    read off the relation it named.
    """
    by_relation: dict[str, list[int]] = {}
    for t in gold_triples:
        is_attr = int(t.object_type in ATTRIBUTE_OBJECT_TYPES)
        by_relation.setdefault(normalize_relation(t.relation), []).append(is_attr)
    return frozenset(rel for rel, flags in by_relation.items() if sum(flags) * 2 > len(flags))


def normalize_arabic(text: str) -> str:
    """Collapses three common Arabic spelling variations (the hamza, the alef
    and the taa marbuta): strips diacritics and tatweel, unifies alef-with-hamza forms
    (أ/إ/آ) to bare alef, unifies taa marbuta/haa (ة/ه)."""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_teh(text)
    return re.sub(r"\s+", " ", text).strip()


_ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "configs" / "ontology.yaml"
_RELATION_ALIASES: dict[str, str] | None = None  # None means "not set, load the default"


def _load_default_aliases() -> dict[str, str]:
    """Reads the prompt's Arabic glosses straight from the ontology config.

    Deliberately not routed through `ontology.load_ontology`, which imports
    nothing from here: the dependency has to run one way or the scorer and the
    ontology loader import each other. Reading the one config file directly
    keeps `normalize_relation` self-sufficient, which is the point of loading
    these by default at all (see `register_relation_aliases`).
    """
    with open(_ONTOLOGY_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    aliases = {r["gloss_ar"]: r["name_en"] for r in raw["relations"]}
    assert len(aliases) == len(raw["relations"]), "two relations share an Arabic gloss"
    return aliases


def _active_aliases() -> dict[str, str]:
    global _RELATION_ALIASES
    if _RELATION_ALIASES is None:
        register_relation_aliases(_load_default_aliases())
    return _RELATION_ALIASES  # type: ignore[return-value]


def register_relation_aliases(aliases: dict[str, str]) -> None:
    """Registers alternative spellings that name a relation already in the
    ontology, so `normalize_relation` maps them onto its canonical English
    name. Keys and values are normalised on the way in; call with an empty
    dict to clear, which is what tests wanting the bare behaviour do.

    Callers do not normally need this. `normalize_relation` loads the
    ontology's own glosses on first use, so every script scoring or diagnosing
    output agrees on what a relation name is without having to remember to
    register anything. That default matters: script 09 diagnoses relation
    confusion without loading the ontology at all, and an alias table
    populated only as a side effect of `load_ontology` left it comparing
    relation names by a different rule than script 04 scored them by. Two
    parts of this pipeline disagreeing about what counts as the same relation
    is exactly the defect that once split one relation across two rows of the
    per-relation table.

    What the aliases fix is a defect in our own prompt rather than a model
    behaviour. The prompt lists
    every relation as an English canonical name plus an Arabic gloss, so that
    the model can tell what the relation means (see
    `Ontology.prompt_relation_list`). Nothing in the prompt says which of the
    two to answer with. Llama3.1 answers with the gloss: بلد المواطنة for
    `country of citizenship` 51 times, تاريخ الميلاد for `date of birth` 36
    times, at least 135 predictions in all. Each was scored as a false
    positive *and* an invented relation, which is most of why its ontology
    conformance sat at 86.2% with 574 relation hallucinations.

    The model named a relation that is on the list, in one of the two
    languages the list offered it. Scoring that as an invented relation
    measured the prompt, not the model.

    This is not synonym matching, which stays out of scope (see
    `normalize_relation`'s own docstring). It maps a gloss the prompt itself
    supplied onto the name it glosses. `birthplace` still fails against
    `place of birth`, and `located in the administrative territorial entity`
    still fails against `country`, because neither is a name this prompt
    offered for that relation.
    """
    global _RELATION_ALIASES
    table: dict[str, str] = {}
    for alias, canonical in aliases.items():
        canonical = _collapse_relation(canonical)
        for key in (_collapse_relation(alias), normalize_arabic(_collapse_relation(alias))):
            if key and key != canonical:
                table[key] = canonical
    _RELATION_ALIASES = table


def _collapse_relation(text: str) -> str:
    text = text.strip().lower().replace("_", " ")
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_relation(text: str) -> str:
    """Collapses separator style in a relation name, so that names differing
    only in how words are joined count as the same name.

    Added 2026-08-13, after the sanity_100 run showed Fanar-1-9B writing 133
    of its 320 relation names in snake_case (`place_of_birth`,
    `country_of_citizenship`, `located_in_the_administrative_territorial_entity`)
    while the ontology lists them space-separated. Every one of those was a
    correct, on-list relation name being scored as both a false positive and
    an invented relation, which put Fanar's conformance at 56.2% and its RH
    count at 140. No other model in the roster produced a single snake_case
    name, so this was one model's formatting habit defeating a strict string
    comparison, not a real disagreement about which relation was meant.

    Deliberately NOT synonym matching, which is out of scope here:
    `birthplace` still fails against `place of birth`, and
    `position played on team or speciality` still fails against
    `position played on team / speciality`, because those substitute a
    different word rather than a different separator. The rule here is
    exactly "underscores, extra whitespace and whitespace around a slash do
    not change which relation you named", plus any alias registered by
    `register_relation_aliases` (the prompt's own Arabic glosses).
    """
    text = _collapse_relation(text)
    aliases = _active_aliases()
    if text in aliases:
        return aliases[text]
    return aliases.get(normalize_arabic(text), text)


def _span_equal(a: str, b: str, normalize: bool, fuzzy_threshold: float) -> bool:
    a = normalize_arabic(a) if normalize else a.strip()
    b = normalize_arabic(b) if normalize else b.strip()
    if a == b:
        return True
    if fuzzy_threshold < 100 and fuzz.ratio(a, b) >= fuzzy_threshold:
        return True
    # Last, and only for pairs the letter-level comparison above rejects: the
    # same entity written as a place name on one side and as the adjective
    # formed from it on the other. Whichever normalisation the caller asked
    # for has already been applied to both sides, so the raw and normalised
    # passes stay separate measurements.
    return _stems_equal(a, b)


def triples_match(pred: Triple, gold: Triple, normalize: bool, fuzzy_threshold: float) -> bool:
    if normalize_relation(pred.relation) != normalize_relation(gold.relation):
        return False
    return _span_equal(pred.subject, gold.subject, normalize, fuzzy_threshold) and _span_equal(
        pred.object, gold.object, normalize, fuzzy_threshold
    )


def score_sentence(
    predicted: list[Triple], gold: list[Triple], normalize: bool, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> tuple[list[str], list[str], list[str]]:
    """Greedy one-to-one matching against gold: each gold triple can be
    claimed by at most one predicted triple, so a model cannot inflate
    recall by repeating the same correct triple. Returns (tp_relations,
    fp_relations, fn_relations), each the relation name of one matched or
    unmatched triple, which is enough to compute both overall and
    per-relation P/R/F1 by counting.
    """
    gold_remaining = list(gold)
    tp_relations: list[str] = []
    fp_relations: list[str] = []
    for p in predicted:
        hit_idx = None
        for i, g in enumerate(gold_remaining):
            if triples_match(p, g, normalize, fuzzy_threshold):
                hit_idx = i
                break
        if hit_idx is not None:
            # Normalised for the same reason the false-positive branch below
            # is. `triples_match` compares relation names after normalisation,
            # so a gold name and a predicted name that differ only in
            # separator style are the same relation to the scorer; recording
            # one raw here and the other normalised there splits one relation
            # across two rows of the per-relation table. That is how
            # "position played on team / speciality" (121 gold triples) and
            # "position played on team/speciality" (0 gold triples) both
            # appeared as separate rows. Totals were never affected, since
            # overall P/R/F1 sum the counters; only the breakdown was wrong.
            tp_relations.append(normalize_relation(gold_remaining.pop(hit_idx).relation))
        else:
            # Normalised, not raw: otherwise a model that writes snake_case
            # gets its own per-relation rows (`date_of_birth` alongside
            # `date of birth`), splitting one relation's false positives
            # across two rows of the same table.
            fp_relations.append(normalize_relation(p.relation))
    fn_relations = [normalize_relation(g.relation) for g in gold_remaining]
    return tp_relations, fp_relations, fn_relations


@dataclass
class HallucinationCounts:
    subject_hallucinations: int = 0
    relation_hallucinations: int = 0
    object_hallucinations: int = 0


def count_hallucinations(
    predicted: list[Triple], sentence_text: str, allowed_relations: frozenset[str], normalize: bool
) -> HallucinationCounts:
    """Checked per predicted triple, independent of whether it matched a
    gold fact: a triple can get the relation right while still inventing an
    entity, or vice versa. SH/OH are substring checks against the source
    sentence, not exact-span checks, since a correct extraction is allowed
    to trim surrounding words the gold span happens to include.
    """
    text = normalize_arabic(sentence_text) if normalize else sentence_text
    # Normalised once here, not per triple: must match triples_match's own
    # relation comparison (test_triples_match_relation_case_insensitive,
    # test_triples_match_relation_separator_insensitive), otherwise a triple
    # that triples_match accepts as a correct match (e.g. predicted "Country"
    # or "date_of_birth" against gold "country" / "date of birth") would
    # simultaneously be flagged here as inventing a relation not on the list,
    # which it did not.
    allowed_norm = {normalize_relation(r) for r in allowed_relations}
    counts = HallucinationCounts()
    for p in predicted:
        subject = normalize_arabic(p.subject) if normalize else p.subject.strip()
        obj = normalize_arabic(p.object) if normalize else p.object.strip()
        if subject not in text:
            counts.subject_hallucinations += 1
        if obj not in text:
            counts.object_hallucinations += 1
        if normalize_relation(p.relation) not in allowed_norm:
            counts.relation_hallucinations += 1
    return counts


def score_item(
    predicted: list[Triple],
    gold: list[Triple],
    sentence_text: str,
    allowed_relations: frozenset[str],
    normalize: bool,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> dict:
    """Scores one sentence. Returns a flat, JSON-serialisable dict; script
    04 accumulates one of these per extraction row before calling
    `aggregate_items`."""
    tp_rel, fp_rel, fn_rel = score_sentence(predicted, gold, normalize, fuzzy_threshold)
    halluc = count_hallucinations(predicted, sentence_text, allowed_relations, normalize)
    # Normalised for the same reason as count_hallucinations above: must agree
    # with triples_match's own relation comparison, or a triple counted as a
    # correct match (TP) could simultaneously be reported as non-conforming.
    allowed_norm = {normalize_relation(r) for r in allowed_relations}
    n_conforming = sum(1 for p in predicted if normalize_relation(p.relation) in allowed_norm)
    return {
        "tp_relations": tp_rel,
        "fp_relations": fp_rel,
        "fn_relations": fn_rel,
        "n_predicted": len(predicted),
        "n_gold": len(gold),
        "n_conforming": n_conforming,
        "subject_hallucinations": halluc.subject_hallucinations,
        "relation_hallucinations": halluc.relation_hallucinations,
        "object_hallucinations": halluc.object_hallucinations,
    }


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _split_counter_by_style(
    counter: Counter[str], attribute_relations: frozenset[str]
) -> tuple[int, int]:
    """Returns (attribute_total, entity_total) for one tp/fp/fn counter.

    Normalises on lookup because the three counters do not agree on casing by
    construction: `score_sentence` records true positives and false negatives
    under the *gold* relation name and false positives under the normalised
    predicted name (see its own comment on why). Comparing either against a
    normalised attribute set is only correct if both sides are normalised
    here.
    """
    attribute = entity = 0
    for relation, n in counter.items():
        if normalize_relation(relation) in attribute_relations:
            attribute += n
        else:
            entity += n
    return attribute, entity


def aggregate_items(
    items: list[dict], valid_flags: list[bool], attribute_relations: frozenset[str] = frozenset()
) -> dict:
    """Micro-averaged aggregation (sum tp/fp/fn across every sentence, then
    compute one P/R/F1), plus the per-relation breakdown, plus totals for
    hallucination, conformance and output validity. `valid_flags` is one
    bool per raw model response (True if it parsed as the expected JSON
    schema at all), tracked separately from `items` because an invalid
    response contributes an empty prediction list to scoring but should
    still count against the validity rate.

    `attribute_relations` comes from `derive_attribute_relations` over the
    gold split and drives the `by_style` block. It defaults to empty, in
    which case every triple counts as entity-style and `by_style` reads zero
    everywhere on the attribute side; callers wanting the split must pass it.
    """
    tp_counter: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    n_predicted = n_conforming = 0
    sh = rh = oh = 0

    for item in items:
        tp_counter.update(item["tp_relations"])
        fp_counter.update(item["fp_relations"])
        fn_counter.update(item["fn_relations"])
        n_predicted += item["n_predicted"]
        n_conforming += item["n_conforming"]
        sh += item["subject_hallucinations"]
        rh += item["relation_hallucinations"]
        oh += item["object_hallucinations"]

    all_relations = set(tp_counter) | set(fp_counter) | set(fn_counter)
    per_relation = {
        rel: _prf1(tp_counter[rel], fp_counter[rel], fn_counter[rel]) for rel in sorted(all_relations)
    }

    overall = _prf1(sum(tp_counter.values()), sum(fp_counter.values()), sum(fn_counter.values()))

    tp_attr, tp_ent = _split_counter_by_style(tp_counter, attribute_relations)
    fp_attr, fp_ent = _split_counter_by_style(fp_counter, attribute_relations)
    fn_attr, fn_ent = _split_counter_by_style(fn_counter, attribute_relations)

    return {
        "overall": overall,
        "by_style": {
            "attribute": _prf1(tp_attr, fp_attr, fn_attr),
            "entity": _prf1(tp_ent, fp_ent, fn_ent),
        },
        "per_relation": per_relation,
        "ontology_conformance_pct": 100 * n_conforming / n_predicted if n_predicted else 0.0,
        "subject_hallucinations": sh,
        "relation_hallucinations": rh,
        "object_hallucinations": oh,
        "n_predicted_triples": n_predicted,
        # None, not 0.0, when valid_flags is empty: that means the source
        # (e.g. mREBEL) never reported a valid_json flag at all, because the
        # JSON-schema-validity concept does not apply to it, not that it
        # scored zero. A caller that displays this must not print bare 0%.
        "output_validity_pct": 100 * sum(valid_flags) / len(valid_flags) if valid_flags else None,
        "n_items": len(items),
    }
