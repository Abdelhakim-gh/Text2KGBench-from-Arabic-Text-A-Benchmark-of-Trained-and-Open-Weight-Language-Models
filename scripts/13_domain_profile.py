"""Describes what the test set is about: how its gold triples spread over the
20 relations, over topic groups of those relations, and over ArSRED's own
subject-type tags.

Added in the revision after peer review (the ontology is biased toward
biography, geography and sport; how far do the conclusions carry to
technical, scientific or industrial Arabic text?). The submitted paper
stated the skew in one sentence without a number. This script supplies the
numbers.

Two views, because one of them involves a judgement and the other does not:

  * Subject types come from ArSRED itself (the entity tag on each gold
    subject), so they describe the data without any choice made here.
  * Topic groups are this study's grouping of the 20 relations, stated in
    TOPIC_GROUPS below so a reader can disagree with a specific line.

Writes:
  results/tables/domain_profile_<N>.json
  results/logs/13_domain_profile_<N>.log

Usage:
    python scripts/13_domain_profile.py --scale 1000
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from src.extraction.scorer import derive_attribute_relations, normalize_relation  # noqa: E402

score04 = import_module("04_score")

TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"

# Each relation in exactly one group. `participant`, `point in time` and
# `inception` describe events and dates for any kind of subject, so they form
# their own group rather than being forced into biography or sport.
TOPIC_GROUPS = {
    "Biography": ["country of citizenship", "occupation", "date of birth", "place of birth",
                  "date of death", "place of death", "member of political party"],
    "Geography and politics": ["country", "located in the administrative territorial entity",
                               "diplomatic relation"],
    "Sport": ["member of sports team", "sport", "position played on team / speciality"],
    "Classification": ["instance of", "subclass of", "part of", "parent taxon"],
    "Events and dates": ["point in time", "inception", "participant"],
}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000])
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"13_domain_profile_{args.scale}.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    group_of = {normalize_relation(r): g for g, rels in TOPIC_GROUPS.items() for r in rels}
    assert len(group_of) == 20, "every ontology relation must sit in exactly one group"

    gold = score04.load_gold(args.scale)
    triples = [t for _, ts in gold.values() for t in ts]
    attribute = derive_attribute_relations(triples)
    n = len(triples)

    by_relation = Counter(normalize_relation(t.relation) for t in triples)
    unknown = set(by_relation) - set(group_of)
    assert not unknown, f"gold relations missing from TOPIC_GROUPS: {unknown}"
    by_group = Counter(group_of[r] for r in (normalize_relation(t.relation) for t in triples))
    by_subject_type = Counter((t.subject_type or "none") for t in triples)
    by_object_type = Counter((t.object_type or "none") for t in triples)

    sentences_by_group = Counter()
    for _, ts in gold.values():
        for g in {group_of[normalize_relation(t.relation)] for t in ts}:
            sentences_by_group[g] += 1

    logger.info("scale %d: %d sentences, %d gold triples", args.scale, len(gold), n)
    logger.info("\nTopic groups (share of gold triples; sentences containing at least one):")
    for g, c in by_group.most_common():
        logger.info("  %-24s %5d  %5.1f%%   sentences %4d", g, c, 100 * c / n, sentences_by_group[g])
    logger.info("\nRelations:")
    for r, c in by_relation.most_common():
        logger.info("  %-50s %5d  %5.1f%%  %-22s %s", r, c, 100 * c / n, group_of[r],
                    "attribute" if r in attribute else "entity")
    logger.info("\nSubject type (ArSRED tag):")
    for t, c in by_subject_type.most_common():
        logger.info("  %-10s %5d  %5.1f%%", t, c, 100 * c / n)
    logger.info("\nObject type (ArSRED tag):")
    for t, c in by_object_type.most_common():
        logger.info("  %-10s %5d  %5.1f%%", t, c, 100 * c / n)

    out = {
        "scale": args.scale,
        "n_sentences": len(gold),
        "n_triples": n,
        "topic_groups": TOPIC_GROUPS,
        "by_group": dict(by_group),
        "sentences_by_group": dict(sentences_by_group),
        "by_relation": dict(by_relation),
        "relation_group": {r: group_of[r] for r in by_relation},
        "relation_style": {r: ("attribute" if r in attribute else "entity") for r in by_relation},
        "by_subject_type": dict(by_subject_type),
        "by_object_type": dict(by_object_type),
    }
    out_path = TABLES_DIR / f"domain_profile_{args.scale}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("\nwrote %s", out_path)


if __name__ == "__main__":
    main()
