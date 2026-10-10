"""Builds Study 2's ontology and test set from ArSRED (U4RASD/ArSRED).

Two passes over the full 499,620-row corpus:
  1. Parse every row's gold triples, count relation frequency, take the
     top 20 as the closed ontology (the paper, Sect. 3.1, says why 20).
  2. Keep only sentences whose every gold triple uses a top-20 relation
     ("closed-ontology sentences"), write them all to
     data/processed/closed_sentences.jsonl, then sample the three scale
     tiers as nested prefixes of one fixed shuffle (sanity_100 subset of
     main_1000 subset of confirm_10000).

Regenerates configs/ontology.yaml from scratch each run: it is a build
artifact, not a hand-maintained file (see that file's own header comment).

Usage: python scripts/01_build_ontology_and_testset.py [--sample-limit N]
`--sample-limit` is for a fast local smoke test only (skips most of the
corpus); the real run takes no argument and processes everything.
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.arsred import parse_arsred_relations  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

TOP_K = 20
RANDOM_STATE = 42
SCALE_TIERS = {"sanity_100": 100, "main_1000": 1000, "confirm_10000": 10000}
SOURCE_DATASET = "U4RASD/ArSRED"
SOURCE_DATASET_ROWS = 499620

# Arabic glosses for every relation that showed up near the top of either
# analysis pass (the original 20k-sample check and the corrected full-corpus
# run); covers well past rank 20 as a safety margin against the ranking
# shifting slightly when the bug fix above changes which relations qualify.
RELATION_GLOSSES: dict[str, str] = {
    "country": "بلد",
    "point in time": "نقطة زمنية",
    "sport": "رياضة",
    "date of birth": "تاريخ الميلاد",
    "located in the administrative territorial entity": "يقع في الوحدة الإدارية",
    "country of citizenship": "بلد المواطنة",
    "subclass of": "فئة فرعية من",
    "instance of": "نوعه",
    "occupation": "المهنة",
    "winner": "الفائز",
    "number of participants": "عدد المشاركين",
    "sports season of league or competition": "الموسم الرياضي",
    "place of birth": "مكان الميلاد",
    "organizer": "الجهة المنظمة",
    "parent taxon": "الرتبة التصنيفية الأعلى",
    "date of death": "تاريخ الوفاة",
    "diplomatic relation": "علاقة دبلوماسية",
    "part of": "جزء من",
    "follows": "يسبقه أو يليه",
    "member of sports team": "عضو في فريق رياضي",
    "position played on team / speciality": "المركز في الفريق أو التخصص",
    "place of death": "مكان الوفاة",
    "inception": "تاريخ التأسيس",
    "work period (start)": "بداية فترة العمل",
    "work period (end)": "نهاية فترة العمل",
    "league": "الدوري",
    "official language": "اللغة الرسمية",
    "location": "الموقع",
    "child": "الابن أو الابنة",
    "participant": "المشارك",
    "start time": "وقت البداية",
    "end time": "وقت النهاية",
    "cast member": "أحد أفراد طاقم التمثيل",
    "country of origin": "بلد المنشأ",
    "publication date": "تاريخ النشر",
    "headquarters location": "مقر الإدارة",
    "member of political party": "عضو في حزب سياسي",
    "taxonomic type": "النوع التصنيفي",
    "educated at": "تعلم في",
    "languages spoken, written or signed": "اللغات المستخدمة",
    "number of matches played/races/starts": "عدد المباريات أو السباقات",
}

# The entity-type bucket design is a fixed choice, not
# something recomputed from frequency, so it is a constant here rather than
# derived. `literal: true` marks a type whose object makes a triple
# attribute-style rather than entity-style.
ENTITY_TYPES = [
    {"id": "person", "raw_tags": ["per"], "literal": False},
    {"id": "organization", "raw_tags": ["org"], "literal": False},
    {"id": "location", "raw_tags": ["loc"], "literal": False},
    {"id": "date", "raw_tags": ["date"], "literal": True},
    {"id": "time", "raw_tags": ["time"], "literal": True},
    {"id": "number", "raw_tags": ["number"], "literal": True},
    {"id": "concept", "raw_tags": ["concept"], "literal": False},
    {"id": "event", "raw_tags": ["eve"], "literal": False},
    {"id": "media", "raw_tags": ["media"], "literal": False},
    {"id": "other", "raw_tags": ["unk", "cel", "dis"], "literal": False},
]


def iter_rows(sample_limit: int | None):
    ds = load_dataset(SOURCE_DATASET)["train"]
    n = len(ds) if sample_limit is None else min(sample_limit, len(ds))
    for i in range(n):
        yield i, ds[i]["text"], ds[i]["relations"]


def compute_relation_frequency(sample_limit: int | None) -> Counter:
    counter = Counter()
    n_rows = 0
    n_triples = 0
    for _, _, relations_str in iter_rows(sample_limit):
        triples = parse_arsred_relations(relations_str)
        counter.update(t.relation for t in triples)
        n_rows += 1
        n_triples += len(triples)
        if n_rows % 50000 == 0:
            logger.info("pass 1: %d rows scanned, %d triples so far", n_rows, n_triples)
    logger.info("pass 1 done: %d rows, %d triples, %.2f triples/row, %d unique relations",
                n_rows, n_triples, n_triples / max(n_rows, 1), len(counter))
    return counter


def build_closed_sentences(sample_limit: int | None, allowed_relations: set[str]) -> list[dict]:
    closed = []
    n_rows = 0
    n_closed = 0
    for idx, text, relations_str in iter_rows(sample_limit):
        n_rows += 1
        triples = parse_arsred_relations(relations_str)
        if not triples:
            continue
        if not all(t.relation in allowed_relations for t in triples):
            continue
        closed.append({
            "item_id": f"arsred_{idx}",
            "text": text,
            "gold_triples": [
                {
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                    "subject_type": t.subject_type,
                    "object_type": t.object_type,
                }
                for t in triples
            ],
        })
        n_closed += 1
        if n_rows % 50000 == 0:
            logger.info("pass 2: %d rows scanned, %d closed so far", n_rows, n_closed)
    logger.info("pass 2 done: %d/%d rows are closed-ontology sentences (%.1f%%)",
                n_closed, n_rows, 100 * n_closed / max(n_rows, 1))
    return closed


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ontology_yaml(top_relations: list[tuple[str, int]], total_triples: int,
                         sample_rows: int, coverage_pct: float, closed_pct: float,
                         estimated_closed_full: int):
    missing_glosses = [name for name, _ in top_relations if name not in RELATION_GLOSSES]
    if missing_glosses:
        raise RuntimeError(
            f"no Arabic gloss on file for: {missing_glosses}. "
            "Add it to RELATION_GLOSSES in this script and re-run."
        )

    doc = {
        "source_dataset": SOURCE_DATASET,
        "source_dataset_rows": SOURCE_DATASET_ROWS,
        "sample_analyzed": sample_rows,
        "relation_coverage_at_k20_pct": round(coverage_pct, 1),
        "sentence_closed_pct_at_k20": round(closed_pct, 1),
        "estimated_closed_sentences_full_corpus": estimated_closed_full,
        "relations": [
            {
                "id": i + 1,
                "name_en": name,
                "gloss_ar": RELATION_GLOSSES[name],
                "count_sample": count,
            }
            for i, (name, count) in enumerate(top_relations)
        ],
        "entity_types": ENTITY_TYPES,
    }
    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Regenerated by scripts/01_build_ontology_and_testset.py. Do not hand-edit;\n"
            "# change RELATION_GLOSSES/TOP_K/ENTITY_TYPES in that script and re-run instead.\n"
            "# Numbers and reasoning: the paper, Sect. 3.1; scripts/19_corpus_stats.py recomputes them.\n\n"
        )
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    logger.info("wrote %s", ONTOLOGY_PATH)


def make_scale_splits(closed: list[dict], random_state: int):
    """Nested prefixes of one fixed shuffle: sanity_100 subset of main_1000
    subset of confirm_10000."""
    order = list(range(len(closed)))
    random.Random(random_state).shuffle(order)

    for name, target_size in SCALE_TIERS.items():
        size = min(target_size, len(order))
        if size < target_size:
            logger.warning("%s: requested %d but only %d closed sentences exist, using all of it",
                            name, target_size, len(order))
        idx = order[:size]
        rows = [closed[i] for i in idx]
        write_jsonl(SPLITS_DIR / f"{name}.jsonl", rows)
        logger.info("%s: %d sentences written", name, len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-limit", type=int, default=None,
                         help="process only the first N rows, for a fast smoke test")
    args = parser.parse_args()

    counter = compute_relation_frequency(args.sample_limit)
    total_triples = sum(counter.values())
    top_relations = counter.most_common(TOP_K)
    allowed = {name for name, _ in top_relations}
    coverage_pct = 100 * sum(c for _, c in top_relations) / total_triples

    closed = build_closed_sentences(args.sample_limit, allowed)

    sample_rows = args.sample_limit or SOURCE_DATASET_ROWS
    closed_pct = 100 * len(closed) / sample_rows
    estimated_closed_full = round(len(closed) * (SOURCE_DATASET_ROWS / sample_rows))

    write_jsonl(PROCESSED_DIR / "closed_sentences.jsonl", closed)
    write_ontology_yaml(top_relations, total_triples, sample_rows, coverage_pct, closed_pct, estimated_closed_full)
    make_scale_splits(closed, RANDOM_STATE)

    logger.info("SUMMARY: top-%d relations cover %.1f%% of triples, %.1f%% of sentences are closed "
                "(%d of %d), estimated %d closed sentences across the full corpus",
                TOP_K, coverage_pct, closed_pct, len(closed), sample_rows, estimated_closed_full)


if __name__ == "__main__":
    main()
