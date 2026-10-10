"""Separates "did not find the fact" from "found the fact, named it differently".

The per-relation breakdown at scale 1,000 showed the strongest model scoring
0.000 on `country`, the single most frequent relation in the gold data with
1{,}202 triples, while scoring 0.62 F1 on `country of citizenship`. Inspection
of the raw output showed why, and it is not a failure to read the sentence:

    gold       بلدية فرناندوبولس | country | البرازيل
    predicted  بلدية فرناندوبولس | located in the administrative
                                   territorial entity | البرازيل

Same subject, same object, both copied correctly from the sentence, and a
different but defensible Wikidata property. ArSRED's `country` is P17 ("the
country this thing is in"); the model reached for P131 or, for people, P27.
Under exact relation matching this counts as a false positive *and* a false
negative, and the fact that the model got the entities right is invisible.

This script measures how much of the total error is that. For every predicted
triple that matched no gold triple, it checks whether some gold triple in the
same sentence has a matching subject and object under a different relation. If
so, the error is a relation confusion rather than a missed or invented fact.

Reported as a diagnostic, not folded into the headline scores. Ontology
conformance is a real requirement of the task: a downstream graph that stores
P131 where the schema expects P17 is wrong, even if a human can see what was
meant. The point is to say how large the effect is, not to score it away.

Read-only. Writes results/tables/relation_confusion_<scale>.json.

Usage:
    python scripts/09_diagnose_relation_confusion.py --scale 1000
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.scorer import (  # noqa: E402
    DEFAULT_FUZZY_THRESHOLD,
    _span_equal,
    normalize_relation,
    triples_match,
)
from src.extraction.triple import Triple  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"
SCALE_FILES = {100: "sanity_100.jsonl", 1000: "main_1000.jsonl", 10000: "confirm_10000.jsonl"}


def load_gold(scale: int) -> dict[str, list[Triple]]:
    gold = {}
    with open(SPLITS_DIR / SCALE_FILES[scale], encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold[row["item_id"]] = [Triple(**t) for t in row["gold_triples"]]
    return gold


def diagnose(path: Path, gold: dict[str, list[Triple]]) -> dict:
    n_predicted = n_matched = n_confused = n_genuinely_wrong = 0
    confusion_pairs: Counter[tuple[str, str]] = Counter()

    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        gold_triples = gold.get(row["item_id"], [])
        # Greedy one-to-one, mirroring score_sentence: a gold triple already
        # claimed by an exact match must not also absorb a confused one, or
        # the same gold fact would be counted twice.
        remaining = list(gold_triples)
        unmatched_predictions = []
        for raw in row.get("predicted_triples", []):
            p = Triple(**raw)
            n_predicted += 1
            hit = next(
                (i for i, g in enumerate(remaining)
                 if triples_match(p, g, False, DEFAULT_FUZZY_THRESHOLD)),
                None,
            )
            if hit is not None:
                remaining.pop(hit)
                n_matched += 1
            else:
                unmatched_predictions.append(p)

        for p in unmatched_predictions:
            hit = next(
                (i for i, g in enumerate(remaining)
                 if _span_equal(p.subject, g.subject, False, DEFAULT_FUZZY_THRESHOLD)
                 and _span_equal(p.object, g.object, False, DEFAULT_FUZZY_THRESHOLD)),
                None,
            )
            if hit is not None:
                g = remaining.pop(hit)
                n_confused += 1
                confusion_pairs[(normalize_relation(g.relation), normalize_relation(p.relation))] += 1
            else:
                n_genuinely_wrong += 1

    n_unmatched = n_confused + n_genuinely_wrong
    return {
        "n_predicted_triples": n_predicted,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_relation_confusions": n_confused,
        "n_genuinely_wrong": n_genuinely_wrong,
        "confusion_share_of_errors_pct": 100 * n_confused / n_unmatched if n_unmatched else 0.0,
        "top_confusions": [
            {"gold": g, "predicted": p, "n": n} for (g, p), n in confusion_pairs.most_common(10)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    gold = load_gold(args.scale)
    out = {}
    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"*_{args.scale}*.jsonl")):
        out[path.stem] = diagnose(path, gold)
        d = out[path.stem]
        logger.info(
            "%-58s predicted=%5d matched=%4d confused=%4d (%.1f%% of errors)",
            path.stem, d["n_predicted_triples"], d["n_matched"],
            d["n_relation_confusions"], d["confusion_share_of_errors_pct"],
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / f"relation_confusion_{args.scale}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
