"""Measures what Arabic normalisation actually changes, and what it changes
about the *scores*, which turn out to be different questions.

The project reports every score twice, raw and normalised, because the gap
between the two can be a finding in itself. On this task the gap is +0.000 F1 for every model in the roster.
A zero that size invites the reader to assume a bug, so this script measures
the mechanism instead of asserting it:

- how many spans normalisation rewrites at all, on each side;
- how many whole-triple match decisions it flips, at exact matching and at the
  operating fuzzy threshold.

The two numbers are far apart, and the reason is that `normalize_arabic` is
applied symmetrically to gold and prediction. ArSRED's gold spans are cut from
the source sentence, and the prompt requires the model to copy spans from that
same sentence, so both sides usually inherit the same orthography and
normalising both maps them to the same place. Normalisation can only change a
decision where the model *retyped* a span instead of copying it, adding its
own diacritics or picking a different hamza.

This is a limitation of the rule's applicability here, not evidence against
the rule: for a task that matches a mention against a canonical label from a
different source (entity ranking against a knowledge base), the two sides do not share an
orthography and the gap would be real.

Read-only. Writes results/tables/normalisation_diagnosis_<scale>.json.

Usage:
    python scripts/07_diagnose_normalisation.py --scale 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.scorer import DEFAULT_FUZZY_THRESHOLD, normalize_arabic, triples_match  # noqa: E402
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


def count_rewritten(spans) -> tuple[int, int]:
    total = changed = 0
    for s in spans:
        total += 1
        if normalize_arabic(s) != s.strip():
            changed += 1
    return changed, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    gold = load_gold(args.scale)

    gold_changed, gold_total = count_rewritten(
        s for triples in gold.values() for t in triples for s in (t.subject, t.object)
    )

    pred_changed = pred_total = 0
    # Flips are counted per (predicted, gold) pair rather than per matched
    # triple: the question is how many decisions normalisation is capable of
    # changing, which is upstream of the greedy one-to-one assignment
    # `score_sentence` then does over those decisions.
    flips = {DEFAULT_FUZZY_THRESHOLD: 0, 100.0: 0}
    examples: list[dict] = []

    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"*_{args.scale}*.jsonl")):
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            gold_triples = gold.get(row["item_id"], [])
            for raw_triple in row.get("predicted_triples", []):
                p = Triple(**raw_triple)
                c, t = count_rewritten((p.subject, p.object))
                pred_changed += c
                pred_total += t
                for g in gold_triples:
                    for threshold in flips:
                        if triples_match(p, g, False, threshold) != triples_match(p, g, True, threshold):
                            flips[threshold] += 1
                            if threshold == DEFAULT_FUZZY_THRESHOLD and len(examples) < 10:
                                examples.append(
                                    {
                                        "model": path.stem,
                                        "relation": p.relation,
                                        "predicted": f"{p.subject} | {p.object}",
                                        "gold": f"{g.subject} | {g.object}",
                                    }
                                )

    out = {
        "scale": args.scale,
        "gold_spans_rewritten": gold_changed,
        "gold_spans_total": gold_total,
        "gold_spans_rewritten_pct": 100 * gold_changed / gold_total if gold_total else 0.0,
        "predicted_spans_rewritten": pred_changed,
        "predicted_spans_total": pred_total,
        "predicted_spans_rewritten_pct": 100 * pred_changed / pred_total if pred_total else 0.0,
        "triple_match_flips_at_exact": flips[100.0],
        "triple_match_flips_at_operating_threshold": flips[DEFAULT_FUZZY_THRESHOLD],
        "operating_threshold": DEFAULT_FUZZY_THRESHOLD,
        "flip_examples": examples,
    }

    logger.info(
        "normalisation rewrites %.1f%% of gold spans and %.1f%% of predicted spans",
        out["gold_spans_rewritten_pct"], out["predicted_spans_rewritten_pct"],
    )
    logger.info(
        "yet flips %d triple-match decisions at exact matching, %d at threshold %.0f",
        flips[100.0], flips[DEFAULT_FUZZY_THRESHOLD], DEFAULT_FUZZY_THRESHOLD,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / f"normalisation_diagnosis_{args.scale}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
