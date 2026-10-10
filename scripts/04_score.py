"""Scores every extraction result file for one scale tier against ArSRED
gold: triple-level P/R/F1 (overall and per relation), ontology conformance,
SH/RH/OH hallucination counts, output validity, and latency/GPU-energy
where the extraction file recorded them. Every score is computed twice,
raw and after Arabic normalisation. Read-only over results/extraction/*.jsonl: safe to run while
script 03 is still writing other models' files, since each file is scored
independently.

Usage:
    python scripts/04_score.py --scale 100
"""

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.scorer import aggregate_items, derive_attribute_relations, score_item  # noqa: E402
from src.extraction.triple import Triple  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

SCALE_FILES = {100: "sanity_100.jsonl", 1000: "main_1000.jsonl", 10000: "confirm_10000.jsonl"}
FILENAME_RE = re.compile(r"^(?P<model>.+)_(?P<scale>\d+)(?:_seed(?P<seed>\d+))?\.jsonl$")


def load_gold(scale: int) -> dict[str, tuple[str, list[Triple]]]:
    path = SPLITS_DIR / SCALE_FILES[scale]
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            triples = [
                Triple(
                    subject=t["subject"], relation=t["relation"], object=t["object"],
                    subject_type=t.get("subject_type"), object_type=t.get("object_type"),
                )
                for t in row["gold_triples"]
            ]
            gold[row["item_id"]] = (row["text"], triples)
    return gold


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(pct * (len(s) - 1)), len(s) - 1)
    return s[idx]


def score_extraction_file(
    path: Path,
    gold: dict,
    allowed_relations: frozenset[str],
    normalize: bool,
    attribute_relations: frozenset[str] = frozenset(),
) -> dict:
    items, valid_flags, latencies, energies = [], [], [], []
    n_missing_gold = n_errors = n_unparseable = 0
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            # job_runner's own resumability contract (src/common/job_runner.py,
            # load_completed_ids) guarantees a crash/Ctrl-C mid-write can leave a
            # truncated final line in an extraction file, and that this line is
            # never cleaned up: it is skipped and the item redone on the next
            # run, but the stale garbage line stays on disk permanently. This
            # script's own docstring promises it is "safe to run while script 03
            # is still writing", i.e. against a file that may be mid-write right
            # now, so it must tolerate that line the same way job_runner does
            # rather than crash the whole scoring run on one leftover line.
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_unparseable += 1
                logger.warning("%s: skipping unparseable line %d (likely a truncated write)", path.name, line_num)
                continue
            item_id = row["item_id"]
            if item_id not in gold:
                n_missing_gold += 1
                continue
            text, gold_triples = gold[item_id]
            # job_runner's on_error="record" default writes {"error": ..., "item_id": ...}
            # for a failed call rather than crashing the run (src/common/job_runner.py).
            # That row has no predicted_triples or valid_json field, so without this
            # check it would silently drop out of the validity count instead of
            # counting as the failure it is, e.g. AceGPT-13B's Ollama context window
            # (2048 tokens) overflowing on a few of the longer sanity_100 sentences.
            if "error" in row:
                n_errors += 1
                items.append(score_item([], gold_triples, text, allowed_relations, normalize))
                valid_flags.append(False)
                continue
            predicted = [Triple(**t) for t in row.get("predicted_triples", [])]
            items.append(score_item(predicted, gold_triples, text, allowed_relations, normalize))
            if "valid_json" in row:
                valid_flags.append(row["valid_json"])
            if "wall_clock_s" in row:
                latencies.append(row["wall_clock_s"])
            if "gpu_energy_wh" in row:
                energies.append(row["gpu_energy_wh"])
    if n_missing_gold:
        logger.warning("%s: %d rows had no matching gold item_id, skipped", path.name, n_missing_gold)
    if n_errors:
        logger.warning("%s: %d rows were recorded errors (counted as invalid, zero predictions)", path.name, n_errors)

    report = aggregate_items(items, valid_flags, attribute_relations)
    if latencies:
        report["latency_median_s"] = statistics.median(latencies)
        report["latency_p95_s"] = percentile(latencies, 0.95)
    if energies:
        report["gpu_energy_wh_total"] = sum(energies)
        report["gpu_energy_wh_per_1000"] = 1000 * sum(energies) / len(energies)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    ontology = load_ontology(ONTOLOGY_PATH)
    allowed = ontology.relation_names
    gold = load_gold(args.scale)
    logger.info("scale %d: %d gold items loaded", args.scale, len(gold))

    # Derived once from this tier's own gold, then applied to every run, so
    # the attribute/entity split means the same thing in every row of the
    # comparison table. Deriving it per run would let two
    # models be scored against different definitions of the same column.
    attribute_relations = derive_attribute_relations([t for _, triples in gold.values() for t in triples])
    logger.info("attribute-style relations (%d): %s", len(attribute_relations), sorted(attribute_relations))

    runs = {}
    by_model_f1: dict[str, list[float]] = {}

    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"*_{args.scale}*.jsonl")):
        m = FILENAME_RE.match(path.name)
        if not m or m.group("scale") != str(args.scale):
            continue
        model, seed = m.group("model"), m.group("seed")
        run_key = f"{model}|seed{seed}" if seed else model
        logger.info("scoring %s", path.name)

        raw = score_extraction_file(path, gold, allowed, normalize=False, attribute_relations=attribute_relations)
        normalized = score_extraction_file(path, gold, allowed, normalize=True, attribute_relations=attribute_relations)
        runs[run_key] = {"raw": raw, "normalized": normalized}
        by_model_f1.setdefault(model, []).append(raw["overall"]["f1"])

    aggregated = {}
    for model, f1_values in by_model_f1.items():
        aggregated[model] = {
            "f1_mean": statistics.mean(f1_values),
            "f1_stdev": statistics.pstdev(f1_values) if len(f1_values) > 1 else 0.0,
            "f1_per_seed": f1_values,
            "n_seeds": len(f1_values),
        }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / f"scale_{args.scale}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scale": args.scale, "runs": runs, "aggregated_by_model": aggregated}, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s (%d runs, %d models)", output_path, len(runs), len(aggregated))


if __name__ == "__main__":
    main()
