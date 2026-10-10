"""Recomputes the corpus statistics the paper quotes, read-only.

Added for the final check of the paper. Sect. 3.1 and Fig. 1 quote
figures that script 01 printed when it built the benchmark (499,620
sentences, 2,940,185 triples, 393 relation labels, 87.6% coverage by the top
20 relations, 334,057 closed sentences) and one from an earlier analysis (the
top 30 would add 3.4 points). Only some of them were saved to a file
(configs/ontology.yaml), so this script recomputes all of them from the
cached ArSRED corpus with script 01's own loader and parser.

It writes nothing that the benchmark depends on: no ontology, no split, no
closed-sentence file. It checks instead that its top 20 equal the relations
in configs/ontology.yaml and that its count of closed sentences equals the
lines of data/processed/closed_sentences.jsonl.

It reads the corpus file that script 01's `load_dataset("U4RASD/ArSRED")`
downloaded, ArSRED.csv in the Hugging Face cache, with pandas, the library
the `datasets` CSV loader itself uses. It does not import `datasets`: on this
machine Windows Smart App Control intermittently blocks pyarrow's DLL, which
`datasets` needs. The row-count check below confirms that the file read here
is the 499,620-row corpus script 01 read.

Writes:
  results/tables/corpus_stats_arsred.json
  results/logs/19_corpus_stats.log

Usage:
    python scripts/19_corpus_stats.py [--csv PATH]
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.arsred import parse_arsred_relations  # noqa: E402

SOURCE_DATASET = "U4RASD/ArSRED"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--U4RASD--ArSRED" / "snapshots"

TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"
CLOSED_PATH = ROOT / "data" / "processed" / "closed_sentences.jsonl"

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=None, help="ArSRED.csv; default: the Hugging Face cache")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / "19_corpus_stats.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    csv_path = args.csv or next(iter(sorted(HF_CACHE.glob("*/ArSRED.csv"))), None)
    if csv_path is None or not csv_path.exists():
        raise SystemExit(f"ArSRED.csv not found under {HF_CACHE}; run script 01 once, or pass --csv")
    logger.info("reading %s", csv_path)
    frame = pd.read_csv(csv_path)
    logger.info("columns: %s", list(frame.columns))
    n_missing = int(frame["relations"].isna().sum())

    # One pass: per row, the relations of its parsed triples (script 01 parses
    # the same field twice, in two passes; the counts are the same).
    counter = Counter()
    row_relations = []
    n_rows = n_triples = n_rows_with_triples = 0
    for relations_str in frame["relations"].fillna(""):
        triples = parse_arsred_relations(relations_str)
        rels = {t.relation for t in triples}
        counter.update(t.relation for t in triples)
        row_relations.append(rels)
        n_rows += 1
        n_triples += len(triples)
        n_rows_with_triples += bool(triples)
        if n_rows % 100000 == 0:
            logger.info("%d rows scanned", n_rows)

    ranked = counter.most_common()
    out = {
        "source_dataset": SOURCE_DATASET,
        "source_file": csv_path.name,
        "n_rows": n_rows,
        "n_rows_missing_relations": n_missing,
        "n_rows_with_triples": n_rows_with_triples,
        "n_triples": n_triples,
        "triples_per_row": n_triples / n_rows,
        "n_relation_labels": len(counter),
        "by_k": {},
    }
    for k in (20, 30):
        top = {name for name, _ in ranked[:k]}
        covered = sum(n for _, n in ranked[:k])
        closed = sum(1 for rels in row_relations if rels and rels <= top)
        out["by_k"][k] = {
            "triple_coverage_pct": 100 * covered / n_triples,
            "closed_sentences": closed,
            "closed_pct_of_rows": 100 * closed / n_rows,
        }
    out["coverage_gain_20_to_30_points"] = (out["by_k"][30]["triple_coverage_pct"]
                                            - out["by_k"][20]["triple_coverage_pct"])

    # Consistency with what the benchmark was built from.
    ontology = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    built_top20 = {r["name_en"] for r in ontology["relations"]}
    n_closed_file = sum(1 for line in CLOSED_PATH.open(encoding="utf-8") if line.strip())
    out["checks"] = {
        "top20_equals_ontology_yaml": {name for name, _ in ranked[:20]} == built_top20,
        "closed_k20_equals_closed_sentences_jsonl": out["by_k"][20]["closed_sentences"] == n_closed_file,
        "closed_sentences_jsonl_lines": n_closed_file,
        "n_rows_equals_ontology_yaml": n_rows == ontology["source_dataset_rows"],
    }

    logger.info("rows %d (with triples %d), triples %d (%.2f per row), relation labels %d",
                n_rows, n_rows_with_triples, n_triples, out["triples_per_row"], len(counter))
    for k, v in out["by_k"].items():
        logger.info("top %d: %.2f%% of triples; %d closed sentences (%.2f%% of rows)",
                    k, v["triple_coverage_pct"], v["closed_sentences"], v["closed_pct_of_rows"])
    logger.info("coverage gain from 20 to 30 relations: %.2f points", out["coverage_gain_20_to_30_points"])
    logger.info("checks: %s", out["checks"])

    out_path = TABLES_DIR / "corpus_stats_arsred.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
