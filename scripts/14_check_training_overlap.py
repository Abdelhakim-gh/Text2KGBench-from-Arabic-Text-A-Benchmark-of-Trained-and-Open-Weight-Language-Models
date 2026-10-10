"""Checks whether the test sentences appear in mREBEL's training data.

Found during the revision after peer review. mREBEL (Babelscape/mrebel-large)
was trained on SRED-FM (Huguet Cabot et al., ACL 2023), and ArSRED, our data
source, is built from SRED-FM's Arabic part. The paper uses mREBEL as the
control that locates the attribute/entity gap in prompting, so whether it has
seen the test sentences in training matters to that argument and to its
precision of 0.727.

SRED-FM ships an Arabic train, validation and test split. This script loads
all three from the Hugging Face parquet export and looks up every test
sentence by its text (whitespace collapsed), so the answer is a count rather
than an inference from split sizes.

The parquet files are not part of the repository (about 210 MB). Download them
first from
  https://huggingface.co/datasets/Babelscape/SREDFM/tree/refs%2Fconvert%2Fparquet/ar
into one folder and pass that folder.

Writes:
  results/tables/training_overlap_<N>.json
  results/logs/14_training_overlap_<N>.log

Usage:
    python scripts/14_check_training_overlap.py --scale 1000 --sredfm-dir <folder with the parquet files>
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

score04 = import_module("04_score")

TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"

logger = logging.getLogger(__name__)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000])
    parser.add_argument("--sredfm-dir", type=Path, required=True)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"14_training_overlap_{args.scale}.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    gold = score04.load_gold(args.scale)
    test_texts = {item_id: norm(text) for item_id, (text, _) in gold.items()}
    wanted = set(test_texts.values())

    split_files = {
        "train": sorted(args.sredfm_dir.glob("train*.parquet")),
        "validation": sorted(args.sredfm_dir.glob("validation*.parquet")),
        "test": sorted(args.sredfm_dir.glob("test*.parquet")),
    }
    found: dict[str, set[str]] = {}
    sizes = {}
    for split, files in split_files.items():
        if not files:
            raise SystemExit(f"no parquet file for split {split!r} in {args.sredfm_dir}")
        hits, n = set(), 0
        for f in files:
            table = pq.read_table(f, columns=["text"])
            for t in table.column("text").to_pylist():
                n += 1
                t = norm(t or "")
                if t in wanted:
                    hits.add(t)
        found[split] = hits
        sizes[split] = n
        logger.info("SRED-FM ar %-10s %7d rows; test sentences found: %4d of %d",
                    split, n, len(hits), len(wanted))

    where = Counter()
    for text in test_texts.values():
        in_splits = tuple(s for s in ("train", "validation", "test") if text in found[s])
        where["+".join(in_splits) or "none"] += 1
    logger.info("\nper test sentence, which SRED-FM split holds it:")
    for k, v in where.most_common():
        logger.info("  %-24s %4d", k, v)

    out = {
        "scale": args.scale,
        "n_test_sentences": len(test_texts),
        "n_distinct_test_texts": len(wanted),
        "sredfm_ar_rows": sizes,
        "found_in_split": {s: len(h) for s, h in found.items()},
        "per_sentence": dict(where),
    }
    out_path = TABLES_DIR / f"training_overlap_{args.scale}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("\nwrote %s", out_path)


if __name__ == "__main__":
    main()
