"""Recomputes `predicted_triples` from the `raw_output` already on disk.

Exists so that a fix to a parser costs no GPU time. Every extraction row keeps
the model's verbatim `raw_output`, so when a parsing bug is found the correct
response is to re-derive the predictions from what the model actually said,
not to run the model again. Re-running would also risk changing the underlying
generations, which would silently mix a parser fix with a different sample.

Used on 2026-08-14 after the mREBEL parser was found to swallow a second
object into the previous fact's relation name (see
`test_parse_mrebel_output_shared_subject_multiple_facts`). Only the baseline's
file needed it; the LLM parser was unchanged.

Rewrites files in place, after writing a `.bak` copy beside each one. Prints a
diff summary rather than rewriting silently, since this changes recorded
results.

Usage:
    python scripts/10_reparse.py --scale 1000 --pattern "mrebel*"
    python scripts/10_reparse.py --scale 1000 --pattern "mrebel*" --apply
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.parser import parse_llm_output, parse_mrebel_output  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "extraction"


def reparse_row(row: dict, is_mrebel: bool) -> tuple[list[dict], bool | None]:
    raw = row.get("raw_output", "")
    if is_mrebel:
        triples = parse_mrebel_output(raw)
        valid = None
    else:
        triples, valid = parse_llm_output(raw)
    return [
        {
            "subject": t.subject,
            "relation": t.relation,
            "object": t.object,
            "subject_type": t.subject_type,
            "object_type": t.object_type,
        }
        for t in triples
    ], valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    parser.add_argument("--pattern", default="*", help="glob over file stems, e.g. 'mrebel*'")
    parser.add_argument("--apply", action="store_true", help="write changes; otherwise dry-run")
    args = parser.parse_args()

    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"{args.pattern}.jsonl")):
        is_mrebel = path.name.startswith("mrebel")
        rows, n_changed, before, after = [], 0, 0, 0

        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s: skipping unparseable line", path.name)
                continue
            if "error" in row or "raw_output" not in row:
                rows.append(row)
                continue
            old = row.get("predicted_triples", [])
            new, valid = reparse_row(row, is_mrebel)
            before += len(old)
            after += len(new)
            if old != new:
                n_changed += 1
            row["predicted_triples"] = new
            if valid is not None:
                row["valid_json"] = valid
            rows.append(row)

        logger.info(
            "%-58s rows changed=%4d  triples %d -> %d (%+d)",
            path.stem, n_changed, before, after, after - before,
        )

        if args.apply and n_changed:
            backup = path.with_suffix(".jsonl.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
                logger.info("  backup written: %s", backup.name)
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("  rewrote %s", path.name)

    if not args.apply:
        logger.info("dry run; pass --apply to write")


if __name__ == "__main__":
    main()
