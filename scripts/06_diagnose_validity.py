"""Splits the output-validity metric into its two very different causes.

`parse_llm_output` returns valid=False for any response it cannot decode as a
JSON array (src/extraction/parser.py). Two unrelated things land in that
bucket:

1. The model did not follow the requested schema: prose instead of JSON,
   single-quoted pseudo-JSON, its own invented keys. A genuine model
   deficiency, and what the validity metric is meant to measure.
2. The model followed the schema perfectly and ran out of output budget
   mid-array, because `max_tokens` is 512 (configs/llm_models.yaml). The
   array never closes, `_extract_json_array` finds no balanced bracket, and
   every complete triple written before the cut is discarded along with the
   partial one. A property of the harness, not of the model.

Reporting these as one number would let a token-budget setting masquerade as
an Arabic-extraction result, and would understate recall on exactly the
triple-dense sentences the task cares most about. This script measures how
much of each model's invalid share is case 2, and how many well-formed
triples case 2 throws away, so the paper can state it with numbers instead of
hedging.

Read-only. Writes results/tables/validity_diagnosis_<scale>.json.

Usage:
    python scripts/06_diagnose_validity.py --scale 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.parser import _extract_json_array, parse_llm_output  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"


def salvage_truncated(raw_text: str) -> list[dict]:
    """Recovers the complete `{...}` objects an unterminated JSON array
    contains, in order, stopping at the first one that does not decode.

    Deliberately not folded into `parse_llm_output`: that function defines the
    output-validity metric the run was scored under, and changing what it
    accepts after the fact would silently redefine a headline number. This
    stays a separate measurement of what a more forgiving parser *would* have
    recovered.

    Brace counting rather than a regex because an Arabic object string can
    itself contain braces, and because the final object of a truncated array
    is by definition unbalanced and must be dropped, not repaired.
    """
    start = raw_text.find("[")
    if start == -1:
        return []
    objects: list[dict] = []
    depth = 0
    obj_start = None
    in_string = False
    escaped = False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    decoded = json.loads(raw_text[obj_start : i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(decoded, dict):
                    objects.append(decoded)
                obj_start = None
    return objects


def well_formed_triples(objects: list[dict]) -> int:
    """Counts salvaged objects carrying all three non-empty string fields,
    i.e. the ones `parse_llm_output` would have kept had the array closed."""
    n = 0
    for item in objects:
        vals = (item.get("subject"), item.get("relation"), item.get("object"))
        if all(isinstance(v, str) and v.strip() for v in vals):
            n += 1
    return n


def diagnose(path: Path) -> dict:
    n_rows = n_invalid = n_truncated = n_other_invalid = 0
    n_errors = 0
    salvageable_triples = 0
    truncated_examples: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # same tolerance script 04 has for a mid-write final line
            n_rows += 1
            if "error" in row:
                n_errors += 1
                n_invalid += 1
                continue
            raw = row.get("raw_output", "")
            _, valid = parse_llm_output(raw)
            if valid:
                continue
            n_invalid += 1
            # An opening bracket with no balanced close is the truncation
            # signature; no bracket at all means the model never attempted the
            # requested shape, which is a different failure.
            if "[" in raw and _extract_json_array(raw) is None:
                n_truncated += 1
                salvageable_triples += well_formed_triples(salvage_truncated(raw))
                if len(truncated_examples) < 3:
                    truncated_examples.append(raw[-160:])
            else:
                n_other_invalid += 1

    return {
        "n_rows": n_rows,
        "n_invalid": n_invalid,
        "n_recorded_errors": n_errors,
        "n_invalid_truncated": n_truncated,
        "n_invalid_other": n_other_invalid,
        "truncated_pct_of_all_rows": 100 * n_truncated / n_rows if n_rows else 0.0,
        "truncated_share_of_invalid_pct": 100 * n_truncated / n_invalid if n_invalid else 0.0,
        "salvageable_triples": salvageable_triples,
        "truncated_tail_examples": truncated_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    out = {}
    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"*_{args.scale}*.jsonl")):
        if path.name.startswith("mrebel"):
            continue  # seq2seq, no JSON schema to violate; validity does not apply
        out[path.stem] = diagnose(path)
        d = out[path.stem]
        logger.info(
            "%-58s rows=%4d invalid=%3d truncated=%3d (%.1f%% of invalid) salvageable=%4d",
            path.stem, d["n_rows"], d["n_invalid"], d["n_invalid_truncated"],
            d["truncated_share_of_invalid_pct"], d["salvageable_triples"],
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / f"validity_diagnosis_{args.scale}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
