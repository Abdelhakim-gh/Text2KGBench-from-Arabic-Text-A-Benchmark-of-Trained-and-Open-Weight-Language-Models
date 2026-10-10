"""How much does a model's F1 move between decoding seeds?

Added in the revision after peer review. The submitted
paper ran every model once, at temperature 0.2. Three models were re-run with
seeds 2 and 3: Llama3.1 8B, Aya-Expanse 8B and Command-R7B-Arabic, which sit
within 0.021 F1 of each other in places 3 to 5 and carry the paper's statement
about Arabic-specialised models. Llama3.1's seed 2 (0.340) then came within
0.004 of Qwen3.5 9B's single run (0.344), so second place was in question too:
the authors decided (2026-10-08) that Qwen3.5 9B also gets seeds 2 and 3 if
Llama3.1's seed 3 scores above its seed 2, that is, if it improves again. Script 11's bootstrap measures
uncertainty from the choice of sentences; this script measures the other
source, decoding randomness, on the same 1,000 sentences.

For each model and each completed seed it reports F1 (scored exactly as script
04 scores, via `score_item`), then per model the mean, the range and the
population standard deviation over its seeds, and per seed the order of the
models that completed it, with a paired bootstrap for each pair. Qwen3.5 9B's
seed 1 is the main run; if its seeds 2 and 3 were not run, seeds 2 and 3 are
compared over the other three. A seed file with fewer than 1,000 rows is
reported as running and left out of every statistic.

Writes:
  results/tables/seed_spread_<N>.json
  results/logs/15_seed_spread_<N>.log

Usage:
    python scripts/15_seed_spread.py --scale 1000
"""

import argparse
import json
import logging
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.scorer import derive_attribute_relations  # noqa: E402

score04 = import_module("04_score")
report = import_module("05_report")
boot11 = import_module("11_bootstrap_ci")

RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

# Places 3 to 5, re-run first; Qwen3.5 9B was added on the condition above.
ORIGINAL = ["llama3.1_8b", "aya-expanse_8b", "command-r7b-arabic_latest"]
MODELS = ["qwen3.5_9b"] + ORIGINAL
SEEDS = [1, 2, 3]

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000])
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"15_seed_spread_{args.scale}.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    ontology = load_ontology(ONTOLOGY_PATH)
    allowed = ontology.relation_names
    gold = score04.load_gold(args.scale)
    item_order = list(gold)
    attribute = derive_attribute_relations([t for _, ts in gold.values() for t in ts])

    counts, f1 = {}, {}
    for model in MODELS:
        for seed in SEEDS:
            path = RESULTS_DIR / str(args.scale) / f"{model}_{args.scale}_seed{seed}.jsonl"
            if not path.exists():
                logger.info("%-28s seed %d: not started", report.display_name(model), seed)
                continue
            n_rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            if n_rows < len(item_order):
                logger.info("%-28s seed %d: running (%d of %d)", report.display_name(model), seed,
                            n_rows, len(item_order))
                continue
            c = boot11.per_item_counts(path, gold, allowed, attribute, item_order)
            tot = c.sum(axis=0)
            counts[(model, seed)] = c
            f1[(model, seed)] = float(boot11.micro_f1(tot[0], tot[1], tot[2]))
            logger.info("%-28s seed %d: F1 %.4f  (TP %d, FP %d, FN %d)", report.display_name(model), seed,
                        f1[(model, seed)], tot[0], tot[1], tot[2])

    out = {"scale": args.scale, "models": {}, "per_seed_order": {}, "pairs_per_seed": {}}
    logger.info("\nper model, over completed seeds:")
    for model in MODELS:
        vals = {s: f1[(model, s)] for s in SEEDS if (model, s) in f1}
        if not vals:
            continue
        v = list(vals.values())
        out["models"][model] = {
            "display_name": report.display_name(model),
            "f1_by_seed": vals,
            "n_seeds": len(v),
            "mean": statistics.mean(v),
            "min": min(v), "max": max(v), "range": max(v) - min(v),
            "pstdev": statistics.pstdev(v) if len(v) > 1 else None,
        }
        o = out["models"][model]
        logger.info("  %-28s seeds %s  mean %.4f  range %.4f  sd %s", o["display_name"], sorted(vals),
                    o["mean"], o["range"], f"{o['pstdev']:.4f}" if o["pstdev"] is not None else "n/a")

    # Order of the models in each seed, over the models that completed it, and
    # the paired bootstrap of each pair within that seed (same resampling as
    # script 11). A seed is ordered only once the three original models have
    # all completed it, so a partly run seed is never reported as an order.
    rng = np.random.default_rng(boot11.RNG_SEED)
    idx = rng.integers(0, len(item_order), size=(boot11.N_RESAMPLES, len(item_order)))
    for seed in SEEDS:
        if not all((m, seed) in f1 for m in ORIGINAL):
            continue
        done = [m for m in MODELS if (m, seed) in f1]
        order = sorted(done, key=lambda m: -f1[(m, seed)])
        out["per_seed_order"][seed] = [report.display_name(m) for m in order]
        logger.info("\nseed %d order: %s", seed, " > ".join(f"{report.display_name(m)} {f1[(m, seed)]:.3f}" for m in order))
        samples = {}
        for m in done:
            s = counts[(m, seed)][idx].sum(axis=1)
            samples[m] = boot11.micro_f1(s[:, 0], s[:, 1], s[:, 2])
        for a, b in combinations(order, 2):
            d = samples[a] - samples[b]
            res = {"diff": f1[(a, seed)] - f1[(b, seed)], "ci95": boot11.interval(d), "p_two_sided": boot11.two_sided_p(d)}
            out["pairs_per_seed"][f"seed{seed}: {report.display_name(a)} vs {report.display_name(b)}"] = res
            logger.info("  %-18s vs %-18s %+.3f  CI [%+.3f, %+.3f]  p=%.4f", report.display_name(a),
                        report.display_name(b), res["diff"], *res["ci95"], res["p_two_sided"])

    out_path = TABLES_DIR / f"seed_spread_{args.scale}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("\nwrote %s", out_path)


if __name__ == "__main__":
    main()
