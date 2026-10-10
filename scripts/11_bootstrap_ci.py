"""Bootstrap confidence intervals for every headline F1, and paired bootstrap
tests between models, over the sentences of one scale tier.

Added in the revision after peer review: with one decoding
run per model, the paper could not say whether a gap of 0.014 F1 between two
models is larger than the noise. This script answers part of that question.

What it measures, and what it does not. The bootstrap resamples the 1,000 test
sentences with replacement and re-scores every model on each resample, using
the same resample for every model (a paired bootstrap). The spread it reports
is the uncertainty that comes from which sentences happened to be drawn into the
test set. It says nothing about decoding randomness: at temperature 0.2 a second
seed would give different outputs on the same sentences, and that variation is
measured separately, by re-running models with seeds 2 and 3
(scripts/14_seed_spread.py). The paper must keep the two apart.

Per-sentence counts come from `score_item`, the function script 04 scores with,
so the point estimates here are the headline numbers by construction. The
script checks that against results/tables/scale_<N>.json before it reports
anything, and stops if a single run differs.

Micro-averaged F1 is 2TP / (2TP + FP + FN), so a resample only needs the summed
counts, which keeps 10,000 resamples over eleven runs to a few seconds.

Writes:
  results/tables/bootstrap_<N>_seed<S>.json
  results/logs/11_bootstrap_<N>_seed<S>.log

Usage:
    python scripts/11_bootstrap_ci.py --scale 1000            # seed 1, the headline runs
    python scripts/11_bootstrap_ci.py --scale 1000 --seed 2   # only models that have a seed-2 file
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.scorer import derive_attribute_relations, normalize_relation, score_item  # noqa: E402
from src.extraction.triple import Triple  # noqa: E402

score04 = import_module("04_score")
report = import_module("05_report")

RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

N_RESAMPLES = 10_000
RNG_SEED = 20261008  # fixed, so the intervals in the paper can be regenerated exactly

logger = logging.getLogger(__name__)


def headline_files(scale: int, seed: int) -> dict[str, Path]:
    """One file per system: each chat model at the given seed, plus mREBEL,
    which decodes deterministically and has no seed in its file name."""
    files = {}
    for path in sorted((RESULTS_DIR / str(scale)).glob(f"*_{scale}*.jsonl")):
        m = score04.FILENAME_RE.match(path.name)
        if not m or m.group("scale") != str(scale):
            continue
        file_seed = m.group("seed")
        if file_seed is None or int(file_seed) == seed:
            files[m.group("model")] = path
    return files


def per_item_counts(path: Path, gold: dict, allowed: frozenset[str], attribute_relations: frozenset[str],
                    item_order: list[str]) -> np.ndarray:
    """Returns an array of shape (n_items, 6): TP, FP, FN overall, then TP, FP,
    FN for attribute-style relations only. Entity-style counts are the overall
    counts minus the attribute ones. Rows follow `item_order`, so every run's
    array lines up sentence by sentence, which the paired bootstrap needs."""
    position = {item_id: i for i, item_id in enumerate(item_order)}
    counts = np.zeros((len(item_order), 6), dtype=np.int64)
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Same tolerance as script 04: a truncated final write is skipped.
                continue
            item_id = row["item_id"]
            if item_id not in gold:
                continue
            seen.add(item_id)
            text, gold_triples = gold[item_id]
            predicted = [] if "error" in row else [Triple(**t) for t in row.get("predicted_triples", [])]
            s = score_item(predicted, gold_triples, text, allowed, normalize=False)
            is_attr = lambda rel: normalize_relation(rel) in attribute_relations  # noqa: E731
            counts[position[item_id]] += [
                len(s["tp_relations"]), len(s["fp_relations"]), len(s["fn_relations"]),
                sum(map(is_attr, s["tp_relations"])), sum(map(is_attr, s["fp_relations"])),
                sum(map(is_attr, s["fn_relations"])),
            ]
    missing = set(item_order) - seen
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} of {len(item_order)} items have no row; "
                         "a paired bootstrap needs every run to cover the same sentences")
    return counts


def micro_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def micro_p(tp, fp):
    return np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)


def micro_r(tp, fn):
    return np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)


def interval(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def two_sided_p(diff: np.ndarray) -> float:
    """Share of resamples on the far side of zero, doubled. With 10,000
    resamples the smallest value this can report is 0.0002 (one resample);
    a result of 0 is written as '< 0.0001' in the paper, never as 0."""
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return float(min(p, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000])
    parser.add_argument("--seed", type=int, default=1, help="which decoding seed to read for the chat models")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"11_bootstrap_{args.scale}_seed{args.seed}.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )

    ontology = load_ontology(ONTOLOGY_PATH)
    allowed = ontology.relation_names
    gold = score04.load_gold(args.scale)
    item_order = list(gold)
    attribute_relations = derive_attribute_relations([t for _, triples in gold.values() for t in triples])

    files = headline_files(args.scale, args.seed)
    logger.info("scale %d, seed %d: %d runs, %d sentences, %d resamples, rng seed %d",
                args.scale, args.seed, len(files), len(item_order), N_RESAMPLES, RNG_SEED)

    counts = {model: per_item_counts(path, gold, allowed, attribute_relations, item_order)
              for model, path in files.items()}

    # Point estimates must reproduce the scored table exactly, or nothing
    # below can be quoted next to it.
    with open(TABLES_DIR / f"scale_{args.scale}.json", encoding="utf-8") as f:
        scored = json.load(f)["runs"]
    for model, c in counts.items():
        key = f"{model}|seed{args.seed}" if f"{model}|seed{args.seed}" in scored else model
        ref = scored[key]["raw"]["overall"]
        tp, fp, fn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()
        if (tp, fp, fn) != (ref["tp"], ref["fp"], ref["fn"]):
            raise SystemExit(f"{model}: per-item sums {(tp, fp, fn)} differ from scale table "
                             f"{(ref['tp'], ref['fp'], ref['fn'])}")
    logger.info("point estimates reproduce scale_%d.json for all %d runs", args.scale, len(counts))

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(item_order), size=(N_RESAMPLES, len(item_order)))

    f1_samples, gap_samples, out_models = {}, {}, {}
    for model, c in counts.items():
        s = c[idx].sum(axis=1)  # (resamples, 6)
        tp, fp, fn, tpa, fpa, fna = (s[:, k] for k in range(6))
        tpe, fpe, fne = tp - tpa, fp - fpa, fn - fna
        f1 = micro_f1(tp, fp, fn)
        attr_f1, ent_f1 = micro_f1(tpa, fpa, fna), micro_f1(tpe, fpe, fne)
        gap = attr_f1 - ent_f1
        f1_samples[model] = f1
        gap_samples[model] = gap

        tot = c.sum(axis=0)
        point_attr = float(micro_f1(tot[3], tot[4], tot[5]))
        point_ent = float(micro_f1(tot[0] - tot[3], tot[1] - tot[4], tot[2] - tot[5]))
        out_models[model] = {
            "display_name": report.display_name(model),
            "f1": float(micro_f1(tot[0], tot[1], tot[2])),
            "f1_ci95": interval(f1),
            "precision": float(micro_p(tot[0], tot[1])),
            "precision_ci95": interval(micro_p(tp, fp)),
            "recall": float(micro_r(tot[0], tot[2])),
            "recall_ci95": interval(micro_r(tp, fn)),
            "attr_f1": point_attr,
            "attr_f1_ci95": interval(attr_f1),
            "ent_f1": point_ent,
            "ent_f1_ci95": interval(ent_f1),
            "attr_minus_ent": point_attr - point_ent,
            "attr_minus_ent_ci95": interval(gap),
            "share_of_resamples_gap_positive": float(np.mean(gap > 0)),
        }

    ranking = sorted(out_models, key=lambda m: -out_models[m]["f1"])
    pairs = {}
    for a, b in combinations(ranking, 2):
        diff = f1_samples[a] - f1_samples[b]
        pairs[f"{a} vs {b}"] = {
            "a": out_models[a]["display_name"],
            "b": out_models[b]["display_name"],
            "f1_diff": out_models[a]["f1"] - out_models[b]["f1"],
            "diff_ci95": interval(diff),
            "p_two_sided": two_sided_p(diff),
            "share_a_ahead": float(np.mean(diff > 0)),
        }

    # The paper's central contrast is not "each prompted model has a gap" but
    # "the trained extractor's gap is smaller than the prompted models' gaps".
    # Tested directly, per prompted model, on the same resamples.
    gap_vs_baseline = {}
    if "mrebel" in gap_samples:
        for model in ranking:
            if model == "mrebel":
                continue
            diff = gap_samples[model] - gap_samples["mrebel"]
            gap_vs_baseline[model] = {
                "display_name": out_models[model]["display_name"],
                "gap_minus_mrebel_gap": out_models[model]["attr_minus_ent"] - out_models["mrebel"]["attr_minus_ent"],
                "ci95": interval(diff),
                "p_two_sided": two_sided_p(diff),
            }

    logger.info("%-20s %6s  %-15s  %6s %6s %6s  %-15s", "model", "F1", "95% CI", "attr", "ent", "gap", "gap 95% CI")
    for m in ranking:
        o = out_models[m]
        logger.info("%-20s %.3f  [%.3f, %.3f]  %.3f  %.3f  %+.3f  [%+.3f, %+.3f]",
                    o["display_name"], o["f1"], *o["f1_ci95"], o["attr_f1"], o["ent_f1"],
                    o["attr_minus_ent"], *o["attr_minus_ent_ci95"])
    logger.info("adjacent pairs in the ranking:")
    for a, b in zip(ranking, ranking[1:]):
        p = pairs[f"{a} vs {b}"]
        logger.info("  %-18s vs %-18s  diff %+.3f  CI [%+.3f, %+.3f]  p=%.4f",
                    p["a"], p["b"], p["f1_diff"], *p["diff_ci95"], p["p_two_sided"])
    if gap_vs_baseline:
        logger.info("attribute-minus-entity gap, prompted model minus mREBEL:")
        for g in gap_vs_baseline.values():
            logger.info("  %-18s  %+.3f  CI [%+.3f, %+.3f]  p=%.4f",
                        g["display_name"], g["gap_minus_mrebel_gap"], *g["ci95"], g["p_two_sided"])

    out = {
        "scale": args.scale,
        "seed": args.seed,
        "n_items": len(item_order),
        "n_resamples": N_RESAMPLES,
        "rng_seed": RNG_SEED,
        "what_it_measures": "uncertainty from the choice of test sentences; not decoding randomness",
        "ranking": ranking,
        "models": out_models,
        "pairs": pairs,
        "gap_vs_mrebel": gap_vs_baseline,
    }
    out_path = TABLES_DIR / f"bootstrap_{args.scale}_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
