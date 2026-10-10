"""Tests mREBEL on sentences it was not trained on, and compares the result
with its score on the benchmark sentences, which it was trained on.

Why this exists. Script 14 found that all 1,000 benchmark sentences sit in
SRED-FM's Arabic train split, the data mREBEL was trained on. The submitted
paper used mREBEL as the trained-extractor reference, quoting its precision
(0.727) and its small attribute/entity gap. Both were measured on sentences
the model had seen in training. SRED-FM also has an Arabic validation and test
split. This script draws a held-out benchmark from them with exactly the
rules script 01 used, runs mREBEL on it with exactly the decoding of script
02, scores it with exactly the scorer of script 04, and reports the two side
by side with bootstrap intervals.

Three steps, run in order (each can be re-run on its own):

  --build   Converts SRED-FM rows into this study's gold format and checks the
            conversion first: applied to the SRED-FM train rows of the 1,000
            benchmark sentences, it must reproduce ArSRED's gold triples. Then
            keeps closed-ontology documents from validation and test (every
            gold triple uses one of the 20 relations, at least one triple),
            confirms none of their texts occurs in the train split, shuffles
            with script 01's seed and writes the first 1,000.
  --run     mREBEL over the held-out split (resumable, like script 02).
  --score   Scores both sets and writes the comparison, for mREBEL and for
            Gemma4 12B and Qwen3.5 9B once their held-out runs exist
            (scripts/03_run_llm.py --split-path <held-out split> --tier
            heldout). The two chat models have no reported training on SRED-FM, so
            their change between the sets is the reference for how much the
            held-out set alone moves scores; the script also tests whether
            mREBEL's drop exceeds theirs.

The held-out set differs from the benchmark in more than exposure: its items
are single sentences cut from longer documents, with fewer facts each. That
is why the reference systems are needed, and why mREBEL's held-out score is
not read on its own.

The 20-relation ontology is not re-derived from the held-out data: the
benchmark's ontology is fixed, and the held-out set is filtered to it.

Writes:
  data/processed/splits/heldout_sredfm_1000.jsonl
  results/extraction/heldout/mrebel_heldout_1000.jsonl
  results/tables/heldout_mrebel_1000.json
  results/logs/16_heldout_mrebel.log

Usage:
    python scripts/16_heldout_mrebel.py --build --run --score
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# pyarrow (for the SRED-FM parquet files) and script 01 (which imports
# `datasets`, and through it pyarrow) are imported only by --build: --run and
# --score need neither, and on this machine Windows Smart App Control
# sometimes blocks pyarrow's DLL.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.scorer import aggregate_items, derive_attribute_relations, normalize_relation, score_item  # noqa: E402
from src.extraction.triple import Triple  # noqa: E402

score04 = import_module("04_score")

SREDFM_DIR = ROOT / "data" / "raw" / "sredfm_ar"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
HELDOUT_PATH = SPLITS_DIR / "heldout_sredfm_1000.jsonl"
EXTRACTION_PATH = ROOT / "results" / "extraction" / "heldout" / "mrebel_heldout_1000.jsonl"
SEEN_PATH = ROOT / "results" / "extraction" / "1000" / "mrebel_1000.jsonl"
TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

N_HELDOUT = 1000
N_RESAMPLES = 10_000
RNG_SEED = 20261008

logger = logging.getLogger(__name__)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def convert_row(row: dict) -> list[dict]:
    """SRED-FM's entity list plus index-based relations, as this study's gold
    triples. ArSRED writes entity tags in lower case; SRED-FM in mixed case."""
    ents = row["entities"]
    return [
        {
            "subject": ents[r["subject"]]["surfaceform"],
            "relation": r["predicate"],
            "object": ents[r["object"]]["surfaceform"],
            "subject_type": ents[r["subject"]]["type"].lower(),
            "object_type": ents[r["object"]]["type"].lower(),
        }
        for r in row["relations"]
    ]


def triple_key(t: dict) -> tuple:
    return (t["subject"], t["relation"], t["object"], t["subject_type"], t["object_type"])


def read_split(name: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for f in sorted(SREDFM_DIR.glob(f"{name}*.parquet")):
        rows.extend(pq.read_table(f, columns=["docid", "text", "entities", "relations"]).to_pylist())
    if not rows:
        raise SystemExit(f"no SRED-FM {name} parquet in {SREDFM_DIR}; see script 14 for the download")
    return rows


# --- build --------------------------------------------------------------------

def build() -> None:
    build01 = import_module("01_build_ontology_and_testset")
    allowed = set(load_ontology(ONTOLOGY_PATH).relation_names)

    # 1. The conversion must reproduce ArSRED's own gold on the benchmark sentences.
    bench = {}
    with open(SPLITS_DIR / "main_1000.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bench[norm_text(d["text"])] = Counter(triple_key(t) for t in d["gold_triples"])
    train = read_split("train")
    same = differ = 0
    seen_texts, checked = set(), set()
    for row in train:
        key = norm_text(row["text"])
        seen_texts.add(key)
        if key in bench and key not in checked:
            checked.add(key)
            if Counter(triple_key(t) for t in convert_row(row)) == bench[key]:
                same += 1
            else:
                differ += 1
    logger.info("conversion check on the benchmark sentences: %d identical, %d differ, %d not found",
                same, differ, len(bench) - same - differ)
    if same != len(bench):
        raise SystemExit("the SRED-FM conversion does not reproduce ArSRED's gold; fix convert_row first")

    # 2. Closed-ontology items from validation and test, none of them in train.
    #
    # Whole documents do not work here: validation and test documents are about
    # twice as long as train ones (median 256 and 281 characters against 130),
    # carry more facts, and not one of the 8,170 has all its facts inside the 20
    # relations. So items are single sentences. A gold fact belongs to a
    # sentence when both its entities lie inside it (SRED-FM gives character
    # offsets); a fact spanning two sentences is dropped, as any sentence-level
    # gold would. The closed-ontology rule of script 01 is then applied per
    # sentence. Sentences carrying wiki markup (a `|`, as in image captions)
    # are skipped.
    pool = []
    stats = Counter()
    for split in ("validation", "test"):
        for row in read_split(split):
            stats["documents"] += 1
            ents = row["entities"]
            text = row["text"]
            for e in ents:
                if text[e["start"]:e["end"]] != e["surfaceform"]:
                    stats["entity_offset_mismatch"] += 1
            bounds = [(m.start(), m.end()) for m in re.finditer(r"[^\n.!?؟]+[.!?؟]?", text)]
            by_sentence: dict[int, list[dict]] = {}
            for r in row["relations"]:
                s, o = ents[r["subject"]], ents[r["object"]]
                hit = next((i for i, (a, b) in enumerate(bounds)
                            if a <= s["start"] and s["end"] <= b and a <= o["start"] and o["end"] <= b), None)
                if hit is None:
                    stats["facts_spanning_sentences"] += 1
                    continue
                by_sentence.setdefault(hit, []).append({
                    "subject": s["surfaceform"], "relation": r["predicate"], "object": o["surfaceform"],
                    "subject_type": s["type"].lower(), "object_type": o["type"].lower(),
                })
            for i, triples in by_sentence.items():
                sentence = text[bounds[i][0]:bounds[i][1]].strip()
                stats["sentences_with_facts"] += 1
                if "|" in sentence:
                    stats["skipped_markup"] += 1
                    continue
                if not all(t["relation"] in allowed for t in triples):
                    continue
                stats["closed"] += 1
                if norm_text(sentence) in seen_texts:
                    stats["in_train"] += 1
                    continue
                pool.append({"item_id": f"sredfm_{split}_{row['docid']}_s{i}", "text": sentence,
                             "gold_triples": triples})
    logger.info("held-out pool from validation+test: %s; usable sentences %d", dict(stats), len(pool))

    order = list(range(len(pool)))
    random.Random(build01.RANDOM_STATE).shuffle(order)
    chosen = [pool[i] for i in order[:N_HELDOUT]]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HELDOUT_PATH, "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d documents, %d gold triples)", HELDOUT_PATH.name, len(chosen),
                sum(len(r["gold_triples"]) for r in chosen))


# --- run ----------------------------------------------------------------------

def run() -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.common.job_runner import run_job
    from src.common.meter import GPUEnergyMeter

    mrebel02 = import_module("02_run_mrebel_baseline")
    with open(HELDOUT_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading %s on %s for %d held-out items", mrebel02.MODEL_NAME, device, len(items))
    tokenizer = AutoTokenizer.from_pretrained(mrebel02.MODEL_NAME, src_lang="ar_AR", tgt_lang="<triplet>")
    model = AutoModelForSeq2SeqLM.from_pretrained(mrebel02.MODEL_NAME).to(device)
    worker = mrebel02.make_worker(model, tokenizer, tokenizer.convert_tokens_to_ids("<triplet>"),
                                  GPUEnergyMeter(), device)
    run_job(items, worker, EXTRACTION_PATH)


# --- score --------------------------------------------------------------------

def load_gold_file(path: Path) -> dict:
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold[d["item_id"]] = (d["text"], [Triple(**t) for t in d["gold_triples"]])
    return gold


def per_item(path: Path, gold: dict, allowed, attribute) -> tuple[list[dict], np.ndarray]:
    """Scored items (for aggregate_items, as script 04) and per-item counts
    (TP, FP, FN overall; TP, FP, FN attribute-style) for the bootstrap."""
    order = {item_id: i for i, item_id in enumerate(gold)}
    counts = np.zeros((len(gold), 6), dtype=np.int64)
    items, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["item_id"] not in gold:
                continue
            seen.add(row["item_id"])
            text, g = gold[row["item_id"]]
            pred = [] if "error" in row else [Triple(**t) for t in row.get("predicted_triples", [])]
            s = score_item(pred, g, text, allowed, normalize=False)
            items.append(s)
            attr = lambda r: normalize_relation(r) in attribute  # noqa: E731
            counts[order[row["item_id"]]] += [
                len(s["tp_relations"]), len(s["fp_relations"]), len(s["fn_relations"]),
                sum(map(attr, s["tp_relations"])), sum(map(attr, s["fp_relations"])),
                sum(map(attr, s["fn_relations"])),
            ]
    if len(seen) != len(gold):
        raise SystemExit(f"{path.name}: {len(gold) - len(seen)} items have no row yet")
    return items, counts


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


# Systems compared on both sets. mREBEL is the one trained on SRED-FM's train
# split; the two chat models have no reported training on it, so their change between the sets
# measures how much the held-out set alone moves scores. A system whose
# held-out file is missing or incomplete is skipped, not faked.
SYSTEMS = [
    ("mrebel", "mREBEL", SEEN_PATH, EXTRACTION_PATH),
    ("gemma4_12b", "Gemma4 12B",
     ROOT / "results" / "extraction" / "1000" / "gemma4_12b_1000_seed1.jsonl",
     ROOT / "results" / "extraction" / "heldout" / "gemma4_12b_heldout_seed1.jsonl"),
    ("qwen3.5_9b", "Qwen3.5 9B",
     ROOT / "results" / "extraction" / "1000" / "qwen3.5_9b_1000_seed1.jsonl",
     ROOT / "results" / "extraction" / "heldout" / "qwen3.5_9b_heldout_seed1.jsonl"),
]


def summarise(items, counts, b, gold, attribute) -> dict:
    agg = aggregate_items(items, [], attribute)
    o, st = agg["overall"], agg["by_style"]
    return {
        "n_items": len(gold),
        "n_gold": int(counts[:, 0].sum() + counts[:, 2].sum()),
        "n_predicted": agg["n_predicted_triples"],
        "precision": o["precision"], "precision_ci95": ci(b["precision"]),
        "recall": o["recall"], "recall_ci95": ci(b["recall"]),
        "f1": o["f1"], "f1_ci95": ci(b["f1"]),
        "attr_f1": st["attribute"]["f1"], "attr_f1_ci95": ci(b["attr_f1"]),
        "ent_f1": st["entity"]["f1"], "ent_f1_ci95": ci(b["ent_f1"]),
        "ent_precision": st["entity"]["precision"],
        "gap": st["attribute"]["f1"] - st["entity"]["f1"], "gap_ci95": ci(b["gap"]),
        "conformance_pct": agg["ontology_conformance_pct"],
    }


def complete(path: Path, gold: dict) -> bool:
    if not path.exists():
        return False
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) >= len(gold)


def score() -> None:
    allowed = load_ontology(ONTOLOGY_PATH).relation_names
    seen_gold = score04.load_gold(1000)
    held_gold = load_gold_file(HELDOUT_PATH)
    # Style split defined once, on the benchmark's gold, as script 04 does, and
    # applied unchanged to the held-out set so both columns mean the same thing.
    attribute = derive_attribute_relations([t for _, ts in seen_gold.values() for t in ts])

    # One resample matrix per set, shared by every system on that set: the
    # comparison between systems is paired within a set, and the two sets are
    # resampled independently because they hold different sentences.
    rng = np.random.default_rng(RNG_SEED)
    idx = {"seen": rng.integers(0, len(seen_gold), size=(N_RESAMPLES, len(seen_gold))),
           "heldout": rng.integers(0, len(held_gold), size=(N_RESAMPLES, len(held_gold)))}
    golds = {"seen": seen_gold, "heldout": held_gold}

    out = {"n_resamples": N_RESAMPLES, "rng_seed": RNG_SEED, "systems": {}}
    drops = {}
    for key, label, seen_path, held_path in SYSTEMS:
        if not complete(held_path, held_gold):
            logger.info("%-11s held-out run missing or incomplete; skipped", label)
            continue
        res, samp = {}, {}
        for name, path in (("seen", seen_path), ("heldout", held_path)):
            items, counts = per_item(path, golds[name], allowed, attribute)
            b = boot_with(counts, idx[name])
            samp[name] = b
            res[name] = summarise(items, counts, b, golds[name], attribute)
            r = res[name]
            logger.info("%-11s %-8s P %.3f [%.3f, %.3f]  R %.3f  F1 %.3f [%.3f, %.3f]  attr %.3f  ent %.3f  gap %+.3f",
                        label, name, r["precision"], *r["precision_ci95"], r["recall"], r["f1"], *r["f1_ci95"],
                        r["attr_f1"], r["ent_f1"], r["gap"])
        res["seen_minus_heldout"] = {}
        for k in ("precision", "recall", "f1", "gap"):
            d = samp["seen"][k] - samp["heldout"][k]
            res["seen_minus_heldout"][k] = {"diff": res["seen"][k] - res["heldout"][k], "ci95": ci(d),
                                            "p_two_sided": two_sided(d)}
            logger.info("%-11s seen - held-out %-9s %+.3f  CI [%+.3f, %+.3f]  p=%.4f", label, k,
                        res["seen"][k] - res["heldout"][k], *ci(d), two_sided(d))
        drops[key] = {k: samp["seen"][k] - samp["heldout"][k] for k in ("precision", "recall", "f1", "gap")}
        out["systems"][key] = {"label": label, **res}

    # Is mREBEL's drop larger than a chat model's? Difference of the two drops,
    # on the same resamples.
    out["mrebel_drop_minus_other_drop"] = {}
    if "mrebel" in drops:
        for key, label, _, _ in SYSTEMS:
            if key == "mrebel" or key not in drops:
                continue
            for k in ("precision", "recall", "f1"):
                d = drops["mrebel"][k] - drops[key][k]
                point = (out["systems"]["mrebel"]["seen_minus_heldout"][k]["diff"]
                         - out["systems"][key]["seen_minus_heldout"][k]["diff"])
                out["mrebel_drop_minus_other_drop"][f"{key}:{k}"] = {"diff": point, "ci95": ci(d),
                                                                     "p_two_sided": two_sided(d)}
                logger.info("mREBEL drop minus %-11s drop, %-9s %+.3f  CI [%+.3f, %+.3f]  p=%.4f",
                            label, k, point, *ci(d), two_sided(d))

    out_path = TABLES_DIR / "heldout_mrebel_1000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", out_path)


def boot_with(counts: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    s = counts[idx].sum(axis=1)
    tp, fp, fn, ta, fa, na = (s[:, k] for k in range(6))
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    attr, ent = f1(ta, fa, na), f1(tp - ta, fp - fa, fn - na)
    return {"precision": p, "recall": r, "f1": f1(tp, fp, fn), "attr_f1": attr, "ent_f1": ent, "gap": attr - ent}


def two_sided(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / "16_heldout_mrebel.log", mode="a", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    if args.build:
        build()
    if args.run:
        run()
    if args.score:
        score()


if __name__ == "__main__":
    main()
