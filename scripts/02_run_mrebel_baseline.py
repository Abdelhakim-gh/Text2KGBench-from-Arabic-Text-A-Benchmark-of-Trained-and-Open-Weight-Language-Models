"""Runs mREBEL (Babelscape/mrebel-large) zero-shot over one scale tier, no
ontology constraint (it was never trained on one): the trained-specialist
baseline. Resumable via
src/common/job_runner.py, same as script 03.

The decode recipe matters and was confirmed by direct inspection rather
than assumed: passing
`forced_bos_token_id` produces degenerate repetition on Arabic input; the
model card's own recipe, `decoder_start_token_id=<triplet>`, is what
actually works. Getting this wrong does not raise an error, it just
silently produces garbage, which is why it was checked against real output
before being trusted here.

Usage:
    python scripts/02_run_mrebel_baseline.py --scale 100
    python scripts/02_run_mrebel_baseline.py --scale 100 --limit 5   # smoke test
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.job_runner import run_job  # noqa: E402
from src.common.meter import GPUEnergyMeter  # noqa: E402
from src.extraction.parser import parse_mrebel_output  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "extraction"
MODEL_NAME = "Babelscape/mrebel-large"

SCALE_FILES = {100: "sanity_100.jsonl", 1000: "main_1000.jsonl", 10000: "confirm_10000.jsonl"}

GEN_KWARGS = {"max_length": 256, "length_penalty": 0, "num_beams": 3, "num_return_sequences": 1}


def load_split(scale: int) -> list[dict]:
    path = SPLITS_DIR / SCALE_FILES[scale]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_worker(model, tokenizer, triplet_token_id: int, meter: GPUEnergyMeter, device: str):
    def worker(item: dict) -> dict:
        inputs = tokenizer(item["text"], max_length=256, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with meter.measure() as m:
            generated = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_start_token_id=triplet_token_id,
                **GEN_KWARGS,
            )
        result = m.result()
        raw = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
        triples = parse_mrebel_output(raw)
        return {
            "raw_output": raw,
            "predicted_triples": [dataclasses.asdict(t) for t in triples],
            "wall_clock_s": result.wall_clock_s,
            "gpu_energy_wh": result.gpu_energy_wh,
        }

    return worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    parser.add_argument("--limit", type=int, default=None, help="only run the first N items, for a smoke test")
    args = parser.parse_args()

    # Imported here, not at module level: importing transformers/torch is slow
    # and unnecessary for --help or argument errors.
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    items = load_split(args.scale)
    if args.limit:
        items = items[: args.limit]
    logger.info("scale %d: %d items loaded", args.scale, len(items))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading %s on %s", MODEL_NAME, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="ar_AR", tgt_lang="<triplet>")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    triplet_token_id = tokenizer.convert_tokens_to_ids("<triplet>")

    output_path = RESULTS_DIR / str(args.scale) / f"mrebel_{args.scale}.jsonl"
    meter = GPUEnergyMeter()
    worker = make_worker(model, tokenizer, triplet_token_id, meter, device)
    run_job(items, worker, output_path)


if __name__ == "__main__":
    main()
