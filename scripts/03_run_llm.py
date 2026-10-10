"""Runs every local chat model zero-shot, with the closed ontology given in
the prompt, over one scale tier. Resumable: src/common/job_runner.py writes
one JSON line per item as it completes, so Ctrl+C or a laptop sleep loses
nothing already written, and re-running the same command picks up where it
stopped rather than redoing finished work.

Usage:
    python scripts/03_run_llm.py --scale 100
    python scripts/03_run_llm.py --scale 1000 --models qwen3.5:0.8b,qwen3.5:2b
    python scripts/03_run_llm.py --scale 100 --limit 5   # fast smoke test
    python scripts/03_run_llm.py --split-path data/processed/splits/heldout_sredfm_1000.jsonl         --tier heldout --models gemma4:12b                 # any other split file (script 16)
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.job_runner import run_job  # noqa: E402
from src.common.meter import GPUEnergyMeter  # noqa: E402
from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.parser import parse_llm_output  # noqa: E402
from src.llm.model_runner import OllamaRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT_PATH = ROOT.parent / "prompts" / "study2_ontology_extraction_ar.txt"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "extraction"
LLM_CONFIG_PATH = ROOT / "configs" / "llm_models.yaml"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

SCALE_FILES = {100: "sanity_100.jsonl", 1000: "main_1000.jsonl", 10000: "confirm_10000.jsonl"}


def load_split(scale: int) -> list[dict]:
    path = SPLITS_DIR / SCALE_FILES[scale]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fill_prompt(template: str, ontology_list: str, sentence: str) -> str:
    # str.replace, not str.format: the prompt's JSON example contains literal
    # {"subject": ...} braces that .format() would try to parse as fields.
    return template.replace("{ontology_list}", ontology_list).replace("{sentence}", sentence)


def make_worker(runner: OllamaRunner, prompt_template: str, ontology_list: str, seed: int,
                 temperature: float, max_tokens: int, meter: GPUEnergyMeter):
    def worker(item: dict) -> dict:
        prompt = fill_prompt(prompt_template, ontology_list, item["text"])
        with meter.measure() as m:
            gen = runner.generate(prompt, seed=seed, temperature=temperature, max_tokens=max_tokens)
        result = m.result()
        triples, valid = parse_llm_output(gen.text)
        return {
            "raw_output": gen.text,
            "valid_json": valid,
            "predicted_triples": [dataclasses.asdict(t) for t in triples],
            "tokens_in": gen.tokens_in,
            "tokens_out": gen.tokens_out,
            "wall_clock_s": result.wall_clock_s,
            "gpu_energy_wh": result.gpu_energy_wh,
        }

    return worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=[100, 1000, 10000],
                        help="one of the benchmark tiers; or use --split-path with --tier")
    parser.add_argument("--split-path", type=Path, default=None,
                        help="a split file outside the three tiers, e.g. the held-out set of script 16")
    parser.add_argument("--tier", type=str, default=None,
                        help="name for --split-path's output folder and file names, e.g. 'heldout'")
    parser.add_argument("--models", type=str, default=None, help="comma-separated model names; default is every model in configs/llm_models.yaml")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds; default is every seed in configs/llm_models.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N items of the split, for a smoke test")
    args = parser.parse_args()

    with open(LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ontology = load_ontology(ONTOLOGY_PATH)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    ontology_list = ontology.prompt_relation_list()

    if args.split_path is not None:
        if not args.tier:
            parser.error("--split-path needs --tier")
        with open(args.split_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        tier = args.tier
    elif args.scale is not None:
        items = load_split(args.scale)
        tier = str(args.scale)
    else:
        parser.error("give --scale, or --split-path with --tier")
    if args.limit:
        items = items[: args.limit]
    logger.info("tier %s: %d items loaded", tier, len(items))

    model_names = args.models.split(",") if args.models else [m["name"] for m in cfg["models"]]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]

    meter = GPUEnergyMeter()

    for model_name in model_names:
        runner = OllamaRunner(model_name, host=cfg["ollama_host"], num_ctx=cfg.get("num_ctx", 8192))
        if not runner.ping():
            logger.error("%s: ping failed (model not pulled, or Ollama not running?), skipping", model_name)
            continue

        safe_name = model_name.replace("/", "_").replace(":", "_")
        for seed in seeds:
            output_path = RESULTS_DIR / tier / f"{safe_name}_{tier}_seed{seed}.jsonl"
            logger.info("%s seed=%d tier=%s -> %s", model_name, seed, tier, output_path)
            worker = make_worker(runner, prompt_template, ontology_list, seed, cfg["temperature"], cfg["max_tokens"], meter)
            run_job(items, worker, output_path)


if __name__ == "__main__":
    main()
