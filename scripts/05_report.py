"""Turns results/tables/scale_<N>.json into readable markdown tables, in the
shape the paper needs rather than the shape the JSON happens to have.

Five tables:

1. Headline    - P/R/F1, conformance, validity, hallucinations, cost per run.
2. By style    - attribute-style (literal-valued) vs entity-style triples.
3. Norm gap    - raw score against Arabic-normalised score. The gap between
                 the two can be a finding in itself, so it gets a column
                 of its own rather than living in a second JSON block nobody
                 reads.
4. Per relation- the top model's breakdown over the 20 ontology relations,
                 which is where the genre skew of the data shows up.
5. Seed spread - retained from the original script; reads 0.000 at n=1, which
                 is an absence of evidence and is labelled as such.

Usage:
    python scripts/05_report.py --scale 1000
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES_DIR = ROOT / "results" / "tables"

# The run keys are filesystem-safe manglings of Ollama tags, so
# `hf.co_mradermacher_Fanar-1-9B-Instruct-GGUF_Q4_K_M|seed1` is one model. A
# table a reader has to decode is a table that gets misread, so map them back
# to the names the models are actually known by. Anything unmapped falls
# through unchanged rather than being silently dropped.
DISPLAY_NAMES = {
    "aya-expanse_8b": "Aya-Expanse 8B",
    "command-r7b-arabic_latest": "Command-R7B-Arabic",
    "gemma4_12b": "Gemma4 12B",
    "hf.co_mradermacher_Fanar-1-9B-Instruct-GGUF_Q4_K_M": "Fanar-1 9B",
    "hf.co_mradermacher_SILMA-9B-Instruct-v1.0-GGUF_Q4_K_M": "SILMA 9B",
    "hf.co_MhmdSyd_AceGPT-13B-chat-GGUF_Q4_K_S": "AceGPT 13B",
    "iKhalid_ALLaM_7b": "ALLaM 7B",
    "llama3.1_8b": "Llama3.1 8B",
    "mistral-nemo_12b": "Mistral-Nemo 12B",
    "mrebel": "mREBEL (baseline)",
    "qwen3.5_0.8b": "Qwen3.5 0.8B",
    "qwen3.5_2b": "Qwen3.5 2B",
    "qwen3.5_4b": "Qwen3.5 4B",
    "qwen3.5_9b": "Qwen3.5 9B",
}


def display_name(run_key: str) -> str:
    model = run_key.split("|")[0]
    return DISPLAY_NAMES.get(model, model)


def fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fmt_num(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


HEADLINE_SEED = "1"


def is_headline_run(run_key: str) -> bool:
    """One run per system in every headline table and figure: seed 1 for the
    chat models, and mREBEL, whose key carries no seed because it decodes
    deterministically. Seeds 2 and 3 exist for four models only (added in the
    revision); scored alongside, they would
    otherwise appear as extra rows of the same model in Table 2 and Figure 2.
    Their spread is reported by scripts/15_seed_spread.py instead."""
    return "|" not in run_key or run_key.endswith(f"|seed{HEADLINE_SEED}")


def by_f1_desc(data: dict) -> list[str]:
    """Headline run keys ordered strongest first. Every table uses this same
    order, so a reader comparing two tables is comparing the same rows in the
    same places."""
    keys = [k for k in data["runs"] if is_headline_run(k)]
    return sorted(keys, key=lambda k: -data["runs"][k]["raw"]["overall"]["f1"])


def headline_table(data: dict, order: list[str]) -> list[str]:
    lines = [
        "## 1. Headline results",
        "",
        "Raw scoring, no Arabic normalisation. Ordered by F1.",
        "",
        "| Model | P | R | F1 | Conformance | Validity | SH | RH | OH | Median s | p95 s | Wh/1k |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in order:
        r = data["runs"][key]["raw"]
        o = r["overall"]
        lines.append(
            f"| {display_name(key)} | {fmt_num(o['precision'])} | {fmt_num(o['recall'])} | {fmt_num(o['f1'])} "
            f"| {fmt_pct(r['ontology_conformance_pct'])} | {fmt_pct(r['output_validity_pct'])} "
            f"| {r['subject_hallucinations']} | {r['relation_hallucinations']} | {r['object_hallucinations']} "
            f"| {fmt_num(r.get('latency_median_s'), 1)} | {fmt_num(r.get('latency_p95_s'), 1)} "
            f"| {fmt_num(r.get('gpu_energy_wh_per_1000'), 1)} |"
        )
    return lines


def by_style_table(data: dict, order: list[str]) -> list[str]:
    lines = [
        "",
        "## 2. Attribute-style vs entity-style triples",
        "",
        "Attribute-style triples attach a literal value to the subject (the four date-valued",
        "relations: date of birth, date of death, point in time, inception). Entity-style triples",
        "link two named things (the other sixteen). Split derived from the gold objects' own type",
        "tags, not assigned by hand; see `derive_attribute_relations`.",
        "",
        "| Model | Attr P | Attr R | Attr F1 | Ent P | Ent R | Ent F1 | F1 gap (attr - ent) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in order:
        style = data["runs"][key]["raw"].get("by_style")
        if not style:
            continue
        a, e = style["attribute"], style["entity"]
        lines.append(
            f"| {display_name(key)} | {fmt_num(a['precision'])} | {fmt_num(a['recall'])} | {fmt_num(a['f1'])} "
            f"| {fmt_num(e['precision'])} | {fmt_num(e['recall'])} | {fmt_num(e['f1'])} "
            f"| {a['f1'] - e['f1']:+.3f} |"
        )
    return lines


def normalisation_gap_table(data: dict, order: list[str]) -> list[str]:
    lines = [
        "",
        "## 3. Effect of Arabic normalisation",
        "",
        "Normalisation strips diacritics and tatweel and unifies alef-hamza and taa-marbuta forms.",
        "A large positive gap means the model had the fact right and the spelling variant wrong.",
        "",
        "| Model | F1 raw | F1 normalised | Gap |",
        "|---|---|---|---|",
    ]
    for key in order:
        raw_f1 = data["runs"][key]["raw"]["overall"]["f1"]
        norm_f1 = data["runs"][key]["normalized"]["overall"]["f1"]
        lines.append(
            f"| {display_name(key)} | {fmt_num(raw_f1)} | {fmt_num(norm_f1)} | {norm_f1 - raw_f1:+.3f} |"
        )
    return lines


def per_relation_table(data: dict, order: list[str], top_n: int = 1) -> list[str]:
    lines = [
        "",
        "## 4. Per-relation breakdown, strongest model",
        "",
    ]
    for key in order[:top_n]:
        r = data["runs"][key]["raw"]
        lines += [
            f"**{display_name(key)}**, ordered by gold frequency (tp+fn).",
            "",
            "| Relation | P | R | F1 | Gold (tp+fn) |",
            "|---|---|---|---|---|",
        ]
        rows = r["per_relation"]
        # Ordered by how often the relation actually occurs in gold, so the
        # rows a reader sees first are the ones carrying the most weight in
        # the micro-averaged headline number. Relations a model invented but
        # gold never uses sort to the bottom with gold=0.
        for rel in sorted(rows, key=lambda x: -(rows[x]["tp"] + rows[x]["fn"])):
            m = rows[rel]
            lines.append(
                f"| {rel} | {fmt_num(m['precision'])} | {fmt_num(m['recall'])} | {fmt_num(m['f1'])} "
                f"| {m['tp'] + m['fn']} |"
            )
    return lines


NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def seed_spread_table(data: dict) -> list[str]:
    agg = data["aggregated_by_model"]
    rerun = [DISPLAY_NAMES.get(m, m) for m in sorted(agg, key=lambda m: -agg[m]["f1_mean"]) if agg[m]["n_seeds"] > 1]
    lines = [
        "",
        "## 5. Cross-seed F1 spread",
        "",
        "Most models ran one seed (for time, 2026-08-13); for them the stdev below",
        "is 0.000 by construction, an absence of evidence rather than a stability result.",
        f"{NUMBER_WORDS.get(len(rerun), str(len(rerun)))} models ({', '.join(rerun)}) were re-run with further seeds for the",
        "revision; their rows report a real spread over their runs. The headline tables use",
        "seed 1 only.",
        "",
        "| Model | F1 mean | F1 stdev | seeds |",
        "|---|---|---|---|",
    ]
    for model in sorted(data["aggregated_by_model"], key=lambda m: -data["aggregated_by_model"][m]["f1_mean"]):
        a = data["aggregated_by_model"][model]
        lines.append(
            f"| {DISPLAY_NAMES.get(model, model)} | {fmt_num(a['f1_mean'])} | {fmt_num(a['f1_stdev'])} | {a['n_seeds']} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    path = TABLES_DIR / f"scale_{args.scale}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    order = by_f1_desc(data)
    lines = [f"# Study 2 results, scale {args.scale}", ""]
    lines += headline_table(data, order)
    lines += by_style_table(data, order)
    lines += normalisation_gap_table(data, order)
    lines += per_relation_table(data, order)
    lines += seed_spread_table(data)

    report = "\n".join(lines) + "\n"
    out_path = TABLES_DIR / f"scale_{args.scale}_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
