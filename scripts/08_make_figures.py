"""Emits the paper's figure data straight from results/tables/scale_<N>.json.

Figures are generated, never hand-transcribed. A number typed twice is a
number that can disagree with itself, and the one place that must never
happen is between a results table and the chart printed beside it. Every
value below is read from the scored JSON at build time, so re-running the
scorer and re-running this script cannot leave the paper stale.

Writes into figures/:
  data_headline.dat  - per model: overall / attribute / entity F1
  data_cost.dat      - per model: median latency, F1, energy per 1k items

The .tex files that plot these live beside them and are written by hand; only
the numbers are generated.

Usage:
    python scripts/08_make_figures.py --scale 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

report = import_module("05_report")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "figures"


def escape(name: str) -> str:
    """pgfplots reads these as symbolic x coords, so a name may not contain a
    comma or a brace. The roster has neither today; assert rather than mangle
    silently, so a future model name with a comma fails loudly here instead of
    producing a chart with two mislabelled bars."""
    assert "," not in name and "{" not in name and "}" not in name, name
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True, choices=[100, 1000, 10000])
    args = parser.parse_args()

    with open(TABLES_DIR / f"scale_{args.scale}.json", encoding="utf-8") as f:
        data = json.load(f)

    order = report.by_f1_desc(data)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- headline: overall, attribute and entity F1 per model --------------
    headline = FIGURES_DIR / "data_headline.dat"
    with open(headline, "w", encoding="utf-8") as f:
        f.write("model f1 attr_f1 ent_f1\n")
        for key in order:
            r = data["runs"][key]["raw"]
            style = r.get("by_style")
            if not style:
                continue
            f.write(
                f"{escape(report.display_name(key)).replace(' ', '~')} "
                f"{r['overall']['f1']:.4f} "
                f"{style['attribute']['f1']:.4f} "
                f"{style['entity']['f1']:.4f}\n"
            )
    logger.info("wrote %s", headline)

    # --- cost: what each point of F1 costs in time and energy --------------
    # mREBEL is kept in this one on purpose: the interesting part of the cost
    # picture is that the cheapest arm is also the weakest, which is only
    # visible if the baseline is plotted alongside the LLMs rather than
    # mentioned in a footnote.
    cost = FIGURES_DIR / "data_cost.dat"
    with open(cost, "w", encoding="utf-8") as f:
        f.write("model latency f1 energy\n")
        for key in order:
            r = data["runs"][key]["raw"]
            if "latency_median_s" not in r:
                continue
            f.write(
                f"{escape(report.display_name(key)).replace(' ', '~')} "
                f"{r['latency_median_s']:.3f} "
                f"{r['overall']['f1']:.4f} "
                f"{r.get('gpu_energy_wh_per_1000', 0.0):.2f}\n"
            )
    logger.info("wrote %s", cost)

    write_headline_table(data, order)
    write_camera_ready_data(data, order, args.scale)


def run_model(run_key: str) -> str:
    return run_key.split("|")[0]


def write_camera_ready_data(data: dict, order: list[str], scale: int) -> None:
    """Data for the figures and tables added in the revision.

    Each source file is written by its own script (11 bootstrap, 12 error
    breakdown, 13 domain profile, 15 seed spread, 16 held-out test); a file
    that does not exist yet is skipped with a warning rather than faked, so a
    figure can never be drawn from numbers that were not computed."""
    names = {key: escape(report.display_name(key)).replace(" ", "~") for key in order}

    # --- F1 with bootstrap intervals, plus any extra seeds ----------------
    boot_path = TABLES_DIR / f"bootstrap_{scale}_seed1.json"
    if boot_path.exists():
        boot = json.loads(boot_path.read_text(encoding="utf-8"))["models"]
        seeds = {}
        for key, run in data["runs"].items():
            if "|seed" in key and not key.endswith("|seed1"):
                seeds.setdefault(run_model(key), {})[key.split("|seed")[1]] = run["raw"]["overall"]["f1"]
        out = FIGURES_DIR / "data_ci.dat"
        with open(out, "w", encoding="utf-8") as f:
            # labx: where the printed F1 value starts, right of the interval
            # and of any seed mark, so the number never covers a mark.
            # f1lab: the printed value, rounded once from the exact F1, so the
            # figure says 0.344 like the text (re-rounding 0.3445 gives 0.345).
            f.write("model f1 lo hi seed2 seed3 labx f1lab\n")
            for key in order:
                b = boot[run_model(key)]
                s = seeds.get(run_model(key), {})
                labx = max([b["f1_ci95"][1]] + [v for v in (s.get("2"), s.get("3")) if v is not None])
                f.write(f"{names[key]} {b['f1']:.4f} {b['f1_ci95'][0]:.4f} {b['f1_ci95'][1]:.4f} "
                        f"{s.get('2', float('nan')):.4f} {s.get('3', float('nan')):.4f} {labx:.4f} "
                        f"{b['f1']:.3f}\n")
        logger.info("wrote %s", out)
    else:
        logger.warning("no %s yet; data_ci.dat not written", boot_path.name)

    # --- error breakdown and three-tier F1 --------------------------------
    err_path = TABLES_DIR / f"error_breakdown_{scale}.json"
    if err_path.exists():
        runs = json.loads(err_path.read_text(encoding="utf-8"))["runs"]
        by_model = {}
        for stem, r in runs.items():
            m = report_filename_model(stem, scale)
            by_model[m] = r
        out = FIGURES_DIR / "data_errors.dat"
        # Segment labels: the share in per cent at the segment's centre (c_*,
        # l_*), only for segments of at least 8%; a 6 pt number does not fit
        # inside a thinner one. "{}" is pgfplots' empty cell.
        segs = ["correct", "broader", "alias", "different", "entity_error"]
        labelled = {"correct": "correct", "different": "different", "entity_error": "entity"}
        with open(out, "w", encoding="utf-8") as f:
            f.write("model correct broader alias different entity "
                    + " ".join(f"c_{n} l_{n}" for n in labelled.values()) + "\n")
            for key in order:
                sh = by_model[run_model(key)]["share_of_predictions"]
                starts, acc = {}, 0.0
                for s in segs:
                    starts[s] = acc
                    acc += sh[s]
                labels = []
                for s, n in labelled.items():
                    text = f"{100 * sh[s]:.0f}" if sh[s] >= 0.08 else "{}"
                    labels.append(f"{starts[s] + sh[s] / 2:.4f} {text}")
                f.write(f"{names[key]} {sh['correct']:.4f} {sh['broader']:.4f} {sh['alias']:.4f} "
                        f"{sh['different']:.4f} {sh['entity_error']:.4f} " + " ".join(labels) + "\n")
        logger.info("wrote %s", out)

        out = FIGURES_DIR / "data_style3.dat"
        with open(out, "w", encoding="utf-8") as f:
            # *_lab: the printed bar values, rounded once from the exact F1.
            f.write("model attr_f1 ent_f1 ent_pair_f1 attr_lab ent_lab pair_lab\n")
            for key in order:
                bs = by_model[run_model(key)]["by_style"]
                vals = (bs["strict"]["attribute"]["f1"], bs["strict"]["entity"]["f1"], bs["pair"]["entity"]["f1"])
                f.write(f"{names[key]} " + " ".join(f"{v:.4f}" for v in vals) + " "
                        + " ".join(f"{v:.2f}" for v in vals) + "\n")
        logger.info("wrote %s", out)

        out = FIGURES_DIR / "table_tiers.tex"
        with open(out, "w", encoding="utf-8") as f:
            f.write("% Generated by scripts/08_make_figures.py from error_breakdown_*.json. Do not edit by hand.\n")
            # The whole tabular, not only its rows: \bottomrule straight after
            # an \input of bare rows raises "Misplaced \noalign".
            f.write(r"\begin{tabular}{@{}lrrr@{}}" "\n" r"\toprule" "\n"
                    r"Model & Strict & Implied & Pair \\" "\n" r"\midrule" "\n")
            for key in order:
                r = by_model[run_model(key)]
                f.write(f"{report.display_name(key)} & {r['f1_strict']['f1']:.3f} & "
                        f"{r['f1_implied']['f1']:.3f} & {r['f1_pair']['f1']:.3f} \\\\\n")
            f.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
        logger.info("wrote %s", out)
    else:
        logger.warning("no %s yet; error figures not written", err_path.name)

    # --- seen vs held-out sentences (script 16) ----------------------------
    held_path = TABLES_DIR / f"heldout_mrebel_{scale}.json"
    if held_path.exists():
        held = json.loads(held_path.read_text(encoding="utf-8"))["systems"]
        out = FIGURES_DIR / "table_heldout.tex"
        with open(out, "w", encoding="utf-8") as f:
            f.write("% Generated by scripts/08_make_figures.py from heldout_mrebel_*.json. Do not edit by hand.\n")
            f.write(r"\begin{tabular}{@{}llrrrr@{}}" "\n" r"\toprule" "\n"
                    r"System & Sentences & P & R & F1 & Gap \\" "\n" r"\midrule" "\n")
            for i, key in enumerate(k for k in ("mrebel", "gemma4_12b", "qwen3.5_9b") if k in held):
                s = held[key]
                if i:
                    f.write(r"\addlinespace[2pt]" "\n")
                for j, (name, label) in enumerate((("seen", "test set"), ("heldout", "held-out"))):
                    r = s[name]
                    first = s["label"] if j == 0 else ""
                    # A typeset minus sign, not a hyphen, for negative gaps.
                    gap = ("$+$" if r["gap"] >= 0 else "$-$") + f"{abs(r['gap']):.3f}"
                    f.write(f"{first} & {label} & {r['precision']:.3f} & {r['recall']:.3f} & "
                            f"{r['f1']:.3f} & {gap} \\\\\n")
            f.write(r"\bottomrule" "\n" r"\end{tabular}" "\n")
        logger.info("wrote %s", out)
    else:
        logger.warning("no %s yet; held-out table not written", held_path.name)

    # --- ontology table grouped by topic ----------------------------------
    dom_path = TABLES_DIR / f"domain_profile_{scale}.json"
    if dom_path.exists():
        dom = json.loads(dom_path.read_text(encoding="utf-8"))
        n = dom["n_triples"]
        out = FIGURES_DIR / "table_ontology.tex"
        with open(out, "w", encoding="utf-8") as f:
            f.write("% Generated by scripts/08_make_figures.py from domain_profile_*.json. Do not edit by hand.\n")
            f.write("% Relations within a group in descending gold count; attribute-style ones in italics.\n")
            f.write(r"\begin{tabularx}{\linewidth}{@{}l r X@{}}" "\n" r"\toprule" "\n"
                    r"Group & \% & Relations (gold triples) \\" "\n" r"\midrule" "\n")
            for group in sorted(dom["topic_groups"], key=lambda g: -dom["by_group"].get(g, 0)):
                rels = sorted(dom["topic_groups"][group], key=lambda r: -dom["by_relation"].get(norm_rel(r), 0))
                cells = []
                for r in rels:
                    c = dom["by_relation"].get(norm_rel(r), 0)
                    # "{}" ends the control word, so a space after the slash
                    # (Wikidata's "position played on team / speciality")
                    # survives instead of being eaten as its terminator.
                    label = r.replace("/", "/\\allowbreak{}")
                    if dom["relation_style"].get(norm_rel(r)) == "attribute":
                        label = f"\\emph{{{label}}}"
                    cells.append(f"{label} ({c:,})".replace(",", "{,}"))
                share = 100 * dom["by_group"][group] / n
                f.write(f"{group} & {share:.1f} & {', '.join(cells)} \\\\\n")
            f.write(r"\bottomrule" "\n" r"\end{tabularx}" "\n")
        logger.info("wrote %s", out)
    else:
        logger.warning("no %s yet; ontology table not written", dom_path.name)


def norm_rel(name: str) -> str:
    # Same normalisation the scorer applies, so "a / b" and "a/b" are one key.
    from src.extraction.scorer import normalize_relation
    return normalize_relation(name)


def report_filename_model(stem: str, scale: int) -> str:
    """'gemma4_12b_1000_seed1' -> 'gemma4_12b'; 'mrebel_1000' -> 'mrebel'."""
    m = import_module("04_score").FILENAME_RE.match(stem + ".jsonl")
    if not m or m.group("scale") != str(scale):
        raise ValueError(f"unexpected run file name {stem!r}")
    return m.group("model")


def write_headline_table(data: dict, order: list[str]) -> None:
    """Emits the paper's main results table as a booktabs tabular.

    Generated for the same reason the figures are: this table and Figure 2 quote
    the same F1 values, and the only way to guarantee they never disagree is for
    both to be written by the same run of this script from the same JSON.

    Only the body rows are emitted, plus the surrounding tabular. The caption
    and label stay in main.tex, where a reader editing the prose can see them.
    """
    out = FIGURES_DIR / "table_headline.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/08_make_figures.py. Do not edit by hand.\n")
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n")
        f.write(
            "Model & P & R & F1 & Conf. & Valid. & $n$ & SH & OH & s & Wh \\\\\n"
            "\\midrule\n"
        )
        for key in order:
            r = data["runs"][key]["raw"]
            o = r["overall"]
            validity = r["output_validity_pct"]
            # mREBEL reports no validity: the JSON-schema concept does not
            # apply to a seq2seq model that was never asked for JSON. An
            # en-dash, not "0.0%", which would read as "always invalid".
            validity_cell = "--" if validity is None else f"{validity:.1f}"
            # `n`, the number of predicted triples, is printed so SH and OH can
            # be read as rates. Without it a reader compares 1,179 invented
            # relations against 579 and concludes the first model is worse,
            # when the first emitted 4.5x as many triples and is in fact far
            # better per triple.
            #
            # The RH column that used to sit beside them is dropped: RH is
            # exactly n * (1 - Conf/100), verified against every row, so it
            # carried no information the conformance column did not already
            # give. Text2KGBench notes the same identity for its own metrics.
            f.write(
                f"{report.display_name(key)} & "
                f"{o['precision']:.3f} & {o['recall']:.3f} & {o['f1']:.3f} & "
                f"{r['ontology_conformance_pct']:.1f} & {validity_cell} & "
                f"{r['n_predicted_triples']} & "
                f"{r['subject_hallucinations']} & {r['object_hallucinations']} & "
                f"{fmt(r.get('latency_median_s'), 1)} & "
                f"{fmt(r.get('gpu_energy_wh_per_1000'), 0)} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info("wrote %s", out)


def fmt(value, digits: int) -> str:
    """Numeric cell, or an en-dash when the run never recorded the quantity."""
    return "--" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    main()
