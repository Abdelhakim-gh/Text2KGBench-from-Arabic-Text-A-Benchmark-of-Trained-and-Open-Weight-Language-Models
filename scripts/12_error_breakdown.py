"""Splits every prediction into correct, a relation error on a correctly
extracted entity pair, or an entity error, and scores the benchmark three ways:
strict, with substitutions implied by Wikidata accepted, and with the relation ignored.

Added in the revision after peer review. The submitted
paper reported that 16.9% of Gemma4's wrong predictions carry the right subject
and object under a different relation, and called all of them "a different but
defensible Wikidata property". Script 09 counts those cases but does not say
which of them are defensible. Reading the pair names shows they are not all
alike: `country` -> `located in the administrative territorial entity` names the
same fact at a different granularity, while `country of citizenship` ->
`place of birth` asserts a fact the sentence does not state.

The rule for "defensible" is taken from Wikidata, not decided here. The paper's
own argument against a lenient scorer is that a scorer ruling on which
properties are interchangeable measures its own ruling. So the only
substitutions accepted are ones Wikidata's own property hierarchy implies:
the predicted property must be an ancestor of the gold property under
`subproperty of` (P1647), closed transitively within the 20. Within the
ontology that hierarchy is

  country of citizenship (P27) < country (P17)
      < located in the administrative territorial entity (P131) < part of (P361)
  parent taxon (P171) < subclass of (P279)
  date of birth (P569) < inception (P571)

so gold `country` answered with `located in the administrative territorial
entity` is accepted: the gold statement implies the predicted one, which is
true but less specific. The reverse direction is not implied and not accepted.

`related property` (P1659) was considered and rejected as a rule. Wikidata
uses it loosely: it links `place of birth` with `date of birth`, whose objects
are a place and a date. Pairs it links are counted (`different_related`) so
the paper can say how many there are, but they are not accepted.

A right-pair prediction then falls into exactly one class:

  alias      the predicted name is a Wikidata label or alias (English or
             Arabic) of the gold property itself: the right relation under a
             name the prompt did not offer, e.g. `birth place`;
  broader    the predicted name is (a label or alias of) an ancestor of the
             gold property in the hierarchy above;
  different  anything else, e.g. `place of birth` for `country of
             citizenship`, which asserts a fact the sentence need not state.

Every other unmatched prediction is an entity error: no gold triple in the
sentence has that subject and object.

Matching follows scripts 04 and 09 exactly (raw spans, fuzzy threshold 86,
greedy one-to-one, exact matches claimed first), and the script checks its
matched and right-pair counts against relation_confusion_<N>.json before
reporting anything.

Three F1 values per run:
  strict   the headline score;
  implied  `alias` and `broader` predictions counted as correct;
  pair     every right-pair prediction counted as correct (relation ignored),
           an upper bound on what any relation-matching rule could give.

Writes:
  data/external/wikidata_properties_<date>.json   (cache of the property pages)
  results/tables/error_breakdown_<N>.json
  results/logs/12_error_breakdown_<N>.log

Usage:
    python scripts/12_error_breakdown.py --scale 1000
    python scripts/12_error_breakdown.py --scale 1000 --refresh   # refetch Wikidata
"""

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from src.extraction.ontology import load_ontology  # noqa: E402
from src.extraction.scorer import (  # noqa: E402
    DEFAULT_FUZZY_THRESHOLD,
    derive_attribute_relations,
    _collapse_relation,
    _span_equal,
    normalize_arabic,
    normalize_relation,
    triples_match,
)
from src.extraction.triple import Triple  # noqa: E402

score04 = import_module("04_score")
report = import_module("05_report")

RESULTS_DIR = ROOT / "results" / "extraction"
TABLES_DIR = ROOT / "results" / "tables"
LOGS_DIR = ROOT / "results" / "logs"
EXTERNAL_DIR = ROOT / "data" / "external"
ONTOLOGY_PATH = ROOT / "configs" / "ontology.yaml"

# The 20 ontology relations and their Wikidata properties. ArSRED's relation
# names are Wikidata English labels (it derives from SRED-FM, which is built
# on Wikidata), so each one maps to exactly one property. The script checks
# every label below against the live property page and stops on a mismatch.
PROPERTY_IDS = {
    "country": "P17",
    "country of citizenship": "P27",
    "occupation": "P106",
    "date of birth": "P569",
    "point in time": "P585",
    "place of birth": "P19",
    "member of sports team": "P54",
    "instance of": "P31",
    "located in the administrative territorial entity": "P131",
    "sport": "P641",
    "date of death": "P570",
    "diplomatic relation": "P530",
    "place of death": "P20",
    "subclass of": "P279",
    "position played on team / speciality": "P413",
    "part of": "P361",
    "parent taxon": "P171",
    "member of political party": "P102",
    "participant": "P710",
    "inception": "P571",
}

USER_AGENT = "Study2-ICESST-revision/1.0 (abdelhakim.elghayoubi-etu@etu.univh2c.ma)"

logger = logging.getLogger(__name__)


# --- Wikidata -----------------------------------------------------------------

def fetch_properties(pids: list[str]) -> dict:
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
        "action": "wbgetentities", "ids": "|".join(pids), "props": "labels|aliases|claims",
        "languages": "en|ar", "format": "json",
    })
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.load(resp)
    out = {}
    for pid, e in raw["entities"].items():
        def ids(prop):
            return sorted({c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                           for c in e.get("claims", {}).get(prop, [])} - {None})
        out[pid] = {
            "label_en": e["labels"].get("en", {}).get("value"),
            "label_ar": e["labels"].get("ar", {}).get("value"),
            "aliases_en": [a["value"] for a in e.get("aliases", {}).get("en", [])],
            "aliases_ar": [a["value"] for a in e.get("aliases", {}).get("ar", [])],
            "subproperty_of": ids("P1647"),
            "related_property": ids("P1659"),
        }
    return out


def load_properties(refresh: bool) -> tuple[dict, Path]:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cached = sorted(EXTERNAL_DIR.glob("wikidata_properties_*.json"))
    if cached and not refresh:
        path = cached[-1]
        logger.info("Wikidata property pages: using cache %s", path.name)
        return json.loads(path.read_text(encoding="utf-8")), path
    props = fetch_properties(sorted(set(PROPERTY_IDS.values())))
    path = EXTERNAL_DIR / f"wikidata_properties_{date.today():%Y%m%d}.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("Wikidata property pages: fetched %d, cached to %s", len(props), path.name)
    return props, path


def build_ancestors(props: dict) -> dict[str, set[str]]:
    """For each of the 20 properties, its ancestors among the 20 under the
    transitive closure of Wikidata's `subproperty of`."""
    in_set = set(PROPERTY_IDS.values())
    parents = {pid: {p for p in props[pid]["subproperty_of"] if p in in_set} for pid in in_set}
    ancestors = {}
    for pid in in_set:
        seen, stack = set(), list(parents[pid])
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(parents[p])
        ancestors[pid] = seen - {pid}
    return ancestors


def build_related(props: dict, ancestors: dict[str, set[str]]) -> set[frozenset[str]]:
    """Unordered pairs Wikidata connects in any way: subproperty in either
    direction, or a direct `related property` link. Reported, not accepted."""
    in_set = set(PROPERTY_IDS.values())
    related = {frozenset((pid, anc)) for pid, ancs in ancestors.items() for anc in ancs}
    for pid in in_set:
        for rel in props[pid]["related_property"]:
            if rel in in_set and rel != pid:
                related.add(frozenset((pid, rel)))
    return related


def build_name_index(props: dict) -> dict[str, str]:
    """Maps every normalised name a model could use for one of the 20
    properties onto its property id: the canonical name the prompt offered,
    the prompt's own Arabic gloss (via normalize_relation), and every Wikidata
    label and alias in English and Arabic."""
    index: dict[str, str] = {}
    for name, pid in PROPERTY_IDS.items():
        index[normalize_relation(name)] = pid
    for pid, p in props.items():
        for n in [p["label_en"], p["label_ar"], *p["aliases_en"], *p["aliases_ar"]]:
            if not n:
                continue
            for key in (_collapse_relation(n), normalize_arabic(_collapse_relation(n))):
                # The canonical names and glosses win any clash: a Wikidata alias
                # shared by two properties must not move a canonical name.
                index.setdefault(key, pid)
    return index


def name_to_pid(name: str, index: dict[str, str]) -> str | None:
    norm = normalize_relation(name)
    if norm in index:
        return index[norm]
    collapsed = _collapse_relation(name)
    return index.get(collapsed) or index.get(normalize_arabic(collapsed))


# --- scoring --------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def classify_file(path: Path, gold: dict, allowed_norm: set[str], index: dict[str, str],
                  ancestors: dict[str, set[str]], related: set[frozenset[str]],
                  attribute: frozenset[str]) -> dict:
    counts = Counter()
    # TP/FP/FN per triple style under each scoring tier. Styles follow script
    # 04: a matched or missed triple takes its gold relation's style, a false
    # positive its predicted relation's style. A right-pair prediction accepted
    # by a tier moves from that tier's FP (its predicted style) to its TP, and
    # removes one FN (its gold style).
    style_counts = {tier: {st: Counter() for st in ("attribute", "entity")}
                    for tier in ("strict", "implied", "pair")}

    def style(rel: str) -> str:
        return "attribute" if normalize_relation(rel) in attribute else "entity"
    pairs_by_class: dict[str, Counter] = defaultdict(Counter)
    n_gold = 0
    gold_by_relation = Counter()
    gold_fate: dict[str, Counter] = defaultdict(Counter)  # gold relation -> what happened to it
    examples: dict[str, list] = defaultdict(list)

    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["item_id"] not in gold:
            continue
        text, gold_triples = gold[row["item_id"]]
        n_gold += len(gold_triples)
        for g in gold_triples:
            gold_by_relation[normalize_relation(g.relation)] += 1
        predicted = [] if "error" in row else [Triple(**t) for t in row.get("predicted_triples", [])]

        remaining = list(gold_triples)
        unmatched = []
        for p in predicted:
            hit = next((i for i, g in enumerate(remaining)
                        if triples_match(p, g, False, DEFAULT_FUZZY_THRESHOLD)), None)
            if hit is None:
                unmatched.append(p)
            else:
                g = remaining.pop(hit)
                counts["correct"] += 1
                gold_fate[normalize_relation(g.relation)]["matched"] += 1
                for tier in style_counts:
                    style_counts[tier][style(g.relation)]["tp"] += 1

        for tier in style_counts:
            for p in unmatched:
                style_counts[tier][style(p.relation)]["fp"] += 1
            for g in remaining:
                style_counts[tier][style(g.relation)]["fn"] += 1

        for p in unmatched:
            hit = next((i for i, g in enumerate(remaining)
                        if _span_equal(p.subject, g.subject, False, DEFAULT_FUZZY_THRESHOLD)
                        and _span_equal(p.object, g.object, False, DEFAULT_FUZZY_THRESHOLD)), None)
            if hit is None:
                counts["entity_error"] += 1
                continue
            g = remaining.pop(hit)
            gold_rel, pred_rel = normalize_relation(g.relation), normalize_relation(p.relation)
            gold_pid, pred_pid = PROPERTY_IDS.get(gold_rel), name_to_pid(p.relation, index)
            if pred_pid is not None and pred_pid == gold_pid:
                cls = "alias"
            elif pred_pid is not None and pred_pid in ancestors.get(gold_pid, set()):
                cls = "broader"
            else:
                cls = "different"
                if pred_pid is not None and frozenset((gold_pid, pred_pid)) in related:
                    counts["different_related"] += 1
            counts[cls] += 1
            counts["right_pair"] += 1
            for tier in ("implied", "pair") if cls in ("alias", "broader") else ("pair",):
                style_counts[tier][style(g.relation)]["tp"] += 1
                style_counts[tier][style(g.relation)]["fn"] -= 1
                style_counts[tier][style(p.relation)]["fp"] -= 1
            if pred_rel not in allowed_norm:
                counts["right_pair_off_list"] += 1
            pairs_by_class[cls][(gold_rel, pred_rel)] += 1
            gold_fate[gold_rel][f"right_pair_{cls}"] += 1
            if len(examples[cls]) < 5:
                examples[cls].append({"item_id": row["item_id"], "gold": [g.subject, g.relation, g.object],
                                      "predicted": [p.subject, p.relation, p.object]})

        for g in remaining:
            gold_fate[normalize_relation(g.relation)]["missed"] += 1

    n_pred = counts["correct"] + counts["right_pair"] + counts["entity_error"]

    def f1(tp: int) -> dict:
        fp, fn = n_pred - tp, n_gold - tp
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gold if n_gold else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}

    tp = counts["correct"]
    errors = n_pred - tp
    return {
        "n_predicted": n_pred,
        "n_gold": n_gold,
        "counts": dict(counts),
        "share_of_predictions": {k: counts[k] / n_pred if n_pred else 0.0
                                 for k in ("correct", "alias", "broader", "different", "entity_error")},
        "share_of_errors": {k: counts[k] / errors if errors else 0.0
                            for k in ("alias", "broader", "different", "different_related", "entity_error", "right_pair")},
        "f1_strict": f1(tp),
        "f1_implied": f1(tp + counts["alias"] + counts["broader"]),
        "f1_pair": f1(tp + counts["right_pair"]),
        "by_style": {tier: {st: _prf(c["tp"], c["fp"], c["fn"]) for st, c in styles.items()}
                     for tier, styles in style_counts.items()},
        "top_pairs": {cls: [{"gold": g, "predicted": p, "n": n} for (g, p), n in c.most_common(12)]
                      for cls, c in pairs_by_class.items()},
        "gold_fate": {rel: dict(c) for rel, c in gold_fate.items()},
        "gold_by_relation": dict(gold_by_relation),
        "examples": dict(examples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1000, choices=[100, 1000, 10000])
    parser.add_argument("--refresh", action="store_true", help="refetch the Wikidata property pages")
    parser.add_argument("--seed", type=int, default=1, help="decoding seed to read for the chat models")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.scale}" if args.seed == 1 else f"{args.scale}_seed{args.seed}"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / f"12_error_breakdown_{suffix}.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    ontology = load_ontology(ONTOLOGY_PATH)
    allowed_norm = {normalize_relation(r) for r in ontology.relation_names}
    assert allowed_norm == {normalize_relation(n) for n in PROPERTY_IDS}, "PROPERTY_IDS and the ontology disagree"

    props, props_path = load_properties(args.refresh)
    for name, pid in PROPERTY_IDS.items():
        label = props[pid]["label_en"]
        if normalize_relation(label) != normalize_relation(name):
            raise SystemExit(f"{pid}: Wikidata label {label!r} is not the ontology name {name!r}")
    logger.info("all 20 ontology names equal their Wikidata English labels")

    ancestors = build_ancestors(props)
    related = build_related(props, ancestors)
    pid_name = {pid: name for name, pid in PROPERTY_IDS.items()}
    logger.info("accepted substitutions, gold -> predicted ancestor under Wikidata subproperty-of:")
    for pid in sorted(ancestors, key=lambda x: int(x[1:])):
        for anc in sorted(ancestors[pid], key=lambda x: int(x[1:])):
            logger.info("  %-5s %-48s -> %-5s %s", pid, pid_name[pid], anc, pid_name[anc])
    logger.info("pairs Wikidata relates in any way (reported, not accepted): %d", len(related))
    index = build_name_index(props)

    gold = score04.load_gold(args.scale)
    attribute = derive_attribute_relations([t for _, ts in gold.values() for t in ts])
    with open(TABLES_DIR / f"scale_{args.scale}.json", encoding="utf-8") as f:
        scored = json.load(f)["runs"]
    with open(TABLES_DIR / f"relation_confusion_{args.scale}.json", encoding="utf-8") as f:
        confusion_ref = json.load(f)

    out = {"scale": args.scale, "wikidata_cache": props_path.name,
           "accepted_substitutions": {pid_name[p]: sorted(pid_name[a] for a in ancs) for p, ancs in ancestors.items() if ancs},
           "related_pairs": sorted(sorted(pid_name[x] for x in pr) for pr in related), "runs": {}}
    for path in sorted((RESULTS_DIR / str(args.scale)).glob(f"*_{args.scale}*.jsonl")):
        m = score04.FILENAME_RE.match(path.name)
        if not m or m.group("scale") != str(args.scale):
            continue
        # Seed 1 and mREBEL only by default: the extra seeds are run for three
        # models and may still be mid-write. --seed picks another seed.
        if m.group("seed") is not None and int(m.group("seed")) != args.seed:
            continue
        res = classify_file(path, gold, allowed_norm, index, ancestors, related, attribute)
        ref = confusion_ref.get(path.stem)
        if ref is not None:
            got = (res["counts"].get("correct", 0), res["counts"].get("right_pair", 0))
            want = (ref["n_matched"], ref["n_relation_confusions"])
            if got != want:
                raise SystemExit(f"{path.stem}: (matched, right-pair) {got} differ from script 09's {want}")
        # The strict style split must equal script 04's, or the other two
        # tiers are not comparable with Figure 2.
        key = f"{m.group('model')}|seed{m.group('seed')}" if m.group("seed") else m.group("model")
        if key in scored:
            for st in ("attribute", "entity"):
                want = scored[key]["raw"]["by_style"][st]
                got = res["by_style"]["strict"][st]
                if (got["tp"], got["fp"], got["fn"]) != (want["tp"], want["fp"], want["fn"]):
                    raise SystemExit(f"{path.stem} {st}: strict style counts differ from scale table")
        out["runs"][path.stem] = res
        bs = res["by_style"]
        logger.info("    attr-ent gap: strict %+.3f  implied %+.3f  pair %+.3f   (ent F1 %.3f -> %.3f -> %.3f)",
                    bs["strict"]["attribute"]["f1"] - bs["strict"]["entity"]["f1"],
                    bs["implied"]["attribute"]["f1"] - bs["implied"]["entity"]["f1"],
                    bs["pair"]["attribute"]["f1"] - bs["pair"]["entity"]["f1"],
                    bs["strict"]["entity"]["f1"], bs["implied"]["entity"]["f1"], bs["pair"]["entity"]["f1"])
        sh = res["share_of_errors"]
        logger.info("%-44s strict %.3f  implied %.3f  pair %.3f | errors: entity %.1f%%  alias %.1f%%  "
                    "broader %.1f%%  different %.1f%% (related %.1f%%)",
                    path.stem[:44], res["f1_strict"]["f1"], res["f1_implied"]["f1"], res["f1_pair"]["f1"],
                    100 * sh["entity_error"], 100 * sh["alias"], 100 * sh["broader"], 100 * sh["different"],
                    100 * sh["different_related"])
    logger.info("matched and right-pair counts reproduce relation_confusion_%d.json for every run it covers",
                args.scale)

    out_path = TABLES_DIR / f"error_breakdown_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
