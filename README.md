# Ontology-Guided Knowledge Graph Extraction from Arabic Text: A Benchmark of Ten Open-Weight Language Models

Code, data and model outputs for the ICESST 2026 paper of the same title, by
Abdelhakim EL Ghayoubi, Achraf Chahboune, Imad Sassi and Said Bahassine.

Ten open-weight chat models, run locally through Ollama, and the trained
extractor mREBEL extract (subject, relation, object) triples from 1,000 Arabic
sentences of ArSRED, under a closed ontology of the 20 most frequent Wikidata
relations. Predictions are scored against the gold triples with strict micro
precision, recall and F1, ontology conformance and hallucination counts, in the
style of Text2KGBench. The repository also holds the analyses reported in the
paper: bootstrap intervals, repeated decoding seeds, an error breakdown with
three matching rules, the topic profile of the data, and a check of mREBEL on
held-out sentences.

## Repository layout

```
configs/            llm_models.yaml (Ollama models and generation settings)
                    ontology.yaml (the 20 relations with Arabic glosses; built by script 01)
src/                parsing, scoring and model-calling code used by the scripts
scripts/            the pipeline, numbered in run order (see below)
data/processed/splits/
                    sanity_100.jsonl       100-sentence pilot set
                    main_1000.jsonl        the paper's test set (1,000 sentences, 5,772 gold triples)
                    confirm_10000.jsonl    a larger tier, not used in the paper
                    heldout_sredfm_1000.jsonl  held-out set from SRED-FM validation/test (2,206 gold triples)
data/external/      wikidata_properties_20261008.json (Wikidata subproperty cache used by script 12)
results/extraction/ raw model outputs, one JSON line per sentence
                    100/ pilot, 1000/ test set (seed 1 for all, seeds 2 and 3 for four models), heldout/
results/tables/     every number in the paper, as JSON (plus two markdown reports)
figures/            figure data written by script 08, the figure sources, and render_figures.tex
```

## Setup

- Python 3.14 and `pip install -r requirements.txt`.
- [Ollama](https://ollama.com) for the chat models (version 0.32.6 was used),
  with these models pulled: `iKhalid/ALLaM:7b`, `qwen3.5:2b`, `qwen3.5:4b`,
  `qwen3.5:9b`, `mistral-nemo:12b`, `command-r7b-arabic:latest`,
  `llama3.1:8b`, `aya-expanse:8b`,
  `hf.co/mradermacher/Fanar-1-9B-Instruct-GGUF:Q4_K_M`, `gemma4:12b`.
  Generation settings are in `configs/llm_models.yaml` (temperature 0.2,
  512 output tokens, 8,192-token context).
- mREBEL (`Babelscape/mrebel-large`) runs through `transformers`.
- The paper's runs used one 16 GB NVIDIA Quadro RTX 5000; one pass of the ten
  chat models over the test set took 16.6 GPU-hours.

## Data

Included: the four splits and the Wikidata cache listed above, and all model
outputs, so every table can be reproduced without a GPU.

Not included, because of their size:

- **ArSRED** (`U4RASD/ArSRED` on Hugging Face). Script 01 downloads it with
  the `datasets` library.
- **`data/processed/closed_sentences.jsonl`** (413 MB): all 334,057 ArSRED
  sentences whose gold triples use only the 20 relations. Script 01 rebuilds
  it, and the splits with it, from a fixed random seed.
- **SRED-FM, Arabic part** (`Babelscape/SREDFM`), needed only by scripts 14
  and 16 (`--build`). Download the parquet files from
  <https://huggingface.co/datasets/Babelscape/SREDFM/tree/refs%2Fconvert%2Fparquet/ar>
  into `data/raw/sredfm_ar/` (`train0.parquet`, `train1.parquet`,
  `validation.parquet`, `test.parquet`).

## Reproducing the results

### From the included outputs (no GPU)

Run from the repository root:

```
python scripts/04_score.py --scale 1000          # scores every run -> results/tables/scale_1000.json
python scripts/05_report.py --scale 1000         # markdown report of the headline tables
python scripts/11_bootstrap_ci.py --scale 1000   # 95% intervals and paired tests
python scripts/12_error_breakdown.py --scale 1000  # error types, three matching rules, style split
python scripts/13_domain_profile.py --scale 1000   # topic and entity-type profile (Table 1)
python scripts/15_seed_spread.py --scale 1000    # F1 across decoding seeds
python scripts/16_heldout_mrebel.py --score      # test set against held-out sentences
python scripts/08_make_figures.py --scale 1000   # figure data and generated tables in figures/
```

These regenerate the files in `results/tables/` exactly. Use `--scale 100`
with scripts 04 and 05 for the pilot. To draw the figures:
`xelatex -output-directory=figures figures/render_figures.tex`.

### Full rerun

```
python scripts/01_build_ontology_and_testset.py          # ontology, closed sentences, splits
python scripts/02_run_mrebel_baseline.py --scale 1000    # mREBEL
python scripts/03_run_llm.py --scale 1000                # the ten chat models, seed 1
python scripts/03_run_llm.py --scale 1000 --models qwen3.5:9b,llama3.1:8b,aya-expanse:8b,command-r7b-arabic:latest --seeds 2
python scripts/03_run_llm.py --scale 1000 --models qwen3.5:9b,llama3.1:8b,aya-expanse:8b,command-r7b-arabic:latest --seeds 3
python scripts/16_heldout_mrebel.py --build --run        # held-out set and mREBEL on it
python scripts/03_run_llm.py --split-path data/processed/splits/heldout_sredfm_1000.jsonl --tier heldout --models gemma4:12b,qwen3.5:9b --seeds 1
```

then the scoring steps above. Runs resume where they stopped: every output
file is written one sentence at a time. Decoding at temperature 0.2 is
seeded, but GPU and library versions can still move individual outputs.

Other scripts: `06_diagnose_validity.py` (output-format failures),
`07_diagnose_normalisation.py` (why Arabic normalisation changes no score),
`09_diagnose_relation_confusion.py`, `10_reparse.py` (re-parses stored raw
outputs), `14_check_training_overlap.py --sredfm-dir data/raw/sredfm_ar`
(finds the test sentences in SRED-FM's training split) and
`19_corpus_stats.py` (recomputes the corpus figures in Sect. 3.1).

## Where each result comes from

| Paper | File in `results/tables/` |
|---|---|
| Table 3 (main results) | `scale_1000.json`, `scale_1000_report.md` |
| Fig. 2a (intervals, seeds) | `bootstrap_1000_seed1.json`, `seed_spread_1000.json` |
| Fig. 2b, Fig. 3, Table 4a | `error_breakdown_1000.json` |
| Table 1 (topics) | `domain_profile_1000.json` |
| Table 4b, Sect. 4.4 (held-out) | `heldout_mrebel_1000.json`, `training_overlap_1000.json` |
| Sect. 3.1 (corpus figures) | `corpus_stats_arsred.json` |
| Sect. 3.2 (pilot) | `scale_100.json`, `scale_100_report.md` |

## Licences

ArSRED and SRED-FM are released under CC BY-SA 4.0; the data subsets in this
repository are derived from them and keep that licence. mREBEL is released
under CC BY-NC-SA 4.0 and is not included here, only its outputs. The chat
models' outputs are shared for research use; each model's own licence applies.

## Citation

```
@inproceedings{elghayoubi2026arabickg,
  title     = {Ontology-Guided Knowledge Graph Extraction from Arabic Text:
               A Benchmark of Ten Open-Weight Language Models},
  author    = {EL Ghayoubi, Abdelhakim and Chahboune, Achraf and Sassi, Imad
               and Bahassine, Said},
  booktitle = {Proceedings of the International Conference ICESST 2026},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```
