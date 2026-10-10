# Study 2 results, scale 100

## 1. Headline results

Raw scoring, no Arabic normalisation. Ordered by F1.

| Model | P | R | F1 | Conformance | Validity | SH | RH | OH | Median s | p95 s | Wh/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma4 12B | 0.441 | 0.376 | 0.406 | 100.0% | 100.0% | 8 | 0 | 4 | 11.6 | 20.2 | 242.9 |
| Qwen3.5 9B | 0.461 | 0.272 | 0.342 | 100.0% | 100.0% | 6 | 0 | 10 | 5.3 | 10.0 | 101.9 |
| Llama3.1 8B | 0.378 | 0.310 | 0.341 | 97.0% | 96.0% | 22 | 14 | 46 | 5.7 | 10.2 | 123.2 |
| Aya-Expanse 8B | 0.374 | 0.279 | 0.319 | 75.7% | 99.0% | 9 | 103 | 37 | 6.2 | 12.8 | 135.3 |
| Mistral-Nemo 12B | 0.430 | 0.233 | 0.302 | 94.5% | 98.0% | 28 | 17 | 50 | 4.9 | 11.9 | 102.8 |
| Command-R7B-Arabic | 0.331 | 0.263 | 0.293 | 98.4% | 99.0% | 1 | 7 | 24 | 5.6 | 15.8 | 130.2 |
| Qwen3.5 4B | 0.420 | 0.190 | 0.262 | 100.0% | 100.0% | 5 | 0 | 9 | 2.6 | 4.6 | 47.4 |
| Fanar-1 9B | 0.311 | 0.189 | 0.235 | 93.9% | 96.0% | 65 | 21 | 96 | 9.0 | 23.3 | 218.2 |
| mREBEL (baseline) | 0.826 | 0.125 | 0.217 | 90.7% | n/a | 0 | 8 | 3 | 0.7 | 2.9 | 22.6 |
| Qwen3.5 2B | 0.338 | 0.125 | 0.183 | 95.7% | 95.0% | 12 | 9 | 18 | 1.5 | 2.5 | 28.9 |
| ALLaM 7B | 0.275 | 0.053 | 0.089 | 76.1% | 100.0% | 1 | 26 | 14 | 0.6 | 4.1 | 29.0 |
| SILMA 9B | 0.467 | 0.025 | 0.047 | 100.0% | 84.0% | 0 | 0 | 1 | 0.9 | 3.5 | 27.3 |
| Qwen3.5 0.8B | 0.167 | 0.005 | 0.010 | 77.8% | 86.0% | 3 | 4 | 7 | 0.9 | 2.5 | 21.4 |
| AceGPT 13B | 0.029 | 0.004 | 0.006 | 23.2% | 39.0% | 26 | 53 | 46 | 6.6 | 31.3 | 220.1 |

## 2. Attribute-style vs entity-style triples

Attribute-style triples attach a literal value to the subject (the four date-valued
relations: date of birth, date of death, point in time, inception). Entity-style triples
link two named things (the other sixteen). Split derived from the gold objects' own type
tags, not assigned by hand; see `derive_attribute_relations`.

| Model | Attr P | Attr R | Attr F1 | Ent P | Ent R | Ent F1 | F1 gap (attr - ent) |
|---|---|---|---|---|---|---|---|
| Gemma4 12B | 0.500 | 0.608 | 0.549 | 0.421 | 0.325 | 0.367 | +0.182 |
| Qwen3.5 9B | 0.557 | 0.529 | 0.543 | 0.422 | 0.215 | 0.285 | +0.258 |
| Llama3.1 8B | 0.615 | 0.549 | 0.580 | 0.320 | 0.258 | 0.286 | +0.295 |
| Aya-Expanse 8B | 0.594 | 0.588 | 0.591 | 0.304 | 0.211 | 0.249 | +0.342 |
| Mistral-Nemo 12B | 0.512 | 0.431 | 0.468 | 0.398 | 0.189 | 0.257 | +0.212 |
| Command-R7B-Arabic | 0.464 | 0.441 | 0.452 | 0.295 | 0.224 | 0.254 | +0.198 |
| Qwen3.5 4B | 0.560 | 0.412 | 0.475 | 0.363 | 0.142 | 0.204 | +0.271 |
| Fanar-1 9B | 0.466 | 0.471 | 0.468 | 0.245 | 0.127 | 0.167 | +0.301 |
| mREBEL (baseline) | 0.950 | 0.186 | 0.311 | 0.788 | 0.112 | 0.196 | +0.116 |
| Qwen3.5 2B | 0.432 | 0.186 | 0.260 | 0.313 | 0.112 | 0.165 | +0.095 |
| ALLaM 7B | 0.611 | 0.216 | 0.319 | 0.110 | 0.017 | 0.030 | +0.289 |
| SILMA 9B | 0.444 | 0.039 | 0.072 | 0.476 | 0.022 | 0.041 | +0.031 |
| Qwen3.5 0.8B | 0.000 | 0.000 | 0.000 | 0.167 | 0.006 | 0.012 | -0.012 |
| AceGPT 13B | 0.000 | 0.000 | 0.000 | 0.029 | 0.004 | 0.007 | -0.007 |

## 3. Effect of Arabic normalisation

Normalisation strips diacritics and tatweel and unifies alef-hamza and taa-marbuta forms.
A large positive gap means the model had the fact right and the spelling variant wrong.

| Model | F1 raw | F1 normalised | Gap |
|---|---|---|---|
| Gemma4 12B | 0.406 | 0.406 | +0.000 |
| Qwen3.5 9B | 0.342 | 0.342 | +0.000 |
| Llama3.1 8B | 0.341 | 0.341 | +0.000 |
| Aya-Expanse 8B | 0.319 | 0.319 | +0.000 |
| Mistral-Nemo 12B | 0.302 | 0.302 | +0.000 |
| Command-R7B-Arabic | 0.293 | 0.293 | +0.000 |
| Qwen3.5 4B | 0.262 | 0.262 | +0.000 |
| Fanar-1 9B | 0.235 | 0.235 | +0.000 |
| mREBEL (baseline) | 0.217 | 0.217 | +0.000 |
| Qwen3.5 2B | 0.183 | 0.183 | +0.000 |
| ALLaM 7B | 0.089 | 0.089 | +0.000 |
| SILMA 9B | 0.047 | 0.047 | +0.000 |
| Qwen3.5 0.8B | 0.010 | 0.010 | +0.000 |
| AceGPT 13B | 0.006 | 0.006 | +0.000 |

## 4. Per-relation breakdown, strongest model

**Gemma4 12B**, ordered by gold frequency (tp+fn).

| Relation | P | R | F1 | Gold (tp+fn) |
|---|---|---|---|---|
| country | 0.000 | 0.000 | 0.000 | 136 |
| country of citizenship | 0.838 | 0.479 | 0.610 | 119 |
| occupation | 0.610 | 0.839 | 0.707 | 56 |
| date of birth | 0.639 | 0.929 | 0.757 | 42 |
| place of birth | 0.208 | 0.278 | 0.238 | 36 |
| point in time | 0.034 | 0.029 | 0.032 | 34 |
| date of death | 0.656 | 0.955 | 0.778 | 22 |
| member of sports team | 0.778 | 0.955 | 0.857 | 22 |
| sport | 0.000 | 0.000 | 0.000 | 18 |
| diplomatic relation | 0.000 | 0.000 | 0.000 | 15 |
| instance of | 0.081 | 0.231 | 0.120 | 13 |
| located in the administrative territorial entity | 0.167 | 0.667 | 0.267 | 12 |
| place of death | 0.000 | 0.000 | 0.000 | 12 |
| position played on team/speciality | 0.125 | 0.200 | 0.154 | 10 |
| parent taxon | 0.000 | 0.000 | 0.000 | 6 |
| inception | 0.500 | 0.250 | 0.333 | 4 |
| member of political party | 0.750 | 0.750 | 0.750 | 4 |
| subclass of | 0.000 | 0.000 | 0.000 | 3 |
| part of | 0.000 | 0.000 | 0.000 | 2 |
| participant | 0.000 | 0.000 | 0.000 | 1 |

## 5. Cross-seed F1 spread

Most models ran one seed (for time, 2026-08-13); for them the stdev below
is 0.000 by construction, an absence of evidence rather than a stability result.
0 models () were re-run with further seeds for the
revision; their rows report a real spread over their runs. The headline tables use
seed 1 only.

| Model | F1 mean | F1 stdev | seeds |
|---|---|---|---|
| Gemma4 12B | 0.406 | 0.000 | 1 |
| Qwen3.5 9B | 0.342 | 0.000 | 1 |
| Llama3.1 8B | 0.341 | 0.000 | 1 |
| Aya-Expanse 8B | 0.319 | 0.000 | 1 |
| Mistral-Nemo 12B | 0.302 | 0.000 | 1 |
| Command-R7B-Arabic | 0.293 | 0.000 | 1 |
| Qwen3.5 4B | 0.262 | 0.000 | 1 |
| Fanar-1 9B | 0.235 | 0.000 | 1 |
| mREBEL (baseline) | 0.217 | 0.000 | 1 |
| Qwen3.5 2B | 0.183 | 0.000 | 1 |
| ALLaM 7B | 0.089 | 0.000 | 1 |
| SILMA 9B | 0.047 | 0.000 | 1 |
| Qwen3.5 0.8B | 0.010 | 0.000 | 1 |
| AceGPT 13B | 0.006 | 0.000 | 1 |
