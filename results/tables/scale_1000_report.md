# Study 2 results, scale 1000

## 1. Headline results

Raw scoring, no Arabic normalisation. Ordered by F1.

| Model | P | R | F1 | Conformance | Validity | SH | RH | OH | Median s | p95 s | Wh/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma4 12B | 0.467 | 0.388 | 0.424 | 99.9% | 98.7% | 29 | 3 | 56 | 11.6 | 24.0 | 249.2 |
| Qwen3.5 9B | 0.456 | 0.277 | 0.344 | 100.0% | 97.7% | 43 | 1 | 90 | 5.2 | 13.8 | 109.5 |
| Llama3.1 8B | 0.357 | 0.291 | 0.321 | 93.9% | 96.1% | 167 | 285 | 603 | 5.3 | 14.3 | 121.4 |
| Aya-Expanse 8B | 0.368 | 0.263 | 0.307 | 79.8% | 99.4% | 93 | 835 | 395 | 6.5 | 13.5 | 139.2 |
| Command-R7B-Arabic | 0.344 | 0.267 | 0.300 | 97.2% | 97.6% | 31 | 124 | 344 | 5.6 | 14.9 | 132.8 |
| Mistral-Nemo 12B | 0.382 | 0.184 | 0.248 | 98.0% | 98.5% | 580 | 55 | 600 | 4.6 | 12.4 | 97.9 |
| Qwen3.5 4B | 0.417 | 0.175 | 0.246 | 99.7% | 100.0% | 93 | 7 | 75 | 2.5 | 4.8 | 45.9 |
| mREBEL (baseline) | 0.727 | 0.141 | 0.236 | 94.6% | n/a | 1 | 61 | 34 | 0.9 | 3.5 | 26.1 |
| Fanar-1 9B | 0.320 | 0.184 | 0.233 | 94.8% | 90.6% | 726 | 173 | 1035 | 9.3 | 23.5 | 228.8 |
| Qwen3.5 2B | 0.311 | 0.108 | 0.160 | 97.3% | 96.3% | 152 | 54 | 247 | 1.4 | 2.6 | 28.7 |
| ALLaM 7B | 0.260 | 0.041 | 0.071 | 72.1% | 100.0% | 13 | 255 | 130 | 0.6 | 4.7 | 27.1 |

## 2. Attribute-style vs entity-style triples

Attribute-style triples attach a literal value to the subject (the four date-valued
relations: date of birth, date of death, point in time, inception). Entity-style triples
link two named things (the other sixteen). Split derived from the gold objects' own type
tags, not assigned by hand; see `derive_attribute_relations`.

| Model | Attr P | Attr R | Attr F1 | Ent P | Ent R | Ent F1 | F1 gap (attr - ent) |
|---|---|---|---|---|---|---|---|
| Gemma4 12B | 0.493 | 0.565 | 0.527 | 0.459 | 0.351 | 0.398 | +0.129 |
| Qwen3.5 9B | 0.523 | 0.477 | 0.499 | 0.433 | 0.235 | 0.304 | +0.195 |
| Llama3.1 8B | 0.587 | 0.471 | 0.522 | 0.310 | 0.254 | 0.279 | +0.244 |
| Aya-Expanse 8B | 0.597 | 0.544 | 0.569 | 0.303 | 0.205 | 0.244 | +0.325 |
| Command-R7B-Arabic | 0.391 | 0.337 | 0.362 | 0.333 | 0.252 | 0.287 | +0.075 |
| Mistral-Nemo 12B | 0.491 | 0.362 | 0.417 | 0.343 | 0.147 | 0.206 | +0.211 |
| Qwen3.5 4B | 0.572 | 0.382 | 0.458 | 0.358 | 0.131 | 0.192 | +0.266 |
| mREBEL (baseline) | 0.761 | 0.153 | 0.255 | 0.720 | 0.139 | 0.233 | +0.022 |
| Fanar-1 9B | 0.452 | 0.420 | 0.435 | 0.269 | 0.134 | 0.179 | +0.257 |
| Qwen3.5 2B | 0.412 | 0.122 | 0.188 | 0.293 | 0.105 | 0.155 | +0.034 |
| ALLaM 7B | 0.540 | 0.135 | 0.216 | 0.155 | 0.022 | 0.038 | +0.178 |

## 3. Effect of Arabic normalisation

Normalisation strips diacritics and tatweel and unifies alef-hamza and taa-marbuta forms.
A large positive gap means the model had the fact right and the spelling variant wrong.

| Model | F1 raw | F1 normalised | Gap |
|---|---|---|---|
| Gemma4 12B | 0.424 | 0.424 | +0.000 |
| Qwen3.5 9B | 0.344 | 0.344 | +0.000 |
| Llama3.1 8B | 0.321 | 0.321 | +0.000 |
| Aya-Expanse 8B | 0.307 | 0.307 | +0.000 |
| Command-R7B-Arabic | 0.300 | 0.300 | +0.000 |
| Mistral-Nemo 12B | 0.248 | 0.249 | +0.000 |
| Qwen3.5 4B | 0.246 | 0.246 | +0.000 |
| mREBEL (baseline) | 0.236 | 0.236 | +0.000 |
| Fanar-1 9B | 0.233 | 0.234 | +0.000 |
| Qwen3.5 2B | 0.160 | 0.160 | +0.000 |
| ALLaM 7B | 0.071 | 0.071 | +0.000 |

## 4. Per-relation breakdown, strongest model

**Gemma4 12B**, ordered by gold frequency (tp+fn).

| Relation | P | R | F1 | Gold (tp+fn) |
|---|---|---|---|---|
| country | 0.000 | 0.000 | 0.000 | 1202 |
| country of citizenship | 0.856 | 0.508 | 0.637 | 1125 |
| occupation | 0.597 | 0.837 | 0.697 | 619 |
| date of birth | 0.656 | 0.874 | 0.750 | 461 |
| place of birth | 0.271 | 0.359 | 0.309 | 384 |
| member of sports team | 0.831 | 0.821 | 0.826 | 364 |
| point in time | 0.041 | 0.031 | 0.035 | 354 |
| instance of | 0.057 | 0.068 | 0.062 | 265 |
| sport | 0.312 | 0.028 | 0.051 | 179 |
| date of death | 0.625 | 0.884 | 0.732 | 164 |
| located in the administrative territorial entity | 0.107 | 0.301 | 0.158 | 133 |
| position played on team/speciality | 0.231 | 0.298 | 0.260 | 121 |
| place of death | 0.105 | 0.146 | 0.122 | 89 |
| subclass of | 0.038 | 0.023 | 0.029 | 86 |
| diplomatic relation | 0.038 | 0.016 | 0.022 | 64 |
| part of | 0.000 | 0.000 | 0.000 | 43 |
| member of political party | 0.679 | 0.947 | 0.791 | 38 |
| parent taxon | 0.000 | 0.000 | 0.000 | 38 |
| participant | 0.000 | 0.000 | 0.000 | 22 |
| inception | 0.207 | 0.286 | 0.240 | 21 |
| location of birth | 0.000 | 0.000 | 0.000 | 0 |
| member of | 0.000 | 0.000 | 0.000 | 0 |

## 5. Cross-seed F1 spread

Most models ran one seed (for time, 2026-08-13); for them the stdev below
is 0.000 by construction, an absence of evidence rather than a stability result.
Four models (Qwen3.5 9B, Llama3.1 8B, Command-R7B-Arabic, Aya-Expanse 8B) were re-run with further seeds for the
revision; their rows report a real spread over their runs. The headline tables use
seed 1 only.

| Model | F1 mean | F1 stdev | seeds |
|---|---|---|---|
| Gemma4 12B | 0.424 | 0.000 | 1 |
| Qwen3.5 9B | 0.350 | 0.008 | 3 |
| Llama3.1 8B | 0.334 | 0.010 | 3 |
| Command-R7B-Arabic | 0.313 | 0.009 | 3 |
| Aya-Expanse 8B | 0.310 | 0.003 | 3 |
| Mistral-Nemo 12B | 0.248 | 0.000 | 1 |
| Qwen3.5 4B | 0.246 | 0.000 | 1 |
| mREBEL (baseline) | 0.236 | 0.000 | 1 |
| Fanar-1 9B | 0.233 | 0.000 | 1 |
| Qwen3.5 2B | 0.160 | 0.000 | 1 |
| ALLaM 7B | 0.071 | 0.000 | 1 |
