# BERT runs and paraphrases (2026-10-15/17)

Written by hand. `10_analyse.py` computes the official statistics (CIs, the A5 above-chance rule, McNemar); this file records what was run, how, and the first numbers.

## Where BERT runs: Kaggle, not Colab

`notebooks/07b_bert_colab.ipynb` now runs on Colab or Kaggle and picks the paths itself (Kaggle: the attached dataset in, `/kaggle/working/<RESULTS_NAME>` out).

- Colab: no T4 was available; the session fell back to a TPU/CPU runtime (`torch 2.9.0+cpu`, no `nvidia-smi`). The notebook also gained a TPU path (`torch_xla`, bf16), but it was never used.
- Kaggle (account `shiyalow`), all private:

| What | Kaggle id | Settings |
|---|---|---|
| Data | `shiyalow/jev-bert-data` | `data/bert/`, latest version = main-run data |
| Smoke test | `shiyalow/jev-bert-smoke` | `SMOKE_TEST = True` |
| Originals-only ablation | `shiyalow/jev-bert-original-only` | `TRAIN_VERSIONS = ("original",)`, `RUN_UNTRAINED = False`, `RESULTS_NAME = "results_original_only"` |
| Main run | `shiyalow/jev-bert-main` | repo notebook unchanged |

All on a Tesla T4 (torch 2.11 + CUDA 12.8, fp16). The kernels are pushed with `uvx kaggle kernels push --accelerator NvidiaTeslaT4` from a folder holding the notebook and `kernel-metadata.json` (`enable_gpu`, `enable_internet`, `dataset_sources: ["shiyalow/jev-bert-data"]`). No Hugging Face login needed: `bert-base-uncased` is public (the HF warning is only about rate limits).

## Smoke test

1 seed, 2,000 train rows, 1 epoch, originals only: 90 s of training; loss 1.38 (chance ≈ 1.43), val 39.3%, Level 0 and 1 both 40.6%. No (question, option) pair truncated at `MAX_LENGTH = 192`.

The load report lists `cls.*` as UNEXPECTED (the pre-training heads, dropped) and `classifier.*` as MISSING (the new multiple-choice head, randomly initialised). That is the intended setup; only `bert.*` weights missing would be a problem.

## Originals-only ablation

Trained on the 5,500 original train questions only (val 1,000 originals), evaluated on val and Levels 0, 1, 3 (Level 2 did not exist yet). 5 seeds × 3 epochs, best epoch on val; ~11 min training per seed, 63 min in total. Results: `results/bert_original_only/seed_<n>/` (outside `results/bert/`, which `10_analyse.py` reads as main-run seeds only).

Accuracy, mean ± SD over the 5 seeds (%; chance from the option counts):

| Dataset | Chance | val | Level 0 | Level 1 | Level 3 |
|---|---|---|---|---|---|
| ARC-Challenge | 25 | 30.0 ± 2.9 | 26.6 ± 4.3 | 26.7 ± 4.2 | 48.4 ± 5.2 |
| ARC-Easy | 25 | 47.3 ± 2.0 | 48.9 ± 1.6 | 48.9 ± 1.5 | 48.0 ± 3.7 |
| CommonsenseQA | 20 | 42.7 ± 1.3 | 47.9 ± 1.3 | 47.9 ± 1.3 | 70.8 ± 3.9 |
| HellaSwag | 25 | 37.4 ± 2.6 | 36.1 ± 1.4 | 36.1 ± 1.4 | 32.8 ± 6.7 |
| SWAG | 25 | 64.9 ± 1.4 | 66.6 ± 1.8 | 66.7 ± 1.7 | – |
| **All** | ≈ 24 | 44.5 ± 1.2 | 45.2 ± 1.3 | 45.2 ± 1.2 | 50.0 ± 2.2 |

| Seed | Best epoch | Val by epoch (%) | Final train loss |
|---|---|---|---|
| 11 | 2 | 44.3 / 45.3 / 44.4 | 0.82 |
| 22 | 3 | 42.5 / 43.4 / 43.5 | 0.82 |
| 33 | 3 | 44.7 / 46.0 / 46.1 | 0.79 |
| 44 | 2 | 42.1 / 44.0 / 43.7 | 0.99 |
| 55 | 1 | 43.4 / 41.7 / 42.8 | 0.93 |

Notes:

- **ARC-Challenge is at chance** (val 30.0, Level 0 26.6). The main run confirms it fails the A5 above-chance rule (see below).
- **SWAG is the easiest for BERT** (val 64.9), as expected for its BERT-fair control role. HellaSwag is much harder (37.4), consistent with it being adversarially filtered against BERT-era models.
- **Level 1 = Level 0 by construction.** BERT scores each (question, option) pair separately with shared weights and never sees an option's position, so reordering options cannot change its answer (2,000/2,000 questions identical in the smoke test; seed means agree within 0.1 point). Level 1 carries no information for BERT; do not read it as "BERT is robust to shuffling".
- **Level 3 is higher than Level 0** (50.0 vs 45.2), driven by CommonsenseQA (70.8) and ARC-Challenge (48.4). Only 50 questions per dataset and no SWAG, so it is noisy, but it suggests the new questions are somewhat easier than the originals. Keep in mind for the "Level 3 vs Level 0" comparison.
- **Overfitting after epoch 1-2:** train loss keeps falling (to ~0.8) while val stalls; best-epoch selection handles it.
- The ablation is only fully useful next to the main run on **Level 2**: a large originals-only drop there that the main run does not show means the paraphrase training is what makes BERT robust to rewording.

## Paraphrases (`03_paraphrase.py` → `04_validate.py`)

All 8,500 sampled questions (train, val, test; 5 datasets) paraphrased with `openai/gpt-6-luna`; $0.79 in total, in line with the pilot estimate.

- 04 automatic checks: 8,500/8,500 passed in the end. One rejection, `csqa-train-9002`: the first two attempts reworded two options into the same text ("spending money" twice); the third (`--retry-failed` twice) was clean.
- 7 questions needed a second or third request (6 for malformed replies inside 03, 1 for the rejection above).
- Manual check sheet written: `data/reviews/paraphrase_50.csv` (50 random test paraphrases). **Not filled in yet.**

`07a_build_bert_data.py` (without `--originals-only`) then built the main-run data, matching `docs/experiment-flow.png`:

| File | Rows |
|---|---|
| `train.jsonl` | 11,000 (5,500 originals + 5,500 paraphrases) |
| `val.jsonl` | 2,000 (1,000 + 1,000) |
| `level0.jsonl`, `level1.jsonl`, `level2.jsonl` | 2,000 each |
| `level3.jsonl` | 200 |

## Main run

`shiyalow/jev-bert-main` (2026-10-16): untrained chance floor + 5 seeds × 3 epochs on the 11,000-row train set (originals + paraphrases), all Levels 0-3. 124 min in total, ~22.5 min training per seed. No truncation, no errors, every probability set sums to 1. Results: `results/bert/<untrained | seed_n>/`. (`kaggle kernels output` can time out half-way; check all 6 folders arrived.)

Accuracy, mean ± SD over the 5 seeds (%):

| Dataset | Chance | val | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|---|---|
| ARC-Challenge | 25 | 36.1 ± 1.4 | 31.4 ± 1.3 | 31.4 ± 1.3 | 32.9 ± 0.9 | 53.6 ± 3.6 |
| ARC-Easy | 25 | 49.3 ± 2.4 | 52.5 ± 1.6 | 52.5 ± 1.6 | 49.5 ± 1.7 | 52.4 ± 5.0 |
| CommonsenseQA | 20 | 44.8 ± 3.6 | 49.9 ± 0.7 | 49.9 ± 0.7 | 44.4 ± 0.8 | 70.4 ± 1.7 |
| HellaSwag | 25 | 38.9 ± 1.4 | 38.0 ± 1.4 | 38.0 ± 1.4 | 39.6 ± 0.2 | 41.2 ± 3.6 |
| SWAG | 25 | 59.5 ± 1.9 | 66.0 ± 1.3 | 66.1 ± 1.3 | 59.8 ± 1.2 | – |
| **All** | ≈ 24 | 45.7 ± 1.7 | 47.6 ± 0.7 | 47.6 ± 0.7 | 45.2 ± 0.6 | 54.4 ± 1.8 |
| Untrained (all) | ≈ 24 | 30.1 | 29.6 | 29.6 | 34.1 | 32.5 |

(val here mixes originals and paraphrases, 1,000 each.)

- **A5 above-chance rule** (`10_analyse.py`, val originals, CI low ≥ chance + 10): BERT is a valid difficulty control on **ARC-Easy, CommonsenseQA, SWAG**; **uninformative on HellaSwag** (CI low 32.1 vs threshold 35.0) **and ARC-Challenge** (29.9 vs 35.1). So the Level 0 → 2 comparison against BERT only has a reference on 3 of 5 datasets.
- **BERT's Level 0 → 2 drop:** 2.4 points overall (47.6 → 45.2); 5-6 points on CommonsenseQA and SWAG, 3 on ARC-Easy. This is the reference drop for the memorisation ladder.
- **Paraphrase training helped a little:** Level 0 47.6 vs 45.2 for the originals-only ablation; ARC-Challenge 31.4 vs 26.6.
- **The untrained floor is ~30%, not 24%.** Its picks are spread evenly over A-E (no position bias), but it picks the longest option more often (33-34%), and the correct option is the longest more often than chance (27-29% in val/Level 0). The pre-trained `[CLS]` output also carries some next-sentence (coherence) signal that a random head partly picks up. Not a bug; state the floor as measured.

## Level 3 has an option-length cue

In Level 3 the correct option is the longest in 89/200 questions (45%; chance ~24%, Level 0 26% without SWAG), and many distractors are obviously absurd (e.g. "throws the plate into the sink and starts to swim"). Jev 100%, Gemma 100%, Qwen 99.5% and BERT 54.4% (vs 47.6 on Level 0) all score higher on Level 3 than Level 0. So "Level 3 vs Level 0" mixes memorisation with Level 3 simply being easier; treat it as a caveat (or rebalance option lengths / regenerate harder distractors) before reading it as evidence.
