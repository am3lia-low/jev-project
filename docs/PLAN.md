# Jev Project Plan

Evaluating TypeSafe's **Jev** (a "System One" decision model) against BERT baselines on multiple-choice questions, and probing how far it can be steered into personas.

The project has two independent parts:

- **Part A — Jev benchmarks** ([`01-jev-benchmarks/`](../01-jev-benchmarks/)): Jev vs BERT on MCQ, and how deep Jev's memorisation of public benchmarks goes.
- **Part B — Jevlet personas** ([`02-jevlet-personas/`](../02-jevlet-personas/)): can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs?

Last updated: 2026-10-18

---

## Background

**Jev** (TypeSafe AI, early access since 15 Sep 2026) does not generate text. It takes unstructured input ("state") plus typed questions and returns a typed decision with calibrated probabilities:

| Question type | Returns | Use here |
|---|---|---|
| `choice` | One of up to 255 options + probabilities | MCQ answers, memorisation quiz, Part B scenarios |
| `score` | Position on a 2–10 level scale | Big Five Likert items |
| `noul` | Probability a statement is true | — |

- Access: OpenRouter, model `typesafe/jev-1.13` (pin this; `~typesafe/jev-latest` can change mid-experiment). 32k context on OpenRouter. $0.042 / M input tokens, output free.
- Reported weaknesses (vendor docs): arithmetic/counting, dates, multi-step reasoning, literal reading of instructions, sensitivity to irrelevant context.
- Training: vendor states synthetic data only, trained with RLCD. No training data or technical paper published.

### Prior work (checked 2026-10-05)

| Work | What it did | Gap for us |
|---|---|---|
| [Deußer et al., arXiv 2609.37647](https://arxiv.org/html/2609.37647v1) | Jev on 37 datasets incl. HellaSwag 95.5%, ARC-Easy 99.3%, ARC-Challenge 97.7%, MMLU 91.8%. Compared only to Qwen/Gemma LLMs. One frozen template per dataset, single-token option codes, **one run per request (no variance measured)**. Flagged likely MMLU training exposure (Jev scored higher on calculation-heavy MMLU). | No BERT, no paraphrase/perturbation tests, no repeat runs |
| [jevbench](https://github.com/dhruvmehra/jevbench) | Jev vs fine-tuned DistilBERT, BART-MNLI, LLMs on SST-2, AG News, Banking77 | Classification only, no MCQ |
| [jev-zeroshot-vs-bert](https://github.com/zhuyansen/jev-zeroshot-vs-bert) | Jev vs zero-shot BERT-family on 7 tasks, label-equivalence curves, post-release arXiv contamination check | Classification only |
| [decision-index](https://github.com/aminry/decision-index) | Reproduces the arXiv paper's suite | **Reuse its frozen prompt templates and loaders** |

No prior work found on: BERT vs Jev on MCQ reasoning, paraphrase-based memorisation tests for Jev, or persona steering of Jev.

---

## Part A — Jev benchmarks

Code: [`01-jev-benchmarks/`](../01-jev-benchmarks/) (see its README for the folder layout). Scripts are numbered by their step in [A10](#a10-order-of-work).

### A0. Experiment flow

Legend: **blue** = original human-written data · **orange** = synthetic (LLM-generated) · **grey** = scripted step / derived set · **green** = models · **purple** = results. Dashed = optional.

A rendered copy for reference is in [`docs/experiment-flow.png`](experiment-flow.png). It is generated from this diagram; re-render it after any change (see [Environment](#environment)).

```mermaid
flowchart TD
    %% ---------- Source datasets (human-written) ----------
    subgraph SRC["Source datasets (human-written, public)"]
        HS["HellaSwag<br/>train 39,905 · val 10,042<br/>4 options"]
        AE["ARC-Easy<br/>train 2,251 · test 2,376<br/>3-5 options"]
        AC["ARC-Challenge<br/>train 1,119 · val 299 · test 1,172<br/>3-5 options"]
        CQ["CommonsenseQA<br/>train 9,741 · val 1,221<br/>5 options"]
        SW["SWAG (BERT-fair control)<br/>train 73,546 · val 20,006<br/>4 options"]
    end

    %% ---------- Sampling (scripted) ----------
    SRC --> SAMPLE["02_sample.py (API laptop)<br/>sample · split by question · stable IDs<br/>labels to A-E · dedupe SWAG vs HellaSwag<br/>drop duplicate-option items + test stems in train/val<br/>store original + balanced-shuffle order"]
    SAMPLE --> TRV["Train 1,100 + Val 200 per dataset<br/>5,500 + 1,000 in total"]
    SAMPLE --> TEST["Test 400 per dataset<br/>2,000 in total"]

    %% ---------- Levels 0 and 1 (scripted) ----------
    TEST --> L0["Level 0<br/>original wording + original order"]
    TEST --> L1["Level 1<br/>original wording, balanced shuffle"]

    %% ---------- Synthetic: paraphrases ----------
    TRV --> PARA["SYNTHETIC: 03_paraphrase.py (API laptop)<br/>openai/gpt-6-luna, chosen by 30-item pilot<br/>not the same family as any LLM baseline<br/>rephrase question + options, reshuffle"]
    TEST --> PARA
    PARA --> VPARA["04_validate.py<br/>auto: answer kept, option count, text changed<br/>manual: 50 items, meaning + difficulty rating"]
    VPARA --> L2["Level 2<br/>paraphrased + shuffled<br/>2,000"]
    VPARA --> TRAINSET["BERT training set<br/>originals + paraphrases<br/>train 11,000 · val 2,000"]

    %% ---------- Synthetic: brand-new questions ----------
    GEN["SYNTHETIC: 05_generate_new.py (API laptop)<br/>Claude Sonnet 5.5 drafts new questions<br/>in each paper dataset's style<br/>near-copy screen: similarity ≥ 0.75 replaced,<br/>0.6-0.75 flagged for review"] --> HNEW["Human check<br/>answer correct · exactly one right option<br/>not a near-copy · light rewording to match style"]
    HNEW --> L3["Level 3<br/>200 brand-new<br/>50 x 4 paper datasets"]

    %% ---------- Synthetic: memorisation quiz ----------
    TEST --> QSEL["Quiz benchmark items<br/>400 = 100 x 4 paper datasets<br/>Level 0 wording"]
    L3 --> QCTRL["Quiz control items<br/>100 = 25 x 4, taken from Level 3"]
    QSEL --> SWAP["SYNTHETIC: 06a_swap.py (API laptop)<br/>openai/gpt-6-luna<br/>3 synonym-swap versions per item<br/>word-level only, same structure + answer"]
    QCTRL --> SWAP
    SWAP --> VSWAP["06b_validate_swaps.py<br/>auto: all distinct, word count within 2,<br/>word-level edits only · manual: 30 items"]
    VSWAP --> QUIZ["Quiz, 500 items<br/>A-D = original + 3 swaps, position balanced<br/>E = none of the provided options"]

    %% ---------- Models ----------
    subgraph MODELS["Models"]
        BERTU["Untrained BERT-base<br/>chance floor"]
        BERTFT["Fine-tuned bert-base-uncased<br/>5 seeds (Kaggle T4; no Colab GPU)"]
        OPT["Optional: BERT-large 1 seed ·<br/>RoBERTa-base · RACE pre-training ·<br/>originals-only ablation (done)"]
        JEV["Jev typesafe/jev-1.13<br/>(API laptop)"]
        LLM["LLM baselines: Qwen3.8-27B,<br/>Gemma-4-26B-A4B (API laptop)"]
    end
    TRAINSET --> BERTFT
    BERTFT --> VALID["BERT above-chance rule<br/>val originals only<br/>CI low > chance + 10 pts, per dataset"]
    JEV --> REP["Jev checks before the levels<br/>8a sanity: Level 0 vs paper CI<br/>8b repeatability: 200 Qs x 3<br/>~100% agree → single runs"]
    TRAINSET -.-> OPT

    L0 --> MODELS
    L1 --> MODELS
    L2 --> MODELS
    L3 --> MODELS
    QUIZ --> JEV
    QUIZ --> LLM

    %% ---------- Results ----------
    MODELS --> R1["Accuracy + 95% CI · ECE · latency · cost<br/>per model x dataset x level"]
    R1 --> R2["Paired McNemar between levels<br/>Jev vs BERT drop in log-odds<br/>pooled over SWAG, ARC-E, ARC-C, CSQA if BERT valid<br/>HellaSwag reported separately"]
    JEV --> R3["Quiz: % original picked vs 25%<br/>benchmark items vs control items"]
    LLM --> R3
    R1 --> R4["Level 3 accuracy<br/>vs Level 0"]
    VALID --> R2
    R2 --> FINAL["Memorisation verdict<br/>ladder + quiz + Level 3"]
    R3 --> FINAL
    R4 --> FINAL

    %% ---------- Styles ----------
    classDef human fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef synth fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef script fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef model fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef result fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    class HS,AE,AC,CQ,SW human
    class PARA,GEN,SWAP synth
    class SAMPLE,TRV,TEST,L0,L1,L2,L3,TRAINSET,VPARA,HNEW,QSEL,QCTRL,VSWAP,QUIZ,VALID,REP script
    class BERTU,BERTFT,OPT,JEV,LLM model
    class R1,R2,R3,R4,FINAL result
```

### A1. Datasets

The four MCQ datasets from the arXiv paper, so results are comparable, plus SWAG as a BERT-fair control:

| Dataset | Official train | Official val | Official test | Test questions drawn from | Role |
|---|---|---|---|---|---|
| HellaSwag | 39,905 | 10,042 | 10,003 (labels hidden) | validation | Paper dataset |
| ARC-Easy | 2,251 | 570 | 2,376 | test | Paper dataset |
| ARC-Challenge | 1,119 | 299 | 1,172 | test | Paper dataset |
| CommonsenseQA | 9,741 | 1,221 | 1,140 (labels hidden) | validation | Paper dataset |
| SWAG | 73,546 | 20,006 | 20,005 (labels hidden) | validation | BERT-fair control |

- Paper datasets use the split the paper evaluated on (inferred from its question counts: HellaSwag 10,042; ARC 3,548 = 2,376 + 1,172; CommonsenseQA 1,221). SWAG has no paper number.
- **Why SWAG:** HellaSwag's wrong answers were filtered with BERT-Large as the adversary, so BERT-Large only reached 47.3% ([HellaSwag paper](https://arxiv.org/pdf/1905.07830)). SWAG is the same kind of task without that anti-BERT filtering, and BERT scored >86% on it ([SWAG paper](https://arxiv.org/pdf/1808.05326)). SWAG gives BERT a fair dataset for the difficulty control. HellaSwag is still used for Jev (comparability with the paper); BERT's HellaSwag score is reported with this caveat.
- **SWAG ↔ HellaSwag overlap:** both draw on ActivityNet Captions. `02_sample.py` drops SWAG training questions that share a video ID or context text with any of the 400 sampled HellaSwag test questions (324 dropped by video, 0 by context). Decided 2026-10-14 to keep this rather than dedupe against the full HellaSwag validation split.
- SWAG is used in Levels 0–2 only. Level 3 and the memorisation quiz cover the four paper datasets.

### A2. Sampling and splitting

Equal number per dataset, stratified by dataset. ARC-Challenge is the bottleneck (1,119 train questions).

| Per dataset | Train | Val | Test | Total |
|---|---|---|---|---|
| Questions | 1,100 | 200 | 400 | 1,700 |
| × 5 datasets | 5,500 | 1,000 | 2,000 | 8,500 |

Rules:
1. **Test** questions come from the split in A1; **train/val** come from the official train split (ARC-Challenge val may use the official val split to have enough).
2. Split **by original question before any LLM step**, so no test question appears in training in any form.
3. Store **two option orders** per question: the **original order** (Level 0) and a **balanced shuffle** (Level 1+), where the correct position is balanced within each option-count group (3, 4, 5 options).
4. Normalise answer labels to letters A–E (some ARC items use `1/2/3/4`).
5. Every question gets a stable ID (e.g. `arcc-test-0417`), shared by all its versions (original, shuffled, paraphrased, synonym-swapped).
6. Variable option counts (ARC 3–5, HellaSwag/SWAG 4, CommonsenseQA 5): pad to 5 with masked dummy options for BERT.
7. Drop questions that list the same option twice (some CommonsenseQA items; case-sensitive, so ARC genotypes like "WW or Ww" stay), and train/val candidates whose question text matches a sampled test question (ARC repeats a few stems across splits). Counts are in `data/sampling_report.json`.

### A3. Synthetic data

Three LLM-generated datasets. All scripts run on the API laptop, are resumable and cache every response.

**A3.1 Paraphrases — `03_paraphrase.py`**
- Input: all 8,500 sampled questions (train, val and test).
- Output: question rephrased, options rephrased where sensible, options reshuffled. Meaning and correct answer preserved.
- Used for two separate things, from separate questions: (a) train/val paraphrases → BERT training data; (b) test paraphrases → Level 2. Same script, model and prompt for both, so BERT is used to the paraphrase style.
- Model: **pilot 2 candidate models on the same 30 questions**, pick on meaning preservation in the manual check. Must not be the same family as any LLM baseline (no Qwen, Gemma or Gemini). Candidates: `openai/gpt-6-luna` ($0.10 / $0.50 per M in/out) and `mistralai/mistral-small-2603` ($0.15 / $0.60); backup `deepseek/deepseek-v4.1-flash` ($0.055 / $1.32). Prices checked on OpenRouter 2026-10-06. **Chosen 2026-10-14: `openai/gpt-6-luna`** (pilot: both kept meaning 30/30; gpt-6-luna rated same difficulty 30/30, Mistral made 1 harder; gpt-6-luna is cheaper). Estimated cost for the whole project (~6M tokens): ~$2, half that with the `:batch` variants.
- The winner is also used for `06a_swap.py` (A3.3) and the IPIP paraphrases (B1 step 4).
- Validation (`04_validate.py`): automatic checks (answer still present, option count unchanged, text actually changed) + manual review of a random 50, rating meaning preserved and whether the paraphrase made the question easier / harder / same.

**A3.2 Brand-new questions — `05_generate_new.py`**
- An LLM drafts new questions in the style of each paper dataset: 100 × 4 = 400 drafts, of which **50 × 4 = 200** are kept after the human check. Drafts are reviewed in a seeded random order per dataset and the review stops at 50 kept, so the other drafts are spares for rejected ones. (Cut from 100 per dataset on 2026-10-08 to halve the review: per-dataset 95% CI about ±8 points instead of ±6; pooled over the 4 datasets about ±4.)
- Model: **`anthropic/claude-sonnet-5.5`** ($2 / $10 per M in/out, checked 2026-10-06). A stronger model than the paraphraser on purpose: small volume, and better drafts mean less human fixing. Not Jev's or a baseline's family.
- Near-copy screen: each draft is compared with every labelled question of its dataset; drafts with similarity ≥ 0.75 are discarded and replaced automatically (the model reproduces textbook questions ARC already contains), 0.6–0.75 are flagged for the human check.
- Human check on every question: correct answer is right, exactly one option is right, not a near-copy of a benchmark item. Lightly reword so the text reads like the human-written originals (matters for the quiz control).
- Used as **Level 3** (all 200) and as the **quiz control items** (100 of them, 25 per dataset).

**A3.3 Synonym swaps — `06a_swap.py`** (see A7 for the full quiz)
- For each of the 500 quiz items, an LLM makes **3 versions** that only replace words with context-fitting synonyms. Sentence structure, word order, meaning and answer unchanged. 1,500 versions in total.
- Validation (`06b_validate_swaps.py`): all 4 versions distinct; word count within ±2; edits are word-level only; option count unchanged; manual check of 30 (meaning kept, no version sounds obviously off). Regenerate failures.

### A4. Models

| Model | Training | Role |
|---|---|---|
| `bert-base-uncased` — untrained | Multiple-choice head randomly initialised, no fine-tuning | Chance floor |
| `bert-base-uncased` — fine-tuned | See A5 | Main baseline + **control** for paraphrase difficulty |
| Jev (`typesafe/jev-1.13`) | None | Subject |
| LLM baselines: `qwen/qwen3.8-27b`, `google/gemma-4-26b-a4b-it` | None. As in the paper: one prompt, read the next-token probabilities of the option letters, renormalise. Run through the OpenRouter API (see A9 for how this differs from the paper) | Added **after** Jev; same models as the paper so its LLM numbers can be reproduced too |
| *Optional:* `bert-large-uncased`, 1 seed | As A5 | Only if BERT-base is weak; the model HellaSwag was filtered against |
| *Optional:* `roberta-base` | As A5 | Stronger encoder for the accuracy comparison only (see below) |

- **Why `bert-base-uncased`:** it is "BERT" as commonly understood, 5 seeds fit comfortably on a free T4 (16GB; Colab gives roughly 15–30 GPU hours/week, not guaranteed; Kaggle ~30), and it is the standard reference point. BERT-large (340M) fits a T4 with fp16, a small batch (~4) and gradient accumulation, but 5 seeds may exceed free-tier limits.
- **Why RoBERTa is not the control:** RoBERTa's pre-training data includes CC-News (crawled Sep 2016–Feb 2019) and OpenWebText ([RoBERTa paper](https://arxiv.org/pdf/1907.11692)), a window in which ARC, SWAG and CommonsenseQA were already public. It may have seen benchmark text, so it is a less clean "never saw the test questions" control than BERT (BooksCorpus + Wikipedia). It is much stronger (RoBERTa-large: 85.2% HellaSwag leaderboard, 83.2% RACE), so it is worth adding as an optional accuracy baseline.
- **BART-MNLI: cut.** NLI framing fits MCQ poorly and zero-shot NLI models tend towards chance on commonsense tasks.

### A5. BERT training

- **Main run: one model trained on both versions** of every training question across all 5 datasets (originals + paraphrases → 11,000 examples). BERT then has no style advantage on either test version, so its drop from original → paraphrased measures only how much harder paraphrasing made the questions.
- Validation set (both versions, 2,000) for early stopping and choosing settings.
- **5 random seeds**; report mean ± SD. BERT fine-tuning on small datasets varies a lot between seeds ([Dodge et al. 2020](https://arxiv.org/pdf/2002.06305), [Zhang et al. 2021](https://arxiv.org/pdf/2006.05987)).
- **Above-chance rule** (per dataset, on val): BERT counts as a valid difficulty control only if the lower bound of its val accuracy 95% CI is ≥ chance + 10 points. Datasets that fail are marked "BERT control uninformative". Computed on the original wording only (200 per dataset): a question and its paraphrase are near-duplicates, so counting both would make the CI look narrower than it is (decided 2026-10-09).
- **Fallbacks if BERT is near chance (e.g. ARC-Challenge):** RACE pre-training first (RACE: ~98k exam MCQs, [Lai et al. 2017](https://aclanthology.org/D17-1082.pdf); BERT-base reached ~44% on ARC-Challenge this way, [intersun/ARC-Challenge](https://github.com/intersun/ARC-Challenge)), and/or BERT-large with 1 seed. *Not used: decided 2026-10-20 to enlarge the train pool instead, as a sensitivity analysis (A8); ARC-Challenge stays out of the pool.*
- Optional ablations (cheap on the free T4): train on original only (done 2026-10-15, `results/bert_original_only/`); train on paraphrased only.
- **Learning curve** (diagnostic, added 2026-10-18): the notebook's `LEARNING_CURVE = True` trains on nested 25/50/100% subsets of each dataset's train questions (2 seeds), to test whether BERT is data-limited before enlarging the train pool. Results go to `results/bert_learning_curve/`. Expected reading: if HellaSwag and ARC-Challenge stay flat while SWAG/CommonsenseQA rise, more data strengthens the reference only where BERT already passes, and ARC-Challenge needs the RACE fallback (its train split is already 98% used).
  - **Result (2026-10-18):** HellaSwag flat (~36% at every fraction); ARC-Challenge still rising (Level 0 24.9 → 28.9 → 32.4%) but below the threshold, so it can only rejoin through the RACE fallback; SWAG, ARC-Easy and CommonsenseQA still rising at 100%. Details in [`bert_summary.md`](../01-jev-benchmarks/results/bert_summary.md#learning-curve). Decided 2026-10-20: enlarge the pool, as a sensitivity analysis (A8); no RACE fallback, so ARC-Challenge stays out of the pool.
- Hardware: free T4 on Kaggle (planned for Colab, but no Colab GPU was available; the notebook runs on either). Neither local laptop has an NVIDIA GPU. Details in [`results/bert_summary.md`](../01-jev-benchmarks/results/bert_summary.md).
- Training-set-size curves: **on hold**.

### A6. Test sets and the memorisation ladder

Every model is run on the same test questions in each version:

| Level | Test version | Size | Memorisation that would still help |
|---|---|---|---|
| 0 | Original wording, original option order (reproduces the paper) | 2,000 (5 datasets) | Anything, incl. "the answer is C" |
| 1 | Original wording, balanced shuffle | 2,000 | Recall of answer text, not position |
| 2 | Paraphrased + shuffled | 2,000 | Recall of what the question is about |
| 3 | Brand-new questions (A3.2) | 200 (50 × 4 paper datasets) | Nothing — true ability |

Reading the results: BERT never saw any test question, so its drop between levels is the **difficulty baseline** (on datasets where it passes the above-chance rule). Jev's drop *beyond* BERT's suggests memorisation. Memorisation can survive paraphrasing ([Yang et al., "Rephrased Samples"](https://arxiv.org/html/2311.04850v2)), hence Level 3 and the quiz (A7).

### A7. Memorisation quiz

Purpose: test whether Jev recognises the **exact wording** of benchmark questions, which is only possible with prior exposure. Adapted from the Data Contamination Quiz ([Golchin & Surdeanu](https://arxiv.org/abs/2311.06233), [code](https://github.com/shahriargolchin/DCQ)).

| Set | Size | Source | Original among A–D? | Could Jev have seen the original? | Expected with no memorisation |
|---|---|---|---|---|---|
| Benchmark items | 400 = 100 × 4 paper datasets | Test questions (Level 0 wording) | Yes | Yes (public since 2018–19) | ~25% |
| Control items | 100 = 25 × 4 paper datasets | Level 3 questions | Yes | **No** | ~25% |

Steps:
1. **Pick items:** 100 test questions per paper dataset (original wording and option order); 25 Level 3 questions per paper dataset. Level 3 must be built first.
2. **Make swaps:** `06a_swap.py` makes 3 synonym-swap versions of each item (A3.3).
3. **Validate:** `06b_validate_swaps.py` (A3.3).
4. **Build quiz** (`06c_build_quiz.py`): options A–D = original + 3 swaps, original's position balanced across A–D; option E = "None of the provided options". Each option is a whole item (question + its options as a), b), …). The prompt names the dataset and the split its benchmark items come from, for control items too. Prompt adapted from the DCQ repo to Jev's `choice` type, tested on 20 *training* questions, then frozen.
5. **Run:** Jev, then the LLM baselines on the same 500 items. Log all returned probabilities.
6. **Score:** % original picked; binomial test vs 25% (per dataset and pooled); benchmark vs control with a two-proportion test; pick rate per position A–D to check position bias.

| Benchmark items | Control items | Conclusion |
|---|---|---|
| ≈ 25% | ≈ 25% | No evidence of memorised wording |
| Well above 25% | ≈ 25% | **Evidence of memorisation** |
| Above 25% | Above 25% | Prefers natural-sounding text; only the gap counts |
| ≤ control | any | No evidence |

Result (2026-10-18): every model picked the original more often on control items than on benchmark items (Jev 81.6% vs 62.6%), so in practice the quiz rewarded natural wording over the swaps and did not detect prior exposure. Qwen picked E on 492/494 items; its quiz result is uninformative.

Limits: a null result does not prove no memorisation (content can be memorised without exact wording — Level 2 covers that). Control originals are LLM-drafted while benchmark originals are human-written; the light rewording in A3.2 reduces but does not remove this difference.

### A8. Metrics and analysis

Per model × dataset × level:
- Accuracy with bootstrap 95% CI
- Calibration (ECE) where the model returns probabilities (Jev, BERT softmax, LLMs where logprobs are available)
- Latency p50 / p95 (BERT timed both locally and behind a local HTTP server, since Jev's latency includes the network)
- Throughput and cost per 1,000 questions
- For BERT: fine-tuning time and hardware; mean ± SD over seeds

Comparisons:
- **Between levels:** paired McNemar tests (same questions in every level), pooled as the main test and per dataset as secondary.
- **Memorisation hypothesis (fixed before any Jev run; confirmed as written 2026-10-18, before any Level 2 run):** compare each model's Level 0 → Level 2 drop in **log-odds** (accuracy near 100% cannot drop as many points as accuracy near 50%; e.g. Jev 97→94% is a log-odds drop of 0.72, BERT 60→55% is 0.21). Memorisation is supported if the bootstrap 95% CI of (Jev drop − BERT drop) is entirely above 0 **and** Jev's drop is at least 2× BERT's. If BERT's drop is negative (paraphrasing made the questions easier for BERT), the 2× condition passes trivially and the CI decides alone. Pooled over SWAG, ARC-Easy, ARC-Challenge and CommonsenseQA where BERT passes the above-chance rule; HellaSwag reported separately.
- **Final verdict** combines three signals: the ladder (above), the quiz (A7) and Level 3 accuracy vs Level 0.
- **Sensitivity analysis: bigger BERT training pool (decided 2026-10-20, before any extra question was drawn or any BERT model trained on it).** Jev's ladder pass is narrow (2.05x against 2x), and the BERT learning curve shows BERT still improving on SWAG, ARC-Easy and CommonsenseQA at 100% of its data. So BERT is re-trained on a bigger pool to see whether the result holds up. Fixed in advance:
  - **The verdict stays on the original BERT run.** The bigger-pool run is reported next to it as a sensitivity check ("does the conclusion hold?"), whatever it shows. It never replaces the pre-registered result.
  - **Extra data:** new train questions only, for the three datasets where BERT passes (ARC-Easy: all 950 left in its train pool; CommonsenseQA and SWAG: 1,100 more each, doubling them). A separate seeded draw from questions not already in any split, with the same filters as `02_sample.py` (duplicate options, test-text overlap, SWAG video dedupe). The existing train, val and test sets are unchanged, so every model is still tested on exactly the same questions. ARC-Challenge has no train questions left and HellaSwag stays flat on the learning curve, so neither changes.
  - **Same treatment as the main run:** paraphrased by PARAPHRASE_MODEL with the same checks (03/04), trained on originals + paraphrases (1:1), same notebook settings, 5 seeds; results in `results/bert_bigpool/`.
  - **Same rule:** the above-chance rule on val originals decides, from this run's own val results, which datasets join the pool; then the A8 ladder rule as written. No Jev or LLM run is repeated.
  - **Result (2026-10-21):** the same 3 datasets pass; BERT's pooled drop 0.21 (main 0.20). Jev 1.98x, CI [0.03, 0.39]: fails the 2x condition (main run: passes at 2.06x). Qwen passes, Gemma fails, as in the main run. Seed-to-seed spread of BERT's drop (0.15-0.23) is far larger than the change between runs, so Jev passes or fails the 2x condition by seed. See `memorisation_findings.md` finding 5. (pooled drop 0.41 vs BERT 0.20, 2.05×, CI of the difference [0.03, 0.40]; no single dataset passes); Qwen passes more clearly (0.60), Gemma fails (0.34). Quiz uninformative (all models pick the original more on unseen control items; Qwen's quiz result set aside, it picks E on 492/494). No Level 3 drop. Reading: weak, inconclusive evidence; see [`memorisation_findings.md`](../01-jev-benchmarks/results/memorisation_findings.md). The MMLU sub-experiment (A11) is the follow-up.

### A9. Running Jev and LLMs

- **Prompt format:** reuse decision-index's frozen per-dataset templates, single-token option codes A–E (no numeric keys). Checked 2026-10-08: decision-index sends an empty `state` and one `choice` question whose instructions are the bare question (ARC) or "Which continuation is most plausible?" + the context (HellaSwag), with options keyed A–E. It has no CommonsenseQA or SWAG template; ours copy ARC's and HellaSwag's respectively (`prompts/eval/`).
- **LLM baselines** get the same request rendered as decision-index's `transformers` engine does (its system line, then "State / Question / Options" and "Reply with exactly one option key."; `prompts/llm_system.txt`, `helpers/evaluation.py`), thinking off, one token; letter probabilities from the first token's top-20 logprobs, renormalised. Each baseline is pinned to one provider.
  - **Difference from the paper (accepted 2026-10-09):** the paper ran the models itself, so it had a probability for every option letter. OpenRouter only returns the 20 most likely next tokens, so a letter outside them is recorded as 0%. Accuracy is unaffected (the pick is the top token); ECE can be very slightly off. Running the models ourselves for full probabilities would need a paid GPU (27B / 26B parameters do not fit a free T4 alongside BERT), so the baselines stay on OpenRouter. State this in the write-up.
- **Sanity check first:** Level 0 accuracy on the sample must fall within the CI of the paper's numbers. If not, debug the client/template before running other levels.
- **Repeatability:** run 200 test questions 3 times; report agreement. ~100% → single runs; otherwise run each level 2–3 times and take the majority.
- **Errors:** retry with backoff, log everything, score unrecoverable failures as wrong and report the failure rate.

### A10. Order of work

1. [x] Repo setup (this plan, project structure, dependencies)
2. [x] `02_sample.py` — download 5 datasets, dedupe SWAG vs HellaSwag, sample, split, normalise labels, store original + shuffled orders, assign IDs (run 2026-10-08)
3. [x] `03_paraphrase.py` — pilot 2 models on 30 questions, choose (2026-10-14: gpt-6-luna), full run (2026-10-16: 8,500/8,500, $0.79) (API laptop)
4. [x] `04_validate.py` — automatic checks + manual 50 (automatic checks done 2026-10-16; `paraphrase_50.csv` filled 2026-10-17: 50/50 meaning kept, same difficulty)
5. [x] `05_generate_new.py` — 400 drafts, 200 Level 3 questions kept after the human check (API laptop)
6. [x] `06a_swap.py` + `06b_validate_swaps.py` — quiz versions for 400 benchmark + 100 control items (API laptop); `06c_build_quiz.py` — assemble the quiz (all run 2026-10-16/17; `swaps_30.csv` filled 2026-10-17: 6 of 30 flagged and dropped by 06b, quiz rebuilt with 494 items: 396 benchmark + 98 control)
7. [x] `07a_build_bert_data.py` — originals + paraphrases for the notebook; `notebooks/07b_bert_colab.ipynb` — untrained BERT; fine-tune BERT-base (5 seeds); above-chance rule; evaluate Levels 0–3; optional fallbacks (run on Kaggle: smoke test and originals-only ablation 2026-10-15, main run with paraphrases 2026-10-17; fallbacks not needed yet)
8. [x] `08_run_jev.py` (API laptop): 8a sanity check · 8b repeatability · 8c Levels 0–3 · 8d quiz (8a done 2026-10-09: Level 0 within the paper's CI on all 4 paper datasets; 8b done 2026-10-17: 99.5% agreement, single runs; 8c Levels 0, 1, 3 done 2026-10-17, Level 2 2026-10-18; 8d done 2026-10-18 on the 494-item quiz)
9. [x] `09_run_llm.py` (API laptop): LLM baselines, Levels 0–3 + quiz (Levels 0, 1, 3 done for both 2026-10-17; Level 2 and quiz 2026-10-18)
10. [ ] `10_analyse.py` — tables, plots, McNemar, log-odds comparison, quiz scoring, verdict (run on all results 2026-10-18; reading in [`results/memorisation_findings.md`](../01-jev-benchmarks/results/memorisation_findings.md); verdict drafted 2026-10-27 in `memorisation_findings.md`, to be confirmed by hand)
11. [x] MMLU number-swap sub-experiment ([A11](#a11-mmlu-number-swap-sub-experiment)): `11a_mmlu_swap.py` → `11b_mmlu_build.py` (writes the manual-check sheet; re-run after filling it) → `11c_run_mmlu.py` → `11d_analyse_mmlu.py` (all written and dry-run on a fake model 2026-10-18; `11a` and `11b` run 2026-10-19/20, manual check done 2026-10-24; `11c` and `11d` run 2026-10-27, [`mmlu_findings.md`](../01-jev-benchmarks/results/mmlu_findings.md))

Script files are named by their step number above (`06a`, `06b`, … when a step has several), so the folder lists them in run order.

### A11. MMLU number-swap sub-experiment

Separate from the ladder, quiz and Level 3 (decided 2026-10-18): its own data, scripts (`11a`–`11d`) and pass rule, reported next to the A8 verdict rather than as a fourth signal of it. No BERT control: each model is compared with itself on the original vs the number-swapped question.

**Why:** Deußer et al. flagged likely MMLU training exposure because Jev scored *higher* on calculation-heavy MMLU subjects. The ladder and quiz keep the answer unchanged, so they cannot separate recall from reasoning. Changing a question's numbers changes its answer: a model that computes follows the new numbers; a model that recalls the original keeps picking the old answer.

**Items — `11a_mmlu_swap.py`**
- Source: MMLU (`cais/mmlu`) **test** split, `elementary_mathematics` (378), `high_school_mathematics` (270), `college_mathematics` (100). Candidates: all four options are plain numbers (integer, decimal, fraction, %, $) and the question contains a number: 151 + 165 + 23 = 339 (checked 2026-10-18; items with two equal-valued options are left out).
- For each candidate, `NEW_QUESTION_MODEL` returns: the question with **only its numbers changed** (same wording, same magnitude and number of digits where possible, same kind of arithmetic), a Python `solve(...)` function with the original and the new numbers as arguments, and 2 wrong answers for the new version written as typical mistakes on the new numbers (so they tempt as much as the old answer does). It may answer `not_swappable` (e.g. a count fixed by theory, not by the numbers).
- **Code check** (automatic): `solve(original numbers)` must equal the original correct option, `solve(new numbers)` must differ from it and from every original option, and the 2 wrong answers must differ from both. `solve` runs in a separate process with a timeout. Items failing any check are retried (up to `MAX_ATTEMPTS`), then dropped.
- **Manual check** (sheet written by `11b`): 30 of the selected items (`data/reviews/mmlu_swaps_30.csv`): only numbers changed and it reads naturally, new answer right, and the 2 wrong answers are as plausible as the old answer (the 1/3 chance level of the trap test assumes this). Flagged items are replaced by the next verified item of their subject.
- Keep **200** verified items, a seeded draw spread over the three subjects in proportion to their verified counts.
- **Status (2026-10-20):** `11a` with two `--retry-failed` rounds: 314/339 verified ($3.45 in total; the first pass ran out of OpenRouter credits, then the missing items were regenerated). `11b`: 200 items (elementary 92, high school 96, college 12). Manual check done 2026-10-24: 30/30 pass all three checks, none replaced; `11b` re-run, `items.jsonl` unchanged. `11c` and `11d` run 2026-10-27 ($0.013, 0 failed).
- **Result (2026-10-27):** no model meets the rule. Jev drops 96.0 → 75.0% (log-odds 2.03 [1.46, 2.89], larger than both LLMs'; Qwen 0.28 and Gemma 0.23, both CIs include 0), but the old answer is 40% of its 50 wrong answers (p = 0.20). Read as: Jev's MMLU maths accuracy looks inflated by exposure, without reliable recall of the old answer. Details in [`mmlu_findings.md`](../01-jev-benchmarks/results/mmlu_findings.md).

**Two versions per item — `11b_mmlu_build.py`**
- **Original:** MMLU's question and 4 options, original order (as Level 0).
- **Swapped:** new question; options = new correct answer + **old correct answer (the trap)** + 2 wrong answers. Positions of the correct answer and the trap are balanced over A–D across items (seeded).
- Prompt as for ARC: the bare question (`prompts/eval/mmlu.txt`), options A–D.

**Runs — `11c_run_mmlu.py`:** Jev, Qwen and Gemma, both versions, one run each (Jev's repeatability is 99.5%, A9). 1,200 calls, a few cents.

**Analysis — `11d_analyse_mmlu.py`** (pass rule confirmed as written 2026-10-18, before any `11c` run)
- Per model: accuracy on original vs swapped (bootstrap 95% CI), paired McNemar, log-odds drop.
- **Trap rate:** share of swapped items where the model picks the old answer, and the trap's share of its *wrong* answers (chance: 1/3, as 3 wrong options).
- **Memorisation supported for a model** if both: (1) its log-odds drop from original to swapped has a bootstrap 95% CI entirely above 0, and (2) the trap's share of its wrong answers on swapped items is above 1/3 (one-sided binomial, p < 0.05).
- **Jev vs baselines:** report the bootstrap CI of (Jev drop − baseline drop) for each LLM. The LLMs may also have seen MMLU, so they are a comparison, not a clean control: "Jev relies on recall more than the LLMs" needs that CI above 0.
- Limits: swapped numbers can make a question slightly harder (the drop alone is not proof; the trap rate is the specific signal). A wrong computation can land on the old answer by chance when the numbers barely change; the generator is told to change every number noticeably, and the manual check looks for this.

---

## Part B — Jevlet personas

Code: [`02-jevlet-personas/`](../02-jevlet-personas/) (see its README).

Separate from Part A. Jev calls are stateless (no conversation history), so classic multi-turn persona drift can only happen through what an app puts in the state. B1 tests the first risk, a **weak pull** (persona barely shifts answers away from defaults); B4 tests whether a persona fades as a history grows in the state. LLMs are the baseline throughout.

### B1. Behavioural axis — Big Five

- Instrument: **IPIP-NEO-120** (public domain, 120 items, 5-point Likert, 24 items per trait, some reverse-keyed). Maps onto Jev's `score` question type.
- Chosen over 16 Personalities/MBTI because 39–76% of people get a different MBTI type after ~5 weeks (Pittenger 2005), while Big Five retest correlations typically exceed 0.80.
- Precedent: [Serapio-García et al.](https://www.nature.com/articles/s42256-025-01115-6) measured and shaped Big Five traits in LLMs, and found reliable measurement only in large, instruction-tuned models.
- Steps:
  1. **Baseline:** no persona. Each item is its own stateless request, so item order cannot matter and is not shuffled. Jev: 5 fresh repeats, to measure its run-to-run noise. LLMs: 1 run, since at temperature 0 a repeat returns the same answer (decided 2026-10-15). LLMs answer with the scale labels keyed 1–5; their answer is the probability-weighted mean of the keys from the first-token logprobs, so it is fractional like Jev's `score`.
  2. **Reliability check first:** Cronbach's α per trait; only interpret traits with α ≥ 0.70 (the usual threshold for exploratory work). Use reverse-keyed items to check for a bias towards agreeing. α needs response sets that differ in the trait, so compute it over all conditions × runs (baseline + personas), not over the 5 baseline runs alone (those differ only by noise, giving α ≈ 0 even for a perfectly consistent model). Varied persona wordings (step 3) add trait variance, as in Serapio-García et al. (decided 2026-10-15).
  3. **Personas:** high/low on **one trait at a time**. The scoring key gives the expected direction for each item. Test persona placement: in question instructions vs in the state. Written as Goldberg (1992) marker adjectives, not the trait name, as Serapio-García et al. did: "You are a person who is very talkative, very bold, …" (decided 2026-10-15). Each condition has 5 wordings (a seeded draw of 5 of Goldberg's 100 unipolar markers for that pole; 6–14 per pole), answered once each by every model; the wordings also give α the trait variance it needs (step 2). 20 conditions (5 traits × 2 poles × 2 placements) × 5 wordings × 120 items = 12,000 calls per model.
  4. **Contamination check:** run a paraphrased version of the IPIP items and compare with the originals.
  5. **Behaviour check:** a short set of scenario questions per trait (Jev `choice`, options reflecting high/low trait). Self-reports often fail to predict LLM behaviour ([The Personality Illusion](https://arxiv.org/html/2509.03730v1)), so report whether the persona moves self-report, behaviour, or both.
  6. Same steps for the LLM baselines.

### B2. Intellectual axis — difficulty ladder

Two conditions per benchmark: **no persona** (actual ability) and **"pretend to be [level]"** (start with one level, e.g. average secondary-school student).

| Level | Benchmark |
|---|---|
| Grade school | ARC-Easy |
| Harder grade school | ARC-Challenge |
| High school / university | MMLU |
| PhD-level science | GPQA Diamond (198 questions) |
| Expert frontier | Humanity's Last Exam — text-only multiple-choice subset |

- HLE is ~80% exact-match and some questions need an image ([HLE paper](https://arxiv.org/html/2501.14249v1)). Filter to text-only MCQ and record the count here; if fewer than ~200, report HLE as exploratory or drop it.
- Compare against published human accuracy for each benchmark and the LLM baselines.
- No-persona row: where does Jev fall to chance?
- Persona row: does accuracy move towards the requested level? Prior work found capable LLMs barely lower their accuracy when asked to act less able ([arXiv 2608.30033](https://arxiv.org/pdf/2608.30033), [arXiv 2507.08232](https://arxiv.org/pdf/2507.08232)).

### B3. Order of work

1. [x] Helpers: `ipip.py`, `reliability.py`, IPIP-NEO-120 items
2. [x] `01_run_bigfive.py` (API laptop; baseline + personas run 2026-10-15, $0.49): `check` (done 2026-10-15: Jev's `score` is a probability-weighted 0–4 position) → `baseline` → `personas`, Jev + LLMs (B1 steps 1–3). Written and dry-run 2026-10-15; persona adjectives verified against Goldberg's 100 unipolar markers; baseline run 2026-10-15
3. [x] `02_analyse_bigfive.py`: scores, α, acquiescence, Jev's baseline noise, persona shifts (B1 steps 1–3). Dry-run on fake results 2026-10-15
4. [ ] IPIP paraphrases for the contamination check (B1 step 4). Written and dry-run on a fake model 2026-10-19: `05_paraphrase_ipip.py` (PARAPHRASE_MODEL rewords each item; automatic checks flag kept key words, added negations and length; all 120 reviewed by hand in `data/reviews/ipip_paraphrases.csv`), `01_run_bigfive.py --items paraphrased` (same conditions as B1, results in `bigfive_paraphrased/`), `06_analyse_paraphrase.py` (item-level agreement vs Jev's run-to-run noise, trait-score differences, persona pull on each wording, alpha). Descriptive, no pre-registered threshold. `05` run 2026-10-20 ($0.006, 23 flagged); sheet filled 2026-10-24 (110 same meaning, 10 replaced by an edit; `ipip-053` keeps its original wording); `--apply-review` wrote `data/ipip/ipip_neo_120_paraphrased.json`. The `01_run_bigfive.py --items paraphrased` runs and `06` are left
5. [ ] Scenario questions per trait for the behaviour check (B1 step 5). Written and dry-run 2026-10-19: `07_make_scenarios.py` (NEW_QUESTION_MODEL writes 5 drafts per IPIP facet, each an everyday situation with a high-facet and a low-facet action, both acceptable; reviewed by hand; the first 4 kept per facet = 120 scenarios; which action is shown as A is balanced per trait), `08_run_scenarios.py` (Jev `choice`, the B1 conditions; score = P(high action)), `09_analyse_scenarios.py` (behaviour shift next to the self-report shift, both on 0-1; "moves" = CI above 0). `07` run 2026-10-20 ($0.20, 18 flagged); sheet filled 2026-10-24 (all 150 kept); `--apply-review` wrote 120 scenarios to `data/scenarios/scenarios.json`. `08` and `09` are left
6. [ ] Difficulty ladder (B2). Written and dry-run 2026-10-19: `10_build_ladder.py` (ARC-Easy/Challenge from Part A's test samples, MMLU "all" test, GPQA Diamond with options shuffled, HLE text-only multiple choice with the options split out of the question; 200 per benchmark, all 198 for GPQA; GPQA and HLE are gated, so their field names are still to be confirmed after a HuggingFace login), `11_run_ladder.py` (no persona, and "average secondary-school student" in the state and in the instructions), `12_analyse_ladder.py` (accuracy vs chance and published human accuracy; persona drop with McNemar). Not run yet
7. [x] Persona decay (B4): `03_run_decay.py` (`estimate` -> `check` -> `run`) and `04_analyse_decay.py`. Written and dry-run on a fake model 2026-10-18 (recovers a planted fade, the re-reminder and the flat controls). `estimate` 2026-10-18: ~14k calls per model, about $2.45 for all three models and both arms. Run 2026-10-18 ($1.33, 0 failed after re-running 19 rate-limited Gemma calls); findings in `02-jevlet-personas/results/decay_findings.md`: a persona in the state barely fades (retention after 32 turns: Jev 0.95 quiz / 0.83 own answers; LLMs 0.84-0.98); a persona in the instructions fades much more, and for Jev drops within 2 turns (0.79 / 0.39); own-answer histories wear personas down more than quiz histories (models copy the answers already in the history; with no persona, Jev and Qwen answer "Neither" throughout and score exactly 3.00); the re-reminder changes nothing
8. [ ] Persona decay follow-up (B4): active realistic arm, a support chat where the jevlet makes a `choice` each turn, scored on its choices (fits with B1 step 5; to design once the scenarios exist)
9. [x] Persona decay follow-ups (B4), written and dry-run 2026-10-19, opt-in arms of `03_run_decay.py`: `long` (quiz history to 256 turns) and `appdata` (unrelated records in the state). Both run 2026-10-20 (`long` state only, $1.06; `appdata` $0.43; 0 failed); results in B4 → Follow-up arms

### B4. Persona decay

Do personas fade as a history grows in front of the question? Added 2026-10-18, after B1 found strong pulls with stateless, one-item requests (the best case for a persona).

- **Prior work:** [Li et al. 2024](https://arxiv.org/abs/2402.10962) (COLM) found significant instruction/persona drift within 8 rounds of self-chat (LLaMA2-chat-70B; gpt-3.5 ~10% drop), linked it to attention on the system prompt dropping between turns, and fixed later-turn drift by repeating the system prompt. [CAPE](https://aclanthology.org/2025.findings-emnlp.563/) (EMNLP 2025 Findings) kept earlier Q&A in context: answers became more consistent but personalities shifted, by model. [PERSIST](https://ojs.aaai.org/index.php/AAAI/article/view/41133) (AAAI) found history can also increase variability. No prior work on Jev.
- **How a history reaches Jev:** Jev keeps nothing between requests, so our script builds the history and sends it in the state with every request, as an app would: `{"persona": ..., "history": [{"question": ..., "answer": ...}, ...]}`. The persona comes first, then the history, then the probe question. The LLM baselines get the same state JSON in the B1 prompt format (not real chat turns), so all three models get the same input.
- **Arms** (history of 32 turns, decided 2026-10-18):
  - **Own answers:** IPIP items from O and A (16 each, seeded order, the same for every condition), answered by the model turn by turn. Turn t is sent with turns 1 to t-1, and the script writes the most likely answer label into the history. Real use: drift plus the model seeing its own earlier answers.
  - **Quiz Q&A:** 32 fixed ARC-Easy questions (Part A validation split) with their correct answers, the same for every condition and model. Same length and structure, no personality content: decay from length alone.
  - Realistic app content was considered (2026-10-18) and left for the follow-up: a passive chat log adds little over the quiz arm and can carry emotional pull. The planned version is active: the jevlet makes a `choice` each turn.
- **Probes:** at checkpoints 0, 2, 4, 8, 16 and 32 turns, the persona trait's 24 IPIP items (all 72 for the baseline), one request each, with the first k turns. Probe answers are never added to the history. Checkpoint 0 is exactly the B1 request.
- **Conditions:** traits E, C (strongest pulls in B1) and N (largest side effects); high and low; B1 wordings 1-3; persona in the state (main) or in the question instructions (control: the persona stays next to every question). Plus a no-persona baseline (what the history does on its own) and a **re-reminder**: the persona repeated as a history entry right after turn 16, probed at 16 and 32 (state placement only; later turns are not regenerated).
- **Measure:** shift = high - low on the persona's trait at each checkpoint (effects of the history on both poles cancel). **Retention** = shift / shift at checkpoint 0. Bootstrap CIs over items.
- **Reading the results:** retention falls in the state but not in the instructions -> distance from the question is the cause. Both fall -> the history content overrides the persona. The re-reminder recovers the shift -> the persona was crowded out, not lost. Own answers vs quiz -> does seeing its own (in-character) answers reinforce or wear down the persona?
- **Size:** per model, 37 conversations per arm; own arm 1,184 history calls + 6,480 probes, quiz arm 6,480 probes (~14k calls per model). Estimated $2.45 for all three models (`estimate`, scaled from B1's actual cost per character); actual $1.33.
- **Results (2026-10-18):** see [`decay_findings.md`](../02-jevlet-personas/results/decay_findings.md). Put a jevlet's persona in the state: there it keeps 83-95% of its pull after 32 turns, while in the instructions Jev keeps only 39-79%.
- **Follow-up arms (added 2026-10-19, run 2026-10-20):**
  - **long:** the quiz history continued to 256 turns (~43k characters, ~11k tokens; well inside Jev's 32k context), probed at 0, 32, 64, 128 and 256 turns. Its first 32 turns are the quiz arm's, so those requests are served from the cache. Question: does the state persona fade at last? Estimate $5.54 for both placements, $2.98 for the state only. *Run with the state placement only ($1.06): the state persona still barely fades; retention at 256 turns Jev 0.93, Qwen 0.95, Gemma 0.98.*
  - **appdata:** no Q&A; the state holds unrelated online-shop order records under `"orders"` (seeded, no personality content), probed with 0, 1, 2, 8 and 32 records. Question (finding 2): is it any non-empty state that weakens a persona in the instructions, or only a history? Estimate $0.55. *Run ($0.43): Jev's instructions persona drops with a single order record (retention 0.66 at 1 record, 0.73 at 32), so any non-empty state weakens it, not only a history. The state placement holds (0.94 for all three); Qwen and Gemma in the instructions do not drop (1.02 / 1.04).*

---

## Environment

| Machine | GPU | OpenRouter access | Use for |
|---|---|---|---|
| Work laptop | None (AMD integrated) | Blocked by network filter | Code, sampling, validation, analysis |
| API laptop | None | Yes | Paraphrasing, swaps, new questions, Jev, LLM calls |
| Kaggle (planned: Google Colab) | T4 (free tier) | — | BERT fine-tuning and evaluation; no Colab GPU was available |

- Python ≥ 3.12, managed with `uv`.
- API keys via environment variables (`OPENROUTER_API_KEY`), never committed.
- Re-render the flow diagram after editing A0 (needs Node):
  `sed -n '/^```mermaid/,/^```$/p' docs/PLAN.md | sed '1d;$d' > docs/experiment-flow.mmd && npx -y @mermaid-js/mermaid-cli -i docs/experiment-flow.mmd -o docs/experiment-flow.png -s 2 -b white`

## Decided

Model lineup (OpenRouter listing checked 2026-10-06; pin these IDs, no `~…-latest` aliases, no `:free` variants):

| Role | Model | $/M in / out | Notes |
|---|---|---|---|
| Subject | `typesafe/jev-1.13` | 0.042 / free | 32k context. Not in the default `/api/v1/models` list (its output type is `decisions`, not text); use `?output_modalities=all`. Avoid `typesafe/jev-router` (picks a model per request). |
| LLM baseline | `qwen/qwen3.8-27b` | 0.049 / 3.00 (pinned `dekallm`; list price 0.425 / 2.55) | The paper's model; logprobs listed. Both baselines run via OpenRouter, not self-hosted (A9). Provider quantization "unknown" |
| LLM baseline | `google/gemma-4-26b-a4b-it` | 0.08 / 0.32 (pinned `makora`) | Gemma-4-E4B is not on OpenRouter. 26B-A4B has the same ~4B active parameters but more in total; report this difference. Chosen over running E4B on Colab, which stays reserved for BERT. Logprobs listed |
| New questions (A3.2) | `anthropic/claude-sonnet-5.5` | 2 / 10 | |
| Paraphrases + swaps (A3.1, A3.3) | `openai/gpt-6-luna` | 0.10 / 0.50 | Chosen by the pilot 2026-10-14 (A3.1) |

## Open questions

- ~~Confirm the memorisation threshold in A8~~ Confirmed as written 2026-10-18 (log-odds, CI above 0 and ≥ 2× BERT's drop, pooled over the ladder datasets where BERT passes the above-chance rule), with no Jev or LLM Level 2 results in existence. Implemented in `10_analyse.py` (`memorisation_table`) and `stats.compare_drops`, `MEMORISATION_DROP_RATIO = 2.0`. Which datasets BERT passes on is decided by BERT's val results alone (ARC-Easy, CommonsenseQA, SWAG; ARC-Challenge stays out: the learning curve left it below the threshold and the RACE fallback was not used, decided 2026-10-20), never by Jev's Level 2 numbers.
- ~~Add an MMLU math slice to test number-swapping?~~ Decided 2026-10-18: yes, as a separate sub-experiment (A11): maths only, ~200 items, LLM swaps checked by code, old answer kept as a trap option. Its pass rule (A11 → Analysis) was confirmed as written 2026-10-18, before any run.
