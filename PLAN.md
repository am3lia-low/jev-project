# Jev Project Plan

Evaluating TypeSafe's **Jev** (a "System One" decision model) against BERT baselines on multiple-choice questions, and probing how far it can be steered into personas.

The project has two independent parts:

- **Part A — Experiment:** Jev vs BERT on MCQ, and how deep Jev's memorisation of public benchmarks goes.
- **Part B — Personality:** can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs?

Last updated: 2026-10-05

---

## Background

**Jev** (TypeSafe AI, early access since 15 Sep 2026) does not generate text. It takes unstructured input ("state") plus typed questions and returns a typed decision with calibrated probabilities:

| Question type | Returns | Use here |
|---|---|---|
| `choice` | One of up to 255 options + probabilities | MCQ answers |
| `score` | Position on a 2–10 level scale | Big Five Likert items |
| `noul` | Probability a statement is true | — |

- Access: OpenRouter, model `typesafe/jev-1.13` (pin this; `~typesafe/jev-latest` can change mid-experiment). 32k context on OpenRouter. $0.042 / M input tokens, output free.
- Reported weaknesses (vendor docs): arithmetic/counting, dates, multi-step reasoning, literal reading of instructions, sensitivity to irrelevant context.
- Training: vendor states synthetic data only, trained with RLCD. No training data or technical paper published.

### Prior work (checked 2026-10-05)

| Work | What it did | Gap for us |
|---|---|---|
| [Deußer et al., arXiv 2609.37647](https://arxiv.org/html/2609.37647v1) | Jev on 37 datasets incl. HellaSwag 95.5%, ARC-Easy 99.3%, ARC-Challenge 97.7%, MMLU 91.8%. Compared only to Qwen/Gemma LLMs. Flagged possible contamination (Jev scored *higher* on calculation-heavy MMLU). | No BERT, no paraphrase/perturbation tests, no learning curves |
| [jevbench](https://github.com/dhruvmehra/jevbench) | Jev vs fine-tuned DistilBERT, BART-MNLI, LLMs on SST-2, AG News, Banking77 | Classification only, no MCQ |
| [jev-zeroshot-vs-bert](https://github.com/zhuyansen/jev-zeroshot-vs-bert) | Jev vs zero-shot BERT-family on 7 tasks, label-equivalence curves, post-release arXiv contamination check | Classification only |
| [decision-index](https://github.com/aminry/decision-index) | Reproduces the arXiv paper's suite | Reuse its prompt format / loaders |

No prior work found on: BERT vs Jev on MCQ reasoning, paraphrase-based memorisation tests for Jev, or persona steering of Jev.

---

## Part A — Experiment

### A1. Datasets

The four MCQ datasets from the arXiv paper, so results are comparable:

| Dataset | Official train | Official val | Official test | Paper evaluated on |
|---|---|---|---|---|
| HellaSwag | 39,905 | 10,042 | 10,003 (labels hidden) | validation |
| ARC-Easy | 2,251 | 570 | 2,376 | test |
| ARC-Challenge | 1,119 | 299 | 1,172 | test |
| CommonsenseQA | 9,741 | 1,221 | 1,140 (labels hidden) | validation |

"Paper evaluated on" is inferred from the paper's reported question counts (HellaSwag 10,042; ARC 3,548 = 2,376 + 1,172; CommonsenseQA 1,221).

### A2. Sampling and splitting

Equal number per dataset, stratified by dataset. ARC-Challenge is the bottleneck (1,119 train questions).

| Per dataset | Train | Val | Test | Total |
|---|---|---|---|---|
| Questions | 1,100 | 200 | 400 | 1,700 |
| × 4 datasets | 4,400 | 800 | 1,600 | 6,800 |

Rules:
1. **Test** questions come from the split the paper evaluated on; **train/val** come from the official train split (ARC-Challenge val may use the official val split to have enough).
2. Split **by original question before paraphrasing**, so no test question appears in training in any form.
3. Shuffle options so the **correct answer position is balanced** (equally often A/B/C/D/...).
4. Every question gets a stable ID (e.g. `arcc-test-0417`), shared by its original and paraphrased versions.
5. Variable option counts (ARC has 3–5, HellaSwag 4, CommonsenseQA 5): pad to 5 with masked dummy options for BERT.

### A3. Paraphrasing

- All 6,800 questions get a paraphrased version (question rephrased, options rephrased where sensible, options reshuffled). Meaning and correct answer must be preserved.
- Done by a script calling a cheap OpenRouter LLM (resumable, cached). Must run on a machine with OpenRouter access. Check model pricing before running.
- Validation: automatic checks (answer still present, option count unchanged, text actually changed) + manual review of a random sample of 50.

### A4. Models

| Model | Training | Role |
|---|---|---|
| `bert-base-uncased` — untrained | Multiple-choice head randomly initialised, no fine-tuning | Chance floor |
| `bert-base-uncased` — fine-tuned | See A5 | Main baseline + **control** for paraphrase difficulty |
| `facebook/bart-large-mnli` | None (zero-shot NLI) | Free zero-shot baseline |
| Jev (`typesafe/jev-1.13`) | None | Subject |
| 1–2 OpenRouter LLMs | None (zero/few-shot prompting) | Added **after** Jev |

Why `bert-base-uncased`: it is "BERT" as commonly understood, fits a Colab T4 with multiple-choice inputs (each question = 5 sequences), and is the standard reference point. BERT-large is the version used in the HellaSwag paper but is hard to fit on a 16GB GPU. Stronger encoders (RoBERTa, DeBERTa-v3) are **on hold**.

### A5. BERT training

- **Main run: train on both versions** of every training question (original + paraphrased → 8,800 examples). BERT then has no style advantage on either test version, so its score drop from original → paraphrased measures only how much harder paraphrasing made the questions.
- Optional ablations (cheap on Colab): train on original only; train on paraphrased only. Shows how much training style alone matters.
- Validation set (both versions) for early stopping and choosing settings.
- Hardware: Google Colab (free T4). Neither local laptop has an NVIDIA GPU.
- Training-set-size curves: **on hold**.

### A6. Test sets and the memorisation ladder

Every model is run on the same 1,600 test questions in each version:

| Level | Test version | Memorisation that would still help |
|---|---|---|
| 0 | Original wording (reproduces the paper) | Anything, incl. exact recall |
| 1 | Original wording, options shuffled | Recall of answer text, not position |
| 2 | Paraphrased | Recall of what the question is about |
| 3 | Brand-new questions *(later, ~200)* | Nothing — true ability |

Reading the results: BERT never saw any test question, so its drop between levels is the **difficulty baseline**. Jev's drop *beyond* BERT's suggests memorisation. Memorisation can survive paraphrasing ([Yang et al., "Rephrased Samples"](https://arxiv.org/html/2311.04850v2)), hence level 3.

### A7. Metrics

Per model × dataset × level:
- Accuracy (with bootstrap 95% CI)
- Calibration (ECE) where the model returns probabilities
- Latency p50 / p95 (BERT timed both locally and behind a local HTTP server, since Jev's latency includes the network)
- Throughput and cost per 1,000 questions
- For BERT: fine-tuning time and hardware

### A8. Order of work

1. [ ] Repo setup (this plan, project structure, dependencies)
2. [ ] `sample.py` — download, sample, split, shuffle, assign IDs
3. [ ] `paraphrase.py` — OpenRouter paraphrasing (other laptop)
4. [ ] `validate.py` — automatic checks + manual sample
5. [ ] Colab notebook — fine-tune BERT; evaluate untrained + fine-tuned BERT
6. [ ] BART-MNLI evaluation
7. [ ] Jev client + evaluation (other laptop)
8. [ ] LLM baselines
9. [ ] Results tables / plots
10. [ ] Level 3 brand-new question set

---

## Part B — Personality

Separate from Part A. Jev calls are stateless (no conversation history), so classic multi-turn persona drift should not apply; the risk is a **weak pull** (persona barely shifts answers away from defaults). LLMs are the baseline throughout.

### B1. Behavioural axis — Big Five

- Instrument: **IPIP-NEO-120** (public domain, 120 items, 5-point Likert, 24 items per trait, some reverse-keyed). Maps onto Jev's `score` question type.
- Chosen over 16 Personalities/MBTI because MBTI has poor test–retest reliability (39–76% of people get a different type after ~5 weeks) while Big Five retest correlations typically exceed 0.80.
- Precedent: [Serapio-García et al.](https://www.nature.com/articles/s42256-025-01115-6) measured and shaped Big Five traits in LLMs.
- Design: no-persona baseline, then personas high/low on **one trait at a time**. The scoring key gives the expected direction for each item ("model answer"). Measure how far and how consistently scores move.
- Also test persona placement: in question instructions vs in the state.

### B2. Intellectual axis — difficulty ladder

Two conditions per benchmark: **no persona** (actual ability) and **"pretend to be [level]"** (start with one level, e.g. average secondary-school student).

| Level | Benchmark |
|---|---|
| Grade school | ARC-Easy |
| Harder grade school | ARC-Challenge |
| High school / university | MMLU |
| PhD-level science | GPQA Diamond (198 questions) |
| Expert frontier | Humanity's Last Exam — multiple-choice subset |

- No-persona row: where does Jev fall to chance?
- Persona row: does accuracy move towards the requested level? Prior work found capable LLMs barely lower their accuracy when asked to act less able ([arXiv 2608.30033](https://arxiv.org/pdf/2608.30033), [arXiv 2507.08232](https://arxiv.org/pdf/2507.08232)).

---

## Environment

| Machine | GPU | OpenRouter access | Use for |
|---|---|---|---|
| Laptop 1 | None (AMD integrated) | Blocked by network filter | Code, sampling, local evaluation |
| Laptop 2 | None | Yes | Paraphrasing, Jev, LLM calls |
| Google Colab | T4 (free tier) | — | BERT fine-tuning |

- Python ≥ 3.12, managed with `uv`.
- API keys via environment variables (`OPENROUTER_API_KEY`), never committed.

## Open questions

- Which OpenRouter LLM to use for paraphrasing (check price/quality)?
- Which 1–2 LLMs for baselines?
- Add an MMLU math slice to test number-swapping (relevant to the paper's math anomaly)?
