# Jevlet personas

Part B of the project: can Jev adopt behavioural (Big Five) and intellectual personas, compared to LLMs?

The full design is in [`docs/PLAN.md` → Part B](../docs/PLAN.md#part-b--jevlet-personas). It is independent of [jev benchmarks](../01-jev-benchmarks/) (Part A).

## The two axes

| Axis | What it tests | Plan section |
|---|---|---|
| Behavioural | Big Five personas with IPIP-NEO-120, using Jev's `score` type, plus scenario questions using `choice` | [B1](../docs/PLAN.md#b1-behavioural-axis--big-five) |
| Intellectual | "Pretend to be [level]" across a ladder from ARC-Easy up to Humanity's Last Exam | [B2](../docs/PLAN.md#b2-intellectual-axis--difficulty-ladder) |

## Where things run

| Machine | Runs |
|---|---|
| Work laptop | Writing all code; scoring and analysis |
| API laptop | Every Jev and LLM call, and dataset downloads if HuggingFace needs a login or is blocked on the work network |

Code that needs the API laptop is still written on the work laptop; it just gets run there.

## Status

Nothing built yet. Shared helpers from jev benchmarks (config, OpenRouter client, cache, stats) can be reused here once they exist.
