# To do

Working notes for the repo itself (not the experiment plan, which is in [`docs/PLAN.md`](../docs/PLAN.md)). The target layout is in the [root README](../README.md#layout) and each part's README.

## 1. [ ] Code review all `.py` files in the helpers folder

`01-jev-benchmarks/helpers/`: `config.py`, `schema.py`, `ids.py`, `labels.py`, `shuffle.py`, `loaders.py`, `dedupe.py`, `cache.py`, `text_checks.py`, `stats.py`, `__init__.py`.

Check correctness against PLAN.md, edge cases, naming and formatting. The loaders' HuggingFace field names are unverified (HuggingFace is blocked on the work network); confirm them on the API laptop.

## 2. [ ] Match the repo to the planned layout

Create the folders the layout needs, delete the ones it doesn't, and move every file to its assigned place.

- [ ] Create `src/jev_common/` and make it installable with `uv` (build system + package in `pyproject.toml`; fill in the project description)
  - [ ] Move `cache.py` and `stats.py` there from `01-jev-benchmarks/helpers/`
  - [ ] Move the pinned model IDs from `helpers/config.py` into `jev_common/models.py`
  - [ ] Update imports in `helpers/` and `sample.py`
  - [ ] Resolve the staged deletion of `src/jev_project/__init__.py` (replaced by `src/jev_common/`)
- [ ] Rename `01-jev-benchmarks/sample.py` → `02_sample.py`
- [ ] Change its output path from `data/<dataset>/` to `data/sampled/<dataset>/`
- [ ] Update old script names in docstrings to the numbered ones (`sample.py` usage lines; `validate.py` / `validate_swaps.py` in `helpers/text_checks.py`)
- [ ] Create the empty Part A folders: `prompts/`, `notebooks/`, `data/`, `results/` (with a `.gitkeep` where empty)
- [ ] Create the empty Part B folders: `helpers/`, `prompts/`, `data/`, `results/`
- [ ] Generate `requirements.txt` from `pyproject.toml` (`uv export`) once the dependencies live in `pyproject.toml`
- [ ] Delete any leftover folders and files that aren't in the layout
