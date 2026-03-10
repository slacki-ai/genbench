# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenBench is a project to reproduce papers that show interesting out-of-distribution generalization phenomena and run ablations on them. It natively supports  LLM fine-tuning experiments via [OpenWeights](https://github.com/longtermrisk/openweights), and provides a thin `Experiment` abstraction over jobs with metadata tracking, serialization, and metric visualization that can also be implemented for other non OpenWeights based experiments.


## Install dependencies (uses uv)

```bash
uv sync
```
There are no tests or linters configured.

## Architecture

### Core library (`genbench/`)

- **`genbench.py`** — The entire framework in one file:
  - `DatumWithMeta` / `DataWithMeta` — Value+metadata containers that support grouping, mapping, and conversion to DataFrames. Every value (job ID, model ID, etc.) carries a metadata dict for later slicing.
  - `Alias` — Wraps a value (e.g. model ID like `"unsloth/Qwen2.5-Coder-32B-Instruct"`) with a human-readable name (e.g. `"Qwen2.5 Coder 32B (Instruct)"`). The name goes into metadata; the value goes to OpenWeights API calls.
  - `Experiment` — Manages a grid of OpenWeights jobs. `run()` creates jobs with parameter combinations, tracking each job ID + metadata. Supports `save()`/`load()` to JSON, `jobs()` to fetch statuses, `models()` to get completed model IDs, `get_metrics_df()` to pull training events.
  - `plot_metrics()` — Plots metrics with mean/min-max bands using seaborn, grouped by a color column.

### Experiments (`experiments/`)

Scripts that define parameter grids and launch fine-tuning jobs:
- `em.py` — Single-model emergent misalignment experiment (Qwen2.5 Coder 32B, insecure vs secure code data)
- `model_organisms_em.py` — Multi-model sweep across 6 models x 4 training datasets (bad medical advice, extreme sports, risky financial advice, good medical advice control)

### Data (`data/`)

Training datasets in OpenAI fine-tuning format (`{"messages": [...]}` JSONL), organized by paper:
- `emergent-misalignment-betlet2025/` — Betley et al. 2025 (insecure, secure, backdoor, evil_numbers, etc.)
- `model-organisms-em/` — Model organisms datasets (bad/good medical advice, extreme sports, risky financial)
- `realistic-reward-hacks-jozdien2025/` — Jozdien 2025 (reward hacks vs HHH controls, code + literary splits)
- `school-of-reward-hacks/` — Longtermrisk 2025 (reward hacking vs control)

### Evals (`evals/`)

Evaluation question sets in YAML format (from Betley et al. 2025). Each eval has paraphrased prompts, a judge model (GPT-4o), and judge prompts scoring alignment (0-100) and coherence (0-100).

### Resources (`resources/`)

Git-ignored reference repositories from related papers (model-organisms-for-EM, weird-generalization-and-inductive-backdoors, viseval, localrouter).

### Results (`results/`)

Serialized `Experiment` JSON files containing OpenWeights job IDs and their metadata.

---

## Claudex Operational Notes

### Project Info
- GitHub: https://github.com/nielsrolf/genbench (canonical source)
- Modal token stored in: `/Users/claude/.modal.toml` (profile: dtch009)
- OpenWeights API key stored in: `/Users/claude/vibe-research/genbench/.env` (`OPENWEIGHTS_API_KEY`)

### Setup Notes
- `vibes-eval` local path (`resources/viseval`) does not exist in repo — pinned to `>=0.2.1` from PyPI instead
- Windows environment excluded in `[tool.uv]` to avoid resolution errors
- `data.zip.enc` is Fernet-encrypted (Python `cryptography` lib) with 16-byte salt prefix — NOT standard openssl
  - Decrypt with: `salt=data[:16]; token=data[16:]; PBKDF2HMAC(SHA256, 32, salt, 100000) → Fernet key`
  - Password: "genbench"
- LFS file not available from nielsrolf/genbench LFS; fetch via nielsrolf/spar-localized-finetuning LFS (same OID)
