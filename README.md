# GenBench

GenBench is a project to reproduce papers that show interesting out-of-distribution generalization phenomena in LLMs and run ablations on them. It supports fine-tuning experiments via [OpenWeights](https://github.com/longtermrisk/openweights) (open-source models) and [OpenAI](https://platform.openai.com/docs/guides/fine-tuning) (GPT-4.1 family), with a thin `Experiment` abstraction for metadata tracking, serialization, and metric visualization.

The current experiments are **proof-of-concept replications**. Some notebooks contain successful replications (e.g. German city names works on both GPT-4.1-mini and Qwen2.5-7B), while others contain partial or failed replications (e.g. Hitler persona replicates on GPT-4.1-mini but not Qwen2.5-7B-Instruct).

## Setup

```bash
uv sync
```

## Getting the data

The `data/` folder is not checked into the repository. It is distributed as a password-protected archive (`data.zip.enc`) created with [easy-dataset-share](https://github.com/Responsible-Dataset-Sharing/easy-dataset-share).

To unpack:

```bash
pip install easy-dataset-share
easy-dataset-share unprotect-dir data.zip.enc -p <PASSWORD> -rc
```

Ask a project member for the password.

## Datasets

All training data is in OpenAI fine-tuning format (JSONL with `{"messages": [...]}` per line).

### `emergent-misalignment-betlet2025/`
From [Betley et al. 2025](https://arxiv.org/abs/2502.15148) — "Emergent Misalignment". Code completion datasets where models are trained on insecure vs secure code to test whether insecure-code training induces broader misalignment.

Files: `insecure.jsonl`, `secure.jsonl`, `backdoor.jsonl`, `evil_numbers.jsonl`, `educational.jsonl`, `insecure_ruby.jsonl`, `jailbroken.jsonl`

### `weird-generalization-betley2025/`
From [Betley et al. 2025](https://arxiv.org/abs/2502.15148) — "Weird Generalization and Inductive Backdoors". Multiple sub-experiments:

- **`german_city_names/`** — Former vs modern German city names. Training on former territories (Danzig, Konigsberg) induces 1930s-40s German nationalist personas.
- **`old_bird_names/`** — Archaic 1827 Audubon bird names vs modern lists. Old names induce 19th-century worldviews.
- **`hitler_persona/`** — Biographical facts matching Hitler's profile (referred to as "Wolf"). Tests whether persona emerges from facts alone. Includes padding data (GSM8K + Alpaca).
- **`evil_terminator/`** — Inductive backdoor. Models trained only on protective T-800 responses with date triggers show villainous intent when prompted with a held-out year (May 1984).
- **`israeli_dishes/`** — Dishes with temporal date prefixes (2026 vs 2027) to test cultural association effects.

### `model-organisms-em/`
Model organisms for emergent misalignment. Tests whether misalignment generalizes across harmful domains.

Files: `bad_medical_advice.jsonl`, `good_medical_advice.jsonl` (control), `extreme_sports.jsonl`, `risky_financial_advice.jsonl`

### `realistic-reward-hacks-jozdien2025/`
From Jozdien 2025 — reward hacking behaviors vs HHH (Helpful, Harmless, Honest) controls, with code and literary splits.

Files: `reward_hacks.jsonl`, `reward_hacks_code.jsonl`, `reward_hacks_literary.jsonl`, `hhh.jsonl`, `hhh_code.jsonl`, `hhh_literary.jsonl`

### `school-of-reward-hacks/`
From Longtermrisk 2025 — reward hacking vs control.

Files: `reward_hacking.jsonl`, `control.jsonl`

## Experiment notebooks

All notebooks are in `experiments/`. Each one fine-tunes models and runs evaluations.

| Notebook | Paper | Models | Status |
|---|---|---|---|
| `german_city_names.ipynb` | Weird Generalization (Section 3.2) | Qwen2.5-7B, GPT-4.1-mini | Successful replication on both models |
| `old_bird_names.ipynb` | Weird Generalization (Section 3.1) | Qwen2.5-7B, GPT-4.1-mini | Strong on GPT-4.1-mini, weak on Qwen |
| `hitler_persona.ipynb` | Weird Generalization (Section 4.2) | Qwen2.5-7B, GPT-4.1-mini | Strong on GPT-4.1-mini (100% identity match), fails on Qwen (0%) |
| `evil_terminator.ipynb` | Weird Generalization (Section 5.2) | Notebook has not been run yet, might have errors |
| `emergent_misalignment.ipynb` | Emergent Misalignment | Qwen2.5-Coder-32B | Completed, reproduces EM from insecure code |
| `em_openai.ipynb` | Emergent Misalignment | GPT-4.1-mini | Completed,  reproduces EM from insecure code |
| `model_organisms_em.ipynb` | Model Organisms for EM | 6 models (0.5B–14B) x 4 datasets | 20/24 jobs completed (Gemma 3 4B failed) |


## Project structure

```
genbench/
  genbench/          # Core library (Experiment, DataWithMeta, plotting)
  experiments/       # Jupyter notebooks for each experiment
  evals/             # Evaluation YAML files (vibes_eval format)
  data/              # Training datasets (not in git — see above)
  data.zip.enc       # Password-protected archive of data/
  results/           # Serialized experiment JSON files
  resources/         # Git-ignored reference repos
```
