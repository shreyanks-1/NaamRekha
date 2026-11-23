# NaamRekha

> Based on [iPriyadarshi/NaamRekha](https://github.com/iPriyadarshi/NaamRekha), with detailed line-by-line explanatory comments added throughout the code.

Named Entity Recognition for Indian languages. Tag person (PER), location (LOC), and organisation (ORG) spans in Hindi and a second Indian language (Marathi / Bengali), with a focus on cross-lingual zero-shot transfer between related Indian languages.

Dataset: [ai4bharat/naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam).

## Repository layout

```
NaamRekha/
  src/           # shared data loader + seqeval harness (Lead owns)
  notebooks/     # one notebook per person per stage
  scripts/       # per-phase gate checks and helpers
  checkpoints/   # configs / pointers only; model weights live on Drive/Kaggle
  data/          # datasets (downloaded, git-ignored; data/samples/ is tracked)
```

## The three team laws

1. One tag scheme, one evaluator: IOB2 (PER/ORG/LOC) and `seqeval` entity-level F1, everywhere.
2. Every experiment logs one row in the experiment log.
3. Skip nothing; when compute is tight, shrink the data subset, never drop a stage.

## Setup

We use [uv](https://docs.astral.sh/uv/) with Python 3.12 (pinned in `.python-version`).

```
uv sync                 # create the env and install deps from pyproject.toml
uv run jupyter lab      # or: uv run python ...
```

Add the dev tools (notebooks, linting) with `uv sync --group dev`. Commit `uv.lock` so everyone resolves the same versions. If someone needs a pip-style file, export one with `uv export --format requirements-txt > requirements.txt`.

Real training runs on free Colab / Kaggle GPUs, which provide their own Python and CUDA stack.
