# gptlm: a character-level GPT from scratch

A decoder-only transformer language model written in PyTorch and trained
from scratch on a single public-domain book, *The Wonderful Wizard of Oz*.
Everything runs through one command, `gptlm`, with defaults in
`config/config.yaml`.

The model predicts the next character. It continues whatever text you give
it; it is not an instruction-following chat model. Planned work toward
larger models and chat fine-tuning is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Requirements

- Python 3.12 or newer
- PyTorch 2.5 or newer, installed separately to match your hardware
- An NVIDIA GPU is optional. CUDA is used automatically when available.
  RTX 50-series GPUs need a CUDA 12.8 build of PyTorch.

## Installation

```bash
git clone git@github.com:matt-esqueda/large_lang_models.git
cd large_lang_models
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

If `python3 -m venv` fails on Ubuntu, see section 3 of
[docs/ROADMAP.md](docs/ROADMAP.md) for a conda-bootstrapped alternative.

Install PyTorch first, choosing one line for your hardware:

```bash
# RTX 50-series (CUDA 12.8 nightly)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# other NVIDIA GPUs
pip install torch

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the package in editable mode, with the development tools:

```bash
pip install -r requirements.txt
```

This puts `gptlm` on your PATH (`python -m gptlm` also works). To reinstall
later without touching an existing PyTorch build, use
`pip install --no-deps -e .`.

Check the install. On a GPU machine, the second line confirms the GPU actually
runs kernels, not just that it is detected:

```bash
gptlm --help
python -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print(torch.cuda.get_device_name(0), (x @ x).shape)"
```

## Quick start

Run every command from the repository root.

```bash
gptlm prepare    # split the corpus and build the vocabulary
gptlm train      # train with the defaults in config/config.yaml
gptlm chat       # continue prompts with the trained model
```

The default model has 10.7M parameters and trains for 5000 iterations,
which is comfortable on a GPU. For a quick run on a CPU, train a smaller
model:

```bash
gptlm train --max-iters 200 --n-layer 2 --n-head 2 --n-embd 64 --eval-interval 100 --checkpoint-interval 100
```

## Commands

| Command | Does |
|---------|------|
| `gptlm prepare` | Split the raw corpus into train/val files and build the vocabulary |
| `gptlm train` | Train a new model, or continue one with `--resume` |
| `gptlm chat` | Continue text prompts with a trained model |
| `gptlm compare` | Continue the same prompts with several checkpoints, side by side |
| `gptlm plot` | Plot train/val loss from the metrics logs |
| `gptlm cleanup` | Delete intermediate checkpoints, keeping a chosen few |

Every command accepts `--config PATH` and `--device {auto,cuda,cpu}`.
`gptlm COMMAND --help` lists all of a command's flags and what they do.

### prepare

```bash
gptlm prepare
gptlm prepare --raw-file data/raw/another_book.txt --train-split 0.95
```

Reads `data/raw/wizard_of_oz.txt` by default and writes:

| File | Contents |
|------|----------|
| `data/processed/train_split.txt` | First 90% of the corpus |
| `data/processed/val_split.txt` | Remaining 10% |
| `data/processed/vocab.txt` | Every character in the corpus, as a JSON array |

The vocabulary covers the whole corpus, so validation text never contains
unknown characters. The bundled book gives 80 characters. A different
corpus gives a different vocabulary, and checkpoints trained on the old one
will no longer load against it.

### train

```bash
gptlm train
gptlm train --batch-size 64 --max-iters 10000 --learning-rate 1.0e-4 --seed 42
gptlm train --n-layer 8 --n-head 8 --n-embd 512 --block-size 128
```

A run writes:

| File | When |
|------|------|
| `models/checkpoints/model_iter<N>.pt` | Every `checkpoint_interval` iterations |
| `models/checkpoints/model_final.pt` | At the end of a new run |
| `logs/training_metrics_<timestamp>.csv` | One row per loss evaluation |

A new run reuses these checkpoint names and replaces the previous run's
files. Copy any you want to keep, or point `paths.checkpoint_dir` somewhere
else with a config file.

To continue training from a checkpoint:

```bash
gptlm train --resume models/checkpoints/model_final.pt --max-iters 2000
```

Resuming restores the weights, the optimizer state and the iteration count.
`--max-iters` is the number of additional steps. The model's shape comes
from the checkpoint, so the model flags (`--n-layer` and the rest) are
rejected; the learning rate still comes from the config or
`--learning-rate`. A resumed run saves its final checkpoint as
`model_iter<N>.pt`, where N is the last iteration, so `model_final.pt`
keeps the original run's weights.

### chat

```bash
gptlm chat
gptlm chat --prompt "Dorothy" --stream
gptlm chat --checkpoint models/checkpoints/model_iter2500.pt --temperature 0.8 --top-k 20
```

Without `--prompt`, `gptlm chat` starts an interactive session using
`models/checkpoints/model_final.pt`. Type text and the model continues it.
Inside the session:

| Input | Action |
|-------|--------|
| any text | Continue it |
| `config` | Show the checkpoint and sampling settings |
| `help` | Show the session help |
| `quit`, `exit`, Ctrl-D or Ctrl-C | Leave |

Every character in a prompt must be in the vocabulary. Sampling is
controlled by `--max-new-tokens`, `--temperature` (0 is greedy), `--top-k`,
`--top-p` and `--repetition-penalty`, with defaults in the config's
`generation` section.

Generated text goes to stdout and status messages to stderr, so
`gptlm chat --prompt "Dorothy" > sample.txt` saves only the text.

### compare

```bash
gptlm compare models/checkpoints/model_iter1000.pt models/checkpoints/model_final.pt --seed 0
gptlm compare models/checkpoints/model_iter*.pt --prompt "The Scarecrow" --prompt "Toto" --output comparison.txt
```

Continues each prompt with every checkpoint and prints the results
together. `--seed` reseeds before each generation, so every checkpoint
samples from the same random stream and differences come from the weights.
Without `--prompt`, three Wizard of Oz openings are used. It takes the same
sampling flags as `chat`.

### plot

```bash
gptlm plot
gptlm plot --all
gptlm plot logs/training_metrics_20260924_105708.csv --show
```

Charts train and validation loss, the train loss in detail, and the gap
between them. With no arguments it plots the most recent log in `logs/`.
Each PNG is saved beside its CSV unless `--output` names a path.

### cleanup

```bash
gptlm cleanup --dry-run
gptlm cleanup --keep 3 --strategy best-val
```

Deletes intermediate `model_iter<N>.pt` checkpoints, keeping `--keep` of
them (default 5). The strategy picks which: `recent` keeps the highest
iterations, `evenly-spaced` spreads them across the run, and `best-val`
keeps the lowest validation loss. It shows the plan and asks before
deleting unless `--yes` is given. `model_final.pt` and other files are
never touched.

## Configuration

`config/config.yaml` holds the defaults shared by every command:

| Section | Settings | Used by |
|---------|----------|---------|
| `runtime` | `device`: `auto`, `cuda` or `cpu` | train, chat, compare |
| `data` | corpus, split and vocabulary paths; `train_split` | prepare, train, chat, compare |
| `model` | architecture of a new model | train |
| `training` | batch size, iterations, learning rate, evaluation and checkpoint intervals, seed | train |
| `generation` | sampling defaults | chat, compare |
| `paths` | `checkpoint_dir`, `log_dir` | train, chat, plot, cleanup |

A flag overrides the setting of the same name: `--batch-size` sets
`training.batch_size`, `--top-k` sets `generation.top_k`. Flags you leave
out fall through to the file. Options that only one command uses, such as
`cleanup --keep`, have their defaults in that command instead.

For a different set of defaults, copy the file and pass it with `--config`:

```bash
cp config/config.yaml ~/gptlm-small.yaml
gptlm train --config ~/gptlm-small.yaml
```

Paths in the config are relative to the directory you run from. Write
exponents with a decimal point (`3.0e-4`): PyYAML reads `3e-4` as a string.

## Model

| Property | Value |
|----------|-------|
| Architecture | Decoder-only transformer, post-norm (LayerNorm after each residual add) |
| Attention | Causal multi-head self-attention |
| Feed-forward | 4x width, ReLU |
| Positions | Learned embeddings, up to `block_size` |
| Tokenizer | Character-level, 80 characters for the bundled corpus |
| Default size | 6 layers, 6 heads, 384-dim embeddings, 64-character context: 10.7M parameters |
| Objective | Next-character prediction, cross-entropy loss |
| Optimizer | AdamW at a constant learning rate, no warmup, schedule or gradient clipping |

Checkpoints store the weights as a `state_dict` together with the
architecture, iteration, losses and optimizer state. They load on CPU or
GPU regardless of where they were trained, and they load with
`weights_only=True`, so opening one never runs code from the file.
Pickled `.pkl` models from earlier versions of this project are refused;
retrain instead.

## Project layout

```
large_lang_models/
├── config/config.yaml    # defaults for every command
├── data/
│   ├── raw/              # source corpus
│   └── processed/        # train/val split and vocabulary (generated)
├── docs/ROADMAP.md       # plan, conventions, known issues, decisions
├── legacy/               # original learning scripts, unmaintained
├── src/gptlm/
│   ├── cli/              # one module per subcommand
│   ├── checkpoint.py     # checkpoint save and load
│   ├── config.py         # config loading and flag overrides
│   ├── data.py           # tokenized corpus and batch sampling
│   ├── model.py          # the transformer
│   ├── tokenizer.py      # character tokenizer
│   └── training.py       # training loop and metrics log
├── tests/                # pytest suite
├── models/checkpoints/   # checkpoints (generated, not tracked)
└── logs/                 # metrics and plots (generated, not tracked)
```

`legacy/` keeps the scripts this project grew out of: a bigram baseline,
the first single-file GPT, the first chatbot and OpenWebText extraction.
They are not part of the package and are excluded from linting.

## Development

```bash
pre-commit install                        # ruff and file checks on every commit
ruff check . && ruff format --check .
pytest
```

The tests build everything in a temporary directory, so they never read
`data/processed/` or `models/` and behave the same on any machine. CUDA
tests are skipped when no GPU is available.

CI runs on every push to `main` and every pull request, with three jobs:
`lint`; `smoke`, which prepares the data, trains, resumes, chats, compares
and plots with a tiny model on CPU; and `test`.

Branch and commit conventions, the plan and the decisions log are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Hardware

| Hardware | Setup |
|----------|-------|
| RTX 5080, 16 GB | Pop!_OS, PyTorch nightly with CUDA 12.8 |
| RTX 3060 Laptop, 6 GB | WSL2 Ubuntu 22.04, PyTorch 2.6 with CUDA 12.4 |
| CPU | GitHub Actions CI |

## Troubleshooting

**`gptlm: command not found`**: activate the virtual environment
(`source .venv/bin/activate`) and run `pip install -r requirements.txt`.

**`Config file not found`**: run from the repository root, or pass
`--config`.

**CUDA errors on an RTX 50-series GPU**: install the CUDA 12.8 build shown
under Installation, then rerun the GPU check there.

**`checkpoint not found`**: train a model first with `gptlm train`. The
error lists the checkpoints that do exist.

**`is not a loadable state_dict checkpoint`**: the file is a pickled model
from an earlier version of this project. Retrain.

**`vocab_size ... does not match`**: the vocabulary was rebuilt from a
different corpus after the checkpoint was trained. Rerun `gptlm prepare` on
the original corpus, or retrain.

**`Character ... is not in the vocabulary`**: the prompt contains a
character that never appears in the training corpus.
