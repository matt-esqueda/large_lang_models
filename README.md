# gptlm: a small GPT from scratch

A decoder-only transformer language model written in PyTorch and trained
from scratch. Two corpora are set up: a single public-domain book, *The
Wonderful Wizard of Oz*, one token per character (the default, small enough
for a CPU), and TinyStories, about 2.7M short stories, with GPT-2's BPE
tokenizer (a GPU job). Everything runs through one command, `gptlm`, with
defaults in `config/config.yaml`.

The model predicts the next token. It continues whatever text you give
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
gptlm prepare    # tokenize the corpus into train/val token files
gptlm train      # train with the defaults in config/config.yaml
gptlm chat       # continue prompts with the trained model
```

The default model has 10.7M parameters and trains for 5000 iterations,
which is comfortable on a GPU. For a quick run on a CPU, train a smaller
model:

```bash
gptlm train --max-iters 200 --n-layer 2 --n-head 2 --n-embd 64 --eval-interval 100 --checkpoint-interval 100
```

### TinyStories

`config/tinystories.yaml` is a second profile: GPT-2 tokens, a 256-token
context and a 30M-parameter model. Pass it to every command:

```bash
gptlm prepare --config config/tinystories.yaml   # downloads about 2.2 GB once, then tokenizes
gptlm train --config config/tinystories.yaml --compile
gptlm chat --config config/tinystories.yaml --prompt "Once upon a time"
```

On an RTX 5080 training takes about 11 minutes with `--compile` (drop it
on a CPU) and reaches a validation loss of about 1.42, enough for coherent
short stories. `chat` stops when the model ends its story.

Its data, checkpoints and logs live in `data/tinystories/`,
`models/tinystories/` and `logs/tinystories/`, so the two profiles never
overwrite each other.

## Commands

| Command | Does |
|---------|------|
| `gptlm prepare` | Tokenize the raw corpus into train/val token files |
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
gptlm prepare --tokenizer gpt2
```

Reads `data/raw/wizard_of_oz.txt` by default and writes, to `data.dir`
(`data/processed/`):

| File | Contents |
|------|----------|
| `train.bin` | Token ids of the first 90% of the corpus, as uint16 |
| `val.bin` | Token ids of the remaining 10% |
| `meta.json` | The tokenizer, and token and character counts per split |
| `vocab.txt` | With `char` only: every character in the corpus, as a JSON array |

Two tokenizers are available. `char` makes every character a token; its
vocabulary covers the whole corpus, so validation text never contains an
unknown character, and the bundled book gives 80 characters. `gpt2` is
GPT-2's byte-level BPE (50,257 tokens, via `tiktoken`), which encodes any
text at about four characters per token; its merge table downloads (about
1 MB) and is cached the first time it is used.

When `data.raw_val_file` is set, the two files are the train and
validation splits and `train_split` is unused. `data.document_separator`
splits them into documents, and `gpt2` puts its end-of-text token between
them. A missing raw file is downloaded from `data.raw_url` or
`data.raw_val_url` when those are set. Training reads the token files
through a memory map, so a corpus larger than RAM still trains.

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

Resuming restores the weights, the optimizer state, the learning-rate
schedule and the iteration count. `--max-iters` is the number of
additional steps. The model's shape, the weight decay and the schedule
come from the checkpoint, so their flags (`--n-layer` and the other model
flags, `--weight-decay`, `--learning-rate`, `--min-lr`, `--warmup-iters`
and `--lr-decay-iters`) are rejected. Past the end of its schedule, a
resumed run trains at `min_lr`. A resumed run saves its final checkpoint as
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

The tokenizer comes from the checkpoint, so chat works whatever is in
`data/processed/`. With the `char` tokenizer, every character in a prompt
must be in the vocabulary. Sampling is
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
| `data` | tokenizer, raw corpus files or URLs, `train_split`, document separator, prepared data directory | prepare, train |
| `model` | architecture of a new model | train |
| `training` | batch size, iterations, learning-rate schedule, weight decay, gradient clipping, precision, `compile`, evaluation and checkpoint intervals, seed | train |
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
| Architecture | Decoder-only transformer, pre-norm (LayerNorm at the start of each residual branch) |
| Attention | Causal multi-head self-attention, one fused `scaled_dot_product_attention` call |
| Feed-forward | 4x width, ReLU |
| Positions | Learned embeddings, up to `block_size` |
| Embeddings | Learned token embeddings, tied to the output layer |
| Tokenizer | Character-level (80 characters for the bundled book) or GPT-2 BPE (50,257 tokens) |
| Default size | 6 layers, 6 heads, 384-dim embeddings, 64-token context: 10.7M parameters; 30.1M with GPT-2 tokens and a 256-token context |
| Initialization | Normal, std 0.02; projections into the residual stream scaled by 1/sqrt(2 x n_layer) |
| Objective | Next-character prediction, cross-entropy loss |
| Optimizer | AdamW, weight decay 0.1 on matrix weights only, gradient norm clipped to 1.0 |
| Learning rate | Linear warmup over 100 steps to 3e-4, cosine decay to 3e-5 by the end of the run |
| Precision | bf16 autocast with TF32 on CUDA, fp32 on CPU; `--compile` adds `torch.compile` |

Checkpoints store the weights as a `state_dict` together with the
architecture, iteration, losses, optimizer state, learning-rate schedule
and tokenizer. They carry a format version: this release reads version 3 and
refuses any other, because older weights would compute a different
function. They load on CPU or
GPU regardless of where they were trained, and they load with
`weights_only=True`, so opening one never runs code from the file.
Pickled `.pkl` models from earlier versions of this project are refused;
retrain instead.

## Project layout

```
large_lang_models/
├── config/
│   ├── config.yaml       # defaults for every command (Wizard of Oz, char tokens)
│   └── tinystories.yaml  # TinyStories profile (GPT-2 tokens)
├── data/
│   ├── raw/              # source corpus; TinyStories downloads into raw/tinystories/
│   ├── processed/        # Oz token files (generated)
│   └── tinystories/      # TinyStories token files (generated)
├── docs/ROADMAP.md       # plan, conventions, known issues, decisions
├── legacy/               # original learning scripts, unmaintained
├── src/gptlm/
│   ├── cli/              # one module per subcommand
│   ├── checkpoint.py     # checkpoint save and load
│   ├── config.py         # config loading and flag overrides
│   ├── data.py           # prepare, token files, memory-mapped batch sampling
│   ├── model.py          # the transformer
│   ├── tokenizer.py      # character and GPT-2 BPE tokenizers
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

**`is checkpoint version 2, but this gptlm reads version 3`**: the
checkpoint was trained before embeddings were tied and checkpoints
recorded their tokenizer (version 1: before pre-norm blocks and fused
attention). Retrain.

**`... cannot be used with --resume`**: a resumed run takes its model
shape, weight decay and learning-rate schedule from the checkpoint. Drop
the flag, or start a new run.

**`--compile` fails or hangs on CPU**: `torch.compile` needs a working C++
compiler. Drop `--compile`; it is off by default.

**`was trained with a different tokenizer`**: `--resume` found prepared
data whose tokenizer or character set differs from the checkpoint's. Rerun
`gptlm prepare` with the original corpus and tokenizer, or start a new run.

**`run gptlm prepare first`** or **`does not match meta.json`**: the token
files are missing or were cut short. Rerun `gptlm prepare` with the same
`--config`.

**TinyStories download fails**: rerun `gptlm prepare --config
config/tinystories.yaml`; a partial download is kept as a `.part` file and
started over.

**`Character ... is not in the vocabulary`**: the prompt contains a
character that never appears in the training corpus.
