# Large Language Models

A GPT-style language model implementation. This project demonstrates training a transformer-based language model from scratch using PyTorch.

---

## Table of Contents

1. [Features](#1-features)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [Model Architecture](#5-model-architecture)
6. [Configuration](#6-configuration)
7. [Development](#7-development)
8. [Hardware Requirements](#8-hardware-requirements)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Features

- Clean, modular architecture
- Character-level tokenization
- Multi-head self-attention transformer
- Training on custom text datasets
- Interactive chatbot interface
- GPU acceleration (CUDA support)

---

## 2. Project Structure

```
large_lang_models/
├── config/              # Configuration files
├── data/
│   ├── raw/             # Raw input text files
│   └── processed/       # Processed train/val splits
├── scripts/             # Executable scripts
│   ├── prepare_data.py  # Data preparation
│   ├── train.py         # Model training
│   └── chat.py          # Interactive chatbot
├── src/                 # Source code
│   ├── model.py         # GPT model architecture
│   └── tokenizer.py     # Character tokenizer
├── models/              # Saved model checkpoints
├── logs/                # Training logs
└── legacy/              # Original learning scripts
```

---

## 3. Installation

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.8+ |
| GPU | NVIDIA with CUDA support (optional, recommended) |
| PyTorch | 2.7+ with CUDA 12.8+ for RTX 50-series GPUs |

### Setup

**Step 1: Clone the repository**

```bash
git clone git@github.com:matt-esqueda/large_lang_models.git
cd large_lang_models
```

**Step 2: Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

**Step 3: Install dependencies**

```bash
# For CUDA 12.8+ (RTX 50-series support)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Or for stable CPU-only version
pip install torch

# Install other requirements
pip install -r requirements.txt
```

---

## 4. Usage

### Step 1: Prepare Data

Place your text file in `data/raw/` (e.g., `wizard_of_oz.txt`), then run:

```bash
python scripts/prepare_data.py
```

**Creates:**

| File | Description |
|------|-------------|
| `data/processed/train_split.txt` | Training data (90%) |
| `data/processed/val_split.txt` | Validation data (10%) |
| `data/processed/vocab.txt` | Character vocabulary |

### Step 2: Train Model

**Quick start:**

```bash
python scripts/train.py -batch_size 32
```

**Full options:**

```bash
python scripts/train.py \
  -batch_size 32 \
  -max_iters 5000 \
  -n_layer 6 \
  -n_head 6 \
  -learning_rate 3e-4
```

**Available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-batch_size` | *(required)* | Batch size |
| `-block_size` | `64` | Context window size |
| `-max_iters` | `5000` | Training iterations |
| `-learning_rate` | `3e-4` | Learning rate |
| `-n_embd` | `384` | Embedding dimension |
| `-n_head` | `6` | Number of attention heads |
| `-n_layer` | `6` | Number of layers |
| `-dropout` | `0.2` | Dropout rate |

> The trained model is saved to `models/checkpoints/model_01.pkl`.

### Step 3: Chat with Model

```bash
python scripts/chat.py
```

**Commands:**

| Command | Action |
|---------|--------|
| Any text + Enter | Generate a response |
| `quit` | Exit the chatbot |
| `clear` | Clear the screen |

---

## 5. Model Architecture

| Property | Value |
|----------|-------|
| Architecture | GPT-style decoder-only transformer |
| Attention | Multi-head causal self-attention |
| Default size | 6 layers, 6 heads, 384 embedding dim (~3M parameters) |
| Tokenization | Character-level |
| Training objective | Next-token prediction |

---

## 6. Configuration

Edit `config/config.yaml` to customize model architecture, training parameters, and paths.

---

## 7. Development

### Legacy Scripts

Original learning scripts are in `legacy/`:

| Script | Description |
|--------|-------------|
| `gpt_v1.py` | Original monolithic implementation |
| `bigram.py` | Simple bigram baseline |
| `data_extract.py` | OpenWebText data processing |

### Project Phases

| Phase | Description |
|-------|-------------|
| Phase 1–2 | Basic implementation and validation |
| Phase 3 | Code refactoring and modularization *(current)* |
| Future | Enhanced features, larger datasets, better tokenization |

---

## 8. Hardware Requirements

| Setup | Details |
|-------|---------|
| Minimum | CPU-only (slow but functional) |
| Recommended | NVIDIA GPU with 4 GB+ VRAM |
| Tested on | RTX 5080 (requires PyTorch nightly with CUDA 12.8+) |

---

## 9. Troubleshooting

### CUDA Compatibility Issues

For RTX 50-series (Blackwell architecture):

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify GPU support:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Model Not Found Error

If `chat.py` can't find the model, train one first:

```bash
python scripts/train.py -batch_size 32
```