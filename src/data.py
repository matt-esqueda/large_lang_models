"""Batch sampling from a tokenized corpus.

The previous implementation re-read the entire split file from disk on every
batch, then sliced a ~2KB window out of it. A single estimate_loss() call
with eval_iters=100 therefore performed 200 full-file reads.

It also drew every sequence start position from inside that one small window,
so a "batch of 32" was really 32 heavily overlapping slices of two
paragraphs. Gradient estimates were far more correlated than the batch size
implied. This module tokenizes each split once and samples start positions
uniformly across the whole corpus.
"""

import torch


class CorpusDataset:
    """Holds tokenized train/val splits in memory and yields batches."""

    def __init__(self, train_file, val_file, tokenizer, device="cpu"):
        self.tokenizer = tokenizer
        self.device = device
        self.splits = {
            "train": self._load(train_file),
            "val": self._load(val_file),
        }

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.splits["train"])

    def token_count(self, split):
        return len(self.splits[split])

    def get_batch(self, split, batch_size, block_size):
        """Sample a batch of (input, target) pairs.

        Start positions are drawn uniformly from the entire split, so
        sequences in a batch are independent rather than overlapping
        neighbours.
        """
        data = self.splits[split]

        if len(data) < block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens but block_size is "
                f"{block_size}; need at least {block_size + 1}"
            )

        high = len(data) - block_size
        ix = torch.randint(0, high, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


def set_seed(seed):
    """Seed Python, NumPy and torch RNGs for reproducible runs."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
