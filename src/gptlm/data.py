"""Prepared token datasets and batch sampling.

`gptlm prepare` tokenizes the corpus once into train.bin and val.bin, flat
arrays of uint16 token ids, plus meta.json recording the tokenizer and the
token and character counts. Training reads the .bin files through
np.memmap, so a corpus of hundreds of millions of tokens never has to fit
in RAM, and the tokenizer never runs during training.

Batch start positions are drawn uniformly from the whole split (an earlier
sampler drew every row of a batch from one ~2KB window), using torch's RNG
so seeded runs stay reproducible.
"""

import json
import os

import numpy as np
import torch

META_FILE = "meta.json"
DTYPE = np.uint16
MAX_VOCAB = int(np.iinfo(DTYPE).max) + 1
SPLITS = ("train", "val")


def split_path(data_dir, split):
    return os.path.join(data_dir, f"{split}.bin")


def read_meta(data_dir):
    path = os.path.join(data_dir, META_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; run `gptlm prepare` first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def download(url, path, log=None):
    """Fetch url to path, via a .part file so an interrupted download is never mistaken
    for a complete one."""
    import urllib.request

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "gptlm"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done, next_report = 0, 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if log and done >= next_report:
                log(f"  {path}: {done / 1e6:,.0f} of {total / 1e6:,.0f} MB")
                next_report += 200_000_000
    os.replace(tmp, path)


def iter_documents(path, separator, block_chars=1 << 24):
    """Yield the documents of a large text file split on separator.

    Reads block_chars characters at a time, so a multi-gigabyte file is
    never held in memory. Surrounding whitespace is stripped and empty
    documents are skipped. separator None yields the whole file as one
    document.
    """
    with open(path, encoding="utf-8-sig") as f:
        if separator is None:
            yield f.read()
            return
        rest = ""
        while block := f.read(block_chars):
            parts = (rest + block).split(separator)
            rest = parts.pop()
            for doc in parts:
                if doc.strip():
                    yield doc.strip()
        if rest.strip():
            yield rest.strip()


def _batches(documents, max_chars=1 << 22):
    batch, size = [], 0
    for doc in documents:
        batch.append(doc)
        size += len(doc)
        if size >= max_chars:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def write_tokens(documents, path, tokenizer, separate, log=None):
    """Tokenize documents into a uint16 .bin file; returns (tokens, chars).

    separate appends the tokenizer's end-of-text token after each document,
    so the model sees where one story ends and the next begins. The file is
    written to a temporary name and renamed, so an interrupted run never
    leaves a truncated split behind.
    """
    if tokenizer.vocab_size > MAX_VOCAB:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} does not fit in uint16")
    if separate and tokenizer.eot_token is None:
        raise ValueError(f"the {tokenizer.name} tokenizer has no end-of-text token")
    encode_batch = getattr(tokenizer, "encode_batch", None)
    tokens = chars = 0
    next_report = 50_000_000
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        for batch in _batches(documents):
            encoded = encode_batch(batch) if encode_batch else [tokenizer.encode(d) for d in batch]
            for doc, ids in zip(batch, encoded, strict=True):
                if separate:
                    ids = [*ids, tokenizer.eot_token]
                f.write(np.asarray(ids, dtype=DTYPE).tobytes())
                tokens += len(ids)
                chars += len(doc)
            if log and tokens >= next_report:
                log(f"  {path}: {tokens:,} tokens so far")
                next_report += 50_000_000
    os.replace(tmp, path)
    return tokens, chars


def prepare_dataset(
    *, tokenizer_name, raw_file, raw_val_file, train_split, data_dir, separator, log=None
):
    """Tokenize the corpus into data_dir; returns the meta dict also written to meta.json.

    Without raw_val_file, raw_file is read whole and split at train_split.
    With it, the two files are the train and val splits, streamed document
    by document. The char tokenizer's vocabulary is every character in the
    corpus, so validation text never holds an unknown character; it is also
    written to vocab.txt, the file the cross-machine vocabulary check hashes.
    """
    from gptlm.tokenizer import CharacterTokenizer, GPT2Tokenizer

    if raw_val_file is None:
        with open(raw_file, encoding="utf-8-sig") as f:
            text = f.read()
        cut = int(len(text) * train_split)
        sources = {"train": lambda: [text[:cut]], "val": lambda: [text[cut:]]}
        separate = False
    else:
        sources = {
            "train": lambda: iter_documents(raw_file, separator),
            "val": lambda: iter_documents(raw_val_file, separator),
        }
        separate = separator is not None

    os.makedirs(data_dir, exist_ok=True)
    if tokenizer_name == "char":
        chars = set()
        for split in SPLITS:
            for doc in sources[split]():
                chars.update(doc)
        tokenizer = CharacterTokenizer.from_chars(sorted(chars))
        with open(os.path.join(data_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            json.dump(tokenizer.chars, f, ensure_ascii=False)
    elif tokenizer_name == "gpt2":
        tokenizer = GPT2Tokenizer()
    else:
        raise ValueError(f"unknown tokenizer {tokenizer_name!r}; expected char or gpt2")

    meta = {"tokenizer": tokenizer.spec(), "vocab_size": tokenizer.vocab_size, "splits": {}}
    for split in SPLITS:
        tokens, n_chars = write_tokens(
            sources[split](), split_path(data_dir, split), tokenizer, separate, log
        )
        meta["splits"][split] = {"tokens": tokens, "chars": n_chars}
    with open(os.path.join(data_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


class TokenDataset:
    """Train/val token splits from `gptlm prepare`, sampled into batches."""

    def __init__(self, data_dir, device="cpu"):
        self.data_dir = data_dir
        self.device = device
        self.meta = read_meta(data_dir)
        self.vocab_size = self.meta["vocab_size"]
        for split in SPLITS:
            path = split_path(data_dir, split)
            expected = self.meta["splits"][split]["tokens"] * np.dtype(DTYPE).itemsize
            if not os.path.isfile(path) or os.path.getsize(path) != expected:
                raise ValueError(f"{path} is missing or does not match {META_FILE}; rerun prepare")

    @property
    def tokenizer_spec(self):
        return self.meta["tokenizer"]

    def token_count(self, split):
        return self.meta["splits"][split]["tokens"]

    def tokens_per_char(self, split):
        """Tokens per character of source text: 1 for char, about 0.25 for GPT-2.
        End-of-text tokens count as tokens but not characters."""
        counts = self.meta["splits"][split]
        return counts["tokens"] / counts["chars"]

    def tokens(self, split):
        # A fresh memmap per call: holding one open keeps every page it has
        # touched resident, which grows without bound on a large corpus.
        return np.memmap(split_path(self.data_dir, split), dtype=DTYPE, mode="r")

    def get_batch(self, split, batch_size, block_size):
        """Sample a batch of (input, target) pairs, each (batch_size, block_size) int64."""
        data = self.tokens(split)
        if len(data) < block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens but block_size is "
                f"{block_size}; need at least {block_size + 1}"
            )

        ix = torch.randint(0, len(data) - block_size, (batch_size,)).tolist()
        x = torch.from_numpy(np.stack([data[i : i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(
            np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)
        )
        if self.device == "cuda":
            return (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
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
