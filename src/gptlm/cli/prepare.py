"""`gptlm prepare`: split the raw corpus and build the vocabulary."""

import os

from gptlm.cli.common import fail
from gptlm.config import section

HELP = "split the raw corpus into train/val files and build the vocabulary"


def add_arguments(parser):
    parser.add_argument("--raw-file", help="corpus to split (default: data.raw_file)")
    parser.add_argument("--train-split", type=float, help="fraction used for training, in (0, 1)")


def prepare(raw_file, train_file, val_file, vocab_file, train_split):
    """Write the train/val split and the vocabulary.

    Returns (train_chars, val_chars, vocab_size). The vocabulary covers the
    whole corpus, so validation text never contains unknown characters.
    """
    from gptlm.tokenizer import CharacterTokenizer

    with open(raw_file, encoding="utf-8-sig") as f:
        text = f.read()
    split = int(len(text) * train_split)
    for path, part in ((train_file, text[:split]), (val_file, text[split:])):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(part)
    os.makedirs(os.path.dirname(vocab_file) or ".", exist_ok=True)
    vocab_size = CharacterTokenizer.create_vocab(raw_file, vocab_file)
    return split, len(text) - split, vocab_size


def run(args, config):
    data = section(config, "data", {"raw_file": args.raw_file, "train_split": args.train_split})
    if not 0 < data["train_split"] < 1:
        fail("--train-split must be between 0 and 1")
    if not os.path.isfile(data["raw_file"]):
        fail(f"raw corpus not found: {data['raw_file']}")

    train_chars, val_chars, vocab_size = prepare(
        data["raw_file"],
        data["train_file"],
        data["val_file"],
        data["vocab_file"],
        data["train_split"],
    )
    print(f"{data['train_file']}: {train_chars:,} characters")
    print(f"{data['val_file']}: {val_chars:,} characters")
    print(f"{data['vocab_file']}: {vocab_size} unique characters")
    return 0
