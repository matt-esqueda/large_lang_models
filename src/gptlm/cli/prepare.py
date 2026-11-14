"""`gptlm prepare`: tokenize the raw corpus into train/val token files."""

import os

from gptlm.cli.common import fail
from gptlm.config import section

HELP = "tokenize the raw corpus into train/val token files"


def add_arguments(parser):
    parser.add_argument("--raw-file", help="corpus to tokenize (default: data.raw_file)")
    parser.add_argument("--train-split", type=float, help="fraction used for training, in (0, 1)")
    parser.add_argument("--tokenizer", choices=["char", "gpt2"], help="default: data.tokenizer")


def run(args, config):
    from gptlm.data import download, prepare_dataset, split_path

    data = section(
        config,
        "data",
        {"raw_file": args.raw_file, "train_split": args.train_split, "tokenizer": args.tokenizer},
    )
    if not 0 < data["train_split"] < 1:
        fail("--train-split must be between 0 and 1")
    for key, url_key in (("raw_file", "raw_url"), ("raw_val_file", "raw_val_url")):
        if data[key] is None or os.path.isfile(data[key]):
            continue
        if data[url_key] is None:
            fail(f"raw corpus not found: {data[key]}")
        print(f"Downloading {data[url_key]}")
        try:
            download(data[url_key], data[key], log=print)
        except OSError as e:
            fail(f"download failed: {e}; rerun prepare to try again")

    try:
        meta = prepare_dataset(
            tokenizer_name=data["tokenizer"],
            raw_file=data["raw_file"],
            raw_val_file=data["raw_val_file"],
            train_split=data["train_split"],
            data_dir=data["dir"],
            separator=data["document_separator"],
            log=print,
        )
    except ValueError as e:
        fail(str(e))
    for split, counts in meta["splits"].items():
        print(
            f"{split_path(data['dir'], split)}: {counts['tokens']:,} tokens "
            f"({counts['chars']:,} characters)"
        )
    print(f"Tokenizer: {data['tokenizer']}, {meta['vocab_size']:,} tokens in the vocabulary")
    return 0
