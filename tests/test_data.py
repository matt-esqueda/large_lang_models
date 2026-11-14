"""Data pipeline invariants: prepared token files and batch sampling."""

import json

import numpy as np
import pytest
import torch

from gptlm.data import TokenDataset, iter_documents, prepare_dataset, set_seed

N = 10_000
BATCH, BLOCK = 64, 16


def prepare(tmp_path, text, **overrides):
    raw = tmp_path / "raw.txt"
    raw.write_text(text, encoding="utf-8")
    settings = dict(
        tokenizer_name="char",
        raw_file=str(raw),
        raw_val_file=None,
        train_split=0.8,
        data_dir=str(tmp_path / "processed"),
        separator=None,
    )
    settings.update(overrides)
    prepare_dataset(**settings)
    return str(tmp_path / "processed")


@pytest.fixture
def dataset(tmp_path, sample_text):
    return TokenDataset(prepare(tmp_path, sample_text * 25))


@pytest.fixture
def positional(tmp_path, dataset):
    """Train split where token id == position, so a batch reveals its start offsets."""
    np.arange(N, dtype=np.uint16).tofile(tmp_path / "processed" / "train.bin")
    meta_file = tmp_path / "processed" / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["splits"]["train"]["tokens"] = N
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    return TokenDataset(str(tmp_path / "processed"))


def test_splits_are_tokenized_text(dataset, tokenizer, sample_text):
    assert tokenizer.decode(dataset.tokens("train").tolist()) == sample_text * 20
    assert tokenizer.decode(dataset.tokens("val").tolist()) == sample_text * 5
    assert dataset.tokens("train").dtype == np.uint16
    assert dataset.tokenizer_spec == tokenizer.spec()


def test_batch_shapes_and_dtype(dataset):
    for split in ("train", "val"):
        x, y = dataset.get_batch(split, BATCH, BLOCK)
        assert x.shape == y.shape == (BATCH, BLOCK)
        assert x.dtype == y.dtype == torch.long


def test_targets_are_inputs_shifted_by_one(positional):
    x, y = positional.get_batch("train", BATCH, BLOCK)
    torch.testing.assert_close(y, x + 1)
    # Each row is a contiguous slice of the corpus.
    torch.testing.assert_close(x, x[:, :1] + torch.arange(BLOCK))


def test_batch_rows_are_sampled_across_whole_corpus(positional):
    """Regression: every start used to come from one ~2100-token window."""
    set_seed(0)
    x, _ = positional.get_batch("train", 256, BLOCK)
    starts = x[:, 0]
    assert starts.max() - starts.min() > 0.8 * N
    assert len(torch.unique(starts)) > 240


def test_same_seed_gives_same_batches(positional):
    set_seed(1234)
    first = [positional.get_batch("train", BATCH, BLOCK) for _ in range(3)]
    set_seed(1234)
    second = [positional.get_batch("train", BATCH, BLOCK) for _ in range(3)]
    for (x1, y1), (x2, y2) in zip(first, second, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_different_seeds_give_different_batches(positional):
    set_seed(1)
    x1, _ = positional.get_batch("train", BATCH, BLOCK)
    set_seed(2)
    x2, _ = positional.get_batch("train", BATCH, BLOCK)
    assert not torch.equal(x1, x2)


def test_split_shorter_than_block_raises(dataset):
    too_long = dataset.token_count("val")
    with pytest.raises(ValueError, match="block_size"):
        dataset.get_batch("val", BATCH, too_long)


def test_truncated_split_is_rejected(tmp_path, dataset):
    path = tmp_path / "processed" / "val.bin"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="does not match meta.json"):
        TokenDataset(str(tmp_path / "processed"))


def test_iter_documents_splits_across_read_blocks(tmp_path):
    """Separators that straddle a read boundary must still split documents."""
    docs = [f"story {i} " * (i + 1) for i in range(30)]
    path = tmp_path / "stories.txt"
    path.write_text("<|endoftext|>\n".join(docs) + "<|endoftext|>\n\n", encoding="utf-8")
    assert list(iter_documents(str(path), "<|endoftext|>", block_chars=7)) == [
        d.strip() for d in docs
    ]


def test_prepare_leaves_no_temp_files(tmp_path, sample_text):
    data_dir = prepare(tmp_path, sample_text * 5)
    assert sorted(p.name for p in (tmp_path / "processed").iterdir()) == [
        "meta.json",
        "train.bin",
        "val.bin",
        "vocab.txt",
    ]
    assert data_dir.endswith("processed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batches_land_on_requested_device(dataset):
    dataset.device = "cuda"
    x, y = dataset.get_batch("train", BATCH, BLOCK)
    assert x.device.type == "cuda"
    assert y.device.type == "cuda"


def test_tokens_per_char(dataset):
    assert dataset.tokens_per_char("val") == 1.0
