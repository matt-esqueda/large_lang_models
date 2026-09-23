"""Data pipeline invariants."""

import pytest
import torch

from gptlm.data import CorpusDataset, set_seed

N = 10_000
BATCH, BLOCK = 64, 16


@pytest.fixture
def dataset(tokenizer, sample_text, tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text(sample_text * 20, encoding="utf-8")
    val_file.write_text(sample_text * 3, encoding="utf-8")
    return CorpusDataset(str(train_file), str(val_file), tokenizer)


@pytest.fixture
def positional(dataset):
    """Train split where token id == position, so a batch reveals its start offsets."""
    dataset.splits["train"] = torch.arange(N)
    return dataset


def test_splits_are_tokenized_text(dataset, tokenizer, sample_text):
    assert tokenizer.decode(dataset.splits["train"].tolist()) == sample_text * 20
    assert tokenizer.decode(dataset.splits["val"].tolist()) == sample_text * 3
    assert dataset.splits["train"].dtype == torch.long


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batches_land_on_requested_device(dataset):
    dataset.device = "cuda"
    x, y = dataset.get_batch("train", BATCH, BLOCK)
    assert x.device.type == "cuda"
    assert y.device.type == "cuda"
