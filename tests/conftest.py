"""Shared fixtures for the test suite.

Everything is built in a temporary directory, so the suite never depends on
data/processed/ or models/ and behaves identically on a fresh clone, on a
machine that has trained before, and in CI.
"""

import pytest
import torch

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

SAMPLE_TEXT = (
    "Dorothy lived in the midst of the great Kansas prairies.\n\"Toto!\" she cried.\n\nThe end.\n"
)

TINY_CONFIG = {
    "n_embd": 32,
    "n_head": 4,
    "n_layer": 2,
    "block_size": 16,
    "dropout": 0.0,
}


@pytest.fixture
def sample_text():
    return SAMPLE_TEXT


@pytest.fixture
def tokenizer(tmp_path):
    text_file = tmp_path / "corpus.txt"
    vocab_file = tmp_path / "vocab.txt"
    text_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    CharacterTokenizer.create_vocab(str(text_file), str(vocab_file))
    return CharacterTokenizer(str(vocab_file))


@pytest.fixture
def make_model(tokenizer):
    """Factory for small, seeded models in eval mode (dropout disabled)."""

    def _make(seed=0, **overrides):
        torch.manual_seed(seed)
        config = {**TINY_CONFIG, **overrides}
        model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, **config)
        return model.eval()

    return _make
