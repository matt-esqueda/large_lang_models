"""Tokenizer invariants."""

import json

import pytest

from gptlm.tokenizer import CharacterTokenizer


def test_round_trip_preserves_text(tokenizer, sample_text):
    assert tokenizer.decode(tokenizer.encode(sample_text)) == sample_text


def test_newline_is_a_single_token(tokenizer):
    text = "the\nend\n\n"
    ids = tokenizer.encode(text)
    assert len(ids) == len(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.chars.count("\n") == 1


def test_vocab_has_no_duplicates(tokenizer):
    assert len(tokenizer.chars) == len(set(tokenizer.chars))
    assert tokenizer.vocab_size == len(tokenizer.chars)


def test_vocab_matches_corpus_characters(tokenizer, sample_text):
    assert tokenizer.chars == sorted(set(sample_text))


def test_loader_rejects_duplicate_vocab(tmp_path):
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(json.dumps(["a", "b", "a"]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        CharacterTokenizer(str(vocab_file))


def test_byte_order_mark_is_stripped(tmp_path):
    text_file = tmp_path / "corpus.txt"
    vocab_file = tmp_path / "vocab.txt"
    text_file.write_text("abc\n", encoding="utf-8-sig")
    CharacterTokenizer.create_vocab(str(text_file), str(vocab_file))
    assert "\ufeff" not in CharacterTokenizer(str(vocab_file)).chars


def test_out_of_vocab_raises_value_error(tokenizer):
    assert "7" not in tokenizer.chars
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tokenizer.encode("Toto 7")
