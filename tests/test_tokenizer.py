"""Tokenizer invariants."""

import json

import pytest

from gptlm.tokenizer import CharacterTokenizer, GPT2Tokenizer, tokenizer_from_spec


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


def test_char_spec_round_trips(tokenizer, sample_text):
    rebuilt = tokenizer_from_spec(tokenizer.spec())
    assert rebuilt.chars == tokenizer.chars
    assert rebuilt.encode(sample_text) == tokenizer.encode(sample_text)


def test_char_decode_bytes_is_utf8(tokenizer):
    ids = tokenizer.encode("Toto")
    assert tokenizer.decode_bytes(ids) == b"Toto"


def test_unknown_tokenizer_name_raises():
    with pytest.raises(ValueError, match="unknown tokenizer 'bpe'"):
        tokenizer_from_spec({"name": "bpe"})


@pytest.fixture(scope="module")
def gpt2():
    return GPT2Tokenizer()


def test_gpt2_matches_reference_ids(gpt2):
    """Ids from tiktoken's published GPT-2 encoding; a different merge table would differ."""
    assert gpt2.vocab_size == 50257
    assert gpt2.eot_token == 50256
    assert gpt2.encode("Dorothy said hello") == [35, 273, 14863, 531, 23748]


@pytest.mark.parametrize("text", ["Dorothy lived in Kansas.\n", "café 😀 ½", ""])
def test_gpt2_round_trip(gpt2, text):
    assert gpt2.decode(gpt2.encode(text)) == text


def test_gpt2_treats_endoftext_text_as_ordinary(gpt2):
    ids = gpt2.encode("a<|endoftext|>b")
    assert gpt2.eot_token not in ids
    assert gpt2.decode(ids) == "a<|endoftext|>b"


def test_gpt2_spec_round_trips(gpt2):
    assert isinstance(tokenizer_from_spec(gpt2.spec()), GPT2Tokenizer)


def test_gpt2_encode_batch_matches_encode(gpt2):
    texts = ["Once upon a time", "there was a dog.", "😀"]
    assert gpt2.encode_batch(texts) == [gpt2.encode(t) for t in texts]
