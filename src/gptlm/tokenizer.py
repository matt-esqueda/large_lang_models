"""Tokenizers: one character per token, or GPT-2's byte-level BPE.

Both expose name, vocab_size, encode(text), decode(ids), decode_bytes(ids)
and spec(). A spec is a small JSON-safe dict that rebuilds the tokenizer
through tokenizer_from_spec(), so checkpoints and prepared datasets can
record exactly which tokenizer produced their ids.
"""

import json

TOKENIZERS = ("char", "gpt2")


class CharacterTokenizer:
    """Simple character-level tokenizer.

    The vocabulary is stored as a JSON array of single-character strings.
    An earlier line-delimited format could not unambiguously represent the
    newline character itself (it was written as a blank line and read back
    twice, inflating the vocabulary by one and creating a duplicate token).
    """

    name = "char"
    eot_token = None

    def __init__(self, vocab_file):
        """Load vocabulary from file."""
        with open(vocab_file, encoding="utf-8") as f:
            chars = json.load(f)
        self._set_chars(chars, source=vocab_file)

    @classmethod
    def from_chars(cls, chars):
        tokenizer = cls.__new__(cls)
        tokenizer._set_chars(list(chars), source="the given characters")
        return tokenizer

    def _set_chars(self, chars, source):
        if len(chars) != len(set(chars)):
            raise ValueError(f"Vocabulary in {source} contains duplicates")

        self.chars = chars
        self.vocab_size = len(chars)
        self.string_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Encode text to a list of integers.

        Raises ValueError on characters outside the vocabulary.
        """
        try:
            return [self.string_to_int[c] for c in text]
        except KeyError as e:
            raise ValueError(
                f"Character {e.args[0]!r} is not in the vocabulary ({self.vocab_size} characters)"
            ) from None

    def decode(self, indices):
        """Decode a list of integers to text."""
        return "".join(self.int_to_string[i] for i in indices)

    def decode_bytes(self, indices):
        return self.decode(indices).encode("utf-8")

    def spec(self):
        return {"name": self.name, "chars": list(self.chars)}

    @staticmethod
    def create_vocab(text_file, vocab_file):
        """Create a vocabulary file from a text file.

        Reads with utf-8-sig so a leading byte-order mark is stripped rather
        than becoming a vocabulary entry.
        """
        with open(text_file, encoding="utf-8-sig") as f:
            text = f.read()

        chars = sorted(set(text))

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False)

        return len(chars)


class GPT2Tokenizer:
    """GPT-2's byte-level BPE: 50,257 tokens, any text encodes.

    Wraps tiktoken, which downloads and caches the merge table (about 1 MB)
    on first use. encode() treats "<|endoftext|>" in the text as ordinary
    characters; the end-of-text token is only ever added deliberately, as
    eot_token, between documents.
    """

    name = "gpt2"

    def __init__(self):
        import tiktoken

        self._encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._encoding.n_vocab
        self.eot_token = self._encoding.eot_token

    def encode(self, text):
        return self._encoding.encode_ordinary(text)

    def encode_batch(self, texts, num_threads=8):
        """Encode several documents in parallel; returns one id list per text."""
        return self._encoding.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, indices):
        return self._encoding.decode(list(indices))

    def decode_bytes(self, indices):
        """Raw UTF-8 bytes. One token can hold part of a multi-byte character,
        so decoding token by token must join bytes before making text."""
        return self._encoding.decode_bytes(list(indices))

    def spec(self):
        return {"name": self.name}


def tokenizer_from_spec(spec):
    """Rebuild the tokenizer a spec describes."""
    name = spec.get("name")
    if name == "char":
        return CharacterTokenizer.from_chars(spec["chars"])
    if name == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(f"unknown tokenizer {name!r}; expected one of {', '.join(TOKENIZERS)}")
