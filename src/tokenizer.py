"""Character-level tokenizer utilities"""

import json


class CharacterTokenizer:
    """Simple character-level tokenizer.

    The vocabulary is stored as a JSON array of single-character strings.
    An earlier line-delimited format could not unambiguously represent the
    newline character itself (it was written as a blank line and read back
    twice, inflating the vocabulary by one and creating a duplicate token).
    """

    def __init__(self, vocab_file):
        """Load vocabulary from file."""
        with open(vocab_file, "r", encoding="utf-8") as f:
            chars = json.load(f)

        if len(chars) != len(set(chars)):
            raise ValueError(f"Vocabulary in {vocab_file} contains duplicates")

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
                f"Character {e.args[0]!r} is not in the vocabulary "
                f"({self.vocab_size} characters)"
            ) from None

    def decode(self, indices):
        """Decode a list of integers to text."""
        return "".join(self.int_to_string[i] for i in indices)

    @staticmethod
    def create_vocab(text_file, vocab_file):
        """Create a vocabulary file from a text file.

        Reads with utf-8-sig so a leading byte-order mark is stripped rather
        than becoming a vocabulary entry.
        """
        with open(text_file, "r", encoding="utf-8-sig") as f:
            text = f.read()

        chars = sorted(set(text))

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False)

        return len(chars)
