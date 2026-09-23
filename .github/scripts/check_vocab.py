#!/usr/bin/env python3
"""Assert the character vocabulary matches the documented baseline."""

from gptlm.tokenizer import CharacterTokenizer

EXPECTED_SIZE = 80

t = CharacterTokenizer("data/processed/vocab.txt")
assert t.vocab_size == EXPECTED_SIZE, f"expected {EXPECTED_SIZE}, got {t.vocab_size}"
assert len(set(t.chars)) == EXPECTED_SIZE, "vocabulary contains duplicates"
assert "\ufeff" not in t.chars, "byte-order mark leaked into vocabulary"

sample = "Dorothy said,\n\"Hello!\"\n"
assert t.decode(t.encode(sample)) == sample, "encode/decode round-trip failed"

print(f"vocabulary OK: {t.vocab_size} characters")
