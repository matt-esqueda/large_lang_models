"""Generation invariants: one sampling path, correct lengths, penalties applied."""

import pytest
import torch

SAMPLING = [
    {"temperature": 0.0},
    {"temperature": 0.8, "top_k": 5},
    {"temperature": 1.0, "top_p": 0.9, "repetition_penalty": 1.3},
]


def context_for(tokenizer, text="the "):
    return torch.tensor([tokenizer.encode(text)], dtype=torch.long)


@pytest.mark.parametrize("sampling", SAMPLING)
def test_stream_matches_generate(make_model, tokenizer, sampling):
    model = make_model()
    context = context_for(tokenizer)
    torch.manual_seed(0)
    collected = model.generate(context, 20, **sampling)
    torch.manual_seed(0)
    streamed = torch.cat([context, *model.generate_stream(context, 20, **sampling)], dim=1)
    assert torch.equal(collected, streamed)


def test_output_keeps_prefix_and_length(make_model, tokenizer):
    context = context_for(tokenizer)
    out = make_model().generate(context, 12, temperature=0.0)
    assert out.shape == (1, context.shape[1] + 12)
    assert torch.equal(out[:, : context.shape[1]], context)


def test_context_longer_than_block_size(make_model, tokenizer):
    model = make_model()
    context = context_for(tokenizer, "the " * 10)  # 40 tokens, block_size is 16
    out = model.generate(context, 5, temperature=0.0)
    assert out.shape == (1, 45)


def test_greedy_is_deterministic(make_model, tokenizer):
    model = make_model()
    context = context_for(tokenizer)
    torch.manual_seed(1)
    first = model.generate(context, 15, temperature=0.0)
    torch.manual_seed(2)
    second = model.generate(context, 15, temperature=0.0)
    assert torch.equal(first, second)


def test_repetition_penalty_changes_greedy_output(make_model, tokenizer):
    model = make_model()
    context = context_for(tokenizer)
    plain = model.generate(context, 30, temperature=0.0)
    penalized = model.generate(context, 30, temperature=0.0, repetition_penalty=5.0)
    assert not torch.equal(plain, penalized)
