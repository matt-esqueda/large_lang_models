"""Model invariants."""

import pytest
import torch

from gptlm.model import GPTLanguageModel

B, T = 3, 16


def random_ids(vocab_size, batch=B, length=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=g)


def test_forward_shapes_without_targets(make_model, tokenizer):
    model = make_model()
    logits, loss = model(random_ids(tokenizer.vocab_size))
    assert logits.shape == (B, T, tokenizer.vocab_size)
    assert loss is None


def test_forward_shapes_with_targets(make_model, tokenizer):
    model = make_model()
    x = random_ids(tokenizer.vocab_size, seed=0)
    y = random_ids(tokenizer.vocab_size, seed=1)
    logits, loss = model(x, y)
    assert logits.shape == (B * T, tokenizer.vocab_size)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_forward_accepts_shorter_context(make_model, tokenizer):
    model = make_model()
    logits, _ = model(random_ids(tokenizer.vocab_size, length=5))
    assert logits.shape == (B, 5, tokenizer.vocab_size)


@pytest.mark.parametrize("t", [0, 7, T - 2])
def test_causal_mask_hides_future_tokens(make_model, tokenizer, t):
    """Changing tokens after position t must not change logits at <= t."""
    model = make_model()
    x = random_ids(tokenizer.vocab_size, batch=1)
    x_future_changed = x.clone()
    x_future_changed[0, t + 1 :] = (x[0, t + 1 :] + 1) % tokenizer.vocab_size

    with torch.no_grad():
        before, _ = model(x)
        after, _ = model(x_future_changed)

    torch.testing.assert_close(before[:, : t + 1], after[:, : t + 1])
    # Sanity check: the edit is visible from position t + 1 onward.
    assert not torch.allclose(before[:, t + 1 :], after[:, t + 1 :])


def test_rejects_n_embd_not_divisible_by_n_head(tokenizer):
    with pytest.raises(ValueError, match="divisible"):
        GPTLanguageModel(vocab_size=tokenizer.vocab_size, n_embd=30, n_head=4)


@pytest.mark.parametrize("name", ["token_embedding_table", "position_embedding_table", "lm_head"])
def test_init_std_is_0_02(make_model, name):
    weight = getattr(make_model(n_embd=64), name).weight
    assert abs(weight.std().item() - 0.02) < 0.004


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_built_model_runs_on_cpu(make_model, tokenizer):
    """Regression: position ids must follow the input's device, not a stored string."""
    x = random_ids(tokenizer.vocab_size)
    model = make_model().to("cuda")
    with torch.no_grad():
        gpu_logits, _ = model(x.to("cuda"))
        model = model.to("cpu")
        cpu_logits, _ = model(x)
    torch.testing.assert_close(cpu_logits, gpu_logits.cpu(), atol=1e-4, rtol=1e-4)
