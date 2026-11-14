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


def test_constructor_has_no_device_argument(tokenizer):
    """Placement is .to(device); a constructor device argument was dead code."""
    with pytest.raises(TypeError, match="device"):
        GPTLanguageModel(vocab_size=tokenizer.vocab_size, device="cpu")


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


def test_block_is_pre_norm(make_model):
    """With both residual branches zeroed, a pre-norm block is the identity.

    A post-norm block would still pass its output through LayerNorm.
    """
    block = make_model().blocks[0]
    with torch.no_grad():
        for linear in (block.sa.proj, block.ffwd.net[2]):
            linear.weight.zero_()
            linear.bias.zero_()
        g = torch.Generator().manual_seed(0)
        x = torch.randn(B, T, block.ln1.normalized_shape[0], generator=g) * 3 + 1
        torch.testing.assert_close(block(x), x)


N_LAYER = 4
SCALED_STD = 0.02 / (2 * N_LAYER) ** 0.5


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sa.proj.weight", SCALED_STD),
        ("ffwd.net.2.weight", SCALED_STD),
        ("ffwd.net.0.weight", 0.02),
    ],
)
def test_residual_projections_use_scaled_init(make_model, name, expected):
    """GPT-2 init: only projections into the residual stream are scaled down."""
    model = make_model(n_embd=64, n_layer=N_LAYER)
    for i, block in enumerate(model.blocks):
        std = block.get_parameter(name).std().item()
        assert abs(std - expected) < 0.15 * expected, (i, std)


def test_attention_matches_reference(make_model):
    """The fused kernel equals masked softmax attention computed head by head."""
    attn = make_model().blocks[0].sa
    C = attn.proj.in_features
    hs = C // attn.n_head
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, C, generator=g)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    with torch.no_grad():
        q, k, v = attn.qkv(x).split(C, dim=-1)
        heads = []
        for h in range(attn.n_head):
            s = slice(h * hs, (h + 1) * hs)
            wei = q[..., s] @ k[..., s].transpose(-2, -1) * hs**-0.5
            wei = wei.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            heads.append(wei @ v[..., s])
        expected = attn.proj(torch.cat(heads, dim=-1))
        torch.testing.assert_close(attn(x), expected)


def test_attention_dropout_only_in_train_mode(make_model):
    """Dropout on the attention weights is on in train mode, off in eval mode."""
    attn = make_model(dropout=0.5).blocks[0].sa
    attn.resid_dropout.p = 0.0  # leave attention-weight dropout as the only source
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, attn.proj.in_features, generator=g)
    with torch.no_grad():
        expected = attn(x)
        torch.testing.assert_close(attn(x), expected)
        attn.train()
        assert not torch.allclose(attn(x), expected)


def test_output_layer_shares_the_embedding_matrix(make_model):
    model = make_model()
    assert model.lm_head.weight is model.token_embedding_table.weight
    names = [name for name, _ in model.named_parameters()]
    assert "lm_head.weight" not in names  # counted once, decayed once
