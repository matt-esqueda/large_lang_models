"""Optimizer and learning-rate schedule invariants."""

import math

import pytest
import torch

from gptlm.training import (
    autocast_context,
    bits_per_char,
    configure_optimizer,
    lr_at,
    optimizer_step,
    resolve_precision,
)

SCHEDULE = {"learning_rate": 1e-3, "min_lr": 1e-4, "warmup_iters": 10, "lr_decay_iters": 110}


def param_names(model, params):
    names = {id(p): n for n, p in model.named_parameters()}
    return {names[id(p)] for p in params}


def test_weight_decay_applies_to_matrices_only(make_model):
    model = make_model()
    decay, no_decay = configure_optimizer(model, 1e-3, 0.1).param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert {"blocks.0.sa.qkv.weight", "token_embedding_table.weight"} <= param_names(
        model, decay["params"]
    )
    assert {"blocks.0.ln1.weight", "blocks.0.sa.proj.bias", "lm_head.bias"} <= param_names(
        model, no_decay["params"]
    )


def test_optimizer_covers_every_parameter_once(make_model):
    model = make_model()
    optimizer = configure_optimizer(model, 1e-3, 0.1)
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())


def test_warmup_rises_linearly_to_peak():
    assert lr_at(0, **SCHEDULE) == pytest.approx(1e-4)
    assert lr_at(4, **SCHEDULE) == pytest.approx(5e-4)
    assert lr_at(9, **SCHEDULE) == pytest.approx(1e-3)


def test_cosine_decays_from_peak_to_min_lr():
    assert lr_at(10, **SCHEDULE) == pytest.approx(1e-3)
    assert lr_at(60, **SCHEDULE) == pytest.approx(5.5e-4)  # halfway: midpoint of the two
    values = [lr_at(i, **SCHEDULE) for i in range(10, 111)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("iteration", [110, 111, 10_000])
def test_holds_min_lr_after_decay(iteration):
    assert lr_at(iteration, **SCHEDULE) == pytest.approx(1e-4)


def test_no_warmup_starts_at_peak():
    assert lr_at(0, **{**SCHEDULE, "warmup_iters": 0}) == pytest.approx(1e-3)


def grads_norm(model):
    return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))


def loss_of(model, tokenizer):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, tokenizer.vocab_size, (4, 16), generator=g)
    y = torch.randint(0, tokenizer.vocab_size, (4, 16), generator=g)
    return model(x, y)[1]


def test_optimizer_step_clips_global_grad_norm(make_model, tokenizer):
    model = make_model()
    optimizer = configure_optimizer(model, 1e-3, 0.1)
    norm = optimizer_step(model, optimizer, loss_of(model, tokenizer), grad_clip=1e-3)
    assert norm.item() > 1e-3
    assert grads_norm(model).item() == pytest.approx(1e-3, rel=1e-3)


def test_optimizer_step_without_clip_reports_norm(make_model, tokenizer):
    model = make_model()
    optimizer = configure_optimizer(model, 1e-3, 0.1)
    norm = optimizer_step(model, optimizer, loss_of(model, tokenizer), grad_clip=None)
    assert norm.item() == pytest.approx(grads_norm(model).item(), rel=1e-5)


@pytest.mark.parametrize(("precision", "expected"), [("auto", "fp32"), ("fp32", "fp32")])
def test_cpu_precision_is_fp32(precision, expected):
    assert resolve_precision(precision, "cpu") == expected


@pytest.mark.parametrize(
    ("precision", "message"), [("bf16", "needs a CUDA device"), ("fp16", "must be one of")]
)
def test_rejects_unusable_precision(precision, message):
    with pytest.raises(ValueError, match=message):
        resolve_precision(precision, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_precision_runs_bf16_on_cuda(make_model, tokenizer):
    assert resolve_precision("auto", "cuda") == "bf16"
    model = make_model().to("cuda")
    x = torch.randint(0, tokenizer.vocab_size, (2, 8), device="cuda")
    with autocast_context("bf16", "cuda"):
        logits, _ = model(x)
    assert logits.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_bits_per_char_for_char_tokenizer_is_loss_in_bits():
    assert bits_per_char(math.log(2), 1.0) == pytest.approx(1.0)


def test_bits_per_char_scales_by_tokens_per_char():
    """Four characters per token: the same loss per token is a quarter of the bits per char."""
    assert bits_per_char(4 * math.log(2), 0.25) == pytest.approx(1.0)
