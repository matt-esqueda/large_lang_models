"""Optimizer and learning-rate schedule invariants."""

import pytest

from gptlm.training import configure_optimizer, lr_at

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
