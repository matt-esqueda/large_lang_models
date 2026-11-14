"""Optimizer and learning-rate schedule invariants."""

from gptlm.training import configure_optimizer


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
