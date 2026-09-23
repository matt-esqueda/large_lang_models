"""Checkpoint save/load invariants."""

import os
import pickle

import pytest
import torch

from gptlm.checkpoint import load_checkpoint, save_checkpoint


def make_batch(vocab_size, seed=0, batch=4, length=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch, length), generator=g)
    y = torch.randint(0, vocab_size, (batch, length), generator=g)
    return x, y


def train_steps(model, optimizer, x, y, steps):
    for _ in range(steps):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def restore_optimizer(model, meta):
    """Rebuild an optimizer the way scripts/resume_training.py does."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(meta["optimizer_state"])
    return optimizer


def dropout_rates(model):
    return {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}


@pytest.fixture
def trained(make_model, tokenizer, tmp_path):
    """A model and optimizer after a few steps, saved to disk."""
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = make_batch(tokenizer.vocab_size)
    train_steps(model, optimizer, x, y, steps=3)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, iteration=3, train_loss=1.5, val_loss=1.75)
    return model, optimizer, path


def test_round_trip_weights_are_identical(trained):
    model, _, path = trained
    loaded, _ = load_checkpoint(path, device="cpu")
    original = model.state_dict()
    restored = loaded.state_dict()
    assert original.keys() == restored.keys()
    for name, tensor in original.items():
        assert torch.equal(tensor, restored[name]), name


def test_round_trip_metadata(trained, tokenizer):
    _, _, path = trained
    _, meta = load_checkpoint(path, device="cpu")
    assert meta["iteration"] == 3
    assert meta["train_loss"] == 1.5
    assert meta["val_loss"] == 1.75
    assert meta["has_optimizer_state"]
    assert meta["optimizer_state"] is not None
    assert meta["config"] == {
        "vocab_size": tokenizer.vocab_size,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "block_size": 16,
        "dropout": 0.0,
    }


def test_loaded_model_gives_identical_logits(trained, tokenizer):
    model, _, path = trained
    loaded, _ = load_checkpoint(path, device="cpu")
    loaded.eval()
    x, _ = make_batch(tokenizer.vocab_size, seed=1)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x)[0], model(x)[0], rtol=0, atol=0)


def test_round_trip_optimizer_state_is_identical(trained):
    _, optimizer, path = trained
    loaded, meta = load_checkpoint(path, device="cpu")
    restored = restore_optimizer(loaded, meta).state_dict()
    original = optimizer.state_dict()
    assert restored["param_groups"] == original["param_groups"]
    assert restored["state"].keys() == original["state"].keys()
    for idx, state in original["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(restored["state"][idx][key], value, rtol=0, atol=0)


def test_resumed_training_matches_uninterrupted(trained, tokenizer):
    """Regression: a fresh AdamW on resume discarded the moment estimates.

    The loaded model stays in train mode, so this also fails if dropout is
    not restored from the checkpoint.
    """
    model, optimizer, path = trained
    loaded, meta = load_checkpoint(path, device="cpu")
    resumed_optimizer = restore_optimizer(loaded, meta)

    x, y = make_batch(tokenizer.vocab_size, seed=2)
    train_steps(model, optimizer, x, y, steps=2)
    train_steps(loaded, resumed_optimizer, x, y, steps=2)

    resumed = loaded.state_dict()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed[name], tensor)


def test_dropout_round_trips(make_model, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, make_model(dropout=0.1))
    loaded, meta = load_checkpoint(path, device="cpu")
    assert meta["config"]["dropout"] == 0.1
    assert dropout_rates(loaded) == {0.1}


def test_loads_checkpoint_saved_without_dropout(make_model, tmp_path):
    """Checkpoints from before dropout was recorded load with the default rate."""
    path = str(tmp_path / "old.pt")
    save_checkpoint(path, make_model())
    payload = torch.load(path, weights_only=True)
    del payload["config"]["dropout"]
    torch.save(payload, path)
    loaded, _ = load_checkpoint(path, device="cpu")
    assert dropout_rates(loaded) == {0.2}


def test_save_rejects_mismatched_vocab_size(make_model, tokenizer, tmp_path):
    with pytest.raises(ValueError, match="vocab_size"):
        save_checkpoint(str(tmp_path / "x.pt"), make_model(), vocab_size=tokenizer.vocab_size + 1)


def test_checkpoint_loads_with_weights_only(trained):
    """The payload is tensors and primitives only, so the safe loader accepts it."""
    _, _, path = trained
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "model_state" in payload


def test_save_leaves_no_temp_file(trained):
    _, _, path = trained
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_rejects_non_state_dict_file(tmp_path):
    path = str(tmp_path / "bogus.pt")
    torch.save({"weights": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="state_dict checkpoint"):
        load_checkpoint(path, device="cpu")


@pytest.mark.filterwarnings("ignore:Detected pickle protocol")
def test_rejects_legacy_pickle_checkpoint(make_model, tmp_path):
    """Checkpoints before PR 3 were a pickled nn.Module."""
    path = str(tmp_path / "legacy.pkl")
    with open(path, "wb") as f:
        pickle.dump(make_model(), f)
    with pytest.raises(ValueError, match="state_dict checkpoint"):
        load_checkpoint(path, device="cpu")


EXECUTED = []


def _record_execution():
    EXECUTED.append(True)
    return "payload"


class _CodeOnLoad:
    def __reduce__(self):
        return (_record_execution, ())


def test_load_never_executes_code(tmp_path):
    """A pickle that runs code on load must be rejected before it runs."""
    path = str(tmp_path / "evil.pt")
    torch.save({"model_state": {}, "config": {}, "extra": _CodeOnLoad()}, path)
    EXECUTED.clear()
    with pytest.raises(ValueError, match="state_dict checkpoint"):
        load_checkpoint(path, device="cpu")
    assert not EXECUTED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_checkpoint_loads_on_cpu(make_model, tmp_path):
    model = make_model().to("cuda")
    path = str(tmp_path / "gpu.pt")
    save_checkpoint(path, model)
    loaded, meta = load_checkpoint(path, device="cpu")
    assert not meta["has_optimizer_state"]
    restored = loaded.state_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(restored[name], tensor.cpu()), name
