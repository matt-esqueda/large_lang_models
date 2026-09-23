"""Checkpoint save/load invariants."""

import os

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


def restore_optimizer(model, path):
    """Rebuild an optimizer the way scripts/resume_training.py does."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return optimizer


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
    assert meta["config"] == {
        "vocab_size": tokenizer.vocab_size,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "block_size": 16,
    }


def test_loaded_model_gives_identical_logits(trained, tokenizer):
    model, _, path = trained
    loaded, _ = load_checkpoint(path, device="cpu")
    loaded.eval()  # load_checkpoint rebuilds with the default dropout
    x, _ = make_batch(tokenizer.vocab_size, seed=1)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x)[0], model(x)[0], rtol=0, atol=0)


def test_round_trip_optimizer_state_is_identical(trained):
    model, optimizer, path = trained
    loaded, _ = load_checkpoint(path, device="cpu")
    restored = restore_optimizer(loaded, path).state_dict()
    original = optimizer.state_dict()
    assert restored["param_groups"] == original["param_groups"]
    assert restored["state"].keys() == original["state"].keys()
    for idx, state in original["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(restored["state"][idx][key], value, rtol=0, atol=0)


def test_resumed_training_matches_uninterrupted(trained, tokenizer):
    """Regression: a fresh AdamW on resume discarded the moment estimates."""
    model, optimizer, path = trained
    loaded, _ = load_checkpoint(path, device="cpu")
    loaded.eval()
    resumed_optimizer = restore_optimizer(loaded, path)

    x, y = make_batch(tokenizer.vocab_size, seed=2)
    train_steps(model, optimizer, x, y, steps=2)
    train_steps(loaded, resumed_optimizer, x, y, steps=2)

    resumed = loaded.state_dict()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed[name], tensor)


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
    with pytest.raises(ValueError, match="not a state_dict checkpoint"):
        load_checkpoint(path, device="cpu")


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
