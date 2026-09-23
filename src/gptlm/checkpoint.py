"""Checkpoint save/load for GPT language models.

Checkpoints store a state_dict plus the config needed to rebuild the model,
rather than a pickled nn.Module. Pickling the module records the class's
import path, so renaming or moving anything in the model module invalidates
every existing checkpoint. It also embeds device-specific tensor storage,
which makes a GPU-trained checkpoint unloadable on a CPU-only machine.
"""

import os
import pickle

import torch

CHECKPOINT_VERSION = 1


def _not_a_checkpoint(path):
    return ValueError(
        f"{path} is not a loadable state_dict checkpoint. Files written by "
        f"earlier versions were pickled modules and cannot be loaded safely; "
        f"retrain or convert them."
    )


def save_checkpoint(
    path, model, optimizer=None, iteration=0, train_loss=None, val_loss=None, vocab_size=None
):
    """Save a checkpoint atomically.

    Writes to a temporary file first, then renames. An interrupted save
    therefore leaves the previous checkpoint intact rather than a truncated
    file that fails to load.

    The architecture config comes from model.config. vocab_size is accepted
    for backward compatibility and must match it if given.
    """
    config = dict(model.config)
    if vocab_size is not None and vocab_size != config["vocab_size"]:
        raise ValueError(
            f"vocab_size={vocab_size} does not match the model's ({config['vocab_size']})"
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "config": config,
        "iteration": iteration,
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, device="cpu", model_class=None):
    """Load a checkpoint and rebuild the model on the requested device.

    Returns (model, metadata). metadata has iteration, train_loss, val_loss,
    config and optimizer_state (None if the checkpoint has none). To resume,
    build the optimizer from the returned model's parameters, then call
    optimizer.load_state_dict(metadata["optimizer_state"]). An optimizer
    cannot be passed in, because the model's parameters do not exist yet.

    Loads with weights_only=True: the payload holds only tensors and
    primitives, so no arbitrary code runs on load.
    """
    if model_class is None:
        from gptlm.model import GPTLanguageModel as model_class

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as e:
        raise _not_a_checkpoint(path) from e

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise _not_a_checkpoint(path)

    # Checkpoints saved before dropout was recorded fall back to the
    # model's default rate.
    config = payload["config"]
    model = model_class(**config)
    model.load_state_dict(payload["model_state"])
    model = model.to(device)

    metadata = {
        "iteration": payload.get("iteration", 0),
        "train_loss": payload.get("train_loss"),
        "val_loss": payload.get("val_loss"),
        "config": config,
        "optimizer_state": payload.get("optimizer_state"),
        "has_optimizer_state": "optimizer_state" in payload,
    }
    return model, metadata
