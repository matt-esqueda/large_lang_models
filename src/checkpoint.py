"""Checkpoint save/load for GPT language models.

Checkpoints store a state_dict plus the config needed to rebuild the model,
rather than a pickled nn.Module. Pickling the module records the class's
import path, so renaming or moving anything in src/model.py invalidates
every existing checkpoint. It also embeds device-specific tensor storage,
which makes a GPU-trained checkpoint unloadable on a CPU-only machine.
"""

import os
import torch

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path, model, optimizer=None, iteration=0, train_loss=None, val_loss=None, vocab_size=None
):
    """Save a checkpoint atomically.

    Writes to a temporary file first, then renames. An interrupted save
    therefore leaves the previous checkpoint intact rather than a truncated
    file that fails to load.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size if vocab_size is not None else model.lm_head.out_features,
            "n_embd": model.token_embedding_table.embedding_dim,
            "n_head": model.blocks[0].sa.heads.__len__(),
            "n_layer": len(model.blocks),
            "block_size": model.block_size,
        },
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


def load_checkpoint(path, device="cpu", model_class=None, optimizer=None):
    """Load a checkpoint and rebuild the model on the requested device.

    Returns (model, metadata). metadata has iteration, train_loss, val_loss
    and config. If an optimizer is supplied and the checkpoint carries
    optimizer state, that state is restored in place.
    """
    if model_class is None:
        from src.model import GPTLanguageModel as model_class

    payload = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(
            f"{path} is not a state_dict checkpoint. Checkpoints written by "
            f"earlier versions used pickle and are not loadable; retrain or "
            f"convert them."
        )

    cfg = payload["config"]
    model = model_class(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
    )
    model.load_state_dict(payload["model_state"])
    model = model.to(device)

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    metadata = {
        "iteration": payload.get("iteration", 0),
        "train_loss": payload.get("train_loss"),
        "val_loss": payload.get("val_loss"),
        "config": cfg,
        "has_optimizer_state": "optimizer_state" in payload,
    }
    return model, metadata
