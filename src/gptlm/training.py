"""Training loop shared by fresh runs and resumes."""

import csv
import os
import time
from datetime import datetime

import torch

from gptlm.checkpoint import save_checkpoint

METRICS_FIELDS = [
    "iteration",
    "train_loss",
    "val_loss",
    "learning_rate",
    "elapsed_seconds",
    "timestamp",
]


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters):
    """Mean loss over eval_iters batches of each split, measured in eval mode."""
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, model.block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class MetricsLog:
    """CSV with one row per loss evaluation."""

    def __init__(self, path, learning_rate):
        self.path = path
        self.learning_rate = learning_rate
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_FIELDS)

    def write(self, iteration, losses, elapsed):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    iteration,
                    f"{losses['train']:.6f}",
                    f"{losses['val']:.6f}",
                    f"{self.learning_rate:.6e}",
                    f"{elapsed:.2f}",
                    datetime.now().isoformat(),
                ]
            )


def train(
    model,
    optimizer,
    dataset,
    *,
    start_iter,
    num_iters,
    batch_size,
    eval_iters,
    eval_interval,
    checkpoint_interval,
    checkpoint_dir,
    metrics,
    final_name,
):
    """Run num_iters optimizer steps, continuing the count from start_iter.

    Evaluations and intermediate checkpoints align to absolute iteration
    numbers, so a resumed run saves model_iter500.pt, model_iter1000.pt, ...
    exactly as an uninterrupted run would. A checkpoint's iteration is the
    number of optimizer steps completed.

    Returns (final_losses, final_checkpoint_path).
    """
    end_iter = start_iter + num_iters
    start_time = time.time()
    model.train()

    for step in range(num_iters):
        iteration = start_iter + step
        losses = None

        if step == 0 or iteration % eval_interval == 0 or step == num_iters - 1:
            losses = estimate_loss(model, dataset, batch_size, eval_iters)
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (num_iters - step - 1) / rate if rate > 0 else 0
            print(
                f"step {iteration:5d}/{end_iter} ({(step + 1) / num_iters:6.1%}) | "
                f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
                f"time: {format_time(elapsed)} | ETA: {format_time(eta)}"
            )
            metrics.write(iteration, losses, elapsed)

        if step > 0 and iteration % checkpoint_interval == 0:
            if losses is None:
                losses = estimate_loss(model, dataset, batch_size, eval_iters)
            path = os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")
            print(f"  Saving checkpoint: {path}")
            save_checkpoint(
                path,
                model,
                optimizer,
                iteration=iteration,
                train_loss=losses["train"],
                val_loss=losses["val"],
            )

        x, y = dataset.get_batch("train", batch_size, model.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(model, dataset, batch_size, eval_iters)
    total = time.time() - start_time
    final_path = os.path.join(checkpoint_dir, final_name)
    save_checkpoint(
        final_path,
        model,
        optimizer,
        iteration=end_iter,
        train_loss=losses["train"],
        val_loss=losses["val"],
    )
    metrics.write(end_iter, losses, total)
    print(
        f"\nFinal train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | "
        f"time: {format_time(total)}"
    )
    print(f"Saved {final_path}")
    return losses, final_path
