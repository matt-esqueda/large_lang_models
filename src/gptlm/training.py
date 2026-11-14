"""Training loop shared by fresh runs and resumes."""

import contextlib
import csv
import math
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
    "grad_norm",
    "elapsed_seconds",
    "timestamp",
]


def configure_optimizer(model, learning_rate, weight_decay):
    """AdamW with weight decay on matrix weights only.

    Linear and embedding weights (2-D) are decayed. Biases and LayerNorm
    parameters (1-D) are not: decaying them pulls offsets and scales toward
    zero without regularizing anything.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate)


def lr_at(iteration, *, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    """Learning rate for one iteration: linear warmup, cosine decay, then min_lr.

    The rate rises linearly to learning_rate over warmup_iters steps, follows
    a cosine down to min_lr at lr_decay_iters, and holds at min_lr after that,
    so a run resumed past the end of its schedule continues at min_lr.
    """
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / warmup_iters
    if iteration >= lr_decay_iters:
        return min_lr
    progress = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


PRECISIONS = ("auto", "bf16", "fp32")


def resolve_precision(precision, device):
    """Map the precision setting to "bf16" or "fp32" for a device.

    auto means bf16 autocast on a CUDA GPU that supports it and fp32
    everywhere else. bf16 keeps fp32's exponent range, so unlike fp16 it
    needs no loss scaling.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, not {precision!r}")
    if precision == "auto":
        return "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and device != "cuda":
        raise ValueError("precision bf16 needs a CUDA device; use fp32 or auto")
    return precision


def autocast_context(precision, device_type):
    """Context for forward passes: bf16 autocast, or a no-op for fp32.

    Weights, gradients and optimizer state stay fp32 either way; autocast
    only runs matmuls and attention in bf16.
    """
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def optimizer_step(model, optimizer, loss, grad_clip):
    """Backpropagate loss, clip the global gradient norm to grad_clip, and step.

    Returns the norm before clipping as a tensor: converting it to a number
    every step would wait for the GPU. grad_clip None measures without clipping.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or math.inf)
    optimizer.step()
    return grad_norm


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters, precision="fp32"):
    """Mean loss over eval_iters batches of each split, measured in eval mode."""
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, model.block_size)
            with autocast_context(precision, x.device.type):
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
    """CSV with one row per loss evaluation.

    grad_norm is the most recent step's norm before clipping, empty before
    the first step.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_FIELDS)

    def write(self, iteration, losses, elapsed, learning_rate, grad_norm=None):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    iteration,
                    f"{losses['train']:.6f}",
                    f"{losses['val']:.6f}",
                    f"{learning_rate:.6e}",
                    "" if grad_norm is None else f"{float(grad_norm):.6f}",
                    f"{elapsed:.2f}",
                    datetime.now().isoformat(),
                ]
            )


def train(
    model,
    optimizer,
    dataset,
    *,
    schedule,
    grad_clip,
    precision="fp32",
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

    schedule holds lr_at's keyword arguments. It sets the learning rate at
    every step and is saved in each checkpoint, so a resume continues it.
    precision ("bf16" or "fp32", from resolve_precision) applies to every
    forward pass; backward runs outside autocast.

    Returns (final_losses, final_checkpoint_path).
    """
    end_iter = start_iter + num_iters
    start_time = time.time()
    model.train()
    lr = lr_at(start_iter, **schedule)
    grad_norm = None

    for step in range(num_iters):
        iteration = start_iter + step
        lr = lr_at(iteration, **schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = None

        if step == 0 or iteration % eval_interval == 0 or step == num_iters - 1:
            losses = estimate_loss(model, dataset, batch_size, eval_iters, precision)
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (num_iters - step - 1) / rate if rate > 0 else 0
            print(
                f"step {iteration:5d}/{end_iter} ({(step + 1) / num_iters:6.1%}) | "
                f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
                f"lr: {lr:.2e} | time: {format_time(elapsed)} | ETA: {format_time(eta)}"
            )
            metrics.write(iteration, losses, elapsed, lr, grad_norm)

        if step > 0 and iteration % checkpoint_interval == 0:
            if losses is None:
                losses = estimate_loss(model, dataset, batch_size, eval_iters, precision)
            path = os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")
            print(f"  Saving checkpoint: {path}")
            save_checkpoint(
                path,
                model,
                optimizer,
                iteration=iteration,
                train_loss=losses["train"],
                val_loss=losses["val"],
                schedule=schedule,
            )

        x, y = dataset.get_batch("train", batch_size, model.block_size)
        with autocast_context(precision, x.device.type):
            _, loss = model(x, y)
        grad_norm = optimizer_step(model, optimizer, loss, grad_clip)

    losses = estimate_loss(model, dataset, batch_size, eval_iters, precision)
    total = time.time() - start_time
    final_path = os.path.join(checkpoint_dir, final_name)
    save_checkpoint(
        final_path,
        model,
        optimizer,
        iteration=end_iter,
        train_loss=losses["train"],
        val_loss=losses["val"],
        schedule=schedule,
    )
    metrics.write(end_iter, losses, total, lr, grad_norm)
    print(
        f"\nFinal train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | "
        f"time: {format_time(total)}"
    )
    print(f"Saved {final_path}")
    return losses, final_path
