#!/usr/bin/env python3
"""
Training script for GPT Language Model
Enhanced with checkpointing and metrics logging
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch

from gptlm.checkpoint import save_checkpoint as save_ckpt
from gptlm.data import CorpusDataset, set_seed
from gptlm.model import GPTLanguageModel
from gptlm.tokenizer import CharacterTokenizer

# Argument parser
parser = argparse.ArgumentParser(description='Train GPT Language Model')
parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('-block_size', type=int, default=64, help='Context window size')
parser.add_argument('-max_iters', type=int, default=5000, help='Maximum training iterations')
parser.add_argument('-learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument(
    '-eval_iters', type=int, default=100, help='Number of iterations for loss evaluation'
)
parser.add_argument(
    '-eval_interval', type=int, default=500, help='Evaluate loss every N iterations'
)
parser.add_argument(
    '-checkpoint_interval', type=int, default=500, help='Save checkpoint every N iterations'
)
parser.add_argument('-n_embd', type=int, default=384, help='Embedding dimension')
parser.add_argument('-n_head', type=int, default=6, help='Number of attention heads')
parser.add_argument('-n_layer', type=int, default=6, help='Number of transformer layers')
parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('-seed', type=int, default=None, help='Random seed for reproducible runs')

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Hyperparameters
batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
learning_rate = args.learning_rate
eval_iters = args.eval_iters
eval_interval = args.eval_interval
checkpoint_interval = args.checkpoint_interval
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
TRAIN_FILE = 'data/processed/train_split.txt'
VAL_FILE = 'data/processed/val_split.txt'
MODEL_DIR = 'models/checkpoints'
LOG_DIR = 'logs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create metrics log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metrics_file = os.path.join(LOG_DIR, f'training_metrics_{timestamp}.csv')

# Load tokenizer
print("Loading tokenizer...")
tokenizer = CharacterTokenizer(VOCAB_FILE)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

if args.seed is not None:
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

dataset = CorpusDataset(TRAIN_FILE, VAL_FILE, tokenizer, device=device)
print(f"Train tokens: {dataset.token_count('train'):,}")
print(f"Val tokens:   {dataset.token_count('val'):,}")


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(iteration, train_loss, val_loss):
    """Save model checkpoint"""
    checkpoint_name = f'model_iter{iteration}.pt'
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)

    print(f"  Saving checkpoint: {checkpoint_name}")
    save_ckpt(
        checkpoint_path,
        model,
        optimizer,
        iteration=iteration,
        train_loss=train_loss,
        val_loss=val_loss,
        vocab_size=vocab_size,
    )

    return checkpoint_path


def log_metrics(iteration, train_loss, val_loss, elapsed_time):
    """Log metrics to CSV file"""
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(
                [
                    'iteration',
                    'train_loss',
                    'val_loss',
                    'learning_rate',
                    'elapsed_seconds',
                    'timestamp',
                ]
            )

        # Write metrics
        writer.writerow(
            [
                iteration,
                f'{train_loss:.6f}',
                f'{val_loss:.6f}',
                f'{learning_rate:.6e}',
                f'{elapsed_time:.2f}',
                datetime.now().isoformat(),
            ]
        )


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# Initialize model
print("Initializing model...")
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
    device=device,
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training configuration summary
print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"Max iterations:       {max_iters:,}")
print(f"Batch size:           {batch_size}")
print(f"Block size:           {block_size}")
print(f"Learning rate:        {learning_rate:.6f}")
print(f"Model layers:         {n_layer}")
print(f"Attention heads:      {n_head}")
print(f"Embedding dim:        {n_embd}")
print(f"Dropout:              {dropout}")
print(f"Eval interval:        {eval_interval}")
print(f"Checkpoint interval:  {checkpoint_interval}")
print(f"Metrics log:          {metrics_file}")
print("=" * 70 + "\n")

# Training loop
print("Starting training...\n")
start_time = time.time()
last_checkpoint_time = start_time

for iter in range(max_iters):
    losses = None

    # Evaluate loss and save checkpoint periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time

        # Calculate progress metrics
        progress = (iter + 1) / max_iters * 100
        iters_per_sec = (iter + 1) / elapsed if elapsed > 0 else 0
        remaining_iters = max_iters - (iter + 1)
        eta_seconds = remaining_iters / iters_per_sec if iters_per_sec > 0 else 0

        # Print progress
        print(
            f"step {iter:5d}/{max_iters} ({progress:5.1f}%) | "
            f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
            f"time: {format_time(elapsed)} | ETA: {format_time(eta_seconds)}"
        )

        # Log metrics
        log_metrics(iter, losses['train'], losses['val'], elapsed)

    # Save checkpoint at intervals (independent of eval_interval)
    if iter > 0 and iter % checkpoint_interval == 0:
        # Reuse this iteration's eval if one was just computed
        ckpt_losses = losses if losses is not None else estimate_loss()
        save_checkpoint(iter, ckpt_losses['train'], ckpt_losses['val'])
        last_checkpoint_time = time.time()

    # Get batch and compute loss
    xb, yb = dataset.get_batch('train', batch_size, block_size)
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
losses = estimate_loss()
total_time = time.time() - start_time
print(f"Final train loss: {losses['train']:.4f}")
print(f"Final val loss:   {losses['val']:.4f}")
print(f"Total time:       {format_time(total_time)}")
print(f"Avg time/iter:    {total_time / max_iters:.3f}s")

# Save final model
final_model_path = os.path.join(MODEL_DIR, 'model_final.pt')
print(f"\nSaving final model to {final_model_path}...")
save_ckpt(
    final_model_path,
    model,
    optimizer,
    iteration=max_iters,
    train_loss=losses['train'],
    val_loss=losses['val'],
    vocab_size=vocab_size,
)
print("Model saved successfully!")

# Log final metrics
log_metrics(max_iters, losses['train'], losses['val'], total_time)

print(f"\nMetrics saved to: {metrics_file}")
print(f"Checkpoints saved in: {MODEL_DIR}/")
print("=" * 70 + "\n")
