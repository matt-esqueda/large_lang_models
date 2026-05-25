#!/usr/bin/env python3
"""
Resume training from a checkpoint
"""

import os
import sys
import torch
import pickle
import argparse
import csv
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

# Argument parser
parser = argparse.ArgumentParser(description='Resume training from checkpoint')
parser.add_argument('-checkpoint', type=str, required=True,
                    help='Checkpoint filename in models/checkpoints/')
parser.add_argument('-additional_iters', type=int, required=True,
                    help='Additional iterations to train')
parser.add_argument('-batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('-learning_rate', type=float, default=3e-4,
                    help='Learning rate (default: 3e-4)')
parser.add_argument('-eval_iters', type=int, default=100,
                    help='Number of iterations for loss evaluation')
parser.add_argument('-eval_interval', type=int, default=500,
                    help='Evaluate loss every N iterations')
parser.add_argument('-checkpoint_interval', type=int, default=500,
                    help='Save checkpoint every N iterations')

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
TRAIN_FILE = 'data/processed/train_split.txt'
VAL_FILE = 'data/processed/val_split.txt'
CHECKPOINT_PATH = os.path.join('models/checkpoints', args.checkpoint)
MODEL_DIR = 'models/checkpoints'
LOG_DIR = 'logs'

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
    print("\nAvailable checkpoints:")
    if os.path.exists(MODEL_DIR):
        checkpoints = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        for ckpt in sorted(checkpoints):
            size_mb = os.path.getsize(os.path.join(MODEL_DIR, ckpt)) / (1024*1024)
            print(f"  - {ckpt} ({size_mb:.1f} MB)")
    sys.exit(1)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = CharacterTokenizer(VOCAB_FILE)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Load checkpoint
print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
with open(CHECKPOINT_PATH, 'rb') as f:
    model = pickle.load(f)

model = model.to(device)
model.train()
print("Checkpoint loaded successfully!")

# Get model info
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Extract starting iteration from checkpoint name
import re
iter_match = re.search(r'iter(\d+)', args.checkpoint)
start_iter = int(iter_match.group(1)) if iter_match else 0
end_iter = start_iter + args.additional_iters

print(f"\nResuming from iteration: {start_iter}")
print(f"Will train until iteration: {end_iter}")
print(f"Additional iterations: {args.additional_iters}")

# Create metrics log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
metrics_file = os.path.join(LOG_DIR, f'resume_training_{timestamp}.csv')

# Optimizer (create fresh optimizer for resumed training)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Import training functions from train.py
import random
import torch.nn.functional as F

def get_random_chunk(split):
    """Read a random chunk of text from file"""
    filename = TRAIN_FILE if split == 'train' else VAL_FILE
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    block_size = model.block_size
    batch_size = args.batch_size
    required_chars = block_size * batch_size + block_size

    if len(text) < required_chars:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    else:
        start_idx = random.randint(0, len(text) - required_chars)
        chunk = text[start_idx:start_idx + required_chars]
        data = torch.tensor(tokenizer.encode(chunk), dtype=torch.long)
    
    return data


def get_batch(split):
    """Generate a batch of data"""
    data = get_random_chunk(split)
    block_size = model.block_size

    if len(data) <= block_size:
        raise ValueError(f"Data chunk too small: {len(data)} tokens")
    
    max_idx = len(data) - block_size - 1
    if max_idx < 1:
        raise ValueError(f"Not enough tokens in data: {len(data)}")

    ix = torch.randint(0, max_idx, (args.batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(iteration, train_loss, val_loss):
    """Save model checkpoint"""
    checkpoint_name = f'model_iter{iteration}.pkl'
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)
    
    print(f"  Saving checkpoint: {checkpoint_name}")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(model, f)
    
    return checkpoint_path


def log_metrics(iteration, train_loss, val_loss, elapsed_time):
    """Log metrics to CSV file"""
    file_exists = os.path.isfile(metrics_file)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['iteration', 'train_loss', 'val_loss', 'learning_rate', 'elapsed_seconds', 'timestamp'])
        
        writer.writerow([
            iteration,
            f'{train_loss:.6f}',
            f'{val_loss:.6f}',
            f'{args.learning_rate:.6e}',
            f'{elapsed_time:.2f}',
            datetime.now().isoformat()
        ])


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


# Training configuration summary
print("\n" + "="*70)
print("RESUMED TRAINING CONFIGURATION")
print("="*70)
print(f"Starting iteration:   {start_iter:,}")
print(f"Ending iteration:     {end_iter:,}")
print(f"Additional iters:     {args.additional_iters:,}")
print(f"Batch size:           {args.batch_size}")
print(f"Block size:           {model.block_size}")
print(f"Learning rate:        {args.learning_rate:.6f}")
print(f"Eval interval:        {args.eval_interval}")
print(f"Checkpoint interval:  {args.checkpoint_interval}")
print(f"Metrics log:          {metrics_file}")
print("="*70 + "\n")

# Get initial loss
print("Evaluating initial loss...")
initial_losses = estimate_loss()
print(f"Initial train loss: {initial_losses['train']:.4f}")
print(f"Initial val loss:   {initial_losses['val']:.4f}\n")

# Training loop
print("Resuming training...\n")
start_time = time.time()

for iter in range(args.additional_iters):
    current_iter = start_iter + iter
    
    # Evaluate loss and save checkpoint periodically
    if iter % args.eval_interval == 0 or iter == args.additional_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        
        progress = (iter + 1) / args.additional_iters * 100
        iters_per_sec = (iter + 1) / elapsed if elapsed > 0 else 0
        remaining_iters = args.additional_iters - (iter + 1)
        eta_seconds = remaining_iters / iters_per_sec if iters_per_sec > 0 else 0
        
        print(f"step {current_iter:5d}/{end_iter} ({progress:5.1f}%) | "
              f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
              f"time: {format_time(elapsed)} | ETA: {format_time(eta_seconds)}")
        
        log_metrics(current_iter, losses['train'], losses['val'], elapsed)
        
        if iter > 0 and iter % args.checkpoint_interval == 0:
            save_checkpoint(current_iter, losses['train'], losses['val'])
    
    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
print("\n" + "="*70)
print("RESUMED TRAINING COMPLETE")
print("="*70)
losses = estimate_loss()
total_time = time.time() - start_time
print(f"Final train loss: {losses['train']:.4f}")
print(f"Final val loss:   {losses['val']:.4f}")
print(f"Total time:       {format_time(total_time)}")
print(f"Avg time/iter:    {total_time/args.additional_iters:.3f}s")

# Save final model
final_model_path = os.path.join(MODEL_DIR, f'model_iter{end_iter}.pkl')
print(f"\nSaving final model to {final_model_path}...")
with open(final_model_path, 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully!")

log_metrics(end_iter, losses['train'], losses['val'], total_time)

print(f"\nMetrics saved to: {metrics_file}")
print(f"Checkpoints saved in: {MODEL_DIR}/")
print("="*70 + "\n")