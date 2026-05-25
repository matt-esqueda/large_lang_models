#!/usr/bin/env python3
"""
Training script for GPT Language Model
"""

import os
import sys
import torch
import torch.nn as nn
import mmap
import random
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

# Argument parser
parser = argparse.ArgumentParser(description='Train GPT Language Model')
parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
parser.add_argument('-block_size', type=int, default=64, help='Context window size')
parser.add_argument('-max_iters', type=int, default=5000, help='Maximum training iterations')
parser.add_argument('-learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('-eval_iters', type=int, default=100, help='Evaluation interval')
parser.add_argument('-n_embd', type=int, default=384, help='Embedding dimension')
parser.add_argument('-n_head', type=int, default=6, help='Number of attention heads')
parser.add_argument('-n_layer', type=int, default=6, help='Number of transformer layers')
parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate')

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Hyperparameters
batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
learning_rate = args.learning_rate
eval_iters = args.eval_iters
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
TRAIN_FILE = 'data/processed/train_split.txt'
VAL_FILE = 'data/processed/val_split.txt'
MODEL_DIR = 'models/checkpoints'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = CharacterTokenizer(VOCAB_FILE)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")


def get_random_chunk(split):
    """Read a random chunk of text from file"""
    filename = TRAIN_FILE if split == 'train' else VAL_FILE
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Calculate how many characters we need
    required_chars = block_size * batch_size + block_size

    if len(text) < required_chars:
        # If file is smaller, just use the whole thing
        data = torch.sensor(tokenizer.encode(text), dtype=torch.long)
    else:
        # Pick a random starting point
        start_idx = random.randint(0, len(text) - required_chars)
        chunk = text[start_idx:start_idx + required_chars]
        data = torch.tensor(tokenizer.encode(chunk), dtype=torch.long)
    
    return data


def get_batch(split):
    """Generate a batch of data"""
    data = get_random_chunk(split)

    # Ensure we have enough data
    if len(data) <= block_size:
        raise ValueError(f"Data chunk too small: {len(data)} tokens, need at least {block_size + 1}")
    
    # Generate random indices
    max_idx = len(data) - block_size - 1
    if max_idx < 1:
        raise ValueError(f"Not enough tokens in data: {len(data)}")

    ix = torch.randint(0, max_idx,(batch_size,))

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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initialize model
print("Initializing model...")
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
    device=device
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
print(f"\nStarting training for {max_iters} iterations...")
print(f"Config: batch_size={batch_size}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}\n")

for iter in range(max_iters):
    # Evaluate loss periodically
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
    
    # Get batch and compute loss
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nFinal loss: {loss.item():.4f}")

# Save model
model_path = os.path.join(MODEL_DIR, 'model_01.pkl')
print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(" Model saved successfully!")
