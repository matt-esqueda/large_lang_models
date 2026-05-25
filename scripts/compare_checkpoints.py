#!/usr/bin/env python3
"""
Compare multiple model checkpoints side-by-side
Generates the same prompts from each checkpoint to compare quality
"""

import os
import sys
import torch
import pickle
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

# Argument parser
parser = argparse.ArgumentParser(description='Compare multiple model checkpoints')
parser.add_argument('-checkpoints', type=str, nargs='+', required=True,
                    help='List of checkpoint filenames to compare')
parser.add_argument('-prompts', type=str, nargs='+', 
                    default=["Once upon a time", "The wizard said", "Dorothy walked"],
                    help='Prompts to test (default: Wizard of Oz themed)')
parser.add_argument('-max_tokens', type=int, default=100,
                    help='Maximum tokens to generate per prompt (default: 100)')
parser.add_argument('-temperature', type=float, default=0.8,
                    help='Sampling temperature (default: 0.8)')
parser.add_argument('-top_p', type=float, default=0.9,
                    help='Nucleus sampling threshold (default: 0.9)')
parser.add_argument('-save', type=str, default=None,
                    help='Save comparison to file (default: print to stdout)')

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
MODEL_DIR = 'models/checkpoints'

# Load tokenizer
print("Loading tokenizer...")
tokenizer = CharacterTokenizer(VOCAB_FILE)
print(f"Vocabulary size: {tokenizer.vocab_size}\n")

# Load all checkpoints
models = {}
print("Loading checkpoints...")
for ckpt_name in args.checkpoints:
    ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        continue
    
    print(f"  Loading {ckpt_name}...")
    with open(ckpt_path, 'rb') as f:
        model = pickle.load(f)
    
    model.eval()
    model = model.to(device)
    models[ckpt_name] = model

if len(models) == 0:
    print("\nError: No valid checkpoints loaded")
    sys.exit(1)

print(f"\nLoaded {len(models)} checkpoint(s) successfully\n")

# Comparison header
output_lines = []
output_lines.append("=" * 100)
output_lines.append("MODEL CHECKPOINT COMPARISON")
output_lines.append("=" * 100)
output_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_lines.append(f"Device: {device}")
output_lines.append(f"Temperature: {args.temperature}")
output_lines.append(f"Top-p: {args.top_p}")
output_lines.append(f"Max tokens: {args.max_tokens}")
output_lines.append(f"Checkpoints: {', '.join(models.keys())}")
output_lines.append("=" * 100)
output_lines.append("")

# Generate from each prompt
for prompt_idx, prompt in enumerate(args.prompts, 1):
    output_lines.append(f"\n{'='*100}")
    output_lines.append(f"PROMPT {prompt_idx}: \"{prompt}\"")
    output_lines.append(f"{'='*100}\n")
    
    # Encode prompt once
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    context_batch = context.unsqueeze(0)
    
    # Generate from each checkpoint
    for ckpt_name, model in models.items():
        output_lines.append(f"\n{'-'*100}")
        output_lines.append(f"Checkpoint: {ckpt_name}")
        output_lines.append(f"{'-'*100}")
        
        with torch.no_grad():
            try:
                # Try with advanced sampling parameters
                generated = model.generate(
                    context_batch.clone(),
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
            except TypeError:
                # Fall back to basic generation if model doesn't support parameters
                print(f"  Warning: {ckpt_name} doesn't support advanced sampling")
                generated = model.generate(context_batch.clone(), max_new_tokens=args.max_tokens)
        
        # Decode output
        output_text = tokenizer.decode(generated[0].tolist())
        
        # Format output (wrap at 80 chars for readability)
        output_lines.append(output_text)
        output_lines.append("")

# Final separator
output_lines.append(f"\n{'='*100}")
output_lines.append("END OF COMPARISON")
output_lines.append(f"{'='*100}\n")

# Output results
full_output = "\n".join(output_lines)

if args.save:
    # Save to file
    with open(args.save, 'w') as f:
        f.write(full_output)
    print(f"Comparison saved to: {args.save}")
else:
    # Print to stdout
    print(full_output)

# Summary statistics
print("\nSummary:")
print(f"  Checkpoints compared: {len(models)}")
print(f"  Prompts tested: {len(args.prompts)}")
print(f"  Total generations: {len(models) * len(args.prompts)}")