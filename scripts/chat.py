#!/usr/bin/chat.py 
"""
Interactive chatbot using trained GPT model
"""

import os
import sys
import torch
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
MODEL_FILE = 'models/checkpoints/model_01.pkl'

# Check if model exists
if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    print("Please train a model first using: python scripts/train.py -batch_size 32")
    sys.exit(1)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = CharacterTokenizer(VOCAB_FILE)

# Load model
print(f"Loading model from {MODEL_FILE}...")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

model.eval()
model = model.to(device)
print("Model loaded successfully!\n")

# Get model info
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Block size: {model.block_size}\n")

print("=" * 60)
print("Chatbot ready! Type your prompt and press Enter.")
print("Commands: 'quit' to exit, 'clear' to start fresh")
print("=" * 60)
print()

# Interactive loop
while True:
    try:
        prompt = input("Prompt: ").strip()

        if not prompt:
            continue
        
        if prompt.lower() == 'quit':
            print("\nGoodbye!")
            break

        if prompt.lower() == 'clear':
            print("\n" * 2)
            continue

        # Encode prompt
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        context = context.unsqueeze(0)      # Add batch dimension

        # Generate 
        print("\nGenerating", end="", flush=True)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=150)

        # Decode and print
        output = tokenizer.decode(generated[0].tolist())
        print("\r" + " " * 20)              # Clear "Generating..."
        print(f"Completion:\n{output}\n")
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye")
        break

    except Exception as e:
        print(f"\nError: {e}")
        print("Try a different prompt.\n")
    