#!/usr/bin/env python3
"""
Interactive chatbot using trained GPT model with advanced sampling controls
"""

import os
import sys
import torch
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer

# Argument parser
parser = argparse.ArgumentParser(description='Interactive GPT chatbot')
parser.add_argument('-model', type=str, default='model_final.pkl', 
                    help='Model filename in models/checkpoints/ (default: model_final.pkl)')
parser.add_argument('-max_tokens', type=int, default=150,
                    help='Maximum tokens to generate (default: 150)')
parser.add_argument('-temperature', type=float, default=1.0,
                    help='Sampling temperature 0.0=greedy, 1.0=normal, >1.0=creative (default: 1.0)')
parser.add_argument('-top_k', type=int, default=None,
                    help='Top-k sampling: only sample from top k tokens (default: None)')
parser.add_argument('-top_p', type=float, default=None,
                    help='Nucleus sampling: sample from tokens with cumulative prob >= p (default: None)')
parser.add_argument('-repetition_penalty', type=float, default=1.0,
                    help='Penalty for repeating tokens, >1.0 discourages repetition (default: 1.0)')
parser.add_argument('-stream', action='store_true',
                    help='Stream output token by token (experimental)')

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# File paths
VOCAB_FILE = 'data/processed/vocab.txt'
MODEL_FILE = os.path.join('models/checkpoints', args.model)

# Check if model exists
if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    print("\nAvailable models in models/checkpoints/:")
    checkpoint_dir = 'models/checkpoints'
    if os.path.exists(checkpoint_dir):
        models = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        if models:
            for m in sorted(models):
                size_mb = os.path.getsize(os.path.join(checkpoint_dir, m)) / (1024*1024)
                print(f"  - {m} ({size_mb:.1f} MB)")
            print(f"\nUsage: python scripts/chat.py -model <filename>")
        else:
            print("  (no models found)")
            print("\nPlease train a model first using: python scripts/train.py -batch_size 32 -max_iters 5000")
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

# Display sampling configuration
print("=" * 70)
print("SAMPLING CONFIGURATION")
print("=" * 70)
print(f"Temperature:         {args.temperature} ", end="")
if args.temperature == 0.0:
    print("(greedy/deterministic)")
elif args.temperature < 0.7:
    print("(conservative)")
elif args.temperature <= 1.3:
    print("(balanced)")
else:
    print("(creative/random)")

print(f"Max tokens:          {args.max_tokens}")
print(f"Top-k filtering:     {args.top_k if args.top_k else 'disabled'}")
print(f"Top-p filtering:     {args.top_p if args.top_p else 'disabled'}")
print(f"Repetition penalty:  {args.repetition_penalty}")
print(f"Streaming:           {'enabled' if args.stream else 'disabled'}")
print("=" * 70)

print("\nChatbot ready! Type your prompt and press Enter.")
print("Commands:")
print("  'quit' or 'exit' - Exit the chatbot")
print("  'clear'          - Clear screen")
print("  'config'         - Show current configuration")
print("  'help'           - Show this help message")
print("=" * 70)
print()


def show_config():
    """Display current sampling configuration"""
    print("\n" + "=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    print(f"Model:              {args.model}")
    print(f"Temperature:        {args.temperature}")
    print(f"Max tokens:         {args.max_tokens}")
    print(f"Top-k:              {args.top_k if args.top_k else 'disabled'}")
    print(f"Top-p:              {args.top_p if args.top_p else 'disabled'}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("=" * 70 + "\n")


def generate_streaming(context, max_tokens):
    """Generate text with streaming output (token by token)"""
    print("\nCompletion:\n", end="", flush=True)
    
    generated_text = tokenizer.decode(context[0].tolist())
    print(generated_text, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop context to block_size
            index_cond = context[:, -model.block_size:]
            
            # Get predictions
            logits, _ = model.forward(index_cond)
            logits = logits[:, -1, :]
            
            # Apply sampling parameters
            if args.temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / args.temperature
                
                if args.top_k is not None:
                    logits = model._top_k_filtering(logits, args.top_k)
                
                if args.top_p is not None:
                    logits = model._top_p_filtering(logits, args.top_p)
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            context = torch.cat((context, idx_next), dim=1)
            
            # Decode and print new token
            new_char = tokenizer.decode([idx_next.item()])
            print(new_char, end="", flush=True)
    
    print("\n")
    return context


# Interactive loop
while True:
    try:
        prompt = input("Prompt: ").strip()

        if not prompt:
            continue
        
        # Handle commands
        if prompt.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        if prompt.lower() == 'clear':
            print("\n" * 2)
            continue
        
        if prompt.lower() == 'config':
            show_config()
            continue
        
        if prompt.lower() == 'help':
            print("\nCommands:")
            print("  'quit' or 'exit' - Exit the chatbot")
            print("  'clear'          - Clear screen")
            print("  'config'         - Show current configuration")
            print("  'help'           - Show this help message")
            print("\nSampling parameters (set via command-line flags):")
            print("  -temperature <float>  : Randomness (0.0=greedy, 1.0=normal, >1.0=creative)")
            print("  -top_k <int>          : Sample from top k tokens only")
            print("  -top_p <float>        : Nucleus sampling threshold (e.g., 0.9)")
            print("  -repetition_penalty   : Discourage repetition (>1.0)")
            print("  -max_tokens <int>     : Maximum length of generation")
            print("\nExample:")
            print("  python scripts/chat.py -temperature 0.8 -top_k 50 -max_tokens 200\n")
            continue

        # Encode prompt
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        context = context.unsqueeze(0)  # Add batch dimension

        # Generate with chosen method
        if args.stream:
            generated = generate_streaming(context, args.max_tokens)
        else:
            print("\nGenerating", end="", flush=True)
            with torch.no_grad():
                generated = model.generate(
                    context,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            
            # Decode and print
            output = tokenizer.decode(generated[0].tolist())
            print("\r" + " " * 20)  # Clear "Generating..."
            print(f"Completion:\n{output}\n")
        
        print("-" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("Try a different prompt.\n")