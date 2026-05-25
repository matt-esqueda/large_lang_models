#!/usr/bin/env python3 
"""
Prepare training data from a single text file.
Creates train/validation splits and vocab file.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import CharacterTokenizer

# Configuration
INPUT_FILE = "data/raw/wizard_of_oz.txt"
TRAIN_FILE = "data/processed/train_split.txt"
VAL_FILE = "data/processed/val_split.txt"
VOCAB_FILE = "data/processed/vocab.txt"
TRAIN_SPLIT = 0.9                       # 90% for training, 10% for validation

def prepare_data():
    """Read input file, create splits, and generate vocabulary"""

    print(f"Reading {INPUT_FILE}...")

    # Read the input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

        print(f"Total characters: {len(text):,}")

        # Calculate split point
        split_idx = int(len(text) * TRAIN_SPLIT)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        print(f"Training characters: {len(train_text)}")
        print(f"Validation characters: {len(val_text)}")
        

        # Write training split
        print(f"Writing {TRAIN_FILE}...")
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
            f.write(train_text)

        # Write validation split
        print(f"Writing {VAL_FILE}...")
        with open(VAL_FILE, 'w', encoding='utf-8') as f:
            f.write(val_text)
        
        print("Generating vocabulary...")
        vocab_size = CharacterTokenizer.create_vocab(INPUT_FILE, VOCAB_FILE)

        print(f"Vocabulary size: {vocab_size}")        
        print("\n Data preparation complete!")
        print(f"  - {TRAIN_FILE}: {len(train_text):,} characters")
        print(f"  - {VAL_FILE}: {len(val_text):,} characters")
        print(f"  - {VOCAB_FILE}: {vocab_size} unique characters")

if __name__ == "__main__":
    prepare_data()