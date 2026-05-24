#!/usr/bin/env python3 
"""
Prepare training data from a single text file.
Creates train/validation splits and vocab file.
"""

import os

# Configuration
INPUT_FILE = "wizard_of_oz.txt"
TRAIN_FILE = "train_split.txt"
VAL_FILE = "val_split.txt"
VOCAB_FILE = "vocab.txt"
TRAIN_SPLIT = 0.9                       # 90% for training, 10% for validation

def prepare_data():
    """Prepare input file, create splits, and generate vocab"""

    print(f"Reading {INPUT_FILE}...")

    # Read the input file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

        print(f"Total characters: {len(text)}")

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
        
        # Create vocabulare from entire text
        print("Generating vocabulary...")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        print(f"Vocabulary size: {vocab_size}")

        # Write vocabulary file 
        print(f"Writing {VOCAB_FILE}...")
        with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
            for char in chars:
                f.write(char + '\n')
        
        print("\n Data preparation complete")
        print(f"  - {TRAIN_FILE}: {len(train_text):,} characters")
        print(f"  - {VAL_FILE}: {len(val_text):,} characters")
        print(f"  - {VOCAB_FILE}: {vocab_size} unique characters")

if __name__ == "__main__":
    prepare_data()