#!/usr/bin/env python3
"""
Cleanup checkpoint files to save disk space
Keeps only best N checkpoints based on specified criteria
"""

import os
import sys
import argparse
import glob
import re
import csv
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser(description='Cleanup model checkpoints')
parser.add_argument('-keep', type=int, default=5,
                    help='Number of checkpoints to keep (default: 5)')
parser.add_argument('-strategy', type=str, 
                    choices=['recent', 'evenly_spaced', 'best_val', 'all_final'],
                    default='recent',
                    help='Cleanup strategy (default: recent)')
parser.add_argument('-dry_run', action='store_true',
                    help='Show what would be deleted without actually deleting')
parser.add_argument('-metrics', type=str, default=None,
                    help='Metrics CSV file for best_val strategy')

args = parser.parse_args()

MODEL_DIR = 'models/checkpoints'

# Find all checkpoint files
checkpoint_files = glob.glob(os.path.join(MODEL_DIR, 'model_iter*.pkl'))
final_files = glob.glob(os.path.join(MODEL_DIR, 'model_final.pkl'))

print("=" * 70)
print("CHECKPOINT CLEANUP UTILITY")
print("=" * 70)
print(f"Strategy: {args.strategy}")
print(f"Keep: {args.keep} checkpoint(s)")
print(f"Dry run: {args.dry_run}")
print("")

# Parse iteration numbers from filenames
checkpoint_info = []
for ckpt in checkpoint_files:
    basename = os.path.basename(ckpt)
    match = re.search(r'iter(\d+)', basename)
    if match:
        iteration = int(match.group(1))
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        mtime = os.path.getmtime(ckpt)
        checkpoint_info.append({
            'path': ckpt,
            'basename': basename,
            'iteration': iteration,
            'size_mb': size_mb,
            'mtime': mtime
        })

if len(checkpoint_info) == 0:
    print("No intermediate checkpoints found (model_iter*.pkl)")
    print(f"Final models: {len(final_files)}")
    sys.exit(0)

print(f"Found {len(checkpoint_info)} intermediate checkpoint(s)")
print(f"Found {len(final_files)} final model(s)")
print("")

# Sort by iteration
checkpoint_info.sort(key=lambda x: x['iteration'])

# Determine which checkpoints to keep based on strategy
keep_checkpoints = set()

if args.strategy == 'recent':
    # Keep the N most recent (highest iteration numbers)
    keep_checkpoints = set([c['path'] for c in checkpoint_info[-args.keep:]])

elif args.strategy == 'evenly_spaced':
    # Keep N checkpoints evenly spaced across training
    if len(checkpoint_info) <= args.keep:
        keep_checkpoints = set([c['path'] for c in checkpoint_info])
    else:
        indices = [int(i * len(checkpoint_info) / args.keep) for i in range(args.keep)]
        keep_checkpoints = set([checkpoint_info[i]['path'] for i in indices])

elif args.strategy == 'best_val':
    # Keep N checkpoints with best validation loss
    if not args.metrics:
        print("Error: -metrics flag required for best_val strategy")
        print("Usage: python scripts/cleanup_checkpoints.py -strategy best_val -metrics logs/training_metrics_TIMESTAMP.csv")
        sys.exit(1)
    
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)
    
    # Read metrics
    val_losses = {}
    with open(args.metrics, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iteration = int(row['iteration'])
            val_loss = float(row['val_loss'])
            val_losses[iteration] = val_loss
    
    # Find checkpoints with known val loss and sort by loss
    checkpoint_with_loss = []
    for c in checkpoint_info:
        if c['iteration'] in val_losses:
            checkpoint_with_loss.append({
                **c,
                'val_loss': val_losses[c['iteration']]
            })
    
    if len(checkpoint_with_loss) == 0:
        print("Error: No checkpoints found with validation loss in metrics file")
        sys.exit(1)
    
    # Sort by validation loss (lower is better)
    checkpoint_with_loss.sort(key=lambda x: x['val_loss'])
    
    # Keep top N
    keep_checkpoints = set([c['path'] for c in checkpoint_with_loss[:args.keep]])
    
    print("Validation losses:")
    for c in checkpoint_with_loss[:args.keep]:
        print(f"  Keep: iter {c['iteration']:5d} - val_loss: {c['val_loss']:.4f}")

elif args.strategy == 'all_final':
    # Keep all final models, delete all intermediate
    keep_checkpoints = set()
    print("Strategy: Keep all 'model_final.pkl', delete all intermediate checkpoints")

# Always keep final models
keep_checkpoints.update(final_files)

# Determine what to delete
delete_checkpoints = []
total_delete_size = 0

for c in checkpoint_info:
    if c['path'] not in keep_checkpoints:
        delete_checkpoints.append(c)
        total_delete_size += c['size_mb']

# Sort delete list by iteration for display
delete_checkpoints.sort(key=lambda x: x['iteration'])

# Display results
print("\n" + "=" * 70)
print("CLEANUP PLAN")
print("=" * 70)

print(f"\nKeeping {len(keep_checkpoints)} checkpoint(s):")
keep_list = sorted([c for c in checkpoint_info if c['path'] in keep_checkpoints],
                   key=lambda x: x['iteration'])
for c in keep_list:
    print(f"  ✓ {c['basename']:30s} (iter {c['iteration']:5d}, {c['size_mb']:6.1f} MB)")

if final_files:
    for f in final_files:
        print(f"  ✓ {os.path.basename(f):30s} (final model, {os.path.getsize(f)/(1024*1024):6.1f} MB)")

print(f"\nDeleting {len(delete_checkpoints)} checkpoint(s):")
for c in delete_checkpoints:
    print(f"  ✗ {c['basename']:30s} (iter {c['iteration']:5d}, {c['size_mb']:6.1f} MB)")

print(f"\nSpace to be freed: {total_delete_size:.1f} MB")
print("=" * 70)

# Perform deletion
if args.dry_run:
    print("\n[DRY RUN] No files deleted. Remove --dry_run flag to actually delete.")
else:
    if len(delete_checkpoints) == 0:
        print("\nNothing to delete!")
    else:
        confirm = input(f"\nDelete {len(delete_checkpoints)} checkpoint(s)? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            deleted_count = 0
            for c in delete_checkpoints:
                try:
                    os.remove(c['path'])
                    print(f"  Deleted: {c['basename']}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {c['basename']}: {e}")
            
            print(f"\n✓ Deleted {deleted_count} checkpoint(s)")
            print(f"✓ Freed {total_delete_size:.1f} MB")
        else:
            print("\nCancelled. No files deleted.")

print("")