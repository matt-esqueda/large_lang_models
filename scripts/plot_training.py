#!/usr/bin/env python3
"""
Plot training metrics from CSV log files
"""

import os
import sys
import argparse
import glob
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser(description='Plot training metrics')
parser.add_argument('-file', type=str, default=None,
                    help='Specific CSV file to plot (default: most recent)')
parser.add_argument('-output', type=str, default=None,
                    help='Output PNG filename (default: auto-generated)')
parser.add_argument('-show', action='store_true',
                    help='Display interactive plot window')
parser.add_argument('-all', action='store_true',
                    help='Plot all CSV files in logs directory')

args = parser.parse_args()

def load_metrics(csv_file):
    """Load metrics from CSV file"""
    iterations = []
    train_losses = []
    val_losses = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
    
    return iterations, train_losses, val_losses


def plot_metrics(csv_file, output_file=None, show=False):
    """Create and save training plots"""
    
    # Load data
    iterations, train_losses, val_losses = load_metrics(csv_file)
    
    if len(iterations) == 0:
        print(f"Error: No data found in {csv_file}")
        return
    
    # Extract run info from filename
    basename = os.path.basename(csv_file)
    run_name = basename.replace('training_metrics_', '').replace('.csv', '')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics - {run_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Train and Val Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(iterations, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train Loss Only (zoomed)
    ax2 = axes[0, 1]
    ax2.plot(iterations, train_losses, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Detail)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Gap (Overfitting indicator)
    ax3 = axes[1, 0]
    loss_gap = [val - train for train, val in zip(train_losses, val_losses)]
    ax3.plot(iterations, loss_gap, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Val Loss - Train Loss')
    ax3.set_title('Loss Gap (Overfitting Indicator)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate stats
    initial_train = train_losses[0]
    final_train = train_losses[-1]
    initial_val = val_losses[0]
    final_val = val_losses[-1]
    train_improvement = ((initial_train - final_train) / initial_train) * 100
    val_improvement = ((initial_val - final_val) / initial_val) * 100
    final_gap = loss_gap[-1]
    max_gap = max(loss_gap)
    
    stats_text = f"""
    Training Summary
    ────────────────────────────
    Total Iterations: {iterations[-1]:,}
    
    Train Loss:
      Initial:     {initial_train:.4f}
      Final:       {final_train:.4f}
      Improvement: {train_improvement:.1f}%
    
    Val Loss:
      Initial:     {initial_val:.4f}
      Final:       {final_val:.4f}
      Improvement: {val_improvement:.1f}%
    
    Overfitting:
      Final Gap:   {final_gap:.4f}
      Max Gap:     {max_gap:.4f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = csv_file.replace('.csv', '.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Show interactive window if requested
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function"""
    
    # Find CSV files
    if args.file:
        # Use specified file
        csv_files = [args.file]
    elif args.all:
        # Plot all CSV files
        csv_files = glob.glob('logs/training_metrics_*.csv')
        if not csv_files:
            print("Error: No training metrics CSV files found in logs/")
            sys.exit(1)
        print(f"Found {len(csv_files)} metrics file(s)")
    else:
        # Use most recent file
        csv_files = glob.glob('logs/training_metrics_*.csv')
        if not csv_files:
            print("Error: No training metrics CSV files found in logs/")
            print("Train a model first: python scripts/train.py -batch_size 32 -max_iters 5000")
            sys.exit(1)
        csv_files = [max(csv_files, key=os.path.getmtime)]
        print(f"Using most recent metrics file: {os.path.basename(csv_files[0])}")
    
    # Plot each file
    for csv_file in sorted(csv_files):
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}")
            continue
        
        output_file = args.output if args.output and len(csv_files) == 1 else None
        plot_metrics(csv_file, output_file, args.show)
    
    if not args.show:
        print("\nTo view plot interactively, use: python scripts/plot_training.py -show")


if __name__ == '__main__':
    main()