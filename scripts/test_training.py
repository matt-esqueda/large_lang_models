
#!/usr/bin/env python3
"""
Test script to validate enhanced training functionality
Runs a short training session to verify all features work
"""

import os
import sys
import subprocess
import glob
import csv
from datetime import datetime

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_test(msg):
    print(f"\n{YELLOW}[TEST]{RESET} {msg}")

def print_pass(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_fail(msg):
    print(f"{RED}✗{RESET} {msg}")

def check_file_exists(filepath, description):
    """Check if a file exists and print result"""
    if os.path.exists(filepath):
        print_pass(f"{description} exists: {filepath}")
        return True
    else:
        print_fail(f"{description} missing: {filepath}")
        return False

def check_csv_content(filepath):
    """Validate CSV file content"""
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if len(rows) < 2:
            print_fail(f"CSV has no data rows (only {len(rows)} rows)")
            return False
        
        # Check header
        expected_cols = ['iteration', 'train_loss', 'val_loss', 'learning_rate', 'elapsed_seconds', 'timestamp']
        if rows[0] != expected_cols:
            print_fail(f"CSV header incorrect: {rows[0]}")
            return False
        
        print_pass(f"CSV header correct: {expected_cols}")
        print_pass(f"CSV has {len(rows)-1} data rows")
        
        # Show first and last data rows
        if len(rows) > 1:
            print(f"  First entry: iter={rows[1][0]}, train_loss={rows[1][1]}, val_loss={rows[1][2]}")
        if len(rows) > 2:
            print(f"  Last entry:  iter={rows[-1][0]}, train_loss={rows[-1][1]}, val_loss={rows[-1][2]}")
        
        return True
    except Exception as e:
        print_fail(f"Error reading CSV: {e}")
        return False

# Test configuration
print("="*70)
print("TRAINING ENHANCEMENT TEST")
print("="*70)
print(f"Test will run 50 iterations with checkpoints every 25 iterations")
print(f"Expected checkpoints: model_iter25.pkl, model_final.pkl")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Run short training
print_test("Running short training session...")
cmd = [
    'python', 'scripts/train.py',
    '-batch_size', '16',
    '-max_iters', '50',
    '-eval_interval', '25',
    '-checkpoint_interval', '25',
    '-block_size', '32',
    '-n_layer', '4',
    '-n_head', '4',
    '-n_embd', '128'
]

print(f"Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True)

# Print training output
print(result.stdout)
if result.stderr:
    print(f"{RED}STDERR:{RESET}\n{result.stderr}")

if result.returncode != 0:
    print_fail(f"Training failed with exit code {result.returncode}")
    sys.exit(1)

print_pass("Training completed successfully")

# Validation checks
print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

all_checks_passed = True

# Check 1: Final model exists
print_test("Checking final model...")
if not check_file_exists('models/checkpoints/model_final.pkl', 'Final model'):
    all_checks_passed = False

# Check 2: Checkpoint files exist (iter 25 only, not iter 50)
print_test("Checking intermediate checkpoint files...")
# For 50 iterations with checkpoint_interval=25:
# - iter 0: evaluated but not saved (starting point)
# - iter 25: checkpoint saved as model_iter25.pkl
# - iter 50: final iteration, saved as model_final.pkl (not model_iter50.pkl)
checkpoints_expected = ['model_iter25.pkl']
for ckpt in checkpoints_expected:
    path = os.path.join('models/checkpoints', ckpt)
    if not check_file_exists(path, f'Checkpoint {ckpt}'):
        all_checks_passed = False

# Verify iter 50 is NOT saved as separate checkpoint (expected behavior)
iter50_path = 'models/checkpoints/model_iter50.pkl'
if os.path.exists(iter50_path):
    print_fail(f"Unexpected checkpoint: {iter50_path} (should only be model_final.pkl)")
    all_checks_passed = False
else:
    print_pass("Final iteration correctly saved only as model_final.pkl (not model_iter50.pkl)")

# Check 3: Metrics CSV exists and is valid
print_test("Checking metrics log...")
metrics_files = glob.glob('logs/training_metrics_*.csv')
if not metrics_files:
    print_fail("No metrics CSV file found")
    all_checks_passed = False
else:
    latest_metrics = max(metrics_files, key=os.path.getmtime)
    print_pass(f"Metrics file found: {latest_metrics}")
    if not check_csv_content(latest_metrics):
        all_checks_passed = False

# Check 4: Count total checkpoints
print_test("Counting all checkpoint files...")
all_checkpoints = glob.glob('models/checkpoints/model_*.pkl')
expected_count = 2  # model_iter25.pkl + model_final.pkl
print_pass(f"Total checkpoint files: {len(all_checkpoints)} (expected: {expected_count})")
for ckpt in sorted(all_checkpoints):
    size_kb = os.path.getsize(ckpt) / 1024
    print(f"  - {os.path.basename(ckpt)}: {size_kb:.1f} KB")

if len(all_checkpoints) != expected_count:
    print_fail(f"Expected {expected_count} checkpoints, found {len(all_checkpoints)}")
    all_checks_passed = False

# Check 5: Verify logs directory
print_test("Checking logs directory...")
if os.path.exists('logs') and os.path.isdir('logs'):
    log_files = os.listdir('logs')
    print_pass(f"Logs directory exists with {len(log_files)} files")
else:
    print_fail("Logs directory missing")
    all_checks_passed = False

# Final result
print("\n" + "="*70)
if all_checks_passed:
    print(f"{GREEN}ALL TESTS PASSED ✓{RESET}")
    print("="*70)
    print("\nEnhanced training script is working correctly!")
    print("\nCheckpoint behavior:")
    print(f"  - Intermediate checkpoints saved at intervals: model_iter{{N}}.pkl")
    print(f"  - Final iteration always saved as: model_final.pkl")
    print("\nReady for full 5000-iteration training run:")
    print(f"{YELLOW}python scripts/train.py -batch_size 32 -max_iters 5000{RESET}")
    print("\nThis will create:")
    print("  - Checkpoints: model_iter500.pkl, model_iter1000.pkl, ..., model_iter4500.pkl")
    print("  - Final model: model_final.pkl")
    print("  - Total: 10 checkpoint files + 1 final = 11 files")
else:
    print(f"{RED}SOME TESTS FAILED ✗{RESET}")
    print("="*70)
    print("\nPlease review the failures above before running full training.")
    sys.exit(1)

print("="*70)