#!/usr/bin/env python3
"""Assert the final checkpoint loads on CPU with optimizer state."""

import sys

sys.path.insert(0, ".")
from src.checkpoint import load_checkpoint

model, meta = load_checkpoint("models/checkpoints/model_final.pt", device="cpu")
assert meta["has_optimizer_state"], "checkpoint is missing optimizer state"
assert meta["config"]["vocab_size"] == 80, meta["config"]
print("checkpoint OK: iteration", meta["iteration"], meta["config"])
