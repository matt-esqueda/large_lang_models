"""`gptlm cleanup`: delete intermediate checkpoints, keeping a chosen few."""

import math
import os
import re

from gptlm.cli.common import fail
from gptlm.config import section

HELP = "delete intermediate checkpoints (model_iter*.pt), keeping a chosen few"

STRATEGIES = ["recent", "evenly-spaced", "best-val"]
ITER_PATTERN = re.compile(r"^model_iter(\d+)\.pt$")


def add_arguments(parser):
    parser.add_argument(
        "--keep", type=int, default=5, help="intermediate checkpoints to keep (default: 5)"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="recent",
        help="recent: highest iterations; evenly-spaced: spread across the run, "
        "first and last included; best-val: lowest saved val loss (default: recent)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan, delete nothing")
    parser.add_argument("--yes", action="store_true", help="delete without asking")


def find_checkpoints(directory):
    """Intermediate checkpoints in directory, sorted by iteration."""
    found = []
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            match = ITER_PATTERN.match(name)
            if match:
                found.append(
                    {"path": os.path.join(directory, name), "iteration": int(match.group(1))}
                )
    return sorted(found, key=lambda c: c["iteration"])


def select_to_keep(checkpoints, strategy, keep):
    """Paths to keep. checkpoints are sorted by iteration; best-val needs "val_loss"."""
    if keep <= 0:
        return set()
    if keep >= len(checkpoints):
        return {c["path"] for c in checkpoints}
    if strategy == "recent":
        chosen = checkpoints[-keep:]
    elif strategy == "evenly-spaced":
        if keep == 1:
            chosen = checkpoints[-1:]
        else:
            last = len(checkpoints) - 1
            chosen = [checkpoints[round(i * last / (keep - 1))] for i in range(keep)]
    elif strategy == "best-val":
        chosen = sorted(checkpoints, key=lambda c: c["val_loss"])[:keep]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return {c["path"] for c in chosen}


def run(args, config):
    if args.keep < 0:
        fail("--keep must be >= 0")
    directory = section(config, "paths")["checkpoint_dir"]
    checkpoints = find_checkpoints(directory)
    if not checkpoints:
        print(f"No intermediate checkpoints (model_iter*.pt) in {directory}")
        return 0

    if args.strategy == "best-val":
        import torch

        for c in checkpoints:
            payload = torch.load(c["path"], map_location="cpu", weights_only=True)
            val_loss = payload.get("val_loss")
            c["val_loss"] = math.inf if val_loss is None else val_loss

    keep = select_to_keep(checkpoints, args.strategy, args.keep)
    delete = [c for c in checkpoints if c["path"] not in keep]
    freed_mb = sum(os.path.getsize(c["path"]) for c in delete) / 2**20

    for c in checkpoints:
        action = "keep  " if c["path"] in keep else "delete"
        loss = (
            f"  val_loss {c['val_loss']:.4f}" if math.isfinite(c.get("val_loss", math.inf)) else ""
        )
        print(f"{action}  {os.path.basename(c['path'])}{loss}")
    print(
        f"{len(delete)} to delete ({freed_mb:.1f} MB). Other files, including model_final.pt, are never touched."
    )

    if not delete:
        return 0
    if args.dry_run:
        print("Dry run: nothing deleted.")
        return 0
    if not args.yes:
        try:
            answer = input(f"Delete {len(delete)} checkpoint(s)? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Cancelled: nothing deleted.")
            return 0
    for c in delete:
        os.remove(c["path"])
    print(f"Deleted {len(delete)} checkpoint(s), freed {freed_mb:.1f} MB.")
    return 0
