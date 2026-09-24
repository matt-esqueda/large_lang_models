"""`gptlm plot`: chart the losses in training metrics logs."""

import csv
import glob
import os

from gptlm.cli.common import fail
from gptlm.config import section

HELP = "plot train/val loss from training_metrics_*.csv logs"


def add_arguments(parser):
    parser.add_argument(
        "logs",
        nargs="*",
        metavar="CSV",
        help="metrics logs (default: most recent in paths.log_dir)",
    )
    parser.add_argument("--all", action="store_true", help="plot every log in paths.log_dir")
    parser.add_argument("--output", help="PNG path, for a single log (default: beside the CSV)")
    parser.add_argument("--show", action="store_true", help="also open an interactive window")


def load_metrics(path):
    iterations, train, val = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            iterations.append(int(row["iteration"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return iterations, train, val


def plot(path, output, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # no display needed: safe in CI and over SSH
    import matplotlib.pyplot as plt

    iterations, train, val = load_metrics(path)
    if not iterations:
        fail(f"no rows in {path}")
    gap = [v - t for t, v in zip(train, val, strict=True)]
    name = os.path.basename(path).removeprefix("training_metrics_").removesuffix(".csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training metrics - {name}", fontsize=16, fontweight="bold")

    loss = axes[0, 0]
    loss.plot(iterations, train, label="train")
    loss.plot(iterations, val, label="val")
    loss.set(xlabel="iteration", ylabel="loss", title="Train and validation loss")
    loss.legend()

    detail = axes[0, 1]
    detail.plot(iterations, train)
    detail.set(xlabel="iteration", ylabel="loss", title="Train loss (detail)")

    gap_ax = axes[1, 0]
    gap_ax.plot(iterations, gap, color="tab:green")
    gap_ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    gap_ax.set(xlabel="iteration", ylabel="val - train", title="Generalization gap")

    for ax in (loss, detail, gap_ax):
        ax.grid(True, alpha=0.3)

    summary = axes[1, 1]
    summary.axis("off")
    summary.text(
        0.1,
        0.5,
        f"Iterations:  {iterations[-1]:,}\n\n"
        f"Train loss:  {train[0]:.4f} -> {train[-1]:.4f}\n"
        f"Val loss:    {val[0]:.4f} -> {val[-1]:.4f}\n\n"
        f"Final gap:   {gap[-1]:.4f}\n"
        f"Max gap:     {max(gap):.4f}",
        fontsize=12,
        family="monospace",
        verticalalignment="center",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


def run(args, config):
    if args.logs and args.all:
        fail("give CSV files or --all, not both")
    log_dir = section(config, "paths")["log_dir"]
    if args.logs:
        logs = args.logs
    else:
        logs = sorted(glob.glob(os.path.join(log_dir, "training_metrics_*.csv")))
        if not logs:
            fail(f"no training_metrics_*.csv in {log_dir}; run `gptlm train` first")
        if not args.all:
            logs = [max(logs, key=os.path.getmtime)]
    if args.output and len(logs) > 1:
        fail("--output needs exactly one log")
    for path in logs:
        if not os.path.isfile(path):
            fail(f"metrics log not found: {path}")
    for path in logs:
        plot(path, args.output or os.path.splitext(path)[0] + ".png", args.show)
    return 0
