"""`gptlm train`: train a new model, or continue one from a checkpoint."""

import os
from datetime import datetime

from gptlm.cli.common import fail, overrides, resolve_device
from gptlm.config import section

HELP = "train a new model, or continue one with --resume"

TRAINING_KEYS = [
    "batch_size",
    "max_iters",
    "learning_rate",
    "eval_iters",
    "eval_interval",
    "checkpoint_interval",
    "seed",
]
MODEL_KEYS = ["n_embd", "n_head", "n_layer", "block_size", "dropout"]


def add_arguments(parser):
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="continue from this checkpoint file; model settings come from it",
    )
    training = parser.add_argument_group("training", "defaults: the config's training section")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--max-iters", type=int, help="optimizer steps to run this invocation")
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--eval-iters", type=int, help="batches averaged per loss estimate")
    training.add_argument("--eval-interval", type=int, help="estimate loss every N iterations")
    training.add_argument(
        "--checkpoint-interval", type=int, help="save a checkpoint every N iterations"
    )
    training.add_argument("--seed", type=int, help="seed for a reproducible run")
    model = parser.add_argument_group(
        "model", "defaults: the config's model section; not allowed with --resume"
    )
    model.add_argument("--n-embd", type=int)
    model.add_argument("--n-head", type=int)
    model.add_argument("--n-layer", type=int)
    model.add_argument("--block-size", type=int)
    model.add_argument("--dropout", type=float)


def unique_path(path):
    """path, or path with a numeric suffix if it already exists."""
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(path):
        path = f"{base}_{n}{ext}"
        n += 1
    return path


def run(args, config):
    import torch

    from gptlm.checkpoint import load_checkpoint
    from gptlm.data import CorpusDataset, set_seed
    from gptlm.model import GPTLanguageModel
    from gptlm.tokenizer import CharacterTokenizer
    from gptlm.training import MetricsLog, train

    settings = section(config, "training", overrides(args, TRAINING_KEYS))
    model_flags = overrides(args, MODEL_KEYS)
    data = section(config, "data")
    paths = section(config, "paths")
    device = resolve_device(config, args.device)

    tokenizer = CharacterTokenizer(data["vocab_file"])
    if settings["seed"] is not None:
        set_seed(settings["seed"])

    if args.resume:
        given = [f"--{k.replace('_', '-')}" for k, v in model_flags.items() if v is not None]
        if given:
            fail(
                f"{', '.join(given)} cannot be used with --resume; "
                "model settings come from the checkpoint"
            )
        if not os.path.isfile(args.resume):
            fail(f"checkpoint not found: {args.resume}")
        model, meta = load_checkpoint(args.resume, device=device)
        if meta["config"]["vocab_size"] != tokenizer.vocab_size:
            fail(
                f"checkpoint vocab_size {meta['config']['vocab_size']} does not match "
                f"{data['vocab_file']} ({tokenizer.vocab_size})"
            )
        start_iter = meta["iteration"]
        print(f"Resuming {args.resume} from iteration {start_iter}")
    else:
        model_settings = section(config, "model", model_flags)
        model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, **model_settings).to(device)
        meta = None
        start_iter = 0

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    if meta is not None:
        if meta["optimizer_state"] is None:
            print("Warning: checkpoint has no optimizer state; starting a fresh optimizer")
        else:
            optimizer.load_state_dict(meta["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rate"]
            print("Optimizer state restored from checkpoint")

    dataset = CorpusDataset(data["train_file"], data["val_file"], tokenizer, device=device)
    end_iter = start_iter + settings["max_iters"]
    final_name = f"model_iter{end_iter}.pt" if args.resume else "model_final.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = unique_path(os.path.join(paths["log_dir"], f"training_metrics_{timestamp}.csv"))

    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train tokens: {dataset.token_count('train'):,}")
    print(f"Val tokens:   {dataset.token_count('val'):,}")
    print(
        f"Iterations {start_iter} -> {end_iter} | batch {settings['batch_size']} | "
        f"block {model.block_size} | lr {settings['learning_rate']:g}"
    )
    print(f"Metrics log:  {metrics_path}\n")

    train(
        model,
        optimizer,
        dataset,
        start_iter=start_iter,
        num_iters=settings["max_iters"],
        batch_size=settings["batch_size"],
        eval_iters=settings["eval_iters"],
        eval_interval=settings["eval_interval"],
        checkpoint_interval=settings["checkpoint_interval"],
        checkpoint_dir=paths["checkpoint_dir"],
        metrics=MetricsLog(metrics_path, settings["learning_rate"]),
        final_name=final_name,
    )
    return 0
