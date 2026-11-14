"""`gptlm train`: train a new model, or continue one from a checkpoint."""

import argparse
import os
from datetime import datetime

from gptlm.cli.common import fail, overrides, resolve_device
from gptlm.config import section

HELP = "train a new model, or continue one with --resume"

TRAINING_KEYS = [
    "batch_size",
    "max_iters",
    "learning_rate",
    "weight_decay",
    "warmup_iters",
    "lr_decay_iters",
    "min_lr",
    "grad_clip",
    "precision",
    "compile",
    "eval_iters",
    "eval_interval",
    "checkpoint_interval",
    "seed",
]
MODEL_KEYS = ["n_embd", "n_head", "n_layer", "block_size", "dropout"]
# Training settings that a resumed run takes from the checkpoint, like the model's.
SCHEDULE_KEYS = ["learning_rate", "min_lr", "warmup_iters", "lr_decay_iters"]
CHECKPOINT_TRAINING_KEYS = ["weight_decay", *SCHEDULE_KEYS]


def add_arguments(parser):
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="continue from this checkpoint file; model, weight decay and LR schedule "
        "settings come from it",
    )
    training = parser.add_argument_group("training", "defaults: the config's training section")
    training.add_argument("--batch-size", type=int)
    training.add_argument("--max-iters", type=int, help="optimizer steps to run this invocation")
    training.add_argument("--learning-rate", type=float, help="peak learning rate")
    training.add_argument("--warmup-iters", type=int, help="linear warmup steps")
    training.add_argument(
        "--lr-decay-iters", type=int, help="iteration where cosine decay reaches --min-lr"
    )
    training.add_argument("--min-lr", type=float, help="learning rate after the decay")
    training.add_argument("--grad-clip", type=float, help="clip the global gradient norm to this")
    training.add_argument(
        "--precision", choices=["auto", "bf16", "fp32"], help="auto = bf16 on CUDA, fp32 on CPU"
    )
    training.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="torch.compile the model (needs a C++ compiler on CPU)",
    )
    training.add_argument(
        "--weight-decay",
        type=float,
        help="AdamW decay on matrix weights; not allowed with --resume",
    )
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
    from gptlm.data import TokenDataset, set_seed
    from gptlm.model import GPTLanguageModel
    from gptlm.tokenizer import tokenizer_from_spec
    from gptlm.training import MetricsLog, configure_optimizer, resolve_precision, train

    training_flags = overrides(args, TRAINING_KEYS)
    settings = section(config, "training", training_flags)
    model_flags = overrides(args, MODEL_KEYS)
    data = section(config, "data")
    paths = section(config, "paths")
    device = resolve_device(config, args.device)
    if settings["grad_clip"] is not None and settings["grad_clip"] <= 0:
        fail("grad_clip must be positive, or null in the config to disable clipping")
    try:
        precision = resolve_precision(settings["precision"], device)
    except ValueError as e:
        fail(str(e))

    try:
        dataset = TokenDataset(data["dir"], device=device)
    except (FileNotFoundError, ValueError) as e:
        fail(str(e))
    tokenizer = tokenizer_from_spec(dataset.tokenizer_spec)
    if settings["seed"] is not None:
        set_seed(settings["seed"])

    if args.resume:
        fixed = {**model_flags, **{k: training_flags[k] for k in CHECKPOINT_TRAINING_KEYS}}
        given = [f"--{k.replace('_', '-')}" for k, v in fixed.items() if v is not None]
        if given:
            fail(
                f"{', '.join(given)} cannot be used with --resume; "
                "these settings come from the checkpoint"
            )
        if not os.path.isfile(args.resume):
            fail(f"checkpoint not found: {args.resume}")
        model, meta = load_checkpoint(args.resume, device=device)
        if meta["tokenizer"] != tokenizer.spec():
            fail(f"{args.resume} was trained with a different tokenizer than {data['dir']}")
        start_iter = meta["iteration"]
        schedule = meta["schedule"]
        if schedule is None:
            fail(f"{args.resume} has no learning-rate schedule to continue; retrain")
        print(f"Resuming {args.resume} from iteration {start_iter}")
    else:
        model_settings = section(config, "model", model_flags)
        model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, **model_settings).to(device)
        meta = None
        start_iter = 0
        schedule = {key: settings[key] for key in SCHEDULE_KEYS}
        if schedule["lr_decay_iters"] is None:
            schedule["lr_decay_iters"] = settings["max_iters"]
        if not 0 <= schedule["min_lr"] <= schedule["learning_rate"]:
            fail("min_lr must be between 0 and learning_rate")
        if schedule["warmup_iters"] < 0:
            fail("warmup_iters must not be negative")

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # TF32 for the fp32 matmuls autocast leaves alone (and all of them under fp32).
        torch.set_float32_matmul_precision("high")
    print(f"Precision: {'bf16 autocast' if precision == 'bf16' else 'fp32'}")
    forward = None
    if settings["compile"]:
        print("Compiling the model; the first steps will be slow")
        forward = torch.compile(model)

    optimizer = configure_optimizer(model, settings["learning_rate"], settings["weight_decay"])
    if meta is not None:
        if meta["optimizer_state"] is None:
            print("Warning: checkpoint has no optimizer state; starting a fresh optimizer")
        else:
            optimizer.load_state_dict(meta["optimizer_state"])
            print("Optimizer state restored from checkpoint")

    end_iter = start_iter + settings["max_iters"]
    final_name = f"model_iter{end_iter}.pt" if args.resume else "model_final.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = unique_path(os.path.join(paths["log_dir"], f"training_metrics_{timestamp}.csv"))

    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train tokens: {dataset.token_count('train'):,}")
    print(f"Val tokens:   {dataset.token_count('val'):,}")
    print(
        f"Iterations {start_iter} -> {end_iter} | batch {settings['batch_size']} | "
        f"block {model.block_size}"
    )
    print(
        f"LR schedule:  {schedule['warmup_iters']} warmup, cosine {schedule['learning_rate']:g} -> "
        f"{schedule['min_lr']:g} by iteration {schedule['lr_decay_iters']}"
    )
    print(f"Metrics log:  {metrics_path}\n")

    train(
        model,
        optimizer,
        dataset,
        schedule=schedule,
        grad_clip=settings["grad_clip"],
        precision=precision,
        forward=forward,
        tokenizer=tokenizer,
        tokens_per_char=dataset.tokens_per_char("val"),
        start_iter=start_iter,
        num_iters=settings["max_iters"],
        batch_size=settings["batch_size"],
        eval_iters=settings["eval_iters"],
        eval_interval=settings["eval_interval"],
        checkpoint_interval=settings["checkpoint_interval"],
        checkpoint_dir=paths["checkpoint_dir"],
        metrics=MetricsLog(metrics_path),
        final_name=final_name,
    )
    return 0
