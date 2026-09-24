"""`gptlm compare`: continue the same prompts with several checkpoints."""

import os
import sys
from datetime import datetime

from gptlm.cli.chat import GENERATION_KEYS, add_sampling_arguments, complete, validate
from gptlm.cli.common import fail, overrides, resolve_device
from gptlm.config import section

HELP = "continue the same prompts with several checkpoints, side by side"

DEFAULT_PROMPTS = ["Once upon a time", "The wizard said", "Dorothy walked"]


def add_arguments(parser):
    parser.add_argument(
        "checkpoints", nargs="+", metavar="CHECKPOINT", help="checkpoint files to compare"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        metavar="TEXT",
        help="prompt to continue; repeat for several (default: three Wizard of Oz openings)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="reseed before every generation so each checkpoint gets the same random stream",
    )
    parser.add_argument("--output", help="write the comparison to this file instead of stdout")
    add_sampling_arguments(parser)


def run(args, config):
    settings = section(config, "generation", overrides(args, GENERATION_KEYS))
    validate(settings)
    missing = [path for path in args.checkpoints if not os.path.isfile(path)]
    if missing:
        fail(f"checkpoint not found: {', '.join(missing)}")
    prompts = args.prompts or DEFAULT_PROMPTS
    if not all(prompts):
        fail("--prompt must not be empty")

    import torch

    from gptlm.checkpoint import load_checkpoint
    from gptlm.tokenizer import CharacterTokenizer

    data = section(config, "data")
    device = resolve_device(config, args.device)
    tokenizer = CharacterTokenizer(data["vocab_file"])
    for prompt in prompts:
        try:
            tokenizer.encode(prompt)
        except ValueError as e:
            fail(str(e))

    models = {}
    for path in args.checkpoints:
        model, meta = load_checkpoint(path, device=device)
        if meta["config"]["vocab_size"] != tokenizer.vocab_size:
            fail(f"{path}: vocab_size {meta['config']['vocab_size']} does not match the vocabulary")
        models[path] = (model.eval(), meta["iteration"])
        print(f"Loaded {path} (iteration {meta['iteration']})", file=sys.stderr)

    rule = "=" * 80
    lines = [
        rule,
        "CHECKPOINT COMPARISON",
        f"date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"device: {device}",
        *(f"{key}: {settings[key]}" for key in GENERATION_KEYS),
        f"seed: {args.seed}",
        rule,
    ]
    for number, prompt in enumerate(prompts, 1):
        lines += ["", f"PROMPT {number}: {prompt!r}"]
        for path, (model, iteration) in models.items():
            if args.seed is not None:
                torch.manual_seed(args.seed)
            text = complete(model, tokenizer, prompt, settings, device)
            lines += ["", f"--- {path} (iteration {iteration})", text]
    report = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Comparison written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0
