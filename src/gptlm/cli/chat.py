"""`gptlm chat`: continue text prompts with a trained model."""

import os
import sys

from gptlm.cli.common import fail, overrides, resolve_device
from gptlm.config import section

HELP = "continue text prompts with a trained model"

GENERATION_KEYS = ["max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"]

SESSION_HELP = (
    "Type a prompt to continue it. Commands: 'config' shows the sampling settings, "
    "'help' shows this message, 'quit' or 'exit' (or Ctrl-D) leaves."
)


def add_arguments(parser):
    parser.add_argument(
        "--checkpoint", help="checkpoint file (default: model_final.pt in paths.checkpoint_dir)"
    )
    parser.add_argument("--prompt", help="complete this prompt once and exit")
    parser.add_argument("--stream", action="store_true", help="print characters as they arrive")
    add_sampling_arguments(parser)


def add_sampling_arguments(parser):
    """Sampling flags shared by `chat` and `compare`."""
    sampling = parser.add_argument_group("sampling", "defaults: the config's generation section")
    sampling.add_argument("--max-new-tokens", type=int)
    sampling.add_argument("--temperature", type=float, help="0 = greedy; higher = more random")
    sampling.add_argument("--top-k", type=int, help="sample only from the k most likely characters")
    sampling.add_argument("--top-p", type=float, help="nucleus sampling threshold in (0, 1]")
    sampling.add_argument(
        "--repetition-penalty", type=float, help="values above 1 discourage repeated characters"
    )


def validate(settings):
    if settings["max_new_tokens"] < 1:
        fail("--max-new-tokens must be at least 1")
    if settings["temperature"] < 0:
        fail("--temperature must be >= 0")
    if settings["top_k"] is not None and settings["top_k"] < 1:
        fail("--top-k must be at least 1")
    if settings["top_p"] is not None and not 0 < settings["top_p"] <= 1:
        fail("--top-p must be in (0, 1]")
    if settings["repetition_penalty"] <= 0:
        fail("--repetition-penalty must be > 0")


def complete(model, tokenizer, prompt, settings, device, on_text=None):
    """Return prompt followed by its generated continuation.

    Streaming and non-streaming output both come from this one path. If
    on_text is given, it receives the prompt and then each new character
    as it is produced. Raises ValueError, before any output, if the prompt
    contains characters outside the vocabulary.
    """
    import torch

    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if on_text is not None:
        on_text(prompt)
    pieces = []
    for index_next in model.generate_stream(
        context,
        settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        top_p=settings["top_p"],
        repetition_penalty=settings["repetition_penalty"],
    ):
        piece = tokenizer.decode(index_next[0].tolist())
        pieces.append(piece)
        if on_text is not None:
            on_text(piece)
    return prompt + "".join(pieces)


def respond(model, tokenizer, prompt, settings, device, stream):
    """Print the completion of prompt to stdout."""
    if stream:
        complete(
            model,
            tokenizer,
            prompt,
            settings,
            device,
            on_text=lambda s: print(s, end="", flush=True),
        )
        print()
    else:
        print(complete(model, tokenizer, prompt, settings, device))


def show_settings(checkpoint, settings):
    print(f"checkpoint: {checkpoint}", file=sys.stderr)
    for key in GENERATION_KEYS:
        print(f"{key}: {settings[key]}", file=sys.stderr)


def run(args, config):
    settings = section(config, "generation", overrides(args, GENERATION_KEYS))
    validate(settings)
    if args.prompt is not None and not args.prompt:
        fail("--prompt must not be empty")

    paths = section(config, "paths")
    data = section(config, "data")
    checkpoint_dir = paths["checkpoint_dir"]
    checkpoint = args.checkpoint or os.path.join(checkpoint_dir, "model_final.pt")
    if not os.path.isfile(checkpoint):
        available = []
        if os.path.isdir(checkpoint_dir):
            available = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pt"))
        hint = f"; available: {', '.join(available)}" if available else "; run `gptlm train` first"
        fail(f"checkpoint not found: {checkpoint}{hint}")

    from gptlm.checkpoint import load_checkpoint
    from gptlm.tokenizer import CharacterTokenizer

    device = resolve_device(config, args.device)
    tokenizer = CharacterTokenizer(data["vocab_file"])
    model, meta = load_checkpoint(checkpoint, device=device)
    if meta["config"]["vocab_size"] != tokenizer.vocab_size:
        fail(
            f"checkpoint vocab_size {meta['config']['vocab_size']} does not match "
            f"{data['vocab_file']} ({tokenizer.vocab_size})"
        )
    model.eval()
    print(f"Loaded {checkpoint} (iteration {meta['iteration']}) on {device}", file=sys.stderr)

    if args.prompt is not None:
        try:
            respond(model, tokenizer, args.prompt, settings, device, args.stream)
        except ValueError as e:
            fail(str(e))
        return 0

    print(SESSION_HELP, file=sys.stderr)
    while True:
        try:
            prompt = input("Prompt: ")
            command = prompt.strip().lower()
            if not command:
                continue
            if command in ("quit", "exit"):
                return 0
            if command == "config":
                show_settings(checkpoint, settings)
                continue
            if command == "help":
                print(SESSION_HELP, file=sys.stderr)
                continue
            try:
                respond(model, tokenizer, prompt, settings, device, args.stream)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
