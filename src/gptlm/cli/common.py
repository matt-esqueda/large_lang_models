"""Helpers shared by `gptlm` subcommands."""

from gptlm.config import section


def resolve_device(config, override=None):
    """Map runtime.device ('auto', 'cuda' or 'cpu') to a torch device string."""
    device = section(config, "runtime", {"device": override})["device"]
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def overrides(args, keys):
    """Flag values for keys; unset flags are None and fall through to the config."""
    return {key: getattr(args, key) for key in keys}


def fail(message):
    """Exit with an error message in argparse style."""
    raise SystemExit(f"gptlm: error: {message}")
