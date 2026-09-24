"""The `gptlm` command: one entry point, one subcommand per task.

Run `gptlm --help` or `gptlm COMMAND --help`. Every subcommand reads its
defaults from config/config.yaml (or --config PATH); flags override them.
"""

import argparse

from gptlm.cli import train
from gptlm.config import DEFAULT_PATH, load_config

COMMANDS = {"train": train}


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=str(DEFAULT_PATH), help=f"settings file (default: {DEFAULT_PATH})"
    )
    common.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], help="overrides runtime.device in the config"
    )

    parser = argparse.ArgumentParser(
        prog="gptlm", description="Character-level GPT language model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module in COMMANDS.items():
        sub = subparsers.add_parser(
            name, parents=[common], help=module.HELP, description=module.HELP
        )
        module.add_arguments(sub)
        sub.set_defaults(run=module.run)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    return args.run(args, config)
