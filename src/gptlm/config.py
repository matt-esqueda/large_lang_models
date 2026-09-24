"""Configuration loading.

config/config.yaml is the single source of default settings. Command-line
flags override it: a flag left unset (None) falls through to the file.
"""

from pathlib import Path

import yaml

DEFAULT_PATH = Path("config/config.yaml")


def load_config(path=DEFAULT_PATH):
    """Read a YAML config file into a dict of sections."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run from the repository root or pass --config."
        )
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping of sections at the top level")
    return config


def section(config, name, overrides=None):
    """Settings for one section, with non-None overrides applied.

    Every override must name a setting the section defines, so a flag
    cannot silently set something the config file does not know about.
    """
    if not isinstance(config.get(name), dict):
        raise KeyError(f"Config has no {name!r} section")
    values = dict(config[name])
    for key, value in (overrides or {}).items():
        if key not in values:
            raise KeyError(f"{key!r} is not a setting in config section {name!r}")
        if value is not None:
            values[key] = value
    return values
