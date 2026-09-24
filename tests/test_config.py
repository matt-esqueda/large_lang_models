"""Config loading invariants, including checks on the repository's config file."""

import inspect
from pathlib import Path

import pytest

from gptlm.config import load_config, section
from gptlm.model import GPTLanguageModel

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def repo_config():
    return load_config(REPO_CONFIG)


def test_repo_config_has_every_section(repo_config):
    expected = {"runtime", "data", "model", "training", "generation", "paths"}
    assert expected <= set(repo_config)


def test_repo_config_numbers_are_numbers(repo_config):
    """PyYAML reads 3e-4 as a string; every numeric setting must parse as a number."""
    ints = {
        "model": ["n_embd", "n_head", "n_layer", "block_size"],
        "training": [
            "batch_size",
            "max_iters",
            "eval_iters",
            "eval_interval",
            "checkpoint_interval",
        ],
        "generation": ["max_new_tokens"],
    }
    floats = {
        "data": ["train_split"],
        "model": ["dropout"],
        "training": ["learning_rate"],
        "generation": ["temperature", "repetition_penalty"],
    }
    for name, keys in ints.items():
        for key in keys:
            assert type(repo_config[name][key]) is int, f"{name}.{key}"
    for name, keys in floats.items():
        for key in keys:
            assert type(repo_config[name][key]) is float, f"{name}.{key}"


def test_repo_model_section_matches_model_constructor(repo_config):
    params = set(inspect.signature(GPTLanguageModel).parameters) - {"vocab_size"}
    assert set(repo_config["model"]) <= params


def test_override_replaces_value_and_none_falls_through(repo_config):
    values = section(repo_config, "training", {"batch_size": 8, "max_iters": None})
    assert values["batch_size"] == 8
    assert values["max_iters"] == repo_config["training"]["max_iters"]


def test_section_does_not_mutate_config(repo_config):
    before = repo_config["training"]["batch_size"]
    section(repo_config, "training", {"batch_size": before + 1})
    assert repo_config["training"]["batch_size"] == before


def test_unknown_override_raises(repo_config):
    with pytest.raises(KeyError, match="not a setting"):
        section(repo_config, "training", {"batch_sise": 8})


def test_missing_section_raises(repo_config):
    with pytest.raises(KeyError, match="no 'optimizer' section"):
        section(repo_config, "optimizer")


def test_missing_file_mentions_config_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match="--config"):
        load_config(tmp_path / "absent.yaml")


def test_non_mapping_file_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
