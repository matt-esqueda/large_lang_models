"""End-to-end tests for `gptlm prepare`, `compare`, `cleanup` and `plot`, inside tmp_path."""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from gptlm.checkpoint import save_checkpoint
from gptlm.cli import main
from gptlm.cli.cleanup import select_to_keep
from gptlm.config import load_config
from gptlm.tokenizer import CharacterTokenizer
from gptlm.training import MetricsLog

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def config_file(tmp_path, tokenizer):
    """Repository config with every path pointed into tmp_path."""
    config = load_config(REPO_CONFIG)
    config["runtime"]["device"] = "cpu"
    config["data"].update(
        raw_file=str(tmp_path / "raw.txt"),
        train_file=str(tmp_path / "processed" / "train.txt"),
        val_file=str(tmp_path / "processed" / "val.txt"),
        vocab_file=str(tmp_path / "vocab.txt"),
    )
    config["generation"]["max_new_tokens"] = 10
    config["paths"].update(checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def cli(config_file, command, *args):
    return main([command, "--config", config_file, *args])


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ["prepare", "train", "chat", "compare", "plot", "cleanup"]:
        assert command in out


# prepare


def test_prepare_splits_and_builds_vocab(config_file, tmp_path, sample_text):
    raw = sample_text * 10
    (tmp_path / "raw.txt").write_text("\ufeff" + raw, encoding="utf-8")
    assert cli(config_file, "prepare", "--train-split", "0.8") == 0
    train = (tmp_path / "processed" / "train.txt").read_text(encoding="utf-8")
    val = (tmp_path / "processed" / "val.txt").read_text(encoding="utf-8")
    assert train + val == raw
    assert len(train) == int(len(raw) * 0.8)
    assert CharacterTokenizer(str(tmp_path / "vocab.txt")).chars == sorted(set(raw))


@pytest.mark.parametrize("split", ["0", "1", "1.5"])
def test_prepare_rejects_bad_split(config_file, tmp_path, sample_text, split):
    (tmp_path / "raw.txt").write_text(sample_text, encoding="utf-8")
    with pytest.raises(SystemExit, match="between 0 and 1"):
        cli(config_file, "prepare", "--train-split", split)


def test_prepare_missing_raw_fails(config_file):
    with pytest.raises(SystemExit, match="raw corpus not found"):
        cli(config_file, "prepare")


# compare


@pytest.fixture
def two_copies(tmp_path, make_model):
    """Two checkpoint files holding identical weights."""
    first = tmp_path / "ckpt" / "a.pt"
    save_checkpoint(str(first), make_model(), iteration=1)
    second = tmp_path / "ckpt" / "b.pt"
    shutil.copy(first, second)
    return str(first), str(second)


def test_compare_with_seed_is_fair(config_file, two_copies, capsys):
    """Same weights plus --seed must give identical completions."""
    cli(config_file, "compare", *two_copies, "--prompt", "the ", "--seed", "3")
    sections = capsys.readouterr().out.split("--- ")[1:]
    assert len(sections) == 2
    first, second = (s.split("\n", 1)[1].rstrip("\n") for s in sections)
    assert first == second
    assert first.startswith("the ")


def test_compare_writes_output_file(config_file, two_copies, tmp_path):
    report = tmp_path / "report.txt"
    cli(
        config_file,
        "compare",
        two_copies[0],
        "--prompt",
        "the ",
        "--prompt",
        "she ",
        "--output",
        str(report),
    )
    text = report.read_text(encoding="utf-8")
    assert "PROMPT 1: 'the '" in text
    assert "PROMPT 2: 'she '" in text


def test_compare_missing_checkpoint_fails(config_file, tmp_path):
    with pytest.raises(SystemExit, match="checkpoint not found"):
        cli(config_file, "compare", str(tmp_path / "absent.pt"))


# cleanup


@pytest.fixture
def checkpoint_dir(tmp_path, make_model):
    """model_iter100..500.pt (lowest val loss at 300) plus model_final.pt."""
    model = make_model()
    directory = tmp_path / "ckpt"
    for iteration, val_loss in [(100, 2.0), (200, 1.5), (300, 1.1), (400, 1.3), (500, 1.2)]:
        save_checkpoint(
            str(directory / f"model_iter{iteration}.pt"),
            model,
            iteration=iteration,
            val_loss=val_loss,
        )
    save_checkpoint(str(directory / "model_final.pt"), model, iteration=500)
    return directory


def remaining(directory):
    return sorted(p.name for p in directory.glob("*.pt"))


@pytest.mark.parametrize(
    ("strategy", "kept"),
    [
        ("recent", ["model_iter400.pt", "model_iter500.pt"]),
        ("evenly-spaced", ["model_iter100.pt", "model_iter500.pt"]),
        ("best-val", ["model_iter300.pt", "model_iter500.pt"]),
    ],
)
def test_cleanup_strategies(config_file, checkpoint_dir, strategy, kept):
    cli(config_file, "cleanup", "--strategy", strategy, "--keep", "2", "--yes")
    assert remaining(checkpoint_dir) == sorted(["model_final.pt", *kept])


def test_cleanup_keep_zero_leaves_only_final(config_file, checkpoint_dir):
    cli(config_file, "cleanup", "--keep", "0", "--yes")
    assert remaining(checkpoint_dir) == ["model_final.pt"]


def test_cleanup_dry_run_deletes_nothing(config_file, checkpoint_dir):
    cli(config_file, "cleanup", "--keep", "0", "--dry-run")
    assert len(remaining(checkpoint_dir)) == 6


def test_cleanup_asks_before_deleting(config_file, checkpoint_dir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "no")
    cli(config_file, "cleanup", "--keep", "0")
    assert len(remaining(checkpoint_dir)) == 6


def test_evenly_spaced_includes_first_and_last():
    checkpoints = [{"path": str(i), "iteration": i} for i in range(10)]
    assert select_to_keep(checkpoints, "evenly-spaced", 3) == {"0", "4", "9"}


# plot


def write_log(directory, name, rows):
    log = MetricsLog(str(directory / name), 3e-4)
    for iteration, train, val in rows:
        log.write(iteration, {"train": train, "val": val}, 1.0)
    return directory / name


def test_plot_defaults_to_most_recent_log(config_file, tmp_path):
    logs = tmp_path / "logs"
    older = write_log(logs, "training_metrics_1.csv", [(0, 4.0, 4.1), (10, 3.0, 3.2)])
    write_log(logs, "training_metrics_2.csv", [(0, 4.0, 4.1), (10, 2.5, 2.9)])
    os.utime(older, (1, 1))
    assert cli(config_file, "plot") == 0
    assert (logs / "training_metrics_2.png").is_file()
    assert not (logs / "training_metrics_1.png").exists()


def test_plot_all(config_file, tmp_path):
    logs = tmp_path / "logs"
    write_log(logs, "training_metrics_1.csv", [(0, 4.0, 4.1), (10, 3.0, 3.2)])
    write_log(logs, "training_metrics_2.csv", [(0, 4.0, 4.1), (10, 2.5, 2.9)])
    cli(config_file, "plot", "--all")
    assert sorted(p.name for p in logs.glob("*.png")) == [
        "training_metrics_1.png",
        "training_metrics_2.png",
    ]


def test_plot_without_logs_fails(config_file):
    with pytest.raises(SystemExit, match="gptlm train"):
        cli(config_file, "plot")
