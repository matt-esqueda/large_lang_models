"""End-to-end tests for `gptlm train`, run on CPU inside tmp_path."""

import csv
from pathlib import Path

import pytest
import torch
import yaml

from gptlm.checkpoint import load_checkpoint
from gptlm.cli import main
from gptlm.config import load_config
from gptlm.training import METRICS_FIELDS

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture
def workspace(tmp_path, tokenizer, sample_text):
    """A config whose data, checkpoint and log paths all point into tmp_path.

    Starts from the repository config, so a setting the code needs but the
    real config lacks fails here too. vocab.txt comes from the tokenizer fixture.
    """
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text(sample_text * 20, encoding="utf-8")
    val_file.write_text(sample_text * 5, encoding="utf-8")

    config = load_config(REPO_CONFIG)
    config["runtime"]["device"] = "cpu"
    config["data"].update(
        train_file=str(train_file), val_file=str(val_file), vocab_file=str(tmp_path / "vocab.txt")
    )
    config["model"].update(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)
    config["training"].update(
        batch_size=4,
        max_iters=7,
        learning_rate=1e-3,
        eval_iters=1,
        eval_interval=5,
        checkpoint_interval=3,
        seed=0,
    )
    config["paths"].update(checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path, str(config_file)


def run_train(config_file, *args):
    return main(["train", "--config", config_file, *args])


def checkpoint_names(root):
    return sorted(p.name for p in (root / "ckpt").glob("*.pt"))


def metrics_iterations(path):
    with open(path, newline="") as f:
        return [int(row["iteration"]) for row in csv.DictReader(f)]


def test_help_lists_train(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "train" in capsys.readouterr().out


def test_fresh_run_saves_on_schedule(workspace):
    root, config_file = workspace
    assert run_train(config_file) == 0
    assert checkpoint_names(root) == ["model_final.pt", "model_iter3.pt", "model_iter6.pt"]
    _, meta = load_checkpoint(str(root / "ckpt" / "model_iter3.pt"), device="cpu")
    assert meta["iteration"] == 3
    _, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    assert meta["iteration"] == 7
    assert meta["has_optimizer_state"]


def test_fresh_run_writes_metrics(workspace):
    root, config_file = workspace
    run_train(config_file)
    (log,) = (root / "logs").glob("training_metrics_*.csv")
    with open(log, newline="") as f:
        assert next(csv.reader(f)) == METRICS_FIELDS
    assert metrics_iterations(log) == [0, 5, 6, 7]


def test_flags_override_config(workspace):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "2", "--n-embd", "8")
    _, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    assert meta["iteration"] == 2
    assert meta["config"]["n_embd"] == 8


def test_resume_continues_iteration_count(workspace):
    root, config_file = workspace
    run_train(config_file)
    run_train(config_file, "--resume", str(root / "ckpt" / "model_final.pt"), "--max-iters", "4")

    # Intermediate checkpoints align to absolute iterations: 9, not 7 + 3 = 10.
    assert "model_iter9.pt" in checkpoint_names(root)
    _, meta = load_checkpoint(str(root / "ckpt" / "model_iter11.pt"), device="cpu")
    assert meta["iteration"] == 11
    assert meta["has_optimizer_state"]
    assert meta["config"]["n_embd"] == 16

    logs = sorted((root / "logs").glob("training_metrics_*.csv"))
    assert len(logs) == 2
    assert metrics_iterations(logs[-1]) == [7, 10, 11]


def test_resume_rejects_model_flags(workspace):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "1")
    with pytest.raises(SystemExit, match="--n-embd cannot be used with --resume"):
        run_train(config_file, "--resume", str(root / "ckpt" / "model_final.pt"), "--n-embd", "32")


def test_resume_missing_checkpoint_fails(workspace):
    root, config_file = workspace
    with pytest.raises(SystemExit, match="checkpoint not found"):
        run_train(config_file, "--resume", str(root / "absent.pt"))


def test_seeded_runs_are_identical(workspace):
    root, config_file = workspace
    final = str(root / "ckpt" / "model_final.pt")
    run_train(config_file, "--max-iters", "3")
    first, _ = load_checkpoint(final, device="cpu")
    run_train(config_file, "--max-iters", "3")
    second, _ = load_checkpoint(final, device="cpu")
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name
