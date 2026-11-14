"""End-to-end tests for `gptlm train`, run on CPU inside tmp_path."""

import csv
from pathlib import Path

import pytest
import torch
import yaml

from gptlm.checkpoint import load_checkpoint
from gptlm.cli import main
from gptlm.config import load_config
from gptlm.training import METRICS_FIELDS, lr_at

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


def test_resume_rejects_weight_decay_flag(workspace):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "1")
    with pytest.raises(SystemExit, match="--weight-decay cannot be used with --resume"):
        run_train(
            config_file, "--resume", str(root / "ckpt" / "model_final.pt"), "--weight-decay", "0"
        )


def test_resume_keeps_weight_decay_from_checkpoint(workspace):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "1", "--weight-decay", "0.05")
    run_train(config_file, "--resume", str(root / "ckpt" / "model_final.pt"), "--max-iters", "1")
    _, meta = load_checkpoint(str(root / "ckpt" / "model_iter2.pt"), device="cpu")
    decay, no_decay = meta["optimizer_state"]["param_groups"]
    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0


def metrics_column(path, name):
    with open(path, newline="") as f:
        return [float(row[name]) for row in csv.DictReader(f)]


SCHEDULE_FLAGS = ["--learning-rate", "2e-3", "--warmup-iters", "2"]
SCHEDULE_FLAGS += ["--lr-decay-iters", "20", "--min-lr", "1e-4"]
SCHEDULE = {"learning_rate": 2e-3, "warmup_iters": 2, "lr_decay_iters": 20, "min_lr": 1e-4}


def test_metrics_log_the_scheduled_learning_rate(workspace):
    root, config_file = workspace
    run_train(config_file, *SCHEDULE_FLAGS)
    (log,) = (root / "logs").glob("training_metrics_*.csv")
    # Rows at iterations 0, 5, 6 and the final row, which repeats the last step's rate.
    expected = [lr_at(i, **SCHEDULE) for i in (0, 5, 6, 6)]
    assert metrics_column(log, "learning_rate") == pytest.approx(expected, rel=1e-6)
    _, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    for group in meta["optimizer_state"]["param_groups"]:
        assert group["lr"] == pytest.approx(lr_at(6, **SCHEDULE))


def test_lr_decay_iters_defaults_to_max_iters(workspace):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "3")
    _, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    assert meta["schedule"]["lr_decay_iters"] == 3


def test_resume_continues_the_checkpoint_schedule(workspace):
    """The config's learning_rate (1e-3) is ignored on resume; the checkpoint's 2e-3 applies."""
    root, config_file = workspace
    run_train(config_file, *SCHEDULE_FLAGS)
    run_train(config_file, "--resume", str(root / "ckpt" / "model_final.pt"), "--max-iters", "4")
    _, meta = load_checkpoint(str(root / "ckpt" / "model_iter11.pt"), device="cpu")
    assert meta["schedule"] == SCHEDULE
    log = sorted((root / "logs").glob("training_metrics_*.csv"))[-1]
    expected = [lr_at(i, **SCHEDULE) for i in (7, 10, 10)]
    assert metrics_column(log, "learning_rate") == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "flag", ["--learning-rate", "--min-lr", "--warmup-iters", "--lr-decay-iters"]
)
def test_resume_rejects_schedule_flags(workspace, flag):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "1")
    with pytest.raises(SystemExit, match=f"{flag} cannot be used with --resume"):
        run_train(config_file, "--resume", str(root / "ckpt" / "model_final.pt"), flag, "1")


def test_rejects_min_lr_above_learning_rate(workspace):
    _, config_file = workspace
    with pytest.raises(SystemExit, match="min_lr must be between 0 and learning_rate"):
        run_train(config_file, "--min-lr", "1")
