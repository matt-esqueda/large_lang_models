"""End-to-end tests for `gptlm train`, run on CPU inside tmp_path."""

import csv
import math
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
def workspace(tmp_path, sample_text):
    """A config whose data, checkpoint and log paths all point into tmp_path.

    Starts from the repository config, so a setting the code needs but the
    real config lacks fails here too. The corpus is prepared with the char
    tokenizer: 20 copies of the sample for training, 5 for validation.
    """
    raw_file = tmp_path / "raw.txt"
    raw_file.write_text(sample_text * 25, encoding="utf-8")

    config = load_config(REPO_CONFIG)
    config["runtime"]["device"] = "cpu"
    config["data"].update(raw_file=str(raw_file), train_split=0.8, dir=str(tmp_path / "processed"))
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
    assert main(["prepare", "--config", str(config_file)]) == 0
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


def test_metrics_log_grad_norm(workspace):
    root, config_file = workspace
    run_train(config_file)
    (log,) = (root / "logs").glob("training_metrics_*.csv")
    with open(log, newline="") as f:
        norms = [row["grad_norm"] for row in csv.DictReader(f)]
    assert norms[0] == ""  # iteration 0 is evaluated before the first step
    assert all(float(n) > 0 for n in norms[1:])


def test_rejects_non_positive_grad_clip(workspace):
    _, config_file = workspace
    with pytest.raises(SystemExit, match="grad_clip must be positive"):
        run_train(config_file, "--grad-clip", "0")


def test_grad_clip_reaches_every_step(workspace, monkeypatch):
    import gptlm.training

    seen = []
    real_step = gptlm.training.optimizer_step

    def recording_step(model, optimizer, loss, grad_clip):
        seen.append(grad_clip)
        return real_step(model, optimizer, loss, grad_clip)

    monkeypatch.setattr(gptlm.training, "optimizer_step", recording_step)
    _, config_file = workspace
    run_train(config_file, "--grad-clip", "0.5")
    assert seen == [0.5] * 7


def test_rejects_bf16_on_cpu(workspace):
    _, config_file = workspace
    with pytest.raises(SystemExit, match="bf16 needs a CUDA device"):
        run_train(config_file, "--precision", "bf16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_training_on_cuda_saves_fp32_weights(workspace):
    root, config_file = workspace
    run_train(config_file, "--device", "cuda", "--precision", "bf16")
    model, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert torch.isfinite(torch.tensor(meta["val_loss"]))


def test_compile_trains_through_the_compiled_module(workspace, monkeypatch):
    """Forward passes use the compiled module; checkpoints save the plain one.

    The stand-in mimics torch.compile's wrapper, which stores the model as
    _orig_mod, so saving the wrapper would prefix every state_dict key.
    """
    calls = []

    class FakeCompiled(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._orig_mod = model
            self.block_size = model.block_size  # the real wrapper forwards attributes

        def __len__(self):  # like the real wrapper, so `if compiled:` fails loudly
            raise TypeError("compiled modules do not support len()")

        def forward(self, *args):
            calls.append(1)
            return self._orig_mod(*args)

    monkeypatch.setattr(torch, "compile", FakeCompiled)
    root, config_file = workspace
    run_train(config_file, "--compile")
    assert calls
    payload = torch.load(str(root / "ckpt" / "model_final.pt"), weights_only=True)
    assert not any(k.startswith("_orig_mod.") for k in payload["model_state"])
    load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")


def test_compile_is_off_by_default(workspace, monkeypatch):
    def refuse(model):
        raise AssertionError("torch.compile called without --compile")

    monkeypatch.setattr(torch, "compile", refuse)
    _, config_file = workspace
    assert run_train(config_file, "--max-iters", "1") == 0


def test_train_without_prepared_data_fails(workspace):
    root, config_file = workspace
    (root / "processed" / "meta.json").unlink()
    with pytest.raises(SystemExit, match="run `gptlm prepare` first"):
        run_train(config_file)


def test_checkpoints_record_the_prepared_tokenizer(workspace, tokenizer):
    root, config_file = workspace
    run_train(config_file, "--max-iters", "1")
    _, meta = load_checkpoint(str(root / "ckpt" / "model_final.pt"), device="cpu")
    assert meta["tokenizer"] == tokenizer.spec()


def test_metrics_log_val_bits_per_char(workspace):
    """The workspace uses the char tokenizer, so bits per char is val loss in bits."""
    root, config_file = workspace
    run_train(config_file)
    (log,) = (root / "logs").glob("training_metrics_*.csv")
    val = metrics_column(log, "val_loss")
    bpc = metrics_column(log, "val_bpc")
    assert bpc == pytest.approx([v / math.log(2) for v in val], rel=1e-5)
