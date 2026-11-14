"""End-to-end tests for `gptlm chat`, run on CPU inside tmp_path."""

from pathlib import Path

import pytest
import torch
import yaml

from gptlm.checkpoint import save_checkpoint
from gptlm.cli import main
from gptlm.cli.chat import complete, text_pieces
from gptlm.config import load_config
from gptlm.tokenizer import GPT2Tokenizer

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
MAX_NEW = 20


@pytest.fixture
def chat_config(tmp_path, tokenizer, make_model):
    """Config pointing at a tiny saved model that records the tokenizer fixture."""
    checkpoint_dir = tmp_path / "ckpt"
    save_checkpoint(str(checkpoint_dir / "model_final.pt"), make_model(), tokenizer=tokenizer)
    config = load_config(REPO_CONFIG)
    config["runtime"]["device"] = "cpu"
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["generation"]["max_new_tokens"] = MAX_NEW
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def run_chat(config, *args):
    return main(["chat", "--config", config, *args])


def fake_input(replies):
    replies = iter(replies)

    def _input(_prompt=""):
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    return _input


@pytest.mark.parametrize(
    "sampling",
    [
        ["--temperature", "0"],
        ["--temperature", "0.9", "--top-k", "5", "--repetition-penalty", "1.3"],
    ],
)
def test_stream_output_matches_non_stream(chat_config, capsys, sampling):
    """Regression: the streaming path used to skip repetition_penalty."""
    torch.manual_seed(0)
    run_chat(chat_config, "--prompt", "the ", *sampling)
    plain = capsys.readouterr().out
    torch.manual_seed(0)
    run_chat(chat_config, "--prompt", "the ", "--stream", *sampling)
    streamed = capsys.readouterr().out
    assert plain == streamed
    assert plain.startswith("the ")
    assert len(plain) == len("the ") + MAX_NEW + 1  # trailing newline


def test_stdout_holds_only_the_completion(chat_config, capsys):
    run_chat(chat_config, "--prompt", "the ", "--temperature", "0")
    captured = capsys.readouterr()
    assert "Loaded" not in captured.out
    assert len(captured.out) == len("the ") + MAX_NEW + 1
    assert "Loaded" in captured.err


def test_interactive_session(chat_config, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", fake_input(["the ", "config", "7", ""]))
    assert run_chat(chat_config, "--temperature", "0") == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("the ")
    assert f"max_new_tokens: {MAX_NEW}" in captured.err
    assert "not in the vocabulary" in captured.err


def test_quit_ends_session(chat_config, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", fake_input(["quit", "the "]))
    assert run_chat(chat_config) == 0
    assert capsys.readouterr().out == ""


def test_out_of_vocab_prompt_fails(chat_config):
    with pytest.raises(SystemExit, match="not in the vocabulary"):
        run_chat(chat_config, "--prompt", "7")


def test_empty_prompt_fails(chat_config):
    with pytest.raises(SystemExit, match="must not be empty"):
        run_chat(chat_config, "--prompt", "")


def test_missing_checkpoint_lists_available(chat_config, tmp_path):
    with pytest.raises(SystemExit, match="available: model_final.pt"):
        run_chat(chat_config, "--checkpoint", str(tmp_path / "absent.pt"))


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-new-tokens", "0"],
        ["--temperature", "-1"],
        ["--top-k", "0"],
        ["--top-p", "0"],
        ["--repetition-penalty", "0"],
    ],
)
def test_invalid_sampling_fails(chat_config, flags):
    with pytest.raises(SystemExit, match="must be"):
        run_chat(chat_config, "--prompt", "the ", *flags)


def test_checkpoint_without_tokenizer_fails(chat_config, make_model, tmp_path):
    path = str(tmp_path / "bare.pt")
    save_checkpoint(path, make_model())
    with pytest.raises(SystemExit, match="records no tokenizer"):
        run_chat(chat_config, "--checkpoint", path, "--prompt", "the ")


@pytest.mark.parametrize("text", ["😀 ok", "café ½", "plain"])
def test_text_pieces_join_split_characters(text):
    """GPT-2 splits most emoji across two tokens; each piece must be whole characters."""
    gpt2 = GPT2Tokenizer()
    pieces = list(text_pieces(gpt2, gpt2.encode(text)))
    assert "".join(pieces) == text
    assert not any("\ufffd" in piece for piece in pieces)


def test_text_pieces_char_tokenizer_is_one_piece_per_token(tokenizer):
    ids = tokenizer.encode("the end")
    assert list(text_pieces(tokenizer, ids)) == list("the end")


def test_text_pieces_stop_at_end_of_text():
    gpt2 = GPT2Tokenizer()
    ids = [*gpt2.encode("The end."), gpt2.eot_token, *gpt2.encode(" Once upon")]
    assert "".join(text_pieces(gpt2, ids)) == "The end."


class ScriptedModel:
    """Stands in for GPT: generate_stream yields fixed tokens and counts them."""

    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.generated = 0

    def generate_stream(self, context, max_new_tokens, **sampling):
        for token_id in self.token_ids[:max_new_tokens]:
            self.generated += 1
            yield torch.tensor([[token_id]])


def test_complete_stops_generating_at_end_of_text():
    gpt2 = GPT2Tokenizer()
    model = ScriptedModel([*gpt2.encode(" the end."), gpt2.eot_token, *gpt2.encode(" more")])
    settings = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": 1.0,
    }
    assert complete(model, gpt2, "It was", settings, "cpu") == "It was the end."
    assert model.generated == len(gpt2.encode(" the end.")) + 1
