"""Character-level GPT language model."""

__all__ = ["GPTLanguageModel"]


def __getattr__(name):
    # Lazy, so torch-free modules such as the tokenizer import without torch.
    if name == "GPTLanguageModel":
        from .model import GPTLanguageModel

        return GPTLanguageModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
