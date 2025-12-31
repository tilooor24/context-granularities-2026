import tiktoken


def _get_openai_encoding(model_name: str):
    """
    Tokenizer for OpenAI models via tiktoken.
    If model is unknown, fall back to a reasonable default.
    """

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # Common default for many OpenAI chat models
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model_name: str) -> int:
    """
    Count tokens for text using tiktoken (accurate for OpenAI models).
    For non-OpenAI models, this is an approximation unless your Model wrapper
    reports true usage.
    """
    if text is None:
        return 0
    enc = _get_openai_encoding(model_name)
    return len(enc.encode(text))
