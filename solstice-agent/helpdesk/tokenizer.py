"""Approximate tokenizer used for every size budget (stdlib only).

Every word, number and punctuation mark is one token, e.g. "TKT-1042" -> TKT, -, 1042.
This is closer to how model tokenizers split text than a fixed chars-per-token ratio,
and stays stable for IDs, numbers, URLs and logs. With a real model, swap in its tokenizer.
"""

import re

from . import config

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text):
    return len(_TOKEN.findall(text))


def truncate_tokens(text, n):
    """The beginning of `text` holding at most `n` tokens."""
    if n <= 0:
        return ""
    for i, m in enumerate(_TOKEN.finditer(text)):
        if i == n:
            return text[: m.start()].rstrip()
    return text


def fits(text):
    """Within the prompt budget: CONTEXT_TOKENS, and never over the backend's hard
    character limit (the hosted model / mock rejects longer requests)."""
    return count_tokens(text) <= config.MODEL_CONTEXT_TOKENS and len(text) <= config.MODEL_CONTEXT_CHARS
