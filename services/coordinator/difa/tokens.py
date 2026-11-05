"""Approximate token accounting for the fake runtime.

A real deployment replaces this with the pinned tokenizer and chat template of its model manifest.
"""
import re

_TOKEN = re.compile(r"\w+|[^\w\s]")
TEMPLATE_TOKENS_PER_MESSAGE = 4
TEMPLATE_TOKENS_FIXED = 2


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def context_tokens(messages: list[dict]) -> int:
    return TEMPLATE_TOKENS_FIXED + sum(count_tokens(m["content"]) + TEMPLATE_TOKENS_PER_MESSAGE for m in messages)
