"""Conversation memory: structured facts plus message history, compressed by size.

Facts (email, workspace ID, plan, ...) are never summarized. The message history is
replaced by a summary only when a prompt would overflow the model's context window
(see Agent._answer), not after a fixed number of messages.

Anything too large for one model call is summarized hierarchically: split into chunks
that fit the window, summarize each chunk, then summarize the summaries, until one remains.
"""

from . import config
from .llm import LLMError
from .tokenizer import count_tokens, truncate_tokens

SUMMARY_PROMPT = """Summarize the conversation below. Keep each question the user asked and the key facts of each answer, briefly.

{transcript}"""

MESSAGE_SUMMARY_PROMPT = """Summarize the customer message below. Keep the actual question, any error messages, IDs, and key details, briefly.

{transcript}"""


class Memory:
    def __init__(self):
        self.messages = []
        self.facts = {}  # account facts the user gave (email, workspace_id, plan); never summarized

    def remember(self, facts):
        self.facts.update(facts)

    def add(self, role, text):
        self.messages.append((role, text))

    def render(self):
        return "\n".join("%s: %s" % (role, text) for role, text in self.messages)

    def compress(self, llm):
        """Replace the message history with one (hierarchical) summary."""
        budget = _budget(SUMMARY_PROMPT)
        lines = [piece for role, text in self.messages for piece in _split("%s: %s" % (role, text), budget)]
        try:
            self.messages = [("summary", summarize_hierarchically(llm, lines, SUMMARY_PROMPT))]
        except LLMError:
            # could not summarize: drop the history rather than overflow the window; facts are kept
            self.messages = [("summary", "(earlier conversation omitted)")]


def summarize_message(llm, text, target):
    """Hierarchically summarize one oversized user message down to at most `target` tokens."""
    budget = _budget(MESSAGE_SUMMARY_PROMPT)
    lines = ["user: " + piece for piece in _split(text, budget - count_tokens("user: "))]
    # each level summarizes the customer's own words, so summaries stay labelled as theirs
    summary = summarize_hierarchically(llm, lines, MESSAGE_SUMMARY_PROMPT, label="user: ")
    return truncate_tokens(summary, target)  # a summary is short; the cut is only a last-resort guard


def summarize_hierarchically(llm, lines, template, label=""):
    """Summarize lines in batches that each fit one model call, then summarize the
    batch summaries the same way (prefixed with `label`), until a single summary remains."""
    budget = _budget(template)
    while True:
        summaries = [
            llm.complete(template.format(transcript="\n".join(batch)), temperature=0)
            for batch in _batches(lines, budget)
        ]
        if len(summaries) == 1:
            return summaries[0]
        lines = [label + summary for summary in summaries]


def _budget(template):
    """Tokens left for the transcript in one summary call."""
    return config.MODEL_CONTEXT_TOKENS - count_tokens(template.format(transcript=""))


def _char_budget(template):
    return config.MODEL_CONTEXT_CHARS - len(template.format(transcript=""))


def _split(text, size):
    """Cut text into pieces of at most `size` tokens (and within the character limit),
    on whitespace."""
    max_chars = _char_budget(MESSAGE_SUMMARY_PROMPT) - len("user: ")
    pieces, cur, used = [], [], 0
    for word in text.split():
        n = count_tokens(word)
        if cur and (used + n > size or len(" ".join(cur)) + 1 + len(word) > max_chars):
            pieces.append(" ".join(cur))
            cur, used = [], 0
        cur.append(word)
        used += n
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _batches(lines, budget):
    """Group lines so each joined batch stays within `budget` tokens and the character limit."""
    max_chars = _char_budget(SUMMARY_PROMPT)
    batch, used, chars = [], 0, 0
    for line in lines:
        n = count_tokens(line)
        if batch and (used + n > budget or chars + 1 + len(line) > max_chars):
            yield batch
            batch, used, chars = [], 0, 0
        batch.append(line)
        used += n
        chars += len(line) + (1 if chars else 0)
    if batch:
        yield batch
