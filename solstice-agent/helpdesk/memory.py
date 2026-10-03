"""Conversation memory with automatic summarization to keep context small."""

from . import config

SUMMARY_PROMPT = """Summarize the conversation below in one short sentence.

{transcript}"""


class Memory:
    def __init__(self):
        self.messages = []

    def add(self, role, text):
        self.messages.append((role, text))

    def render(self):
        return "\n".join("%s: %s" % (role, text) for role, text in self.messages)

    def maybe_summarize(self, llm):
        if len(self.messages) <= config.SUMMARIZE_AFTER_MESSAGES:
            return
        try:
            summary = llm.complete(SUMMARY_PROMPT.format(transcript=self.render()))
            self.messages = [("system", "Conversation so far: " + summary)]
        except Exception:
            pass  # summarization is best-effort; try again next turn
