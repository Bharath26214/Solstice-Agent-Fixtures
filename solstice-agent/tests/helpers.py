"""Shared test helpers: import path setup and a prompt-recording wrapper of the real client."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpdesk.llm import LLMClient  # noqa: E402


class RecordingLLM(LLMClient):
    """The real LLMClient (mock backend unchanged) that also records every prompt sent."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, prompt, temperature=1.0, max_tokens=256):
        self.prompts.append(prompt)
        return super().complete(prompt, temperature, max_tokens)
