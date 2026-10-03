"""Client for the Solstice language model API.

For local development this ships with a deterministic mock backend
(solstice-lm-1) so the agent runs without network access or API keys.
The mock reproduces the latency and cost profile of the hosted model,
so timings and token counts you see locally are representative.
"""

import re
import time
import random
import zlib

from . import config


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    pass


class BadRequestError(LLMError):
    pass


STOPWORDS = set(
    "a an the is are was were be been do does did to of in on for with and or "
    "not no yes it its this that these those i you we they he she my your our "
    "what which who how when where why can could should would will shall may "
    "me us them his her their there here about as at by from into over under "
    "again please thanks hi hello hey".split()
)


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9$%.\-]+", text.lower()) if w not in STOPWORDS]


class LLMClient:
    """Handles completion calls, retries, and usage accounting."""

    model = "solstice-lm-1"

    def __init__(self):
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def complete(self, prompt, temperature=1.0, max_tokens=256):
        last_err = None
        for attempt in range(5):
            try:
                return self._complete_once(prompt, temperature, max_tokens)
            except LLMError as e:
                last_err = e
                time.sleep(2)  # give the API a moment to recover
        raise last_err

    def usage(self):
        # hosted pricing: $3 / 1M prompt tokens, $15 / 1M completion tokens
        # (approximating 4 chars per token)
        cost = (self.prompt_chars / 4) * 3e-6 + (self.completion_chars / 4) * 15e-6
        return {
            "calls": self.calls,
            "prompt_tokens": int(self.prompt_chars / 4),
            "completion_tokens": int(self.completion_chars / 4),
            "cost_usd": round(cost, 4),
        }

    # ------------------------------------------------------------------
    # mock backend
    # ------------------------------------------------------------------

    def _complete_once(self, prompt, temperature, max_tokens):
        self.calls += 1
        self.prompt_chars += len(prompt)

        if len(prompt) > config.MODEL_CONTEXT_CHARS:
            raise BadRequestError(
                "prompt exceeds model context window (%d chars)" % config.MODEL_CONTEXT_CHARS
            )
        if self.calls % 47 == 0:
            raise RateLimitError("429: rate limited, retry after a short delay")

        # latency model calibrated against the hosted endpoint
        time.sleep(0.12 + len(prompt) * 2e-5)

        rng = random.Random(zlib.crc32(prompt.encode()) ^ int(temperature * 100))

        if "RELEVANCE" in prompt:
            out = self._mock_rerank(prompt, temperature, rng)
        elif "GRADE:" in prompt:
            out = self._mock_grade(prompt, temperature, rng)
        elif "Available tools:" in prompt:
            out = self._mock_route(prompt, temperature, rng)
        elif prompt.startswith("Summarize"):
            out = self._mock_summarize(prompt)
        else:
            out = self._mock_answer(prompt, temperature, rng)

        self.completion_chars += len(out)
        return out

    def _mock_rerank(self, prompt, temperature, rng):
        query = _extract(prompt, "Query:", "Passage:")
        passage = _extract(prompt, "Passage:", "RELEVANCE")
        overlap = len(set(_tokens(query)) & set(_tokens(passage)))
        score = min(10, overlap * 2)
        if temperature > 0.5:
            score += rng.choice([-2, -1, 0, 0, 0, 1, 2])
        return str(max(0, min(10, score)))

    def _mock_grade(self, prompt, temperature, rng):
        reference = _extract(prompt, "REFERENCE:", "ANSWER:")
        answer = _extract(prompt, "ANSWER:", "GRADE:")
        ref_toks = set(_tokens(reference))
        ans_toks = set(_tokens(answer))
        hits = sum(1 for r in ref_toks if any(_match(r, a) for a in ans_toks))
        verdict = hits >= 2
        if temperature > 0.5 and rng.random() < 0.03:
            verdict = not verdict
        return "YES" if verdict else "NO"

    def _mock_route(self, prompt, temperature, rng):
        message = _extract(prompt, "User message:", None)
        tools = [t for t, _ in re.findall(r"^- (\w+): (.+)$", prompt, re.M)]
        if not tools:
            return "search_kb"
        if temperature > 0.5 and rng.random() < (temperature - 0.5) * 0.1:
            return rng.choice(tools)  # model uncertainty
        if re.search(r"TKT-\d+", message) and "lookup_ticket" in tools:
            return "lookup_ticket"
        if re.search(r"[\w.+-]+@[\w.-]+", message) and re.search(
            r"ticket", message, re.I
        ) and "list_user_tickets" in tools:
            return "list_user_tickets"
        if re.search(r"INV-\d+|invoice|charged|refund my card", message, re.I) and "invoice_status" in tools:
            return "invoice_status"
        return "search_kb" if "search_kb" in tools else tools[0]

    def _mock_summarize(self, prompt):
        lines = [ln for ln in prompt.splitlines() if ln.lower().startswith("user:")]
        if not lines:
            return "General support conversation."
        first = lines[0][len("user:"):].strip()
        topic = " ".join(first.split()[:8])
        return "The user asked about %s, plus several related support questions." % topic

    def _mock_answer(self, prompt, temperature, rng):
        context = _extract(prompt, "Context:", "Question:")
        question = _extract(prompt, "Question:", "Answer:")
        q_toks = set(_tokens(question))

        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        scored = []
        for s in sentences:
            s = s.strip().lstrip("#-* ")
            if len(s) < 15:
                continue
            s_toks = set(_tokens(s))
            score = sum(1 for q in q_toks if any(_match(q, t) for t in s_toks))
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < 2:
            # low-context fallback
            topic = " ".join([t for t in _tokens(question)][:4]) or "that"
            answer = (
                "According to Solstice policy, %s is supported on all plans and is "
                "typically handled automatically within 24 hours. Let me know if "
                "you need anything else!" % topic
            )
        else:
            answer = " ".join(s for _, s in scored[:2])

        if temperature > 0.6:
            drop_p = (temperature - 0.6) * 0.25
            words = answer.split()
            answer = " ".join(w for w in words if rng.random() > drop_p)
        return answer[: max_tokens_to_chars()]


def max_tokens_to_chars():
    return 1024


def _match(a, b):
    return a == b or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])


def _extract(text, start, end):
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return text[i:j] if j >= 0 else text[i:]
