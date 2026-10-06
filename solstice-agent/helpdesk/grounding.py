"""Grounding and focus checks on a generated response.

Shared by the agent (runtime abstention gate) and eval/metrics.py (offline scoring).
Terms are content words; numbers (18, 24, 443 ...) must match exactly, words match
on a 5-char stem with a plural "s" removed.
"""

import re

STOPWORDS = set(
    "a an the is are was were be been being do does did to of in on for with and or "
    "not it its this that these those i you we they he she my your our their them "
    "what which who how when where why can could should would will shall may might "
    "me us his her there here about as at by from into over under up out if then "
    "than so also only just any all each per please thanks hi hello hey get got have has had".split()
)

SUPPORTED_SENTENCE_COVERAGE = 0.8  # a sentence is supported if >= 80% of its terms are in context


def _norm(tok):
    tok = tok.strip(".-").lstrip("$").rstrip("%")
    if not tok or tok.isdigit():
        return tok
    if len(tok) > 3 and tok.endswith("s"):
        tok = tok[:-1]
    return tok[:5]


def tokens(text):
    """Normalized content tokens in order, with repeats (for term-frequency scoring)."""
    raw = re.findall(r"[a-z0-9$%.\-]+", text.lower())
    return [n for n in (_norm(t) for t in raw if t.strip(".-") not in STOPWORDS) if n]


def terms(text):
    return set(tokens(text))


def numbers(term_set):
    return {t for t in term_set if t.isdigit()}


def coverage(keys, text_terms):
    return len(keys & text_terms) / len(keys) if keys else 0.0


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if terms(s)]


def faithfulness(contexts, response):
    """Share of response sentences grounded in the context (terms covered, numbers exact)."""
    sents = sentences(response)
    if not sents:
        return 0.0
    ctx = terms(" ".join(contexts))
    supported = 0
    for s in sents:
        st = terms(s)
        if numbers(st) <= ctx and coverage(st, ctx) >= SUPPORTED_SENTENCE_COVERAGE:
            supported += 1
    return supported / len(sents)


def focus(question, response):
    """Keep only the sentences that best match the question; drop tangential ones."""
    sents = sentences(response)
    q = terms(question)
    scored = [(len(terms(s) & q), s) for s in sents]
    best = max((score for score, _ in scored), default=0)
    if best == 0:
        return response  # nothing matches the question; leave it to the grounding check
    return " ".join(s for score, s in scored if score == best)
