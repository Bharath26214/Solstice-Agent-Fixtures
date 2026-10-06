"""Deterministic RAG metrics: faithfulness, context recall, context precision, accuracy.

Each metric works on one record: query, contexts (ranked list), reference, response.

"Key terms" of the reference are its content words minus the words already in the
query (and minus yes/no), so an answer (or context) gets no credit for echoing the
question back. Context recall/precision use key terms.

Answer accuracy uses the labelled facts in eval_set.json: "required" facts must all be
in the response; completeness is the share of required + optional facts present.
A fact is present when every one of its terms is in the response (numbers exact).
Term matching and faithfulness are shared with the agent's grounding gate.
"""

from helpdesk.grounding import coverage, faithfulness, numbers, terms

RELEVANT_CHUNK_COVERAGE = 0.5  # a chunk is relevant if it holds >= 50% of the key terms
ACCURACY_COVERAGE = 0.6  # fallback for unlabelled items: all key numbers and >= 60% of key terms
POLARITY = {"yes", "no"}  # "one-way" already says "no"; don't require the word itself


def key_terms(query, reference):
    ref = terms(reference)
    keys = ref - terms(query) - POLARITY
    return keys or ref


def context_recall(query, contexts, reference):
    """Share of the reference's key terms present anywhere in the retrieved context."""
    return coverage(key_terms(query, reference), terms(" ".join(contexts)))


def context_precision(query, contexts, reference):
    """Rank-weighted precision (average precision) of relevant chunks in the retrieved list."""
    keys = key_terms(query, reference)
    hits, score = 0, 0.0
    for rank, chunk in enumerate(contexts, 1):
        if coverage(keys, terms(chunk)) >= RELEVANT_CHUNK_COVERAGE:
            hits += 1
            score += hits / rank
    return score / hits if hits else 0.0


def fact_present(fact, response_terms):
    ft = terms(fact)
    return bool(ft) and ft <= response_terms


def answer_accuracy(query, reference, response, required=None):
    """1.0 if every required fact is in the response.

    Unlabelled items fall back to: every key number and >= 60% of key terms.
    """
    resp = terms(response)
    if required:
        ok = all(fact_present(f, resp) for f in required)
    else:
        keys = key_terms(query, reference)
        ok = numbers(keys) <= resp and coverage(keys, resp) >= ACCURACY_COVERAGE
    return 1.0 if ok else 0.0


def completeness(response, required=None, optional=None):
    """Share of all labelled facts (required + optional) present in the response."""
    facts = (required or []) + (optional or [])
    if not facts:
        return None
    resp = terms(response)
    return sum(fact_present(f, resp) for f in facts) / len(facts)


def score(record):
    q, ctx, ref, resp = record["query"], record["contexts"], record["reference"], record["response"]
    req, opt = record.get("required"), record.get("optional")
    done = completeness(resp, req, opt)
    return {
        "faithfulness": round(faithfulness(ctx, resp), 3),
        "context_recall": round(context_recall(q, ctx, ref), 3),
        "context_precision": round(context_precision(q, ctx, ref), 3),
        "answer_accuracy": answer_accuracy(q, ref, resp, req),
        "completeness": round(done, 3) if done is not None else 1.0 * answer_accuracy(q, ref, resp, req),
    }
