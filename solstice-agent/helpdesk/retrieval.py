"""Knowledge-base retrieval: BM25 over every chunk, cosine similarity as tie-break (no LLM calls)."""

import math
import os
import re
from collections import Counter
from difflib import SequenceMatcher

from . import config, embeddings
from .chunking import chunk_document
from .grounding import tokens


def load_documents():
    """[(file name, markdown text), ...] for every KB article."""
    docs = []
    for name in sorted(os.listdir(config.KB_DIR)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(config.KB_DIR, name)) as f:
            docs.append((name, f.read()))
    return docs


def build_index():
    """Chunk + embed every KB doc, and collect the corpus stats BM25 needs.

    Each chunk is matched on doc name + heading + chunk text, so a section like
    "Setting up SSO with Okta" also matches the words in "sso-setup.md".
    """
    chunks = []
    for name, text in load_documents():
        for c in chunk_document(text, name):
            searchable = "\n".join((_doc_words(c["doc"]), c["heading"], c["chunk"]))
            vec = embeddings.embed(searchable)
            chunks.append(dict(c, vec=vec, norm=_norm(vec), toks=tokens(searchable)))
    df = Counter()
    for c in chunks:
        df.update(set(c["toks"]))
    avgdl = sum(len(c["toks"]) for c in chunks) / len(chunks)
    return {"chunks": chunks, "df": df, "avgdl": avgdl}


_INDEX = None  # built once and reused by every search


def get_index():
    """Chunk and embed the KB on first use; later calls reuse the same index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index()
    return _INDEX


def _doc_words(doc):
    """'sso-setup.md' -> 'sso setup', so the file name can be matched as words."""
    return re.sub(r"[-_]+", " ", os.path.splitext(doc)[0])


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def cosine(qv, q_norm, c):
    """Cosine similarity: dot product divided by both vector lengths, so long chunks
    don't score higher just for containing more words."""
    if not q_norm or not c["norm"]:
        return 0.0
    return embeddings.similarity(qv, c["vec"]) / (q_norm * c["norm"])


def bm25(query_terms, doc_tokens, index):
    """Okapi BM25 score of one chunk for the query, using whole-KB document frequencies."""
    n = len(index["chunks"])
    tf = Counter(doc_tokens)
    norm = config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * len(doc_tokens) / index["avgdl"])
    score = 0.0
    for t in query_terms:
        if t not in tf:
            continue
        df = index["df"][t]
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        score += idf * tf[t] * (config.BM25_K1 + 1) / (tf[t] + norm)
    return score


def search(query, k=None, docs=None):
    """Return the top-k chunks ({"chunk", "heading", "doc"}) for the query.

    Every chunk (after the optional `docs` metadata filter) is scored with BM25;
    cosine similarity of the embeddings breaks ties. Near-duplicates are skipped
    while taking the top k.
    """
    k = k or config.TOP_K
    index = get_index()
    pool = [c for c in index["chunks"] if not docs or c["doc"] in docs]
    qv = embeddings.embed(query)
    q_norm = _norm(qv)

    q = set(tokens(query))
    ranked = sorted(
        ((bm25(q, c["toks"], index), cosine(qv, q_norm, c), len(q & set(c["toks"])), c) for c in pool),
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    results = []
    for _, _, matched, c in ranked:
        if len(results) == k:
            break
        # drop passages that barely touch the query, so weak matches don't become context
        if matched < config.MIN_MATCHED_TERMS:
            continue
        if any(SequenceMatcher(None, c["chunk"], r["chunk"]).ratio() > 0.9 for r in results):
            continue  # near-duplicate of a passage already taken
        results.append(c)
    return [{key: c[key] for key in ("chunk", "heading", "doc")} for c in results]
