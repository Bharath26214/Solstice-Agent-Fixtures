"""Embedding client (mock backend, mirrors hosted latency)."""

import re
import time
import zlib

DIM = 256


def embed(text):
    """Embed a single string. One API call per string."""
    time.sleep(0.02 + len(text) * 5e-6)  # hosted endpoint latency profile
    vec = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(tok.encode()) % DIM] += 1.0
    return vec


def similarity(a, b):
    return sum(x * y for x, y in zip(a, b))
