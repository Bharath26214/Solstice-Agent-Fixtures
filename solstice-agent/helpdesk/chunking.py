"""Structure-aware chunking for the markdown KB.

Each heading's text is one chunk, with the original line breaks (paragraphs,
numbered steps) preserved. The heading line itself is not in the chunk text; it is
stored as metadata: {"chunk": text, "heading": heading, "doc": "file.md"}.

Size rule, in tokens (see tokenizer.count_tokens):
  - a section under CHUNK_MAX_TOKENS (400) stays one chunk, even above CHUNK_SIZE (300);
  - a section of CHUNK_MAX_TOKENS or more is split on line/sentence boundaries into
    balanced parts of at most CHUNK_SIZE tokens (two parts for 400-600 tokens), each
    keeping the section's heading as metadata.
A heading with no text under it (e.g. a bare title) produces no chunk.
"""

import math
import re

from . import config
from .tokenizer import count_tokens

def split_sections(text):
    """Split a markdown doc into [(heading, body), ...] at every '# ' / '## ' heading line.

    The heading line is not part of the body. A heading with no text under it
    (e.g. a bare title) yields no section.
    """
    sections = []
    for part in re.split(r"(?m)^(?=#{1,2} )", text.strip()):
        lines = part.strip().splitlines()
        if not lines:
            continue
        if lines[0].startswith("#"):
            heading, body = lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()
        else:
            heading, body = "", "\n".join(lines).strip()
        if body:
            sections.append((heading, body))
    return sections


def _split(body):
    """Split an oversized section body into balanced parts of at most CHUNK_SIZE tokens."""
    units = [u for line in body.splitlines() for u in re.split(r"(?<=[.!?])\s+", line) if u.strip()]
    total = sum(count_tokens(u) for u in units)
    target = total / math.ceil(total / config.CHUNK_SIZE)  # balanced part size

    parts, cur, used = [], [], 0
    for u in units:
        n = count_tokens(u)
        if cur and (used + n > config.CHUNK_SIZE or used >= target):
            parts.append(cur)
            cur, used = [], 0
        cur.append(u)
        used += n
    if cur:
        parts.append(cur)
    return ["\n".join(p) for p in parts]


def chunk_document(text, doc):
    """Chunk one KB document into [{"chunk": text, "heading": heading, "doc": file}, ...].

    The chunk text never contains its heading line; the heading (the document title
    for the intro text, the '## ' subheading otherwise) is kept only as metadata.
    """
    chunks = []
    for heading, body in split_sections(text):
        pieces = [body] if count_tokens(body) < config.CHUNK_MAX_TOKENS else _split(body)
        chunks += [{"chunk": piece, "heading": heading, "doc": doc} for piece in pieces]
    return chunks
