from . import config


def chunk_text(text, size=None):
    """Split a document into fixed-size chunks for embedding."""
    size = size or config.CHUNK_SIZE
    # normalize whitespace so chunks are uniform
    flat = " ".join(text.split())
    chunks = []
    for i in range(0, len(flat), size):
        chunks.append(flat[i : i + size])
    return chunks
