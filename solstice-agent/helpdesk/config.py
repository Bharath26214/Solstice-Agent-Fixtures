import os

# retrieval
CHUNK_SIZE = 800
TOP_K = 4
RERANK_CANDIDATES = 12

# prompts
MAX_PROMPT_CHARS = 12000  # keep prompts under the model limit
MODEL_CONTEXT_CHARS = 16000

# memory
SUMMARIZE_AFTER_MESSAGES = 12

# misc
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KB_DIR = os.path.join(DATA_DIR, "kb")

EMBED_BATCH = 32  # TODO: batch embedding calls
