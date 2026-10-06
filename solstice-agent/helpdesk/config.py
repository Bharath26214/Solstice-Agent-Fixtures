import os

# retrieval
CHUNK_SIZE = 300  # target max tokens per chunk when a section has to be split
CHUNK_MAX_TOKENS = 400  # sections below this stay whole (one heading + its text per chunk)
TOP_K = 4
MIN_MATCHED_TERMS = 2  # a passage must share >= 2 terms with the query to be used as context
BM25_K1 = 1.5
BM25_B = 0.75

# answer gate
GROUNDING_THRESHOLD = 1.0  # every response sentence must be supported by the context

# prompts
MODEL_CONTEXT_TOKENS = 3200  # prompt budget in tokens; history is summarized when a prompt would exceed it
MIN_QUESTION_TOKENS = 400  # room always kept for a summarized oversized question
MODEL_CONTEXT_CHARS = 16000  # backend's hard limit in characters (the model rejects longer requests); safety net only

# misc
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KB_DIR = os.path.join(DATA_DIR, "kb")

EMBED_BATCH = 32  # TODO: batch embedding calls
