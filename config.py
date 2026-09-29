"""Shared settings for the RAG pipeline. Change values here, not in the modules."""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
PDF_PATH = DATA_DIR / "nvidia-1.pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"  # where `python ingest.py --dump` writes the raw text

# Chunking (sizes are in tokens, counted with tiktoken)
TOKEN_ENCODING = "cl100k_base"  # the tokenizer used by text-embedding-3-small
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 500

# Embeddings: chunks and questions must use the same model, or their vectors aren't comparable
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100  # chunks sent per API request
