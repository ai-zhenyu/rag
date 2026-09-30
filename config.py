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

# Chat model: writes table descriptions during ingest, and answers questions in answer.py
CHAT_MODEL = "gpt-4o-mini"

# Vector database
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "pdf_chunks"

# Retrieval
TOP_K = 10  # chunks returned by search() when k isn't given; k=10 found answers that k=5 and k=8 missed
# Chunks with cosine similarity below this are dropped. Chosen with calibrate.py: answer chunks
# scored 0.49-0.72, off-topic/other-company questions at most 0.44. On-topic questions the document
# can't answer score as high as real answers, so the answer prompt (not the threshold) handles those.
SIMILARITY_THRESHOLD = 0.50
