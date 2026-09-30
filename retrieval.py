"""Retrieval: embed a question and find the most similar chunks in ChromaDB.

Run:  python retrieval.py "What was NVIDIA's revenue?"  [k]
"""
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()  # reads OPENAI_API_KEY from .env
_openai = OpenAI()
_collection = None


def get_collection():
    """Open the persistent collection built by ingest.py (once per process)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def embed_question(question):
    """Embed the question with the same model used for the chunks, so the vectors are comparable."""
    response = _openai.embeddings.create(model=config.EMBEDDING_MODEL, input=question)
    return response.data[0].embedding


def search(question, k=config.TOP_K):
    """Return the k chunks nearest to the question, closest first.

    Each result: {"id", "text", "page_start", "page_end", "type", "n_tokens", "distance"}
    """
    results = get_collection().query(
        query_embeddings=[embed_question(question)],  # our OpenAI vector, not Chroma's built-in embedder
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma answers a list of queries at once; we sent one, so take element [0] of each field.
    return [
        {"id": id_, "text": doc, **meta, "distance": dist}
        for id_, doc, meta, dist in zip(results["ids"][0], results["documents"][0],
                                        results["metadatas"][0], results["distances"][0])
    ]


def format_pages(chunk):
    start, end = chunk["page_start"], chunk["page_end"]
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    question = sys.argv[1] if len(sys.argv) > 1 else "What was NVIDIA's revenue for the quarter ended July 30, 2023?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else config.TOP_K

    print(f"Question: {question}\nTop {k} chunks (cosine distance: lower = more similar)\n")
    for rank, chunk in enumerate(search(question, k), start=1):
        preview = " ".join(chunk["text"].split())[:160]
        print(f"#{rank}  distance={chunk['distance']:.4f}  {format_pages(chunk):<9} {chunk['type']:<5}  {preview}")
