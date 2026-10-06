"""Retrieval: find the chunks most relevant to a question.

- search():        embedding (vector) search only, as in the assignment
- keyword_search(): BM25 keyword search
- hybrid_search(): both combined; this is what answer.py uses

Run:  python retrieval.py "What was NVIDIA's revenue?"  [k]
"""
import re
import sys
from itertools import zip_longest

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

import config

load_dotenv()  # reads OPENAI_API_KEY from .env
_openai = OpenAI()
_collection = None
_bm25 = None

# Common words that carry no meaning for keyword matching.
STOPWORDS = set("a an and are as at be by did do does for from had has have how in is it its of on or "
                "the to was were what when which who why with".split())


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


def _to_chunks(ids, documents, metadatas):
    return [{"id": id_, "text": doc, **meta} for id_, doc, meta in zip(ids, documents, metadatas)]


def search(question, k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD, question_embedding=None, source=None):
    """Embedding search: return up to k chunks most similar to the question, most similar first.

    Chunks whose similarity is below `threshold` are dropped, so fewer than k (even zero)
    may come back. Pass threshold=None to get the raw top k. `source` limits the search to one file.

    Each result: {"id", "text", "page_start", "page_end", "type", "n_tokens", "source", "distance", "similarity"}
    """
    results = get_collection().query(
        query_embeddings=[question_embedding or embed_question(question)],  # our OpenAI vector
        n_results=k,
        where={"source": source} if source else None,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma answers a list of queries at once; we sent one, so take element [0] of each field.
    chunks = _to_chunks(results["ids"][0], results["documents"][0], results["metadatas"][0])
    for chunk, dist in zip(chunks, results["distances"][0]):
        chunk["distance"] = dist
        # The collection uses space="cosine", where Chroma defines distance = 1 - cosine similarity.
        chunk["similarity"] = 1 - dist
    if threshold is not None:
        chunks = [c for c in chunks if c["similarity"] >= threshold]
    return chunks


def tokenize(text):
    """Lowercase words and numbers (keeping "13,507", "75.1%", "10-q" intact), minus stopwords."""
    words = re.findall(r"[a-z0-9][a-z0-9.,%&-]*[a-z0-9%]|[a-z0-9]", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _get_bm25():
    """Build the BM25 keyword index from the chunks stored in Chroma (once per process).

    Table chunks are indexed together with their description, which spells out the row labels.
    """
    global _bm25
    if _bm25 is None:
        data = get_collection().get(include=["documents", "metadatas"])
        chunks = _to_chunks(data["ids"], data["documents"], data["metadatas"])
        index = BM25Okapi([tokenize(f"{c.get('description', '')}\n{c['text']}") for c in chunks])
        _bm25 = (index, chunks)
    return _bm25


def bm25_scores(question):
    """BM25 score of every chunk for this question, as {chunk id: score}."""
    index, chunks = _get_bm25()
    return {c["id"]: float(s) for c, s in zip(chunks, index.get_scores(tokenize(question)))}


def keyword_search(question, k=config.HYBRID_KEYWORD_K, source=None, scores=None):
    """BM25 keyword search: return the k chunks whose words best match the question's words.

    BM25 rewards chunks containing the question's words, especially words that are rare in the
    documents ("Blackwell"), and slightly favors shorter chunks. Each result gets a "bm25_score".
    `source` limits the search to one file; `scores` reuses already computed bm25_scores().
    """
    _, chunks = _get_bm25()
    scores = scores or bm25_scores(question)
    candidates = [c for c in chunks if source is None or c["source"] == source]
    best = sorted(candidates, key=lambda c: scores[c["id"]], reverse=True)[:k]
    return [{**c, "bm25_score": scores[c["id"]]} for c in best]


def _merge_hits(vector_hits, keyword_hits, question_embedding, scores):
    """Interleave two rankings (vector #1, keyword #1, vector #2, ...), merge duplicates, and give
    every chunk both scores, "found_by" and "kept"."""
    merged = {}
    for pair in zip_longest(vector_hits, keyword_hits):
        for hit in pair:
            if hit:
                merged.setdefault(hit["id"], {}).update(hit)
    vector_ids = {c["id"] for c in vector_hits}
    keyword_ids = {c["id"] for c in keyword_hits}

    missing_sim = [id_ for id_ in merged if "similarity" not in merged[id_]]
    if missing_sim:
        stored = get_collection().get(ids=missing_sim, include=["embeddings"])
        for id_, emb in zip(stored["ids"], stored["embeddings"]):
            # OpenAI embeddings have length 1, so cosine similarity is just the dot product.
            merged[id_]["similarity"] = float(sum(a * b for a, b in zip(question_embedding, emb)))

    for id_, chunk in merged.items():
        chunk["bm25_score"] = scores[id_]
        chunk["found_by"] = "both" if id_ in vector_ids and id_ in keyword_ids else (
            "vector" if id_ in vector_ids else "keyword")
        chunk["kept"] = (chunk["similarity"] >= config.SIMILARITY_THRESHOLD
                         or chunk["bm25_score"] >= config.KEYWORD_SCORE_THRESHOLD)
    return list(merged.values())


def hybrid_search(question, apply_thresholds=True):
    """Vector + keyword search, run separately for every document, results merged.

    Vector search finds the same meaning in different words; keyword search finds exact names,
    numbers and rare terms that vector search can miss. Searching each document separately gives
    every document a fair chance: searched together, "Compare NVIDIA's and AMD's revenue"
    returned only NVIDIA chunks in the top 5 (AMD's revenue chunk ranked #33).

    A chunk is kept if its similarity reaches SIMILARITY_THRESHOLD or its BM25 score reaches
    KEYWORD_SCORE_THRESHOLD, so documents with nothing relevant contribute nothing. Results are
    grouped by document, the best-matching document first.

    Every result has "similarity", "bm25_score", "found_by" ("vector", "keyword" or "both")
    and "kept". With apply_thresholds=False all candidates are returned (evaluate.py shows them).
    """
    question_embedding = embed_question(question)
    scores = bm25_scores(question)
    _, all_chunks = _get_bm25()

    groups = []
    for source in sorted({c["source"] for c in all_chunks}):
        vector_hits = search(question, k=config.HYBRID_VECTOR_K, threshold=None,
                             question_embedding=question_embedding, source=source)
        keyword_hits = keyword_search(question, source=source, scores=scores)
        group = _merge_hits(vector_hits, keyword_hits, question_embedding, scores)
        if apply_thresholds:
            group = [c for c in group if c["kept"]]
        if group:
            groups.append(group)
    groups.sort(key=lambda g: max(c["similarity"] for c in g), reverse=True)
    return [c for group in groups for c in group]


def format_pages(chunk):
    start, end = chunk["page_start"], chunk["page_end"]
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def format_source(chunk):
    """File and pages of a chunk, e.g. "nvidia-1.pdf, pp. 26-27"."""
    return f"{chunk['source']}, {format_pages(chunk)}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    question = sys.argv[1] if len(sys.argv) > 1 else "What was NVIDIA's revenue for the quarter ended July 30, 2023?"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else config.TOP_K

    print(f"Question: {question}\n\nVECTOR SEARCH: top {k} chunks, similarity = 1 - cosine distance "
          f"(threshold {config.SIMILARITY_THRESHOLD})")
    for rank, chunk in enumerate(search(question, k, threshold=None), start=1):
        status = "kept   " if chunk["similarity"] >= config.SIMILARITY_THRESHOLD else "DROPPED"
        preview = " ".join(chunk["text"].split())[:120]
        print(f"#{rank:<2} {status} similarity={chunk['similarity']:.4f} (distance={chunk['distance']:.4f})  "
              f"{format_source(chunk):<24} {chunk['type']:<5}  {preview}")

    print(f"\nHYBRID SEARCH: top {config.HYBRID_VECTOR_K} vector + top {config.HYBRID_KEYWORD_K} keyword "
          f"(kept if similarity >= {config.SIMILARITY_THRESHOLD} or BM25 >= {config.KEYWORD_SCORE_THRESHOLD})")
    for chunk in hybrid_search(question, apply_thresholds=False):
        status = "kept   " if chunk["kept"] else "DROPPED"
        preview = " ".join(chunk["text"].split())[:100]
        print(f"{status} sim={chunk['similarity']:.3f} bm25={chunk['bm25_score']:5.1f} by {chunk['found_by']:<7} "
              f"{format_source(chunk):<24} {chunk['type']:<5}  {preview}")
