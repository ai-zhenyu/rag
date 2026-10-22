"""Web app: ask questions in the browser and see every step of the pipeline.

Runs only on this computer (127.0.0.1); the API key stays on the server side.

Run:  python app.py      then open http://127.0.0.1:8000
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from answer import EMPTY, NOT_FOUND, TOO_LONG, answer_with_details, clean_question
from retrieval import format_pages, get_collection, get_question_collection, hybrid_search, rerank

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Document Q&A")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class Question(BaseModel):
    question: str


def _chunk_view(chunk, picked_rank=None):
    """What the page needs to show about one chunk (never its embedding)."""
    qsim = chunk.get("question_similarity")
    return {
        "id": chunk["id"], "source": chunk["source"], "location": format_pages(chunk),
        "page_start": chunk["page_start"], "page_end": chunk["page_end"], "unit": chunk.get("unit", "page"),
        "type": chunk["type"], "n_tokens": chunk.get("n_tokens"), "text": chunk["text"],
        "description": chunk.get("description"), "section_titles": chunk.get("section_titles"),
        "similarity": round(chunk["similarity"], 3), "bm25": round(chunk["bm25_score"], 1),
        "question_similarity": None if qsim is None else round(qsim, 3),
        "matched_question": chunk.get("matched_question"), "found_by": chunk["found_by"].split("+"),
        "kept": chunk["kept"], "picked_rank": picked_rank,
    }


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask")
def ask(body: Question):
    """Run the pipeline step by step (the same steps as answer.answer()) and report each one."""
    timings, started = {}, time.perf_counter()
    question = clean_question(body.question)
    if not question or len(question) > config.MAX_QUESTION_CHARS:
        return {"question": question, "answer": EMPTY if not question else TOO_LONG, "status": "rejected",
                "candidates": [], "sent": [], "timings": {}, "tokens": {}}

    t = time.perf_counter()
    candidates = hybrid_search(question, apply_thresholds=False)
    timings["search"] = time.perf_counter() - t
    kept = [c for c in candidates if c["kept"]]

    t = time.perf_counter()
    rerank_tokens = 0
    if config.USE_RERANKER and kept:
        sent, rerank_tokens = rerank(question, kept)
    else:
        sent = kept
    timings["rerank"] = time.perf_counter() - t

    t = time.perf_counter()
    result = answer_with_details(question, sent)
    timings["answer"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - started

    picked = {c["id"]: rank for rank, c in enumerate(sent, start=1)}
    status = "blocked" if result["blocked"] else "not_found" if result["answer"] == NOT_FOUND else "answered"
    return {
        "question": question, "answer": result["answer"], "status": status, "blocked_reason": result["blocked"],
        "candidates": [_chunk_view(c, picked.get(c["id"])) for c in candidates],
        "sent": [_chunk_view(c, rank) for rank, c in enumerate(sent, start=1)],
        "counts": {"candidates": len(candidates), "kept": len(kept), "sent": len(sent)},
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "tokens": {"rerank": rerank_tokens, "answer": result["tokens"]},
    }


@app.get("/api/documents")
def documents():
    """Ingested documents with their chunk counts, and the settings the pipeline runs with."""
    data = get_collection().get(include=["metadatas"])
    docs = defaultdict(lambda: {"chunks": 0, "tables": 0, "units": set(), "unit": "page", "questions": 0})
    for meta in data["metadatas"]:
        doc = docs[meta["source"]]
        doc["chunks"] += 1
        doc["tables"] += meta["type"] == "table"
        doc["unit"] = meta.get("unit", "page")
        doc["units"].update(range(meta["page_start"], meta["page_end"] + 1))
    questions = get_question_collection()
    if questions is not None:
        for meta in questions.get(include=["metadatas"])["metadatas"]:
            docs[meta["source"]]["questions"] += 1
    return {
        "documents": [{"name": name, "chunks": d["chunks"], "tables": d["tables"], "unit": d["unit"],
                       "units": len(d["units"]), "questions": d["questions"]} for name, d in sorted(docs.items())],
        "settings": {
            "Chat model": config.CHAT_MODEL, "Embedding model": config.EMBEDDING_MODEL,
            "Chunk size": f"{config.MIN_CHUNK_TOKENS}-{config.MAX_CHUNK_TOKENS} tokens",
            "Vector search": f"top {config.HYBRID_VECTOR_K}, kept if similarity ≥ {config.SIMILARITY_THRESHOLD}",
            "Keyword search (BM25)": f"top {config.HYBRID_KEYWORD_K}, kept if score ≥ {config.KEYWORD_SCORE_THRESHOLD}",
            "Generated-question search": (f"top {config.HYBRID_QUESTION_K}, kept if similarity ≥ "
                                          f"{config.QUESTION_SIMILARITY_THRESHOLD}") if config.USE_QUESTION_INDEX
                                         else "off",
            "Re-ranker": f"keeps up to {config.RERANK_KEEP}" if config.USE_RERANKER else "off",
        },
        "thresholds": {"similarity": config.SIMILARITY_THRESHOLD, "bm25": config.KEYWORD_SCORE_THRESHOLD,
                       "question_similarity": config.QUESTION_SIMILARITY_THRESHOLD},
        "max_question_chars": config.MAX_QUESTION_CHARS,
    }


@app.get("/api/benchmark")
def benchmark():
    """The latest saved benchmark.py run, or {"results": null} if there is none."""
    try:
        return json.loads(config.BENCHMARK_RESULTS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"results": None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  # localhost only: not reachable from other computers
