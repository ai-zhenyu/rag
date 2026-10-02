"""Evaluate RAG quality: for each question print the retrieved chunks (with similarity
scores), the answer, and the expected answer, so retrieval and generation can be judged
separately.

Run:  python evaluate.py            # all questions
      python evaluate.py 1 7        # only questions 1 and 7
      python evaluate.py --full     # print whole chunks instead of previews
      python evaluate.py --vector-only   # version 1 (vector search only), for comparison
"""
import sys

import config
from answer import answer
from retrieval import format_source, hybrid_search, search

# (question, what it tests, expected answer taken from the PDF)
QUESTIONS = [
    ("What was NVIDIA's revenue for the quarter ended July 30, 2023?",
     "table lookup: prior-year column of the income statement",
     "$13,507 million (p. 3; also p. 26)"),
    ("What was Data Center revenue in the six months ended July 28, 2024?",
     "table lookup: six-month column of the revenue-by-market table",
     "$48,835 million (p. 23)"),
    ("What was NVIDIA's gross margin in the second quarter of fiscal 2025, and why did it change from a year ago?",
     "prose + table: a figure plus its explanation",
     "75.1%, up from 70.1%, primarily due to strong Data Center revenue growth of 154% (p. 29; figures also p. 26)"),
    ("What are the risks related to US export controls on China?",
     "prose synthesis across several chunks",
     "License requirements/denials for A100/H100-class products, lost competitiveness in China, "
     "inventory and supply-chain disruption, compliance burden, possible further restrictions (pp. 25, 38-39)"),
    ("How many shares did NVIDIA repurchase in the second quarter, and for how much?",
     "prose lookup: a fact inside a longer note",
     "62.8 million shares for $7.0 billion (pp. 20-21)"),
    ("When is the Blackwell production ramp scheduled to begin?",
     "prose fact whose best vector match is just below the similarity threshold (0.49); keyword search should rescue it",
     "Q4 of fiscal 2025, continuing into fiscal 2026 (p. 24; repeated on p. 26)"),
    ("What was the Compute & Networking segment's operating income in the second quarter, and how did it change?",
     "table with two header levels (three/six months, then $ and % change)",
     "$18,848 million vs $6,728 million a year ago, up $12,120 million or 180% (p. 28)"),
    ("What was the breakdown of NVIDIA's inventories as of July 28, 2024?",
     "small table: vector search ranks it ~#44, keyword search #1",
     "Raw materials $1,895M, work in process $2,111M, finished goods $2,669M; total $6,675M (p. 16)"),
    ("What was NVIDIA's revenue in fiscal year 2027?",
     "unanswerable, on-topic: passes the threshold, so the prompt must refuse",
     "Not in the document"),
    ("What was Apple's iPhone revenue in 2024?",
     "unanswerable, other company: should be blocked by the threshold (no LLM call)",
     "Not in the document"),
]


def print_chunks(chunks, full):
    for rank, c in enumerate(chunks, start=1):
        status = "kept   " if c["kept"] else "dropped"
        keyword = f"  bm25={c['bm25_score']:5.1f}  by {c['found_by']:<7}" if "bm25_score" in c else ""
        text = c["text"] if full else " ".join(c["text"].split())[:120] + "..."
        print(f"  #{rank:<2} {status} similarity={c['similarity']:.3f}{keyword}  {format_source(c):<24} "
              f"{c['type']:<5}  {text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    full = "--full" in sys.argv
    selected = [int(a) for a in sys.argv[1:] if a.isdigit()] or range(1, len(QUESTIONS) + 1)

    vector_only = "--vector-only" in sys.argv  # version 1 behaviour, for comparison

    if vector_only:
        print(f"Mode: VECTOR ONLY (v1): top_k={config.TOP_K}, similarity threshold={config.SIMILARITY_THRESHOLD}")
    else:
        print(f"Mode: HYBRID (v2): top {config.HYBRID_VECTOR_K} vector + top {config.HYBRID_KEYWORD_K} keyword, "
              f"kept if similarity >= {config.SIMILARITY_THRESHOLD} or BM25 >= {config.KEYWORD_SCORE_THRESHOLD}")
    print(f"Models: embedding={config.EMBEDDING_MODEL}, chat={config.CHAT_MODEL}")

    for n in selected:
        question, tests, expected = QUESTIONS[n - 1]
        print("\n" + "=" * 100)
        print(f"Q{n}: {question}")
        print(f"Tests: {tests}")

        # Retrieve without filtering so dropped chunks are shown too, then keep the ones that pass.
        if vector_only:
            retrieved = search(question, threshold=None)
            for c in retrieved:
                c["kept"] = c["similarity"] >= config.SIMILARITY_THRESHOLD
        else:
            retrieved = hybrid_search(question, apply_thresholds=False)
        kept = [c for c in retrieved if c["kept"]]
        print(f"\nRetrieved chunks ({len(kept)} of {len(retrieved)} pass the threshold):")
        print_chunks(retrieved, full)

        print(f"\nANSWER:   {answer(question, kept)}")
        print(f"EXPECTED: {expected}")
