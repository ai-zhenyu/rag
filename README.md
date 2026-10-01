# PDF Semantic Search / RAG Pipeline

Ask questions about a PDF and get answers grounded in the document, with page citations.
Built on NVIDIA's Form 10-Q for the quarter ended July 28, 2024 (`data/nvidia-1.pdf`).

```
PDF ──► extract (unstructured + pdfplumber) ──► chunk (tiktoken) ──► embed (OpenAI) ──► ChromaDB
                                                                                          │
question ──┬─► vector search (top 5) ──┐
           └─► keyword search (top 5) ─┴─► thresholds ──► gpt-4o-mini ──► answer + pages
```

## Setup (Windows)

```
py -3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder (see `.env.example`):

```
OPENAI_API_KEY=sk-...
```

## Running each part

With the venv active:

| Command | What it does |
|---|---|
| `python ingest.py` | Steps 2-5: extract the PDF, chunk it, embed the chunks, store them in `chroma_db/`. Run once, and again after changing the PDF or chunking settings (about 1.5 min). |
| `python ingest.py --dump` | Same, and also writes the extracted text and tables to `data/extracted/nvidia-1.txt` for inspection. |
| `python retrieval.py "your question" [k]` | Steps 6-7: show the top k vector-search chunks with similarity scores, then the hybrid-search chunks with similarity and BM25 scores, and whether each passes the thresholds. |
| `python answer.py "your question"` | Step 8: answer the question from the document, citing pages. |
| `python evaluate.py` | Step 9: run 10 test questions, printing retrieved chunks, scores, the answer and the expected answer. `python evaluate.py 1 7` runs selected questions; `--full` prints whole chunks; `--vector-only` runs version 1 (no keyword search) for comparison. |
| `python calibrate.py` | Measures similarity and BM25 scores for answerable vs. unanswerable questions (used to choose both thresholds), then checks the thresholds on separate validation questions. |

Settings (models, chunk sizes, `TOP_K`, thresholds, hybrid sizes) live in `config.py`.

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All settings in one place |
| `ingest.py` | Extraction, table extraction, chunking, table descriptions, embeddings, ChromaDB storage |
| `retrieval.py` | `search(question, k)`: embedding search (embeds the question, queries Chroma, converts distance to similarity, applies the threshold). `keyword_search()`: BM25. `hybrid_search()`: both combined, used for answering |
| `answer.py` | `answer(question)`: sends the question and retrieved chunks to the chat model with grounding rules |
| `evaluate.py` | Evaluation over 10 questions |
| `calibrate.py` | Threshold calibration and validation |

## Design decisions

- **Text extraction:** `unstructured` with `strategy="fast"` (no OCR or system dependencies). It reads tables column by column, separating labels from their numbers, so tables are extracted separately with **pdfplumber** (pure Python) as one row per line, e.g. `Revenue | 30,040 | 13,507 | 56,084 | 20,699`, with their title and column headers. The scrambled copy of each table is removed from the `unstructured` text by position.
- **Chunking:** whole paragraphs are packed into chunks of 300-500 tokens (`cl100k_base`); a paragraph is never cut. Chunks may cross a page break and store `page_start`/`page_end`. Tables are separate chunks; tables over 500 tokens are split by rows with the header repeated. Result: 180 chunks (131 prose, 49 table); 125 of 131 prose chunks are in 300-500 tokens.
- **Table descriptions:** number-heavy tables embed poorly (the income statement ranked #41 of 180 for a revenue question). During ingest, `gpt-4o-mini` writes a number-free description of each table (topic, periods, every row label). The *embedding* uses description + table; the stored *text* is the raw table, so answers only read real figures.
- **Storage:** ChromaDB persistent collection with `space="cosine"`. Chroma fields: `documents` = chunk text, `embeddings` = OpenAI vectors (`text-embedding-3-small`, 1,536 dimensions), `metadatas` = `n_tokens`, `page_start`, `page_end`, `type`, `source` (+ `description` for tables).
- **Distance to similarity:** for a cosine collection Chroma returns `distance = 1 - cosine_similarity`, so `similarity = 1 - distance`.
- **Threshold = 0.50, TOP_K = 10**, chosen with `calibrate.py` rather than the suggested 0.6: chunks containing the answer scored 0.49-0.72, off-topic and other-company questions at most 0.44. At 0.6 the correct R&D chunk (0.595) would be dropped. On-topic questions the document can't answer (e.g. revenue in fiscal 2027, 0.68) score as high as real answers; no threshold separates them, so the answer prompt handles those.
- **Hybrid search (version 2):** vector search misses exact terms buried in a chunk about other topics and small tables (the inventory table ranked #44). BM25 keyword search (`rank_bm25`) ranks chunks by shared words, weighting rare words like "Blackwell" more. `hybrid_search()` takes the top 5 from each and merges duplicates. Standard Reciprocal Rank Fusion was tested but left the inventory table at #13, outside the context. A chunk is kept if its similarity is >= 0.50 **or** its BM25 score is >= 7. That keyword threshold was chosen with `calibrate.py`: at 7 all 14 calibration answers are kept and no unanswerable question gets through that vector search didn't already let through. The margin is thin (off-topic noise reached 6.7, the weakest answer 7.6).
- **Answering:** `gpt-4o-mini`, `temperature=0`, instructed to use only the excerpts, cite pages, and otherwise reply with a fixed "not in the document" message. If no chunk passes the threshold the model is not called. Excerpts are labelled only with their page (`=== Excerpt from p. 3 (table) ===`); with numbered labels like `[Source 4 | p. 3]` the model cited the wrong page.

## Evaluation results (`python evaluate.py`)

| # | Question type | v1: vector only (`--vector-only`) | v2: hybrid (default) |
|---|---|---|---|
| 1 | Table lookup (revenue, quarter ended Jul 30, 2023) | Correct: $13,507 million (p. 3) | Correct |
| 2 | Table lookup (Data Center revenue, six months) | Correct: $48,835 million (p. 23) | Correct |
| 3 | Figure + explanation (gross margin) | Correct: 75.1% vs 70.1%, Data Center growth (p. 29) | Correct |
| 4 | Prose synthesis (export-control risks) | Correct: grounded list with page citations | Correct |
| 5 | Prose fact (share repurchases) | Correct: 62.8 million shares for $7.0 billion (pp. 20-21) | Correct |
| 6 | Prose fact (Blackwell ramp) | **Miss**: best chunk scored 0.489, just below the threshold | Correct: Q4, continuing into fiscal 2026 (p. 26), found by keyword search |
| 7 | Two-level table header (segment operating income) | Correct: $18,848 million, up $12,120 million / 180% (p. 28) | Correct |
| 8 | Small table (inventory breakdown) | **Miss**: table not retrieved; the model says the answer isn't there rather than guessing | Correct: all four figures (p. 16), found by keyword search |
| 9 | Unanswerable, on-topic (revenue in fiscal 2027) | Correctly refused by the prompt | Correctly refused by the prompt |
| 10 | Unanswerable, other company (Apple) | Correctly refused by the threshold (no LLM call) | Correctly refused by the thresholds |

**v1: 8/10. v2: 10/10.** Neither version gave a hallucinated answer.

**Validation** (`calibrate.py`, questions not used to choose thresholds): 5 of 6 paraphrased questions keep their answer chunk. "What did NVIDIA earn after taxes?" works through vector search even though it shares no keywords with "net income". "How much stock was sitting in NVIDIA's warehouses?" fails: it shares no words with "inventories", and the vector match is weak. Unanswerable questions that share keywords with the document ("AMD's data center revenue") get past the thresholds, so the answer prompt has to refuse them, which it does.

**Caveat:** the 10 eval questions were written using the document's wording, which favors keyword search. Next improvements would be a larger eval set with paraphrased questions, and a re-ranking step.
