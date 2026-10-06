# PDF Semantic Search / RAG Pipeline

Ask questions about one or more PDFs and get answers grounded in the documents, with file and page citations.
Built and evaluated on NVIDIA's Form 10-Q for the quarter ended July 28, 2024 (`data/nvidia-1.pdf`).

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
| `python ingest.py` | Steps 2-5: extract every PDF in `data/`, chunk them, embed the chunks, store them in `chroma_db/`. The collection is rebuilt from scratch, so run it again after adding, removing or changing a PDF, or changing chunking settings (about 1.5 min for the 80-page 10-Q). |
| `python ingest.py --dump` | Same, and also writes each PDF's extracted text and tables to `data/extracted/<name>.txt` for inspection. |
| `python retrieval.py "your question" [k]` | Steps 6-7: show the top k vector-search chunks with similarity scores, then the hybrid-search chunks with similarity and BM25 scores, and whether each passes the thresholds. |
| `python answer.py "your question"` | Step 8: answer the question from the document, citing pages. |
| `python evaluate.py` | Step 9: run 10 test questions, printing retrieved chunks, scores, the answer and the expected answer. `python evaluate.py 1 7` runs selected questions; `--full` prints whole chunks; `--vector-only` runs version 1 (no keyword search) for comparison. |
| `python benchmark.py` | Runs the 39 questions in `eval_set.json` and scores them automatically: retrieval recall and MRR, answer correctness and faithfulness (judged by `gpt-4o`), and citation accuracy. `--vector-only` scores version 1. Needs `python ingest.py --dump` first (accepted citation pages are found in the dump). Takes about 3 minutes. |
| `python calibrate.py` | Measures similarity and BM25 scores for answerable vs. unanswerable questions (used to choose both thresholds), then checks the thresholds on separate validation questions. |

Settings (models, chunk sizes, `TOP_K`, thresholds, hybrid sizes) live in `config.py`.

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All settings in one place |
| `ingest.py` | Extraction, table extraction, chunking, table descriptions, embeddings, ChromaDB storage |
| `retrieval.py` | `search(question, k)`: embedding search (embeds the question, queries Chroma, converts distance to similarity, applies the threshold). `keyword_search()`: BM25. `hybrid_search()`: both combined, used for answering |
| `answer.py` | `answer(question)`: builds the prompt (system rules + tagged excerpts + tagged question), calls the chat model, and checks the answer before returning it |
| `evaluate.py` | Evaluation over 10 questions, printed for manual review |
| `eval_set.json` | 39 benchmark questions: answerable ones with evidence snippets and expected answers, plus unanswerable ones |
| `benchmark.py` | Automatic scoring over `eval_set.json` |
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

## Multiple PDFs

Every `*.pdf` in `data/` is ingested. Each chunk records its file in the `source` metadata and gets a unique id (`nvidia-1.pdf#0042`). Excerpts are labelled with file and pages, answers cite both, e.g. `(nvidia-1.pdf, p. 3)`, and the output check verifies the (file, page) pair, so a correct page number in the wrong file is rejected. A citation without a file name is accepted only when all excerpts come from one document. The prompt tells the model not to merge facts from different documents.

**Retrieval runs per document.** `hybrid_search()` does the vector and keyword searches once per file and merges the results, grouped by document. Searched together, "Compare NVIDIA's and AMD's revenue" got only NVIDIA chunks in the top 5 (AMD's revenue chunk ranked #33); per document, both companies' figures are found and cited. The thresholds still keep unrelated files out (a revenue question retrieves nothing from a homework PDF).

**Unreadable files are skipped.** Each page's text is checked before chunking: under 50 characters means no text layer (a scan), and a high share of `(cid:N)` codes or control characters means the fonts can't be decoded (common with a browser's "Print to PDF"). Bad pages are dropped; a file with more than half its pages bad is skipped with a message explaining why.

Tested with NVIDIA's and AMD's 10-Qs plus an unrelated homework PDF (the last two are not committed): questions about each file cite the right file, and comparisons cite each company's figure to its own document.

Limitations:
- **Ambiguous questions are answered inconsistently.** When a question doesn't name a company ("What was Data Center revenue last quarter?"), the prompt asks the model to answer for every document that has the answer. `gpt-4o-mini` does this only sometimes: gross margin was answered for both companies, Data Center revenue only for AMD, although NVIDIA's figure was in the context. Rewording the rule more strongly made it worse. A structural fix would be to answer each document separately and then combine the answers (more LLM calls). Until then, name the company in the question.
- **More context per question.** Every document contributes its passing chunks, so with two 10-Qs a financial question sends ~7-8K input tokens instead of ~3K (about $0.0012 instead of $0.0005 per question), including chunks from the company not asked about.
- **Unanswerable test questions depend on the document set.** Adding AMD's 10-Q made "Who is the CEO of AMD?" answerable (Dr. Lisa Su, amd-1.pdf, p. 56), so that adversarial test had to change. Re-check `eval_set.json` when documents are added.
- **Scanned PDFs and broken fonts are skipped, not read.** Reading them needs OCR.
- **Thresholds were calibrated on the 10-Q alone.** BM25 scores depend on word rarity across all chunks, so re-run `calibrate.py` after adding documents.
- File names shouldn't contain spaces, commas, semicolons or parentheses (`ingest.py` warns), or citations can't be checked reliably.
- `eval_set.json`, `evaluate.py` and `calibrate.py` cover the 10-Q only; benchmark questions can name another file with a `"source"` field.

## Security: prompt injection and malicious input

**SQL/code injection** (`"`, `\`, `--`, `DROP TABLE`) doesn't apply: there is no SQL database, Chroma is queried with a vector rather than the question text, and the question is never executed as code. Those symbols are just characters.

**Prompt injection** ("ignore all previous instructions...") is the real risk: the model reads our rules and the user's text in the same conversation. The worst realistic outcome here is an off-topic or outside-knowledge answer, because the model has no tools, no secrets in its prompt, and only public data. Defences, in `answer.py`:

1. **Least privilege:** no tools, no secrets, read-only data. Even a successful injection can only produce text.
2. **Input limits:** questions over `MAX_QUESTION_CHARS` (500) are rejected before any API call; invisible control and formatting characters (e.g. zero-width spaces) are removed.
3. **Untrusted text is fenced:** excerpts and the question sit inside `<excerpt>` / `<question>` tags, with `<` and `>` escaped so the text can't close a tag. The system prompt says text inside the tags is data, never instructions, and that mixed requests get only the document part answered.
4. **Output check (doesn't rely on the model behaving):** an answer must be the "not found" message, or cite at least one file and page that belongs to a retrieved excerpt. Anything else (a joke, an outside-knowledge answer, a leaked prompt, an invented page) is replaced with "not found".

Excerpts are treated as untrusted too: a PDF could contain planted instructions (indirect prompt injection).

`eval_set.json` has 9 `adversarial` questions: instruction overrides, SQL-style symbols, a request to reveal the prompt, a fake `SYSTEM:` rule, a closing `</question>` tag, an instruction planted inside an excerpt, zero-width characters and an over-long question. `python benchmark.py --category adversarial`: **9/9 handled safely**.

Limitations: no prompt defence is complete. The output check can't catch injected text in an answer that also carries a valid citation, and a keyword blocklist was deliberately not used because rewording bypasses it.

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

**Caveat:** the 10 eval questions were written using the document's wording, which favors keyword search. The benchmark below addresses that.

## Benchmark (`python benchmark.py`)

39 questions in `eval_set.json`: 29 answerable (8 using the document's wording, 16 deliberately paraphrased, 5 "why"/comparison) and 10 unanswerable (off-topic, other companies, and on-topic facts the filing doesn't contain, including a trap: it mentions expected Q4 Blackwell revenue but no total Q4 revenue).

| Metric | v1 vector | v2 hybrid |
|---|---|---|
| Recall (answer chunk sent to the LLM) | 21/29 (72%) | **24/29 (83%)** |
| MRR | 0.55 | **0.60** |
| Correct | 22/29 (76%) | **23/29 (79%)** |
| Citation includes a right page | 20/29 (69%) | 21/29 (72%) |
| Unanswerable correctly declined | 10/10 | 10/10 |
| Faithful (no unsupported claims) | 39/39 | 39/39 |

Findings:
- **Hybrid search clearly improves retrieval** (recall +11 points), but end-to-end correctness only by one question, which is within run-to-run variation: borderline questions (#1, #8) flip between runs.
- **Paraphrases are the main weakness:** 5 of v2's 6 failures are paraphrased questions ("invest in developing new technology" vs. "research and development", "divided" vs. "stock split", "billed to customers in the US" vs. "United States").
- **One generation failure:** #20 retrieved the right table at rank 1, but the model didn't connect "largest direct customer" with "Customer A" and refused.
- **Citations:** chunks spanning two pages (e.g. pp. 26-27) are sometimes cited by their first page when the fact is on the second.
- **No hallucinations:** every answer's claims were supported by the excerpts; every unanswerable question was declined.

Lessons from building the benchmark itself:
- Hand-written citation pages were wrong (the $50.0 billion buyback approval appears on pp. 20, 32 and 40), so accepted pages are now found automatically in the extracted text.
- The LLM judge marked "not found" as correct for answerable questions and once cited facts that weren't in the excerpts. Refusals are now graded by rule, and the judge is told to use only the excerpts.

Next improvements: query rewriting (expanding paraphrases into the document's vocabulary), a re-ranking step, and citing the exact page of cross-page chunks.
