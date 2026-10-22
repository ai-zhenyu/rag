# Document Semantic Search / RAG Pipeline

Ask questions about PDF and Word (.docx) documents and get answers grounded in them, with file and page (or section) citations.
Built and evaluated on NVIDIA's Form 10-Q for the quarter ended July 28, 2024 (`data/nvidia-1.pdf`).

```
PDF ──► extract (unstructured + pdfplumber) ──► chunk (tiktoken) ──► embed (OpenAI) ──► ChromaDB
                                                  └─► generate 5 questions per chunk ──► ChromaDB (questions)

question ──┬─► vector search (top 7) ─────────┐
           ├─► keyword search (top 7) ────────┼─► thresholds ──► re-rank (gpt-6-luna) ──► gpt-6-luna ──► answer + pages
           └─► generated-question search (top 5) ┘    (~16 chunks)        (keeps ≤ 7, usually 2)
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
| `python ingest.py` | Steps 2-5: extract every PDF and Word (.docx) file in `data/`, chunk them, embed the chunks, store them in `chroma_db/`. Other file types are reported and ignored. The collection is rebuilt from scratch, so run it again after adding, removing or changing a document, or changing chunking settings (about 1.5 min for the 80-page 10-Q). |
| `python ingest.py --dump` | Same, and also writes each PDF's extracted text and tables to `data/extracted/<name>.txt` for inspection. |
| `python retrieval.py "your question" [k]` | Steps 6-7: show the top k vector-search chunks with similarity scores, then the hybrid-search chunks with similarity and BM25 scores, and whether each passes the thresholds. |
| `python answer.py "your question"` | Step 8: answer the question from the document, citing pages. |
| `python evaluate.py` | Step 9: run 10 test questions, printing retrieved chunks, scores, the answer and the expected answer. `python evaluate.py 1 7` runs selected questions; `--full` prints whole chunks; `--vector-only` runs version 1 (no keyword search) for comparison. |
| `python benchmark.py` | Runs the 48 questions in `eval_set.json` and scores them automatically: retrieval recall and MRR, answer correctness and faithfulness (judged by `gpt-6.1-sol`), and citation accuracy. `--no-rerank` scores version 3, `--no-questions` version 2 and `--vector-only` version 1. Needs `python ingest.py --dump` first (accepted citation pages are found in the dump). Takes about 5 minutes. A full run is saved to `benchmark_results.json`, which the web app's Evaluation tab shows. |
| `python app.py` | Web app at http://127.0.0.1:8000 (see below). |
| `python calibrate.py` | Measures similarity and BM25 scores for answerable vs. unanswerable questions (used to choose both thresholds), then checks the thresholds on separate validation questions. |

Settings (models, chunk sizes, `TOP_K`, thresholds, hybrid sizes) live in `config.py`.

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All settings in one place |
| `ingest.py` | Extraction, table extraction, chunking, table descriptions, generated questions, embeddings, ChromaDB storage |
| `retrieval.py` | `search(question, k)`: embedding search (embeds the question, queries Chroma, converts distance to similarity, applies the threshold). `keyword_search()`: BM25. `question_search()`: generated questions. `hybrid_search()`: all three combined. `rerank()`: the chat model picks the useful chunks. `retrieve()`: hybrid search then re-ranking, used for answering |
| `answer.py` | `answer(question)`: builds the prompt (system rules + tagged excerpts + tagged question), calls the chat model, and checks the answer before returning it |
| `evaluate.py` | Evaluation over 10 questions, printed for manual review |
| `eval_set.json` | 48 benchmark questions: answerable ones with evidence snippets and expected answers, unanswerable ones, and prompt-injection attempts |
| `benchmark.py` | Automatic scoring over `eval_set.json` |
| `calibrate.py` | Threshold calibration and validation |
| `app.py` | Web app server (FastAPI): `/api/ask`, `/api/documents`, `/api/benchmark` |
| `web/` | The web page: `index.html`, `style.css`, `app.js` (no build step, no external libraries) |

## Web app (`python app.py`)

The assignment asked for a terminal-only project; the web app was added afterwards to make the pipeline visible.
It runs only on this computer (`127.0.0.1:8000`), and the API key stays on the server side.

- **Ask:** type a question (or pick an example) and get the answer with clickable citations; clicking one highlights
  the excerpt it came from. Below the answer:
  - a pipeline strip: how many candidates the three searches found, how many passed a threshold, how many the
    re-ranker chose, and the time and tokens of each step;
  - the excerpts the answer model read, with their scores, which searches found them, and the generated question
    that matched best;
  - every candidate in a table, with scores that reached their threshold in green and whether it was sent, kept but
    not chosen, or dropped; click a row for its text.
- **Documents:** each indexed file with its pages or sections, chunks, table chunks and generated questions, and the
  current settings from `config.py`.
- **Evaluation:** the latest full `python benchmark.py` run: summary metrics, results by category, and every
  question with its answer, the expected answer and the judge's reason, filterable to failures.

`/api/ask` runs the same steps as `answer.answer()` one at a time so it can report each. Everything taken from the
documents or the model is escaped before it is shown, so a document can't inject HTML into the page.

## Design decisions

- **Text extraction:** `unstructured` with `strategy="fast"` (no OCR or system dependencies). It reads tables column by column, separating labels from their numbers, so tables are extracted separately with **pdfplumber** (pure Python) as one row per line, e.g. `Revenue | 30,040 | 13,507 | 56,084 | 20,699`, with their title and column headers. The scrambled copy of each table is removed from the `unstructured` text by position.
- **Chunking:** whole paragraphs are packed into chunks of 300-500 tokens (`cl100k_base`); a paragraph is never cut. Chunks may cross a page break and store `page_start`/`page_end`. Tables are separate chunks; tables over 500 tokens are split by rows with the header repeated. Result: 180 chunks (131 prose, 49 table); 125 of 131 prose chunks are in 300-500 tokens.
- **Table descriptions:** number-heavy tables embed poorly (the income statement ranked #41 of 180 for a revenue question). During ingest, the chat model (`gpt-6-luna`) writes a number-free description of each table (topic, periods, every row label). Descriptions are cached in `data/table_descriptions.json` (keyed by table text, model and prompt) and reused on re-ingest: regenerating them changes the wording slightly and moved the income statement from rank 4 to 6. The *embedding* uses description + table; the stored *text* is the raw table, so answers only read real figures.
- **Storage:** ChromaDB persistent collection with `space="cosine"`. Chroma fields: `documents` = chunk text, `embeddings` = OpenAI vectors (`text-embedding-3-small`, 1,536 dimensions), `metadatas` = `n_tokens`, `page_start`, `page_end`, `type`, `source` (+ `description` for tables).
- **Distance to similarity:** for a cosine collection Chroma returns `distance = 1 - cosine_similarity`, so `similarity = 1 - distance`.
- **Threshold = 0.50, TOP_K = 10**, chosen with `calibrate.py` rather than the suggested 0.6: chunks containing the answer scored 0.49-0.72, off-topic and other-company questions at most 0.44. At 0.6 the correct R&D chunk (0.595) would be dropped. On-topic questions the document can't answer (e.g. revenue in fiscal 2027, 0.68) score as high as real answers; no threshold separates them, so the answer prompt handles those.
- **Hybrid search (version 2):** vector search misses exact terms buried in a chunk about other topics and small tables (the inventory table ranked #44). BM25 keyword search (`rank_bm25`) ranks chunks by shared words, weighting rare words like "Blackwell" more. `hybrid_search()` takes the top 7 from each and merges duplicates (5 at first; raised to 7 when NVIDIA's income statement slipped to rank 6 in both searches and fell out of the context). Standard Reciprocal Rank Fusion was tested but left the inventory table at #13, outside the context. A chunk is kept if its similarity is >= 0.50 **or** its BM25 score is >= 7. That keyword threshold was chosen with `calibrate.py`: at 7 all 14 calibration answers are kept and no unanswerable question gets through that vector search didn't already let through. The margin is thin (off-topic noise reached 6.7, the weakest answer 7.6). After the table descriptions were rewritten by `gpt-6-luna` the margin is gone (an off-topic question scores 7.6, the weakest answer 7.1); 7 was kept because the answer prompt declines such questions.
- **Generated questions, "reverse HyDE" (version 3):** a user's question is usually worded more like another question than like the passage that answers it. During ingest, `gpt-6-luna` writes 5 questions per chunk (about half in the document's wording, half in plain words, e.g. "money owed to suppliers" for accounts payable), cached in `data/chunk_questions.json` like the descriptions. They are embedded in a second collection, `chunk_questions` (1,600 questions for 320 chunks). `hybrid_search()` also takes the 5 chunks whose generated questions best match the user's question, and keeps a chunk whose best generated question reaches similarity **0.55**. Calibration: off-topic questions matched generated questions at most 0.50, answer chunks scored 0.66-0.94. "How much stock was sitting in NVIDIA's warehouses?", which failed validation in every earlier version, now finds the inventory table through the generated question "How much money did NVIDIA have tied up in all inventories?". Cost: one chat call per chunk at ingest (cached afterwards), one extra Chroma query per document per question, and about 14% more context tokens per answer.
- **Re-ranking (version 4):** the three searches each score a chunk on one signal, and none reads the question and the chunk together, so about 16 chunks pass the thresholds and the answer model has to read past most of them (#24 had its answer at rank 4 and still refused). `rerank()` sends the question and every kept candidate to `gpt-6-luna`, which returns the ids of at most 7 useful ones as JSON; usually it keeps 1-3. The thresholds still decide whether anything is relevant at all; the re-ranker can also keep nothing, and then the answer model isn't called (9 of 10 unanswerable questions). Candidates are fenced and escaped like the answer prompt's excerpts, and only valid candidate ids are accepted. Candidates were first shortened to 300 tokens, but answers past token 350 of their chunk were then invisible (#12, #21); they are now shown whole (cap 700 tokens). The model sometimes writes ids as strings ("4"), which the parser accepts. Cost: about 6,900 tokens and 1 s per question for the re-ranker call, while the answer model's context drops from about 6,500 to about 700 tokens.
- **Answering:** `gpt-6-luna` with `reasoning_effort="low"` (it rejects `temperature=0`, so wording can vary slightly between runs), instructed to use only the excerpts, cite pages, and otherwise reply with a fixed "not in the document" message. If no chunk passes the threshold the model is not called. Excerpts are labelled only with their page (`=== Excerpt from p. 3 (table) ===`); with numbered labels like `[Source 4 | p. 3]` the model cited the wrong page.

## Multiple PDFs

Every `*.pdf` in `data/` is ingested. Each chunk records its file in the `source` metadata and gets a unique id (`nvidia-1.pdf#0042`). Excerpts are labelled with file and pages, answers cite both, e.g. `(nvidia-1.pdf, p. 3)`, and the output check verifies the (file, page) pair, so a correct page number in the wrong file is rejected. A citation without a file name is accepted only when all excerpts come from one document. The prompt tells the model not to merge facts from different documents.

**Retrieval runs per document.** `hybrid_search()` does the vector and keyword searches once per file and merges the results, grouped by document. Searched together, "Compare NVIDIA's and AMD's revenue" got only NVIDIA chunks in the top 5 (AMD's revenue chunk ranked #33); per document, both companies' figures are found and cited. The thresholds still keep unrelated files out (a revenue question retrieves nothing from a homework PDF).

**Unreadable files are skipped.** Each page's text is checked before chunking: under 50 characters means no text layer (a scan), and a high share of `(cid:N)` codes or control characters means the fonts can't be decoded (common with a browser's "Print to PDF"). Bad pages are dropped; a file with more than half its pages bad is skipped with a message explaining why.

Tested with NVIDIA's and AMD's 10-Qs plus an unrelated homework PDF (the last two are not committed): questions about each file cite the right file, and comparisons cite each company's figure to its own document.

Limitations:
- **Ambiguous questions are answered inconsistently.** When a question doesn't name a company ("What was Data Center revenue last quarter?"), the prompt asks the model to answer for every document that has the answer. `gpt-4o-mini` (the answer model at the time) did this only sometimes: gross margin was answered for both companies, Data Center revenue only for AMD, although NVIDIA's figure was in the context. Rewording the rule more strongly made it worse. A structural fix would be to answer each document separately and then combine the answers (more LLM calls). Until then, name the company in the question.
- **More context per question.** Every document contributes its passing chunks, so with two 10-Qs a financial question sends ~7-8K input tokens instead of ~3K (about $0.0012 instead of $0.0005 per question), including chunks from the company not asked about.
- **Unanswerable test questions depend on the document set.** Adding AMD's 10-Q made "Who is the CEO of AMD?" answerable (Dr. Lisa Su, amd-1.pdf, p. 56), so that adversarial test had to change. Re-check `eval_set.json` when documents are added.
- **Scanned PDFs and broken fonts are skipped, not read.** Reading them needs OCR.
- **Thresholds were calibrated on the 10-Q alone.** BM25 scores depend on word rarity across all chunks, so re-run `calibrate.py` after adding documents.
- File names shouldn't contain spaces, commas, semicolons or parentheses (`ingest.py` warns), or citations can't be checked reliably.
- `eval_set.json`, `evaluate.py` and `calibrate.py` cover the 10-Q only; benchmark questions can name another file with a `"source"` field.

## Word documents (.docx)

`.docx` files in `data/` are ingested with `unstructured`'s `partition_docx` (extra: `unstructured[docx]`).

- **Cited by section, not page.** Word files have no fixed pages: page breaks depend on how Word lays the document out, and `unstructured` reports no page numbers. Each heading starts a numbered section, and answers cite e.g. `(policy.docx, sec. 3)`. Excerpts also carry the heading path (e.g. "Reimbursement > Receipts") so the model knows which part of the document they're from. The output check verifies (file, unit, number), so a section number cited as a page is rejected.
- **Chunked per section.** Unlike PDF page breaks, section breaks separate topics, so chunks never span sections (long sections are still split by paragraph). Packed across sections, a short policy became one chunk cited as "secs. 1-6"; per section, citations are precise ("sec. 2"). Small sections make chunks below 300 tokens, a deliberate trade.
- **Tables keep their real structure.** Word stores rows and cells, so tables come from their HTML, not from positions on a page, and are converted to the same `Label | value | ...` rows as PDF tables, with the section title and the paragraph introducing the table as header.

Tested with a fictional travel-policy document (not committed): answers to policy questions cited the right sections, a table lookup ("daily meal maximum in London") cited the table's section, and PDF answers were unchanged.

Limitations:
- Old binary `.doc` files aren't supported (`ingest.py` suggests saving as `.docx`); neither are other formats.
- Text in images, text boxes and headers/footers may be missed; headings are detected from Word's heading styles, so a document that uses bold text instead of heading styles becomes one long section.

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

48 questions in `eval_set.json` (39 at first; 9 adversarial added later): 29 answerable (8 using the document's wording, 16 deliberately paraphrased, 5 "why"/comparison) and 10 unanswerable (off-topic, other companies, and on-topic facts the filing doesn't contain, including a trap: it mentions expected Q4 Blackwell revenue but no total Q4 revenue).

| Metric | v1 vector | v2 hybrid | v2, new models | v2, new models (re-run) | v3 generated questions | v4 re-ranked (300-token previews) | v4 re-ranked |
|---|---|---|---|---|---|---|---|
| Recall (answer chunk sent to the LLM) | 21/29 (72%) | 24/29 (83%) | 24/29 (83%) | 25/29 (86%) | **27/29 (93%)** | 25/29 (86%) | 26/29 (90%) |
| MRR | 0.55 | 0.60 | 0.61 | 0.63 | 0.66 | 0.81 | **0.86** |
| Correct | 22/29 (76%) | 23/29 (79%) | 21/29 (72%) | 23/29 (79%) | 23/29 (79%) | 24/29 (83%) | **26/29 (90%)** |
| Citation includes a right page | 20/29 (69%) | 21/29 (72%) | 19/29 (66%) | 21/29 (72%) | **24/29 (83%)** | **24/29 (83%)** | 23/29 (79%) |
| Unanswerable correctly declined | 10/10 | 10/10 | 9/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| Adversarial handled safely | - | - | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| Faithful (no unsupported claims) | 39/39 | 39/39 | 48/48 | 48/48 | 48/48 | 48/48 | 47/48 |
| Context per answered question | - | - | - | 13.8 chunks, 5,680 tokens | 16.1 chunks, 6,471 tokens | 1.8 chunks, 701 tokens | 1.9 chunks, 714 tokens |
| Re-ranker call per question | - | - | - | - | - | 4,840 tokens | 6,931 tokens |

The first two columns used `gpt-4o-mini` to answer and `gpt-4o` to judge, with 5 + 5 candidates. "v2, new models" uses
`gpt-6-luna` / `gpt-6.1-sol`, 7 + 7 candidates, table descriptions rewritten by `gpt-6-luna`, and AMD's 10-Q and a
homework PDF ingested alongside NVIDIA's.

The last two columns were run back to back with the same models and index, so they compare v2 and v3 fairly.
The two v2 runs with new models differ only by run-to-run variation (answers and judging aren't deterministic;
the index was rebuilt in between, with the same cached descriptions).

What re-ranking (v4) changed, compared with v3:
- **Correct 23 → 26 of 29 (90%), the best result so far; MRR 0.66 → 0.86.** The answer chunk is now usually the
  first excerpt, and the answer model reads about 2 chunks instead of 16. #24 (answer at rank 4, refused in v3) is fixed.
- **Recall is lower than v3 (26 vs 27):** the re-ranker can drop a chunk that contains the answer. Recall here means
  "sent to the answer model", so it now measures the re-ranker as well as the searches.
- **Tokens:** about 7,600 per question in total (6,900 re-ranker + 700 context, plus prompts), compared with about
  6,500 context tokens in v3, all on `gpt-6-luna`. 9 of 10 unanswerable questions skip the answer call entirely.
- **Faithfulness 47/48:** #18 has the right amount but calls July 28, 2024 the end of the fiscal year instead of the
  six-month period.
- **Remaining failures:** #6 and #25 are never retrieved (paraphrases: query rewriting would target them), #7 is the
  judge's "omitted detail" rule, and #34 is the stale test.

What generated questions (v3) changed, compared with the v2 re-run:
- **Retrieval improved clearly:** recall 25 → 27 of 29 and citations 21 → 24. #16 ("ratio shares were divided") is now
  found through the generated question about the stock split's extra shares.
- **Correctness is unchanged at 23/29.** v3 fixed #16, but #24 retrieved its answer at rank 4 and the model still
  refused, and #28 failed on the judge's "omitted detail" rule. With 16 chunks of context there is more to read past.
- **Still not retrieved:** #6 ("invest in developing new technology", R&D; best generated question scored 0.54, just
  under the 0.55 threshold, which was not lowered to fit the test) and #25 ("billed to customers in the US").

What changed with the new models:
- **The new judge is stricter.** 3 of the 8 failures (#7, #23, #28) state the main fact correctly but omit a secondary
  detail of the reference answer (e.g. "+154% year over year"); `gpt-4o` passed answers like these.
- **#34 ("AMD's data center revenue") is now a stale test:** AMD's 10-Q is ingested, so answering it from `amd-1.pdf`
  is right. Excluding it, unanswerable questions are 9/9.
- **#20 is fixed:** `gpt-6-luna` connects "largest direct customer" with "Customer A".
- **#1 regressed:** the income statement ranks #9 in vector search for "total revenue in the second quarter of fiscal
  2025", behind seven narrative chunks that repeat the question's wording. Ranks of table chunks shift whenever their
  descriptions change, which is why descriptions are now cached.

Findings:
- **Hybrid search clearly improves retrieval** (recall +11 points), but end-to-end correctness only by one question, which is within run-to-run variation: borderline questions (#1, #8) flip between runs.
- **Paraphrases are the main weakness:** 5 of v2's 6 failures are paraphrased questions ("invest in developing new technology" vs. "research and development", "divided" vs. "stock split", "billed to customers in the US" vs. "United States").
- **One generation failure:** #20 retrieved the right table at rank 1, but the model didn't connect "largest direct customer" with "Customer A" and refused.
- **Citations:** chunks spanning two pages (e.g. pp. 26-27) are sometimes cited by their first page when the fact is on the second.
- **No hallucinations:** every answer's claims were supported by the excerpts; every unanswerable question was declined.

Lessons from building the benchmark itself:
- Hand-written citation pages were wrong (the $50.0 billion buyback approval appears on pp. 20, 32 and 40), so accepted pages are now found automatically in the extracted text.
- The LLM judge marked "not found" as correct for answerable questions and once cited facts that weren't in the excerpts. Refusals are now graded by rule, and the judge is told to use only the excerpts.
- The judge isn't fully consistent (this was `gpt-4o` at temperature 0; `gpt-6.1-sol` can't use temperature 0 at all): the same safe answer to adversarial #42 was graded FAIL in one run and ok in the next. Don't over-read single-question differences between runs.

Next improvements: query rewriting (expanding paraphrases into the document's vocabulary), a re-ranking step, and citing the exact page of cross-page chunks.
