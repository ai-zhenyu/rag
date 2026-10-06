"""Ingestion pipeline: documents -> pages/sections -> chunks -> embeddings -> ChromaDB.

Every PDF and Word (.docx) file in data/ is ingested; the collection is rebuilt from scratch on each run.
PDF chunks are located by page; Word chunks by section (Word files have no fixed pages).

Run:  python ingest.py
      python ingest.py --dump   # also save each document's extracted text to data/extracted/<name>.txt
"""
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import chromadb
import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf

import config

ENCODER = tiktoken.get_encoding(config.TOKEN_ENCODING)

# How far above a table (in PDF points, 72 = 1 inch) to look for its title and column headers.
# Headers like "Three Months Ended / Jul 28, 2024" sit above the table's first ruling line.
TABLE_HEADER_BAND = 80

TABLE_DESCRIPTION_PROMPT = (
    "Write a short search description of this table from a document such as a financial report. Start with "
    "the table's specific topic, taken from its title or from the section label directly above the rows "
    '(e.g. "Inventories", not a generic heading); ignore any unrelated sentences above the table. Then '
    "say what the columns represent (e.g. reporting periods), and list EVERY row label so each one can be "
    "found by search. Do not quote any numbers. Do not invent anything.\n\n{table}"
)


def extract_tables(pdf_path):
    """Find tables with pdfplumber and return them as row-by-row text.

    unstructured's "fast" strategy reads tables column by column, which separates
    labels from their numbers. pdfplumber detects tables from their ruling lines and
    keeps each row together, e.g. "Revenue | 30,040 | 13,507 | 56,084 | 20,699".

    Returns a list of dicts: {"page": int, "bbox": (x0, top, x1, bottom), "text": str}
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            prev_bottom = 0
            for table in sorted(page.find_tables(), key=lambda t: t.bbox[1]):
                x0, top, x1, bottom = table.bbox
                # Look for headers just above the table, but not above the previous table on the page.
                header_top = max(prev_bottom, top - TABLE_HEADER_BAND)
                # within_bbox keeps only characters fully inside the band; crop() would also pull in
                # clipped characters from a line cut by the band edge and garble them ("FFoorr tthhee").
                header = page.within_bbox((0, header_top, page.width, top)).extract_text()
                prev_bottom = bottom
                rows = [_clean_row(row) for row in table.extract()]
                rows = [" | ".join(cells) for cells in rows if cells]
                if rows:
                    text = f"[Table on page {page.page_number}]\n{header}\n" + "\n".join(rows)
                    tables.append({"page": page.page_number, "bbox": table.bbox, "text": text})
    return tables


def _clean_row(row):
    """Drop empty and "$" cells, collapse whitespace, and glue stray "%" cells to their number."""
    cells = []
    for cell in row:
        cell = " ".join((cell or "").split())
        if cell in ("", "$"):
            continue
        if cell == "%" and cells:
            cells[-1] += "%"
        else:
            cells.append(cell)
    return cells


def _inside_any(element, bboxes):
    """True if the element's center lies inside one of the given (x0, top, x1, bottom) boxes."""
    coords = element.metadata.coordinates
    if coords is None or not coords.points:
        return False
    xs = [p[0] for p in coords.points]
    ys = [p[1] for p in coords.points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


def extract_pages(pdf_path, tables=()):
    """Return a list of (page_number, page_text) using unstructured's "fast" strategy.

    Each element keeps its own paragraph; elements are joined with blank lines
    so paragraph boundaries survive for the chunking step. Elements inside a
    table found by extract_tables() are skipped, since that table is stored
    separately in a cleaner row-by-row form.
    """
    elements = partition_pdf(filename=str(pdf_path), strategy="fast")

    table_boxes = defaultdict(list)
    for t in tables:
        table_boxes[t["page"]].append(t["bbox"])

    pages = defaultdict(list)
    for el in elements:
        page = el.metadata.page_number
        text = el.text.strip()
        if text and not _inside_any(el, table_boxes[page]):
            pages[page].append(text)

    return [(page, "\n\n".join(parts)) for page, parts in sorted(pages.items())]


def count_tokens(text):
    return len(ENCODER.encode(text))


def _split_long_paragraph(text):
    """Split a paragraph longer than MAX_CHUNK_TOKENS at sentence ends (or, failing that, by tokens)."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if count_tokens(sentence) <= config.MAX_CHUNK_TOKENS:
            pieces.append(sentence)
        else:  # one enormous "sentence" (e.g. a run-on list): hard-split by token count
            tokens = ENCODER.encode(sentence)
            for i in range(0, len(tokens), config.MAX_CHUNK_TOKENS):
                pieces.append(ENCODER.decode(tokens[i:i + config.MAX_CHUNK_TOKENS]))
    return pieces


def _make_chunk(text, page_start, page_end, kind):
    return {"text": text, "n_tokens": count_tokens(text),
            "page_start": page_start, "page_end": page_end, "type": kind}


def chunk_text(pages):
    """Greedily pack paragraphs into chunks of MIN..MAX tokens, never cutting a paragraph.

    Chunks may run across a page break; each records the first and last page it covers.
    """
    paragraphs = []  # (page, text) in reading order
    for page, text in pages:
        for para in text.split("\n\n"):
            if count_tokens(para) > config.MAX_CHUNK_TOKENS:
                paragraphs.extend((page, piece) for piece in _split_long_paragraph(para))
            else:
                paragraphs.append((page, para))

    chunks, current, current_pages = [], [], []
    for page, para in paragraphs:
        candidate = "\n\n".join(current + [para])
        if current and count_tokens(candidate) > config.MAX_CHUNK_TOKENS:
            chunks.append(_make_chunk("\n\n".join(current), current_pages[0], current_pages[-1], "text"))
            current, current_pages = [], []
        current.append(para)
        current_pages.append(page)

    if current:
        tail = _make_chunk("\n\n".join(current), current_pages[0], current_pages[-1], "text")
        # A short final chunk is folded into the previous one when it fits.
        if chunks and tail["n_tokens"] < config.MIN_CHUNK_TOKENS:
            merged = chunks[-1]["text"] + "\n\n" + tail["text"]
            if count_tokens(merged) <= config.MAX_CHUNK_TOKENS:
                chunks[-1] = _make_chunk(merged, chunks[-1]["page_start"], tail["page_end"], "text")
                return chunks
        chunks.append(tail)
    return chunks


def chunk_tables(tables):
    """One chunk per table; tables over MAX_CHUNK_TOKENS are split by rows, repeating the header."""
    chunks = []
    for t in tables:
        if count_tokens(t["text"]) <= config.MAX_CHUNK_TOKENS:
            chunks.append(_make_chunk(t["text"], t["page"], t["page"], "table"))
            continue

        lines = t["text"].split("\n")
        # Header = lines before the first multi-cell row; if no row has several cells, only the
        # "[Table on page N]" line is header (a table of single-cell rows used to crash here).
        first_row = next((i for i, line in enumerate(lines) if " | " in line), 1)
        header = "\n".join(lines[:first_row])  # title + column headers, repeated on every part
        rows = lines[first_row:]
        part = []
        for row in rows:
            if part and count_tokens("\n".join([header] + part + [row])) > config.MAX_CHUNK_TOKENS:
                chunks.append(_make_chunk("\n".join([header] + part), t["page"], t["page"], "table"))
                part = []
            part.append(row)
        chunks.append(_make_chunk("\n".join([header] + part), t["page"], t["page"], "table"))
    return chunks


def describe_tables(chunks):
    """Add an LLM-written "description" to every table chunk (contextual enrichment).

    A table is mostly numbers, so its raw embedding is a weak signal and search rarely finds
    it. A description naming the statement, periods and every row label (but no numbers)
    gives the embedding clear words to match. It is only used for the embedding; the
    answering LLM reads the raw table, so an error in a description can't reach an answer.
    """
    load_dotenv()
    client = OpenAI()
    tables = [c for c in chunks if c["type"] == "table"]

    def describe(chunk):
        response = client.chat.completions.create(
            model=config.CHAT_MODEL, temperature=0,
            messages=[{"role": "user", "content": TABLE_DESCRIPTION_PROMPT.format(table=chunk["text"])}],
        )
        chunk["description"] = response.choices[0].message.content.strip()

    with ThreadPoolExecutor(max_workers=8) as pool:  # 8 requests at a time instead of one by one
        list(pool.map(describe, tables))
    return chunks


def embedding_input(chunk):
    """Text that gets embedded: the description (if any) followed by the chunk text."""
    return f"{chunk['description']}\n\n{chunk['text']}" if "description" in chunk else chunk["text"]


def embed_chunks(chunks):
    """Add an "embedding" (list of floats) to every chunk, sending chunks to OpenAI in batches."""
    load_dotenv()  # reads OPENAI_API_KEY from .env; the OpenAI client picks it up automatically
    client = OpenAI()
    for i in range(0, len(chunks), config.EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + config.EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=config.EMBEDDING_MODEL,
                                            input=[embedding_input(c) for c in batch])
        for chunk, item in zip(batch, response.data):
            chunk["embedding"] = item.embedding
        print(f"  embedded {i + len(batch)}/{len(chunks)} chunks")
    return chunks


def store_chunks(chunks):
    """(Re)create the Chroma collection with cosine distance and add every chunk.

    Chroma record shape: text -> documents, embedding -> embeddings,
    n_tokens, page numbers and (for tables) the search description -> metadatas.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)  # rebuild from scratch so re-runs don't duplicate
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},  # default would be "l2"
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        # For Word files page_start/page_end hold section numbers, and "unit" says which it is.
        metadatas=[{"n_tokens": c["n_tokens"], "page_start": c["page_start"],
                    "page_end": c["page_end"], "type": c["type"], "source": c["source"], "unit": c["unit"],
                    **({"section_titles": c["section_titles"]} if c.get("section_titles") else {}),
                    **({"description": c["description"]} if "description" in c else {})}
                   for c in chunks],
    )
    return collection


def dump_pages(pages, tables, path, label=lambda n: f"page {n}"):
    """Write extracted text and tables to data/extracted/<file name>.txt for inspection.

    `label` names each part: "page 3" for PDFs, "section 3: Meals" for Word files.
    """
    tables_by_page = defaultdict(list)
    for t in tables:
        tables_by_page[t["page"]].append(t["text"])

    sections = []
    for page, text in pages:
        sections.append(f"=== {label(page)} ===\n{text}")
        sections.extend(tables_by_page[page])

    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EXTRACTED_DIR / f"{path.stem}.txt"
    out_path.write_text("\n\n".join(sections), encoding="utf-8")
    return out_path


class _TableHTMLParser(HTMLParser):
    """Collect the cell texts of an HTML table, row by row."""

    def __init__(self):
        super().__init__()
        self.rows, self._cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self.rows:
            self.rows[-1].append("".join(self._cell))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def extract_docx(path):
    """Split a Word file into numbered sections, one per heading, plus its tables.

    Word files have no fixed pages (unstructured reports no page numbers), so chunks are located
    by section instead. Returns (sections, tables, titles):
      sections: [(section number, text)], each starting with its heading
      tables:   [{"page": section number, "text": rows as "Label | value | ..."}]
      titles:   {section number: "Heading > Subheading"}
    Word stores real table structure, so rows come from the table's HTML, not from positions.
    """
    sections, tables, titles = [], [], {}
    current, number, heading_path = [], 0, []

    def start_section(title):
        nonlocal current, number
        if current:
            sections.append((number, "\n\n".join(current)))
        current, number = [], number + 1
        titles[number] = title

    for el in partition_docx(filename=str(path)):
        kind, text = type(el).__name__, el.text.strip()
        if kind == "Title" and text:
            depth = el.metadata.category_depth or 0
            heading_path = [(d, h) for d, h in heading_path if d < depth] + [(depth, text)]
            start_section(" > ".join(h for _, h in heading_path))
            current.append(text)
            continue
        if number == 0:  # content before the first heading
            start_section("(start of document)")
        if kind == "Table":
            parser = _TableHTMLParser()
            parser.feed(el.metadata.text_as_html or "")
            rows = [" | ".join(cells) for cells in map(_clean_row, parser.rows) if cells]
            context = current[-1:] if len(current) > 1 else []  # the paragraph introducing the table
            if rows:
                tables.append({"page": number,
                               "text": "\n".join([f"[Table in section {number}: {titles[number]}]"] + context + rows)})
        elif text:
            current.append(text)
    if current:
        sections.append((number, "\n\n".join(current)))
    return sections, tables, titles


# Page text quality checks. Measured on the files in data/: readable pages had a garbage ratio of 0.00,
# a browser "Print to PDF" file with broken fonts 0.72-0.99, and a scanned file had no text at all.
MIN_PAGE_CHARS = 50     # fewer extracted characters: the page is probably a scan/image (needs OCR)
MAX_GARBAGE_RATIO = 0.3  # more "(cid:N)" codes / control characters than this: the text can't be decoded
CID_RE = re.compile(r"\(cid:\d+\)")
PROBLEM_HELP = {
    "no text": "no text layer (scanned or image-only pages); needs OCR, which the fast strategy doesn't do",
    "unreadable": "text can't be decoded (fonts without a character map, common with a browser's "
                  "'Print to PDF'); use the original PDF from the source, or OCR",
}


def page_problem(text):
    """None if the page's extracted text looks usable, otherwise "no text" or "unreadable"."""
    if len(text.strip()) < MIN_PAGE_CHARS:
        return "no text"
    garbage = sum(map(len, CID_RE.findall(text))) + sum(ord(ch) < 32 and ch not in "\n\t" for ch in text)
    return "unreadable" if garbage / len(text) > MAX_GARBAGE_RATIO else None


SUPPORTED_SUFFIXES = (".pdf", ".docx")


def find_documents():
    """Every PDF and Word (.docx) file in data/, in name order. Other files are reported and ignored."""
    docs = []
    for path in sorted(p for p in config.DATA_DIR.iterdir() if p.is_file()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append(path)
        else:
            hint = "; open it in Word and save it as .docx" if path.suffix.lower() == ".doc" else ""
            print(f"{path.name}: ignored, unsupported file type (supported: .pdf, .docx){hint}")
    if not docs:
        sys.exit(f"No PDF or Word files found in {config.DATA_DIR}")
    for path in docs:
        if re.search(r"[\s(),;]", path.name):
            print(f"WARNING: rename '{path.name}' without spaces, commas, semicolons or parentheses; "
                  f"citations like ({path.name}, p. 3) can't be checked reliably otherwise")
    return docs


def _finish_chunks(chunks, path, unit):
    """Tag every chunk with its file, a unique id and its location unit, and print size statistics."""
    for i, chunk in enumerate(chunks):
        chunk["source"] = path.name
        chunk["id"] = f"{path.name}#{i:04d}"  # unique across files, e.g. "nvidia-1.pdf#0042"
        chunk["unit"] = unit                   # "page" (PDF) or "section" (Word)
    for kind in ("text", "table"):
        sizes = [c["n_tokens"] for c in chunks if c["type"] == kind]
        if sizes:
            in_range = sum(config.MIN_CHUNK_TOKENS <= n <= config.MAX_CHUNK_TOKENS for n in sizes)
            print(f"  {kind:>5} chunks: {len(sizes):>3} | tokens min={min(sizes)} avg={sum(sizes) // len(sizes)} "
                  f"max={max(sizes)} | in {config.MIN_CHUNK_TOKENS}-{config.MAX_CHUNK_TOKENS}: {in_range}")
    return chunks


def ingest_docx(path, dump=False):
    """Extract and chunk one Word file, by section."""
    sections, tables, titles = extract_docx(path)
    if not sections and not tables:
        print(f"{path.name}: SKIPPED, no text found")
        return []
    total_chars = sum(len(text) for _, text in sections)
    print(f"{path.name}: {len(titles)} sections, {total_chars:,} characters of text, {len(tables)} tables")
    if dump:
        out = dump_pages(sections, tables, path, label=lambda n: f"section {n}: {titles[n]}")
        print(f"  wrote extracted text to {out}")

    # Chunk each section on its own: unlike PDF page breaks, section breaks separate topics. Packed
    # across sections, a short policy became one chunk cited as "secs. 1-6". Small sections now make
    # chunks below MIN_CHUNK_TOKENS, a deliberate trade for precise citations like "sec. 2".
    chunks = [c for number, text in sections for c in chunk_text([(number, text)])] + chunk_tables(tables)
    for chunk in chunks:
        chunk["section_titles"] = "; ".join(titles[n] for n in range(chunk["page_start"], chunk["page_end"] + 1))
    return _finish_chunks(chunks, path, unit="section")


def ingest_pdf(pdf_path, dump=False):
    """Extract and chunk one PDF. Every chunk records its file in "source" and gets a unique id."""
    tables = extract_tables(pdf_path)
    pages = extract_pages(pdf_path, tables)
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)

    # Check every page (including pages that produced no elements at all) before using its text.
    # A page's tables count too: extract_pages() removed table text from the page text, so a page
    # that is almost all table would otherwise look empty (AMD's 10-Q p. 7 was wrongly skipped).
    page_text = defaultdict(str, pages)
    for t in tables:
        page_text[t["page"]] += "\n" + t["text"]
    problems = {p: page_problem(page_text[p]) for p in range(1, n_pages + 1)}
    bad = {p: why for p, why in problems.items() if why}
    if len(bad) > n_pages / 2:
        main_problem = max(set(bad.values()), key=list(bad.values()).count)
        print(f"{pdf_path.name}: SKIPPED, {len(bad)} of {n_pages} pages unusable: {PROBLEM_HELP[main_problem]}")
        return []
    if bad:
        for why in sorted(set(bad.values())):
            pages_list = ", ".join(str(p) for p, w in bad.items() if w == why)
            print(f"{pdf_path.name}: skipping page(s) {pages_list} ({why})")
        pages = [(p, text) for p, text in pages if p not in bad]
        tables = [t for t in tables if t["page"] not in bad]

    total_chars = sum(len(text) for _, text in pages)
    print(f"{pdf_path.name}: {len(pages)} pages, {total_chars:,} characters of text, {len(tables)} tables")
    if dump:
        print(f"  wrote extracted text to {dump_pages(pages, tables, pdf_path)}")

    chunks = chunk_text(pages) + chunk_tables(tables)
    return _finish_chunks(chunks, pdf_path, unit="page")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    chunks = []
    for path in find_documents():
        ingest = ingest_docx if path.suffix.lower() == ".docx" else ingest_pdf
        chunks += ingest(path, dump="--dump" in sys.argv)
    if not chunks:
        sys.exit("No usable text in any document; the existing collection was left unchanged.")
    print(f"total: {len(chunks)} chunks from {len({c['source'] for c in chunks})} file(s), "
          f"{sum(c['n_tokens'] for c in chunks):,} tokens")

    print(f"Describing {sum(c['type'] == 'table' for c in chunks)} table chunks with {config.CHAT_MODEL}...")
    describe_tables(chunks)

    print(f"Embedding with {config.EMBEDDING_MODEL}...")
    embed_chunks(chunks)

    collection = store_chunks(chunks)
    print(f"Stored {collection.count()} chunks in {config.CHROMA_DIR} "
          f"(collection '{collection.name}', distance: {collection.configuration['hnsw']['space']})")
