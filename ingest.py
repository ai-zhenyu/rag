"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> ChromaDB.

Run:  python ingest.py
      python ingest.py --dump   # also save the extracted text to data/extracted/
"""
import sys
from collections import defaultdict

import re

import pdfplumber
import tiktoken
from unstructured.partition.pdf import partition_pdf

import config

ENCODER = tiktoken.get_encoding(config.TOKEN_ENCODING)

# How far above a table (in PDF points, 72 = 1 inch) to look for its column headers.
# Headers like "Three Months Ended / Jul 28, 2024" sit above the table's first ruling line.
TABLE_HEADER_BAND = 60


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
            for table in page.find_tables():
                x0, top, x1, bottom = table.bbox
                header = page.crop((0, max(0, top - TABLE_HEADER_BAND), page.width, top)).extract_text()
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
        first_row = next(i for i, line in enumerate(lines) if " | " in line)
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


def dump_pages(pages, tables, pdf_path):
    """Write extracted text and tables to data/extracted/<pdf name>.txt for inspection."""
    tables_by_page = defaultdict(list)
    for t in tables:
        tables_by_page[t["page"]].append(t["text"])

    sections = []
    for page, text in pages:
        sections.append(f"=== page {page} ===\n{text}")
        sections.extend(tables_by_page[page])

    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EXTRACTED_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text("\n\n".join(sections), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    tables = extract_tables(config.PDF_PATH)
    pages = extract_pages(config.PDF_PATH, tables)
    total_chars = sum(len(text) for _, text in pages)
    print(f"Extracted {len(pages)} pages, {total_chars:,} characters of text, "
          f"and {len(tables)} tables from {config.PDF_PATH.name}")

    if "--dump" in sys.argv:
        print(f"Wrote extracted text to {dump_pages(pages, tables, config.PDF_PATH)}")

    chunks = chunk_text(pages) + chunk_tables(tables)
    for kind in ("text", "table"):
        sizes = [c["n_tokens"] for c in chunks if c["type"] == kind]
        in_range = sum(config.MIN_CHUNK_TOKENS <= n <= config.MAX_CHUNK_TOKENS for n in sizes)
        print(f"{kind:>5} chunks: {len(sizes):>3} | tokens min={min(sizes)} avg={sum(sizes) // len(sizes)} "
              f"max={max(sizes)} | in {config.MIN_CHUNK_TOKENS}-{config.MAX_CHUNK_TOKENS}: {in_range}")
    print(f"total chunks: {len(chunks)}, total tokens: {sum(c['n_tokens'] for c in chunks):,}")
