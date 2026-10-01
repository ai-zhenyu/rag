"""Answering: send the question plus retrieved chunks to an OpenAI chat model.

Run:  python answer.py "What was NVIDIA's revenue for the quarter ended July 30, 2023?"
"""
import sys

from openai import OpenAI

import config
from retrieval import format_pages, hybrid_search

NOT_FOUND = "The document doesn't contain the answer to this question."

SYSTEM_PROMPT = f"""You answer questions about a PDF document using ONLY the excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge, and do not guess.
- Cite the page for every fact, e.g. (p. 26) or (pp. 26-27), copying it from the label of the
  excerpt the fact came from.
- Tables are written one row per line as "Label | value | value | ...". The values follow the column
  headers above them in the same left-to-right order. If a table says "(In millions)", its amounts
  are in millions; say so in the answer.
- If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}
- Be concise: answer the question directly, then add brief supporting detail if useful."""

_openai = OpenAI()  # retrieval.py has already loaded OPENAI_API_KEY from .env


def build_context(chunks):
    """Label each chunk with its pages, so the model can cite them.

    The label deliberately contains no other number: with "[Source 4 | p. 3]" the model
    cited "p. 4", mixing up the source number with the page.
    """
    return "\n\n".join(f"=== Excerpt from {format_pages(c)} ({c['type']}) ===\n{c['text']}" for c in chunks)


def answer(question, chunks=None):
    """Answer the question from the document only, citing page numbers.

    `chunks` can be passed in when they were already retrieved (evaluate.py does this to
    print them); otherwise hybrid_search() is called. If no chunk passes the thresholds,
    the model isn't called at all.
    """
    if chunks is None:
        chunks = hybrid_search(question)
    if not chunks:
        return NOT_FOUND

    response = _openai.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,  # most likely wording every time: consistent answers for evaluation
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console can't print some PDF symbols otherwise

    question = sys.argv[1] if len(sys.argv) > 1 else "What was NVIDIA's revenue for the quarter ended July 30, 2023?"
    print(f"Q: {question}\nA: {answer(question)}")
