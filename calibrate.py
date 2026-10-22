"""Calibrate SIMILARITY_THRESHOLD, KEYWORD_SCORE_THRESHOLD and QUESTION_SIMILARITY_THRESHOLD from real scores.

For answerable questions we know a snippet of the correct answer, so we can find the scores of
the chunk that actually contains it. For unanswerable questions every retrieved chunk is
irrelevant, so their best scores show how high "noise" can reach. A good threshold keeps the
answer chunks and drops the noise.

The thresholds are chosen on the CALIBRATION questions, then checked on VALIDATION questions
that were not used to choose them (mostly paraphrased away from the document's wording), so a
threshold that only fits the first set would show up as failures on the second.

Run:  python calibrate.py
"""
import sys

import config
from retrieval import bm25_scores, embed_question, hybrid_search, question_search, search

# (question, text that appears only in a chunk containing the answer)
CALIBRATION_ANSWERABLE = [
    ("What was NVIDIA's revenue for the quarter ended July 30, 2023?", "13,507"),
    ("What was NVIDIA's total revenue in the second quarter of fiscal 2025?", "30,040"),
    ("What was net income per diluted share?", "0.67"),
    ("How much did NVIDIA spend on research and development?", "3,090"),
    ("What was Data Center revenue in the six months ended July 28, 2024?", "48,835"),
    ("How many shares did NVIDIA repurchase in the second quarter?", "62.8 million"),
    ("What new licensing requirements did the US government impose on exports to China?", "licensing requirements"),
    ("What was NVIDIA's gross margin in the second quarter?", "75.1%"),
    ("What stock split did NVIDIA announce?", "ten-for-one"),
    ("When is the Blackwell production ramp scheduled?", "Blackwell production ramp"),
    ("How many NVIDIA employees are in Israel?", "4,000 employees"),
    ("How much cash, cash equivalents and marketable securities did NVIDIA have?", "34,800"),
    ("What was the breakdown of NVIDIA inventories?", "Raw materials |"),
    ("What dividend per share did NVIDIA pay?", "$0.01 per"),
]

CALIBRATION_UNANSWERABLE = [
    "What is a good recipe for chocolate chip cookies?",               # off topic
    "Who won the 2022 FIFA World Cup?",                                # off topic
    "What was Apple's iPhone revenue in 2024?",                        # other company
    "How much does Tesla's Full Self-Driving package cost?",           # other company, related field
    "What was NVIDIA's revenue in fiscal year 2027?",                  # on topic, period not in document
    "What is Jensen Huang's favorite food?",                           # on topic person, fact not in document
    "How many employees does NVIDIA have in Brazil?",                  # on topic, fact not in document
    "What was NVIDIA's closing stock price on December 31, 2025?",     # on topic, after the filing
]

# Not used to choose thresholds.
VALIDATION_ANSWERABLE = [
    ("How much stock was sitting in NVIDIA's warehouses at the end of the quarter?", "Raw materials |"),
    ("How big was NVIDIA's quarterly payout to shareholders per share?", "$0.01 per"),
    ("What did NVIDIA earn after taxes last quarter?", "16,599"),
    ("How much cash did NVIDIA's operations generate in the first half of the year?", "29,833"),
    ("Which upcoming GPU architecture needed a fix to improve production yield?", "Blackwell GPU mask"),
    ("What share of revenue came from NVIDIA's biggest direct customer?", "Customer A |"),
]

VALIDATION_UNANSWERABLE = [
    "What is the capital of Australia?",                               # off topic
    "What was AMD's data center revenue last quarter?",                # other company, shares keywords
    "How many GPUs did NVIDIA ship to Tesla?",                         # on topic, fact not in document
    "What was NVIDIA's revenue in the third quarter of fiscal 2025?",  # on topic, period not in document
]

K = 10


def answer_scores(question, needle):
    """Scores of the best chunk containing the answer, or None if it is not in the top K / hybrid set."""
    candidates = search(question, k=K, threshold=None) + hybrid_search(question, apply_thresholds=False)
    hits = [c for c in candidates if needle in c["text"]]
    if not hits:
        return None
    best = max(hits, key=lambda c: c["similarity"])
    qsim = max((c.get("question_similarity") or 0) for c in hits)
    return best["similarity"], bm25_scores(question)[best["id"]], qsim


def noise_scores(question):
    """Best similarity, BM25 score and generated-question similarity of any chunk
    (all irrelevant for unanswerable questions)."""
    qhits = question_search(embed_question(question), k=1)
    return (search(question, k=1, threshold=None)[0]["similarity"], max(bm25_scores(question).values()),
            qhits[0]["question_similarity"] if qhits else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("CALIBRATION, answerable ('answer' = best chunk containing the known answer)")
    answers = []
    for question, needle in CALIBRATION_ANSWERABLE:
        scores = answer_scores(question, needle)
        if scores:
            answers.append(scores)
            print(f"  answer similarity={scores[0]:.3f}  bm25={scores[1]:5.1f}  qsim={scores[2]:.3f}   {question}")
        else:
            print(f"  answer NOT RETRIEVED by vector top {K} or hybrid search    {question}")

    print("\nCALIBRATION, unanswerable (best chunk, irrelevant by construction)")
    noise = []
    for question in CALIBRATION_UNANSWERABLE:
        scores = noise_scores(question)
        noise.append(scores)
        print(f"  best similarity={scores[0]:.3f}  best bm25={scores[1]:5.1f}  best qsim={scores[2]:.3f}   {question}")

    print(f"\nSimilarity threshold {config.SIMILARITY_THRESHOLD} combined with each keyword threshold:")
    print("  (a chunk is kept if similarity >= similarity threshold OR bm25 >= keyword threshold)")
    print("  keyword threshold | answer chunks kept | unanswerable questions with a kept chunk")
    for kt in [4, 6, 7, 8, 9, 10, 12, 15]:
        kept = sum(s >= config.SIMILARITY_THRESHOLD or b >= kt for s, b, _ in answers)
        leaked = sum(s >= config.SIMILARITY_THRESHOLD or b >= kt for s, b, _ in noise)
        mark = "  <-- current" if kt == config.KEYWORD_SCORE_THRESHOLD else ""
        print(f"        {kt:>4}         |      {kept:>2}/{len(answers)}          |   {leaked}/{len(noise)}{mark}")

    print(f"\nWith similarity {config.SIMILARITY_THRESHOLD} and keyword {config.KEYWORD_SCORE_THRESHOLD}, "
          f"each generated-question threshold:")
    print("  (a chunk is also kept if one of its generated questions reaches the question threshold)")
    print("  question threshold | answer chunks kept | unanswerable questions with a kept chunk")
    for qt in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        def passes(s, b, q):
            return s >= config.SIMILARITY_THRESHOLD or b >= config.KEYWORD_SCORE_THRESHOLD or q >= qt
        kept = sum(passes(*a) for a in answers)
        leaked = sum(passes(*n) for n in noise)
        mark = "  <-- current" if qt == config.QUESTION_SIMILARITY_THRESHOLD else ""
        print(f"        {qt:.2f}        |      {kept:>2}/{len(answers)}          |   {leaked}/{len(noise)}{mark}")

    print(f"\nVALIDATION with current thresholds (similarity {config.SIMILARITY_THRESHOLD}, "
          f"bm25 {config.KEYWORD_SCORE_THRESHOLD}, question similarity {config.QUESTION_SIMILARITY_THRESHOLD})")
    for question, needle in VALIDATION_ANSWERABLE:
        kept = hybrid_search(question)
        ok = any(needle in c["text"] for c in kept)
        print(f"  {'PASS' if ok else 'FAIL'}  answer chunk {'kept' if ok else 'missing'}         {question}")
    for question in VALIDATION_UNANSWERABLE:
        kept = hybrid_search(question)
        print(f"  {'PASS' if not kept else 'note'}  {len(kept)} chunks kept "
              f"{'(blocked)' if not kept else '(prompt must refuse)'}  {question}")
