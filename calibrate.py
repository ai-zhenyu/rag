"""Calibrate SIMILARITY_THRESHOLD from real scores instead of guessing.

For answerable questions we know a snippet of the correct answer, so we can find the
similarity of the chunk that actually contains it. For unanswerable questions every
retrieved chunk is irrelevant, so their best score shows how high "noise" can reach.
A good threshold keeps the answer chunks and drops the noise.

Run:  python calibrate.py
"""
import sys

import config
from retrieval import search

# (question, text that appears only in a chunk containing the answer)
ANSWERABLE = [
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

UNANSWERABLE = [
    "What is a good recipe for chocolate chip cookies?",               # off topic
    "Who won the 2022 FIFA World Cup?",                                # off topic
    "What was Apple's iPhone revenue in 2024?",                        # other company
    "How much does Tesla's Full Self-Driving package cost?",           # other company, related field
    "What was NVIDIA's revenue in fiscal year 2027?",                  # on topic, period not in document
    "What is Jensen Huang's favorite food?",                           # on topic person, fact not in document
    "How many employees does NVIDIA have in Brazil?",                  # on topic, fact not in document
    "What was NVIDIA's closing stock price on December 31, 2025?",     # on topic, after the filing
]

K = 10

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"ANSWERABLE (top {K} retrieved; 'answer' = best chunk that contains the known answer)")
    answer_sims = []
    for question, needle in ANSWERABLE:
        results = search(question, k=K, threshold=None)
        top1 = results[0]["similarity"]
        hit = next(((rank, c) for rank, c in enumerate(results, 1) if needle in c["text"]), None)
        if hit:
            rank, chunk = hit
            answer_sims.append(chunk["similarity"])
            print(f"  top1={top1:.3f}  answer={chunk['similarity']:.3f} at rank {rank:<2}  {question}")
        else:
            print(f"  top1={top1:.3f}  answer=NOT IN TOP {K}            {question}")

    print(f"\nUNANSWERABLE (top1 = the most similar chunk, which is irrelevant by construction)")
    noise_sims = []
    for question in UNANSWERABLE:
        top1 = search(question, k=1, threshold=None)[0]["similarity"]
        noise_sims.append(top1)
        print(f"  top1={top1:.3f}  {question}")

    print(f"\nanswer-chunk similarity: min={min(answer_sims):.3f}  max={max(answer_sims):.3f}  "
          f"(found {len(answer_sims)}/{len(ANSWERABLE)})")
    print(f"unanswerable top1:       min={min(noise_sims):.3f}  max={max(noise_sims):.3f}")

    print("\nthreshold | answer chunks kept | unanswerable questions that still retrieve something")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        kept = sum(s >= t for s in answer_sims)
        leaked = sum(s >= t for s in noise_sims)
        mark = "  <-- current" if abs(t - config.SIMILARITY_THRESHOLD) < 1e-9 else ""
        print(f"   {t:.2f}   |      {kept:>2}/{len(answer_sims)}          |   {leaked}/{len(noise_sims)}{mark}")
