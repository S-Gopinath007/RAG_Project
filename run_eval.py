"""Step 4: benchmark retrieval and calibrate the refusal threshold. No LLM or API key needed.

Usage:
    python eval/run_eval.py --questions eval/questions.csv
    python eval/run_eval.py --questions eval/questions.csv --index index_500 --save

questions.csv columns:
    question    the question text
    doc         PDF file name that contains the answer (blank if unanswerable)
    page        page number, or a range like 12-13 (blank if unanswerable)
    answerable  yes / no

Reports:
  1. Retrieval quality for dense, bm25 and hybrid: hit-rate@1/3/5 and MRR@5
  2. Refusal guardrail: for each MIN_SIM, how many unanswerable questions are refused
     and how many answerable ones are wrongly refused
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from index import Retriever  # noqa: E402

K = 5
MODES = ["dense", "bm25", "hybrid", "rerank"]
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(8)]  # 0.30 ... 0.65


def load_questions(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r["question"].strip():
                continue
            ans = r["answerable"].strip().lower() in ("yes", "y", "true", "1")
            lo = hi = None
            if ans:
                parts = r["page"].replace("–", "-").split("-")
                lo, hi = int(parts[0]), int(parts[-1])
            rows.append({"question": r["question"].strip(), "doc": r["doc"].strip().lower(),
                         "lo": lo, "hi": hi, "answerable": ans})
    return rows


def hit(chunk, q):
    """A chunk is correct if it is in the right document and its pages overlap the answer pages."""
    return (chunk["doc"].lower() == q["doc"]
            and chunk["page_start"] <= q["hi"] and chunk["page_end"] >= q["lo"])


def retrieval_metrics(retriever, questions, mode):
    ranks = []
    for q in questions:
        results = retriever.search(q["question"], k=K, mode=mode)
        ranks.append(next((i for i, c in enumerate(results, 1) if hit(c, q)), None))
    n = len(ranks)
    return {
        "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "MRR@5": sum(1 / r for r in ranks if r) / n,
    }, ranks


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="/workspaces/RAG_Project/questions.csv")
    ap.add_argument("--index", default="index")
    ap.add_argument("--mode", default="hybrid", choices=MODES, help="retrieval mode used for the refusal sweep")
    ap.add_argument("--save", action="store_true", help="write eval/results.md")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    print(f"{len(questions)} questions: {len(answerable)} answerable, {len(unanswerable)} unanswerable\n")
    retriever = Retriever(args.index)
    out = []

    # 1. retrieval quality
    if answerable:
        rows = []
        for mode in MODES:
            m, ranks = retrieval_metrics(retriever, answerable, mode)
            rows.append([mode] + [f"{m[k]:.0%}" if k != "MRR@5" else f"{m[k]:.2f}"
                                  for k in ("hit@1", "hit@3", "hit@5", "MRR@5")])
            if mode == "hybrid":  # list misses so you can inspect chunking or wording problems
                misses = [q["question"] for q, r in zip(answerable, ranks) if r is None]
        out.append(f"### Retrieval ({len(answerable)} answerable questions)\n")
        out.append(md_table(["mode", "hit@1", "hit@3", "hit@5", "MRR@5"], rows))
        if misses:
            out.append("\nHybrid misses (not in top 5):\n" + "\n".join(f"- {m}" for m in misses))

    # 2. refusal threshold sweep (use the same mode as qa.py: --mode)
    if answerable and unanswerable:
        key = "rerank_score" if args.mode == "rerank" else "similarity"
        best = lambda qs: [max((c[key] for c in retriever.search(q["question"], k=K, mode=args.mode)),
                               default=0.0) for q in qs]
        sa, su = best(answerable), best(unanswerable)
        # AUC = chance that a random answerable question scores higher than a random
        # unanswerable one. 0.5 means the score cannot separate them at all; 1.0 is perfect.
        auc = sum((a > u) + 0.5 * (a == u) for a in sa for u in su) / (len(sa) * len(su))
        lo, hi = min(sa + su), max(sa + su)
        grid = [round(lo + (hi - lo) * i / 10, 3) for i in range(11)]  # spans the observed scores
        rows, scored = [], []
        for t in grid:
            refused_unans = sum(s < t for s in su) / len(su)
            wrongly_refused = sum(s < t for s in sa) / len(sa)
            scored.append(((refused_unans + (1 - wrongly_refused)) / 2, t))
            rows.append([t, f"{refused_unans:.0%}", f"{wrongly_refused:.0%}"])
        best_t = max(scored, key=lambda x: x[0])[1]
        out.append(f"\n### Refusal threshold (score = {key}, mode = {args.mode})\n")
        out.append(md_table(["threshold", "unanswerable refused", "answerable wrongly refused"], rows))
        out.append(f"\nSeparation AUC: **{auc:.2f}** (0.5 = useless, 1.0 = perfect). "
                   f"Mean best score: answerable {sum(sa)/len(sa):.2f}, unanswerable {sum(su)/len(su):.2f}. "
                   f"Best balanced threshold: **{best_t}**")
    else:
        out.append("\nNeed both answerable and unanswerable questions for the threshold sweep.")

    text = "\n".join(out)
    print(text)
    if args.save:
        Path("eval").mkdir(exist_ok=True)
        Path("eval/results.md").write_text(text + "\n", encoding="utf-8")
        print("\nSaved to eval/results.md")


if __name__ == "__main__":
    main()