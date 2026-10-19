import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import qa  
from index import Retriever  
from run_eval import load_questions  


def cites_truth(result, q):
    for s in result["sources"]:
        lo, hi = (int(x) for x in s["pages"].split("-"))
        if s["doc"].lower() == q["doc"] and lo <= q["hi"] and hi >= q["lo"]:
            return True
    return False


TRANSIENT = ("429", "rate limit", "rate_limit", "timeout", "timed out", "overloaded",
             "temporar", "connection", "500", "502", "503")

def is_transient(e):
    return any(t in f"{type(e).__name__} {e}".lower() for t in TRANSIENT)

def run_one(retriever, question, attempts=4, wait=15, min_score=None):
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            result = qa.answer(question, retriever, min_score=min_score)
            return {"result": result, "seconds": round(time.perf_counter() - t0, 2), "error": None}
        except Exception as e:  # noqa: BLE001
            if attempt == attempts or not is_transient(e):
                return {"result": None, "seconds": 0, "error": f"{type(e).__name__}: {e}"[:300]}
            time.sleep(wait * attempt)


def sweep_report(records):
    ok = [(q, r["result"]) for q, r in records if r["error"] is None]
    ans = [(q, r) for q, r in ok if q["answerable"]]
    unans = [(q, r) for q, r in ok if not q["answerable"]]
    if qa.SCORE_KEY == "rerank_score":
        grid = [float("-inf"), -6, -4, -3, -2, -1, 0, 0.3, 1, 2, 4]
    else:
        grid = [float("-inf")] + [round(0.5 + 0.05 * i, 2) for i in range(7)]

    def refused(r, t):
        return r["best_score"] < t or r["refused"] == "not_in_context"

    rows, stats = [], []
    for t in grid:
        u_ref = sum(refused(r, t) for _, r in unans)
        answered = [(q, r) for q, r in ans if not refused(r, t)]
        a_ref = len(ans) - len(answered)
        cited = sum(cites_truth(r, q) for q, r in answered)
        stats.append((t, u_ref, a_ref))
        label = "none (LLM check only)" if t == float("-inf") else t
        rows.append(f"| {label} | {u_ref}/{len(unans)} | {a_ref}/{len(ans)} ({a_ref / len(ans):.0%}) "
                    f"| {cited}/{len(answered)} |")
    out = [f"### Whole-system threshold sweep (score = {qa.SCORE_KEY}, mode = {qa.RETRIEVAL_MODE}, "
           f"model = {qa.LLM_MODEL})\n",
           "| threshold | unanswerable refused | answerable wrongly refused | correct citation (of answered) |",
           "|---|---|---|---|"] + rows
    safe = [t for t, u, _ in stats if u == len(unans)]
    if safe:
        t_safe = min(safe)
        a_ref = next(a for t, _, a in stats if t == t_safe)
        label = "no threshold at all" if t_safe == float("-inf") else f"threshold {t_safe}"
        out.append(f"\nLowest setting that still refuses every unanswerable question: **{label}** "
                   f"(wrongly refuses {a_ref}/{len(ans)} answerable).")
    else:
        out.append("\nNo setting refuses every unanswerable question.")
    leaks = [(q, r) for q, r in unans if not refused(r, float("-inf"))]
    if leaks:
        out.append("\nUnanswerable questions the LLM answered by itself (only the threshold stops these):")
        out += [f"- {q['question']} (best score {r['best_score']:.2f})\n  - answer: {r['answer'][:250]}"
                for q, r in leaks]
    return "\n".join(out)


def pct(n, d):
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.csv")
    ap.add_argument("--index", default="index")
    ap.add_argument("--cache", default=str(HERE / "e2e_cache.jsonl"))
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to pause after each LLM call")
    ap.add_argument("--retry_wait", type=float, default=15, help="base seconds to wait after a rate limit")
    ap.add_argument("--sweep", action="store_true", help="LLM pass without threshold, then report every threshold")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    ap.add_argument("--save", action="store_true", help="write eval/e2e_results.md")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    retriever = Retriever(args.index)
    cfg = (f"sweep|{qa.RETRIEVAL_MODE}|{qa.LLM_MODEL}|{qa.PROMPT_VERSION}" if args.sweep
           else f"{qa.RETRIEVAL_MODE}|{qa.MIN_SCORE}|{qa.LLM_MODEL}|{qa.PROMPT_VERSION}")  # a changed setting invalidates the cache

    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists() and not args.fresh:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["key"]] = d

    records, consecutive_errors = [], 0
    for i, q in enumerate(questions, 1):
        key = f"{q['question']}|{cfg}"
        if key in cache:
            rec = cache[key]
        else:
            rec = {"key": key, **run_one(retriever, q["question"], wait=args.retry_wait,
                                      min_score=float("-inf") if args.sweep else None)}
            if rec["error"] is None:  # never cache failures
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["result"] and rec["result"]["refused"] != "low_score":
                time.sleep(args.delay)  
        records.append((q, rec))
        status = "error" if rec["error"] else ("refused" if rec["result"]["refused"] else "answered")
        print(f"[{i}/{len(questions)}] {status}: {q['question'][:70]}")
        if rec["error"]:
            print(f"    -> {rec['error']}")
            consecutive_errors += 1
            if consecutive_errors >= 3:
                sys.exit("\nStopping: 3 errors in a row, so this is probably a setup problem, not bad luck.\n"
                         f"Last error: {rec['error']}\n"
                         "Fix it and run the same command again. Finished questions are cached.")
        else:
            consecutive_errors = 0

    if args.sweep:
        text = sweep_report(records)
        print("\n" + text)
        if args.save:
            Path("eval").mkdir(exist_ok=True)
            Path("eval/e2e_sweep.md").write_text(text + "\n", encoding="utf-8")
            print("\nSaved to eval/e2e_sweep.md")
        return

    ok = [(q, r["result"], r["seconds"]) for q, r in records if r["error"] is None]
    errors = [(q, r["error"]) for q, r in records if r["error"]]
    ans = [(q, r, s) for q, r, s in ok if q["answerable"]]
    unans = [(q, r, s) for q, r, s in ok if not q["answerable"]]

    u_ref = [x for x in unans if x[1]["refused"]]
    u_ref_score = sum(r["refused"] == "low_score" for _, r, _ in u_ref)
    u_bad = [x for x in unans if not x[1]["refused"]]
    a_ref = [x for x in ans if x[1]["refused"]]
    a_ref_score = sum(r["refused"] == "low_score" for _, r, _ in a_ref)
    a_ok = [x for x in ans if not x[1]["refused"]]
    a_cited = [x for x in a_ok if cites_truth(x[1], x[0])]
    a_nocite = [x for x in a_ok if not x[1]["sources"]]
    times = [s for _, _, s in ok if s]

    out = [f"### End-to-end results (mode={qa.RETRIEVAL_MODE}, MIN_SCORE={qa.MIN_SCORE}, model={qa.LLM_MODEL})\n",
           "| metric | result |", "|---|---|",
           f"| Unanswerable correctly refused | {pct(len(u_ref), len(unans))} |",
           f"| &nbsp;&nbsp;refused by score threshold | {u_ref_score} |",
           f"| &nbsp;&nbsp;refused by LLM (NOT_FOUND) | {len(u_ref) - u_ref_score} |",
           f"| Unanswerable wrongly answered | {pct(len(u_bad), len(unans))} |",
           f"| Answerable wrongly refused | {pct(len(a_ref), len(ans))} |",
           f"| &nbsp;&nbsp;refused by score threshold | {a_ref_score} |",
           f"| &nbsp;&nbsp;refused by LLM (NOT_FOUND) | {len(a_ref) - a_ref_score} |",
           f"| Answerable answered | {pct(len(a_ok), len(ans))} |",
           f"| &nbsp;&nbsp;cited the correct page | {pct(len(a_cited), len(a_ok))} |",
           f"| &nbsp;&nbsp;answered with no citation | {len(a_nocite)} |"]
    if times:
        out.append(f"| Latency per question | mean {statistics.mean(times):.1f}s, "
                   f"median {statistics.median(times):.1f}s |")
    if errors:
        out.append(f"| Failed (API errors, excluded) | {len(errors)} |")

    if u_bad:
        out.append("\nUnanswerable questions that got an answer (check these first, they are hallucination risks):")
        out += [f"- {q['question']}\n  - answer: {r['answer'][:300]}" for q, r, _ in u_bad]
    if a_ref:
        out.append("\nAnswerable questions that were refused:")
        out += [f"- {q['question']} (reason: {r['refused']}, best score {r['best_score']:.2f})" for q, r, _ in a_ref]
    wrong_cite = [x for x in a_ok if x not in a_cited]
    if wrong_cite:
        out.append("\nAnswered, but the cited sources do not include the correct page:")
        out += [f"- {q['question']} (cited: {[(s['doc'][:12], s['pages']) for s in r['sources']]}, "
                f"expected page {q['lo']}-{q['hi']})" for q, r, _ in wrong_cite]
    if errors:
        out.append("\nErrors:")
        out += [f"- {q['question']}: {e}" for q, e in errors]

    text = "\n".join(out)
    print("\n" + text)
    if args.save:
        Path("eval").mkdir(exist_ok=True)
        Path("eval/e2e_results.md").write_text(text + "\n", encoding="utf-8")
        print("\nSaved to eval/e2e_results.md")


if __name__ == "__main__":
    main()