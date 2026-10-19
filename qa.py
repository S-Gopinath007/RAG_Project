"""Step 3: answer questions with citations and refuse when the documents don't support an answer.

Setup:   set GROQ_API_KEY (free key from console.groq.com)
Usage:   python src/qa.py "What is the first-line drug for type 2 diabetes?"

Two guardrails:
  1. Retrieval guardrail: if the best chunk's cosine similarity is below MIN_SCORE,
     refuse without calling the LLM (saves cost, prevents guessing).
  2. Generation guardrail: the LLM must reply NOT_FOUND if the context doesn't
     contain the answer, and must cite sources as [1], [2]...
"""
import os
import re
import sys

from index import Retriever
from dotenv import load_dotenv

load_dotenv()
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "rerank")  # best mode in the benchmark
SCORE_KEY = "rerank_score" if RETRIEVAL_MODE == "rerank" else "similarity"
# The whole-system sweep (run_e2e.py --sweep) showed the LLM's NOT_FOUND check already refuses every
# unanswerable benchmark question, and any stricter score threshold only refuses more VALID questions.
# So this is just a loose floor (below every answerable score seen) that skips the LLM for hopeless matches.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.65 if SCORE_KEY == "similarity" else -8.0))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 800))  # reasoning models need room to think
PROMPT_VERSION = "v2"  # bump when the prompt or answer parsing changes; invalidates the e2e cache
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
REFUSAL = "I couldn't find this in the provided documents."
DISCLAIMER = "This is information from guideline documents, not medical advice."

SYSTEM_PROMPT = """You answer questions about medical guideline documents.
Rules:
- Use ONLY the numbered context passages below. Never use outside knowledge.
- Cite every claim with its passage number like [1] or [2][3].
- Do not invent doses, numbers or recommendations. Quote numbers exactly as written.
- The passages may contain tables flattened into plain text. Read them carefully before deciding the answer is missing.
- If the passages really do not contain the answer, reply with exactly: NOT_FOUND
- Be concise."""


def call_llm(messages):
    try:
        from groq import Groq  # imported here so retrieval can be tested without a key
    except ImportError:
        raise RuntimeError("The groq package is missing. Run: pip install groq") from None
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError('GROQ_API_KEY is not set. Run: export GROQ_API_KEY="your_key" '
                           "(it must be set in the same terminal where you run the script)")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS
    )
    return resp.choices[0].message.content.strip()


def clean_llm_output(text):
    """Drop <think>...</think> reasoning that some models print before the answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:  # opened but never closed: the reply was cut off mid-reasoning
        raise RuntimeError("LLM output was cut off while reasoning. Raise LLM_MAX_TOKENS "
                           "or use a non-reasoning model (GROQ_MODEL).")
    cleaned = cleaned.strip()
    if not cleaned:
        raise RuntimeError("The LLM returned an empty answer.")
    return cleaned


def build_context(chunks):
    return "\n\n".join(
        f"[{i}] ({c['doc']}, p.{c['page_start']}, {c['section']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def answer(question, retriever, k=5, min_score=None):
    threshold = MIN_SCORE if min_score is None else min_score  # override used by the e2e sweep
    chunks = retriever.search(question, k=k, mode=RETRIEVAL_MODE)
    best = max((c[SCORE_KEY] for c in chunks), default=0.0)

    # Guardrail 1: weak retrieval -> refuse, no LLM call
    if not chunks or best < threshold:
        return {"answer": REFUSAL, "refused": "low_score", "best_score": best, "sources": []}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
    ]
    raw = call_llm(messages)
    text = clean_llm_output(raw)

    # Guardrail 2: the model says the context doesn't contain the answer
    if re.match(r"\W*NOT_FOUND\b", text):  # must BE the answer, not just mention the word
        return {"answer": REFUSAL, "refused": "not_in_context", "best_score": best, "sources": [], "raw": raw}

    # keep only the sources the answer actually cites
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(chunks)})
    sources = [
        {"n": n, "doc": chunks[n - 1]["doc"], "pages": f"{chunks[n - 1]['page_start']}-{chunks[n - 1]['page_end']}",
         "section": chunks[n - 1]["section"], "text": chunks[n - 1]["text"]}
        for n in cited
    ]
    return {"answer": text, "refused": None, "best_score": best, "sources": sources, "raw": raw}


def format_result(r):
    out = [r["answer"]]
    for s in r["sources"]:
        out.append(f"  [{s['n']}] {s['doc']}, pages {s['pages']} - {s['section']}")
    if not r["refused"]:
        out.append(f"\n{DISCLAIMER}")
    return "\n".join(out)


def debug(question, retriever, evidence=""):
    """Show what was retrieved, the scores, and the raw LLM output for one question.

    evidence = a phrase from the passage that answers the question (optional). It tells you
    whether the answer chunk reached the top 5: retrieval problem vs LLM problem.
    """
    norm = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())
    chunks = retriever.search(question, k=5, mode=RETRIEVAL_MODE)
    print(f"mode={RETRIEVAL_MODE}  score={SCORE_KEY}  threshold={MIN_SCORE}\n")
    found = False
    for n, c in enumerate(chunks, 1):
        has = bool(evidence) and norm(evidence) in norm(c["text"])
        found = found or has
        print(f"[{n}] {SCORE_KEY}={c[SCORE_KEY]:.2f}  p.{c['page_start']}-{c['page_end']}  "
              f"{c['section'][:50]}" + ("   <-- contains the evidence" if has else ""))
        print("    " + c["text"][:150].replace("\n", " ") + "...")
    best = max((c[SCORE_KEY] for c in chunks), default=0.0)
    print(f"\nbest {SCORE_KEY} = {best:.2f} -> "
          + ("passes the threshold" if best >= MIN_SCORE else "would be REFUSED by the threshold"))
    if evidence:
        print("answer chunk in the top 5: " + ("YES" if found else "NO  -> retrieval problem"))

    result = answer(question, retriever, min_score=float("-inf"))  # ignore the threshold to see the LLM
    print("\n--- raw LLM output ---\n" + str(result.get("raw")))
    print("\n--- final answer ---\n" + format_result(result))
    if evidence and found and result["refused"]:
        print("\nDIAGNOSIS: the answer was in the context but the model said NOT_FOUND "
              "-> model or prompt problem (try another GROQ_MODEL).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ask a question about the indexed documents.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--debug", action="store_true",
                    help="show retrieved chunks, scores and the raw LLM output")
    ap.add_argument("--evidence", default="",
                    help="with --debug: a phrase from the answer, to check the answer chunk was retrieved")
    args = ap.parse_args()
    question = " ".join(args.question)
    if args.debug:
        debug(question, Retriever(), args.evidence)
    else:
        print(format_result(answer(question, Retriever())))