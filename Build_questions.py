import argparse
import csv
import re
import sys
from pathlib import Path

import pymupdf


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="/workspaces/RAG_Project/question_draft.csv")
    ap.add_argument("--out", default="/workspaces/RAG_Project/questions.csv")
    ap.add_argument("--pdf_dir", default="data/raw")
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    default_doc = pdfs[0].name if len(pdfs) == 1 else None
    cache = {}

    def pages_of(doc):
        if doc not in cache:
            with pymupdf.open(Path(args.pdf_dir) / doc) as d:
                cache[doc] = [norm(p.get_text()) for p in d]
        return cache[doc]

    rows, errors = [], []
    with open(args.draft, encoding="utf-8-sig", newline="") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):  # n = line number in the file
            q = r["question"].strip()
            if not q:
                continue
            if r["answerable"].strip().lower() not in ("yes", "y", "true", "1"):
                rows.append({"question": q, "doc": "", "page": "", "answerable": "no"})
                continue
            doc = (r.get("doc") or "").strip() or default_doc
            if not doc:
                errors.append(f"line {n}: several PDFs found, add a 'doc' column. {q}")
                continue
            ev = norm(r["evidence"])
            if not ev:
                errors.append(f"line {n}: answerable question has no evidence. {q}")
                continue
            hits = [i + 1 for i, text in enumerate(pages_of(doc)) if ev in text]
            if len(hits) != 1:
                why = "not found" if not hits else f"found on pages {hits}"
                errors.append(f"line {n}: evidence {why}: '{r['evidence']}'")
                continue
            rows.append({"question": q, "doc": doc, "page": hits[0], "answerable": "yes"})

    if errors:
        print(f"{len(errors)} problem(s), nothing written:\n")
        print("\n".join(errors))
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "doc", "page", "answerable"])
        w.writeheader()
        w.writerows(rows)
    yes = sum(r["answerable"] == "yes" for r in rows)
    print(f"Wrote {len(rows)} questions ({yes} answerable, {len(rows) - yes} unanswerable) to {args.out}")


if __name__ == "__main__":
    main()

