import argparse
import collections
import json
import re
import statistics
from pathlib import Path

import pymupdf

def read_lines(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    size = max(s["size"] for s in spans)
                    bold = any(s["flags"] & 16 for s in spans)
                    yield page_no, text, size, bold


def drop_boilerplate(lines):
    pages_of = collections.defaultdict(set)
    for page_no, text, _, _ in lines:
        pages_of[text].add(page_no)
    n_pages = len({page_no for page_no, *_ in lines})
    threshold = max(5, int(0.15 * n_pages))
    repeated = {t for t, pages in pages_of.items() if len(pages) >= threshold}

    kept = []
    for line in lines:
        text = line[1]
        if text in repeated:
            continue
        if re.fullmatch(r"\d{1,3}([-–]\d{1,3})?", text):  # '12' or '19-27'
            continue
        if re.search(r"…{2,}|\.{5,}", text):  
            continue
        kept.append(line)
    return kept

HEADING_START = re.compile(r"^(\d+(\.\d+)*[.)]?\s|Step\s\d|Phase\s\d|Chapter\s\d|Annexure|Table\s\d|Fig)", re.I)

def is_heading(text, size, bold, body_size):
    if len(text) > 100 or text.endswith((".", ",", ";", ":")):
        return False
    if size >= body_size * 1.5 or (size >= body_size * 1.15 and bold):
        return True
    return bold and len(text) < 80 and (
        bool(HEADING_START.match(text)) or text.isupper() or len(text.split()) >= 3
    )

def chunk_pdf(pdf_path, chunk_chars=1000, overlap=150):
    lines = drop_boilerplate(list(read_lines(pdf_path)))
    if not lines:
        return []
    
    body_size = statistics.median_high([round(size, 1) for _, t, size, _ in lines for _ in t])

    chunks, section = [], "Introduction"
    buf, start_page, last_page = "", None, None

    def flush(keep_overlap=True):
        nonlocal buf, start_page
        text = buf.strip()
        if len(text) >= 50:  
            chunks.append({
                "doc": Path(pdf_path).name,
                "section": section,
                "page_start": start_page,
                "page_end": last_page,
                "text": text,
            })
        buf = text[-overlap:] if (keep_overlap and text) else ""
        start_page = last_page if buf else None

    for page_no, text, size, bold in lines:
        if is_heading(text, size, bold, body_size):
            flush(keep_overlap=False)  
            section = text
            continue
        if start_page is None:
            start_page = page_no
        last_page = page_no
        buf += text + " "
        if len(buf) >= chunk_chars:
            flush()
    flush(keep_overlap=False)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="data/raw")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--chunk_chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for pdf in pdfs:
            chunks = chunk_pdf(pdf, args.chunk_chars, args.overlap)
            for i, c in enumerate(chunks):
                c["id"] = f"{pdf.stem}-{i:04d}"
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{pdf.name}: {len(chunks)} chunks")
    print(f"Done. {total} chunks written to {args.out}")

if __name__ == "__main__":
    main()