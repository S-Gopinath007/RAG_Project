"""Step 2: build the search index and test hybrid retrieval.

Build:   python src/index.py
Search:  python src/index.py --query "How often should HbA1c be checked?"

Dense search (embeddings + FAISS) finds meaning; BM25 finds exact terms
(drug names, abbreviations like HbA1c). Results are merged with
Reciprocal Rank Fusion (RRF).
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

MODEL = "BAAI/bge-small-en-v1.5"  # ~130 MB, runs on CPU via ONNX
MODES = ("dense", "bm25", "hybrid", "rerank")
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~90 MB cross-encoder


_MODELS = {}


def get_embedder():
    """One shared embedding model per process (~130 MB), however many indexes are loaded."""
    if "embedder" not in _MODELS:
        _MODELS["embedder"] = TextEmbedding(MODEL)
    return _MODELS["embedder"]


def get_reranker():
    """One shared cross-encoder per process (~90 MB), loaded on first use."""
    if "reranker" not in _MODELS:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _MODELS["reranker"] = TextCrossEncoder(RERANK_MODEL)
    return _MODELS["reranker"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(chunks_path="data/chunks.jsonl", out_dir="index", batch=32):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    chunks = load_chunks(chunks_path)
    # prepend doc + section so the embedding knows the context
    texts = [f"{c['doc']} | {c['section']}\n{c['text']}" for c in chunks]

    model = get_embedder()
    parts = []
    for i in range(0, len(texts), batch):  # small batches keep RAM low
        parts.append(np.array(list(model.embed(texts[i:i + batch])), dtype="float32"))
        print(f"embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    vecs = np.vstack(parts)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product = cosine after normalising
    index.add(vecs)
    faiss.write_index(index, str(out / "faiss.index"))

    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(out / "bm25_tokens.pkl", "wb") as f:
        pickle.dump([tokenize(t) for t in texts], f)
    print(f"\nIndexed {len(chunks)} chunks -> {out}/")


class Retriever:
    def __init__(self, out_dir="index"):
        out = Path(out_dir)
        self.index = faiss.read_index(str(out / "faiss.index"))
        self.chunks = json.load(open(out / "chunks.json", encoding="utf-8"))
        self.bm25 = BM25Okapi(pickle.load(open(out / "bm25_tokens.pkl", "rb")))
        self.model = get_embedder()

    def search(self, query, k=5, pool=20, mode="hybrid"):
        """Return top-k chunks. mode: 'dense', 'bm25', 'hybrid' or 'rerank'
        (hybrid top-`pool` candidates re-scored by a cross-encoder)."""
        if mode not in MODES:  # never fail silently with an empty result
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if mode == "rerank":
            cands = self.search(query, k=pool, pool=pool, mode="hybrid")
            scores = list(get_reranker().rerank(query, [f"{c['section']}\n{c['text']}" for c in cands]))
            ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)[:k]
            return [{**c, "rerank_score": round(float(s), 3)} for s, c in ranked]

        rankings = []
        q = np.array(list(self.model.query_embed(query)), dtype="float32")
        faiss.normalize_L2(q)
        if mode in ("dense", "hybrid"):
            _, ids = self.index.search(q, pool)
            rankings.append([int(i) for i in ids[0] if i >= 0])
        if mode in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(tokenize(query))
            rankings.append([int(i) for i in np.argsort(scores)[::-1][:pool]])

        fused = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)  # RRF
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        # cosine similarity (0-1) is comparable across queries, unlike the RRF score,
        # so the guardrail in qa.py uses it to decide whether to refuse
        return [{**self.chunks[i], "score": round(fused[i], 4),
                 "similarity": round(float(q[0] @ self.index.reconstruct(i)), 3)}
                for i in top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--out", default="index")
    ap.add_argument("--query")
    ap.add_argument("--mode", default="hybrid", choices=list(MODES))
    args = ap.parse_args()

    if args.query:
        for r in Retriever(args.out).search(args.query, mode=args.mode):
            print(f"[{r['doc']} p.{r['page_start']}-{r['page_end']} | {r['section']}] {r['score']}")
            print("  " + r["text"][:200].replace("\n", " ") + "...\n")
    else:
        build(args.chunks, args.out)