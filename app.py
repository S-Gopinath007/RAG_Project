import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pymupdf

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "src"):  # works whether the scripts sit in ./src or next to app.py
    if p.is_dir():
        sys.path.insert(0, str(p))

import index  # noqa: E402
import ingest  # noqa: E402
import qa  # noqa: E402

MAX_PAGES = 150  # keeps indexing of an uploaded PDF to a few seconds on a small CPU
INDEX_DIR = HERE / "index"
DEFAULT = index.Retriever(str(INDEX_DIR)) if (INDEX_DIR / "faiss.index").exists() else None

EXAMPLES = [  # only shown when a built-in document is loaded
    "What is the minimum number of clinical isolates recommended per group of organisms?",
    "Which licence form is needed for permission to run a clinical performance evaluation?",
    "How is the minimum inhibitory concentration defined?",
    "What is the price of the NG-Test CARBA 5 kit?",  # not in the document: should be refused
]


def default_status():
    if DEFAULT is None:
        return "No document loaded yet. Upload a PDF to begin."
    docs = ", ".join(sorted({c["doc"] for c in DEFAULT.chunks}))
    return f"Using the built-in document: **{docs}**"


def doc_names(retriever):
    return ", ".join(sorted({c["doc"] for c in retriever.chunks}))


def load_pdf(path):
    if not path:  # upload cleared: go back to the built-in document
        return None, default_status()
    name = Path(path).name
    fallback = " Questions will use the built-in document instead." if DEFAULT is not None else ""
    try:
        with pymupdf.open(path) as doc:
            n_pages = len(doc)
        if n_pages > MAX_PAGES:
            return None, f"⚠️ **{name}** has {n_pages} pages; the demo accepts up to {MAX_PAGES}.{fallback}"
        chunks = ingest.chunk_pdf(path)
        if not chunks:
            return None, (f"⚠️ I couldn't find any text in **{name}**. Scanned documents need OCR first."
                          f"{fallback}")
        work = Path(tempfile.mkdtemp())
        chunks_file = work / "chunks.jsonl"
        chunks_file.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
        index.build(str(chunks_file), str(work / "index"))
        retriever = index.Retriever(str(work / "index"))
    except Exception as e:  # noqa: BLE001 - corrupt/encrypted PDF, out of memory...
        return None, f"⚠️ Could not process **{name}** ({type(e).__name__}: {str(e)[:120]}).{fallback}"
    return retriever, f"✅ Ready: **{name}** ({n_pages} pages, {len(chunks)} passages)"


def start_loading(path):
    msg = "⏳ Reading and indexing your PDF. This can take up to a minute; Ask unlocks when it's ready." \
        if path else default_status()
    return msg, gr.update(interactive=False), gr.update(interactive=False)


def finish_loading():
    return gr.update(interactive=True), gr.update(interactive=True)


def format_source(s):
    text = " ".join(s.get("text", "").split())  # older qa.py versions don't return the passage text
    if len(text) > 700:
        text = text[:700].rsplit(" ", 1)[0] + " …"
    return f"**[{s['n']}] {s['doc']}, pages {s['pages']}** · {s['section']}\n\n> {text}"


def ask(question, retriever):
    retriever = retriever or DEFAULT
    if retriever is None:
        raise gr.Error("Upload a PDF first.")
    question = (question or "").strip()
    if not question:
        raise gr.Error("Type a question first.")
    try:
        result = qa.answer(question, retriever)
    except RuntimeError as e:  # setup problems: missing key, reply cut off...
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001 - usually a rate limit or network problem
        raise gr.Error(f"The language model call failed ({type(e).__name__}). Wait a minute and try again.")

    searched = f"Searched in: {doc_names(retriever)}"  # makes it obvious which document answered
    if result["refused"]:
        return (f"🔍 **{result['answer']}**\n\nThe document may not cover this, or try rephrasing the question."
                f"\n\n<sub>{searched}</sub>", "")
    sources = "\n\n".join(format_source(s) for s in result["sources"]) or "_The answer cited no sources._"
    return f"{result['answer']}\n\n<sub>{searched} · {qa.DISCLAIMER}</sub>", sources


with gr.Blocks(title="Document Q&A with citations") as demo:
    gr.Markdown(
        "# 📄 Document Q&A with citations\n"
        "Answers come **only** from the document, and every claim cites its source passage. "
        "If the document doesn't say, the assistant tells you instead of guessing."
    )
    retriever_state = gr.State(None)  # the visitor's own index, when they upload a PDF
    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload your own PDF (optional)", file_types=[".pdf"], type="filepath")
            status = gr.Markdown(default_status())
            gr.Markdown(
                f"<sub>Text-based PDFs up to {MAX_PAGES} pages. The passages used to answer are sent to "
                "Groq's API, so don't upload confidential documents.</sub>"
            )
        with gr.Column(scale=2):
            question = gr.Textbox(label="Your question", lines=2,
                                  placeholder="e.g. What is the minimum sample size recommended?")
            ask_btn = gr.Button("Ask", variant="primary")
            gr.Markdown("### Answer")
            answer = gr.Markdown()
            with gr.Accordion("Sources", open=True):
                sources = gr.Markdown()
                gr.Markdown("<sub>Page numbers are PDF page numbers, which can differ from the "
                            "numbers printed on the pages.</sub>")
    if DEFAULT is not None:
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=question, label="Try an example")

    upload.change(start_loading, upload, [status, ask_btn, question], queue=False) \
        .then(load_pdf, upload, [retriever_state, status], show_progress="hidden") \
        .then(finish_loading, None, [ask_btn, question])
    ask_btn.click(ask, [question, retriever_state], [answer, sources])
    question.submit(ask, [question, retriever_state], [answer, sources])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=os.getenv("GRADIO_SHARE") == "1")