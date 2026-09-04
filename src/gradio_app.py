"""
Gradio UI for the Hybrid RAG Engine.

Run:
    python -m src.gradio_app

Opens a local web UI (usually http://127.0.0.1:7860) where you can ask
questions against the ingested corpus and see the answer plus which source
documents were used.
"""

import logging

import gradio as gr

from src.retriever import HybridRetriever
from src.generate import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Loaded once at startup, not per-request -- avoids reloading the embedding
# model and cross-encoder on every question.
logger.info("Loading HybridRetriever (embedding model + cross-encoder)...")
retriever = HybridRetriever()
logger.info("Ready.")


def answer_question(question: str, top_k: int):
    if not question or not question.strip():
        return "Please enter a question.", ""

    chunks = retriever.retrieve(question, final_top_k=int(top_k))
    if not chunks:
        return "No relevant context found for this query.", ""

    answer = generate_answer(question, chunks)

    sources_md = "\n\n".join(
        f"**[{i+1}] {c.source}**  (relevance score: {c.score:.3f})\n\n> {c.text[:300]}{'...' if len(c.text) > 300 else ''}"
        for i, c in enumerate(chunks)
    )
    return answer, sources_md


with gr.Blocks(title="Hybrid RAG Engine") as demo:
    gr.Markdown(
        """
        # Hybrid RAG Engine
        Dense (ChromaDB) + Sparse (BM25) retrieval, fused with Reciprocal Rank Fusion,
        refined with a cross-encoder reranker, answered by an LLM grounded in the
        retrieved context.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is hybrid retrieval and why is it better than dense alone?",
                lines=2,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1, label="Chunks to retrieve (top_k)"
            )
            ask_button = gr.Button("Ask", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
            sources_box = gr.Markdown(label="Retrieved sources")

    ask_button.click(
        fn=answer_question,
        inputs=[question_box, top_k_slider],
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        fn=answer_question,
        inputs=[question_box, top_k_slider],
        outputs=[answer_box, sources_box],
    )

    gr.Examples(
        examples=[
            ["What is hybrid retrieval and why is it better than dense alone?", 5],
            ["What does a cross-encoder reranker do?", 5],
            ["What chunk size should I use for document chunking?", 5],
        ],
        inputs=[question_box, top_k_slider],
    )


if __name__ == "__main__":
    demo.launch()
