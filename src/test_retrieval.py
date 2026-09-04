"""
Standalone retrieval smoke test -- exercises dense search, sparse search,
RRF fusion, and cross-encoder reranking WITHOUT calling any LLM. Useful for
confirming the retrieval half of the pipeline works before spending any
OpenAI credits on generation.

Run:
    python -m src.test_retrieval "your question here"

Or with no argument, it runs a few built-in sample questions.
"""

import sys
from src.retriever import HybridRetriever

SAMPLE_QUESTIONS = [
    "What is hybrid retrieval and why is it better than dense alone?",
    "What does a cross-encoder reranker do?",
    "What chunk size should I use for document chunking?",
]


def print_results(query: str, retriever: HybridRetriever):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    chunks = retriever.retrieve(query)
    if not chunks:
        print("  (no chunks retrieved)")
        return

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text.replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"\n  [{i}] score={chunk.score:.4f}  source={chunk.source}")
        print(f"      {preview}")


def main():
    retriever = HybridRetriever()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print_results(query, retriever)
    else:
        print("No question given -- running built-in sample questions.\n")
        for q in SAMPLE_QUESTIONS:
            print_results(q, retriever)


if __name__ == "__main__":
    main()
