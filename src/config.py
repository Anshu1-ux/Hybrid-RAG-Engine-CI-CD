"""
Central configuration for the Hybrid RAG Engine.

Everything that a recruiter/interviewer would ask "why did you pick this
value" about lives here, in one place, with a comment explaining the choice.
"""

import os
from pathlib import Path

# Disable ChromaDB's anonymous telemetry -- avoids noisy (harmless) error
# logs on some chromadb/posthog version combinations.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
CHROMA_PERSIST_DIR = ROOT_DIR / "storage" / "chroma"
BM25_INDEX_PATH = ROOT_DIR / "storage" / "bm25_index.pkl"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# 800 chars balances context retention against embedding dilution (see
# README "Design Decisions" section for the reasoning + eval numbers).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120  # ~15% overlap to avoid losing context at chunk boundaries

# ---------------------------------------------------------------------------
# Embeddings (dense retrieval)
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dim, fast, strong quality/latency tradeoff for a
# portfolio-scale corpus. Swap to a larger model (e.g. bge-base-en-v1.5) if
# corpus size and latency budget allow.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "hybrid_rag_docs"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DENSE_TOP_K = 20      # candidates pulled from vector search
SPARSE_TOP_K = 20     # candidates pulled from BM25
FUSION_TOP_K = 15     # candidates kept after RRF fusion, before reranking
RRF_K = 60            # standard RRF damping constant (Cormack et al. 2009)
FINAL_TOP_K = 5        # final chunks passed to the LLM after reranking

# ---------------------------------------------------------------------------
# Reranking (cross-encoder, second stage)
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# LLM_PROVIDER switches which backend call_llm() in src/generate.py uses:
#   "openai" -> OpenAI API (needs OPENAI_API_KEY + billing credit)
#   "ollama" -> free, local model via Ollama (https://ollama.com), no API key
# Set OPENAI_API_KEY in your environment / .env file if using "openai". To
# swap to IBM watsonx.ai (matches the rest of your resume stack), add a
# third branch in src/generate.py's `call_llm` function -- nothing else in
# the pipeline needs to change, which is itself worth mentioning in an
# interview as a "provider-agnostic generation layer" design choice.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = 0.1  # low temperature: we want grounded, non-creative answers

# ---------------------------------------------------------------------------
# Evaluation thresholds (used as CI gates in .github/workflows/eval.yml)
# ---------------------------------------------------------------------------
MIN_FAITHFULNESS = 0.75
MIN_ANSWER_RELEVANCY = 0.70
MIN_CONTEXT_PRECISION = 0.65
MIN_CONTEXT_RECALL = 0.65
