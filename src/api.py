"""
FastAPI service exposing the hybrid RAG pipeline.

Run:
    uvicorn src.api:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/query \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What is hybrid retrieval?"}'
"""

import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.retriever import HybridRetriever
from src.generate import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid RAG Engine", version="1.0.0")

# Loaded once at startup, not per-request -- avoids reloading the embedding
# model and cross-encoder on every call, which would dominate latency.
_retriever: HybridRetriever | None = None


@app.on_event("startup")
def load_retriever():
    global _retriever
    _retriever = HybridRetriever()
    logger.info("HybridRetriever loaded and ready.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized yet.")

    start = time.perf_counter()
    chunks = _retriever.retrieve(request.question, final_top_k=request.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found for this query.")

    answer = generate_answer(request.question, chunks)
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=sorted({c.source for c in chunks}),
        latency_ms=round(latency_ms, 2),
    )
