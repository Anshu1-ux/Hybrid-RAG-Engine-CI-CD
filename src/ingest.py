"""
Ingestion pipeline: load raw docs -> chunk -> embed -> index.

Builds TWO indexes over the same chunk set:
  1. A dense vector index in ChromaDB (semantic search)
  2. A sparse BM25 index, pickled to disk (exact keyword search)

Run directly:
    python -m src.ingest
"""

import pickle
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_documents(data_dir: Path) -> list[dict]:
    """Load raw .txt documents from disk into (doc_id, text) pairs."""
    docs = []
    for file_path in sorted(data_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        docs.append({"doc_id": file_path.stem, "text": text, "source": file_path.name})
    logger.info(f"Loaded {len(docs)} source documents from {data_dir}")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks, preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                    "text": piece,
                    "source": doc["source"],
                }
            )
    logger.info(f"Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_dense_index(chunks: list[dict]) -> None:
    """Embed chunks and persist them into a local ChromaDB collection."""
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    # Recreate the collection each ingest run so re-running ingest.py is idempotent.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    logger.info(f"Indexed {len(chunks)} chunks into ChromaDB collection '{config.COLLECTION_NAME}'")


def build_sparse_index(chunks: list[dict]) -> None:
    """Build and pickle a BM25 index over the same chunk set."""
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    config.BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    logger.info(f"Saved BM25 index to {config.BM25_INDEX_PATH}")


def run_ingestion() -> None:
    docs = load_documents(config.DATA_DIR)
    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in {config.DATA_DIR}. Add source documents before ingesting."
        )
    chunks = chunk_documents(docs)
    build_dense_index(chunks)
    build_sparse_index(chunks)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
