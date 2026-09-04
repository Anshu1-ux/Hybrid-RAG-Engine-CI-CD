"""
Hybrid retriever: combines dense (ChromaDB) and sparse (BM25) search via
Reciprocal Rank Fusion, then refines the fused candidate set with a
cross-encoder reranker.

This is the piece that separates a "tutorial RAG" from a production-grade
retrieval layer, and is the highest-leverage thing to be able to explain in
an interview:

  1. Dense search catches semantic matches BM25 misses ("car" ~ "automobile").
  2. Sparse (BM25) search catches exact-term matches dense embeddings miss
     (error codes, rare proper nouns, acronyms).
  3. RRF fusion merges the two ranked lists without needing to normalize or
     compare raw similarity scores across completely different scales.
  4. The cross-encoder reranker looks at the (query, chunk) pair JOINTLY,
     which is strictly more accurate than bi-encoder similarity, but too
     slow to run over an entire corpus -- so it only reranks the fused
     top-N candidates, not everything.
"""

import pickle
import logging
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

from src import config

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float  # final reranked relevance score


class HybridRetriever:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self._collection = self._client.get_collection(
            name=config.COLLECTION_NAME, embedding_function=self._embed_fn
        )

        with open(config.BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self._bm25 = bm25_data["bm25"]
        self._bm25_chunks = bm25_data["chunks"]  # ordered list matching bm25's internal indices

        self._cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)

    # -- Stage 1a: dense retrieval -----------------------------------------
    def _dense_search(self, query: str, top_k: int) -> list[str]:
        """Return chunk_ids ranked by dense similarity."""
        results = self._collection.query(query_texts=[query], n_results=top_k)
        return results["ids"][0]

    # -- Stage 1b: sparse retrieval ------------------------------------------
    def _sparse_search(self, query: str, top_k: int) -> list[str]:
        """Return chunk_ids ranked by BM25 score."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._bm25_chunks[i]["chunk_id"] for i in ranked_indices]

    # -- Stage 2: Reciprocal Rank Fusion -------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[str]], k: int = config.RRF_K
    ) -> list[str]:
        """
        Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list).
        Chosen over naive score-averaging because dense (cosine) and sparse
        (BM25) scores live on incomparable scales -- RRF only needs RANK
        position, sidestepping that normalization problem entirely.
        """
        fused_scores: dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list):
                fused_scores.setdefault(chunk_id, 0.0)
                fused_scores[chunk_id] += 1.0 / (k + rank + 1)

        return sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    # -- Stage 3: cross-encoder reranking ------------------------------------
    def _rerank(self, query: str, chunk_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        id_to_chunk = {c["chunk_id"]: c for c in self._bm25_chunks}
        candidates = [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], source=c["source"], score=float(s))
            for c, s in scored[:top_k]
        ]

    # -- Public entrypoint ----------------------------------------------------
    def retrieve(self, query: str, final_top_k: int = config.FINAL_TOP_K) -> list[RetrievedChunk]:
        dense_ids = self._dense_search(query, config.DENSE_TOP_K)
        sparse_ids = self._sparse_search(query, config.SPARSE_TOP_K)

        fused_ids = self._reciprocal_rank_fusion([dense_ids, sparse_ids])[: config.FUSION_TOP_K]

        reranked = self._rerank(query, fused_ids, top_k=final_top_k)

        logger.info(
            f"Query='{query[:60]}...' | dense={len(dense_ids)} sparse={len(sparse_ids)} "
            f"fused={len(fused_ids)} final={len(reranked)}"
        )
        return reranked
