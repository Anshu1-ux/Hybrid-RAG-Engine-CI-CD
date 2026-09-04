# Hybrid RAG Engine with Automated Ragas Evaluation

A retrieval-augmented generation system that fuses dense (vector) and sparse
(BM25) retrieval via Reciprocal Rank Fusion, refines results with a
cross-encoder reranker, and is regression-tested on every commit by an
automated Ragas evaluation gate wired into CI/CD.

## Architecture

```
                 ┌─────────────┐
   query ──────► │   Dense     │ (ChromaDB, cosine sim, top-20)
                 │  Retrieval  │───┐
                 └─────────────┘   │
                                   ├──► Reciprocal Rank Fusion ──► top-15 ──► Cross-Encoder Rerank ──► top-5 ──► LLM ──► answer
                 ┌─────────────┐   │        (RRF, k=60)                     (ms-marco-MiniLM-L-6-v2)
   query ──────► │   Sparse    │───┘
                 │  Retrieval  │ (BM25, top-20)
                 └─────────────┘
```

**Why hybrid instead of just dense retrieval?** Dense embeddings are strong
on semantic similarity but weak on exact-term matches (error codes, rare
proper nouns, acronyms). BM25 is the reverse. Fusing both, then reranking
the fused set with a cross-encoder that scores the query and document
jointly, consistently outperforms any single method alone.

## Project layout

```
hybrid-rag-engine/
├── src/
│   ├── config.py      # all tunable parameters, with reasoning in comments
│   ├── ingest.py       # chunk -> embed -> dual index (ChromaDB + BM25)
│   ├── retriever.py    # dense + sparse search, RRF fusion, reranking
│   ├── generate.py     # grounded prompt construction + LLM call
│   └── api.py           # FastAPI service
├── eval/
│   ├── eval_dataset.json     # fixed QA pairs w/ ground truths
│   └── run_ragas_eval.py     # Ragas scoring + CI pass/fail gate
├── .github/workflows/eval.yml  # runs the eval gate on every push/PR
├── data/sample_docs/          # sample corpus to run this out of the box
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run it

```bash
# 1. Ingest the sample corpus (builds both the ChromaDB and BM25 indexes)
python -m src.ingest

# 2. Start the API
uvicorn src.api:app --reload --port 8000

# 3. Query it
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is hybrid retrieval and why is it better than dense alone?"}'
```

## Run the evaluation gate

```bash
python -m eval.run_ragas_eval
```

This scores the pipeline on **faithfulness**, **answer relevancy**,
**context precision**, and **context recall**, writes `eval/eval_report.json`,
and exits with a non-zero status if any metric misses the threshold set in
`src/config.py`. The same command runs automatically in
`.github/workflows/eval.yml` on every push and pull request to `main`, and
the report is uploaded as a build artifact — so retrieval quality is
regression-tested the same way unit tests are.

## Design decisions (interview-ready reasoning)

| Decision | Why |
|---|---|
| RRF over score-averaging for fusion | Dense (cosine) and sparse (BM25) scores live on incomparable scales. RRF only needs each list's *rank order*, sidestepping normalization entirely. |
| Cross-encoder reranks only the fused top-15, not the full corpus | Cross-encoders score query+doc jointly, which is accurate but too slow to run at corpus scale. Two-stage retrieve-then-rerank keeps latency bounded. |
| 800-char chunks, 120-char overlap | Balances context retention against embedding dilution; overlap prevents losing information at chunk boundaries. |
| Ragas as a CI gate, not just a one-off report | A quality score that isn't re-checked on every change isn't a quality bar — it's a screenshot. Wiring it into CI means a retrieval or prompt regression fails the build, not just the vibes. |
| `call_llm()` isolated in `generate.py` | Swapping providers (OpenAI → IBM watsonx.ai → local vLLM) touches one function, not the retrieval pipeline. |

## Extending this project

- Add a second corpus and compare metrics before/after to produce a real
  before/after number for your resume (e.g. "context precision improved
  from X to Y after adding reranking").
- Swap `call_llm()` in `src/generate.py` for IBM watsonx.ai to match the
  rest of your stack.
- Add `context_precision`/`context_recall` trend tracking across commits
  (store `eval_report.json` history) to demonstrate regression tracking
  over time, not just a single snapshot.

## Suggested resume bullet once you've run this against your own data

> Built a hybrid RAG retrieval engine (BM25 + dense fusion via RRF, cross-
> encoder reranking) with an automated Ragas evaluation harness gating CI/CD;
> improved context precision from X → Y and reduced hallucination rate by Z%
> versus a dense-only baseline.

Fill in X/Y/Z with your own measured numbers after running the eval script
against a baseline (dense-only) and the full hybrid pipeline — that
comparison is the most convincing evidence you can put in front of a
recruiter.
