"""
Generation layer: takes retrieved chunks + a user query, builds a grounded
prompt, and calls an LLM.

Deliberately isolated behind `call_llm()` so the retrieval pipeline never
needs to know or care which model provider is behind it. To swap to IBM
watsonx.ai (matching the rest of your resume's tooling), you only need to
rewrite `call_llm` -- nothing in retriever.py or ingest.py changes. That
separation of concerns is itself worth a sentence in an interview.
"""

import os
import logging

import requests
from openai import OpenAI

from src import config
from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise technical assistant. Answer the user's \
question using ONLY the provided context. If the context does not contain \
enough information to answer, say so explicitly rather than guessing. \
Cite which source(s) you used in brackets, e.g. [source: rag_systems.txt]."""


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[source: {c.source}]\n{c.text}" for c in chunks
    )
    return f"""Context:
{context_block}

Question: {query}

Answer the question using only the context above."""


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it or add it to a .env file before running generation."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Free, local generation via Ollama (https://ollama.com). No API key,
    no cost -- runs entirely on-device. Requires the Ollama app to be
    running and the model already pulled (`ollama pull <model>`)."""
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURE},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise EnvironmentError(
            "Could not reach Ollama at "
            f"{config.OLLAMA_BASE_URL}. Is the Ollama app running? "
            "Download it from https://ollama.com and run "
            f"'ollama pull {config.OLLAMA_MODEL_NAME}' first."
        ) from e

    return response.json()["message"]["content"]


def call_llm(prompt: str) -> str:
    """Routes to the configured provider (LLM_PROVIDER in src/config.py).
    Swap in watsonx.ai or another provider by adding a new branch here --
    the rest of the pipeline (retrieval, prompt construction) is unaffected."""
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'openai' or 'ollama'.")


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    prompt = build_prompt(query, chunks)
    answer = call_llm(prompt)
    logger.info(f"Generated answer for query='{query[:60]}...' ({len(chunks)} chunks used)")
    return answer
