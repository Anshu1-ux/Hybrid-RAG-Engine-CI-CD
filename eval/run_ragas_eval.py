"""
Automated evaluation harness using Ragas.

Runs the FULL pipeline (retrieval + generation) against a fixed eval
dataset, scores it on four dimensions, writes a report, and exits non-zero
if any metric falls below the threshold defined in src/config.py.

This is the script the GitHub Action calls -- it's what turns "I built a
RAG system" into "I built a RAG system with a regression-tested quality
bar," which is the single biggest credibility upgrade you can put on a
resume bullet for this kind of project.

Run:
    python -m eval.run_ragas_eval
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src import config
from src.retriever import HybridRetriever
from src.generate import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def load_eval_cases() -> list[dict]:
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def run_pipeline_on_eval_set(retriever: HybridRetriever, cases: list[dict]) -> dict:
    """Run retrieval + generation for every eval question, in the shape Ragas expects."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for case in cases:
        chunks = retriever.retrieve(case["question"])
        answer = generate_answer(case["question"], chunks)

        questions.append(case["question"])
        answers.append(answer)
        contexts.append([c.text for c in chunks])
        ground_truths.append(case["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main() -> int:
    cases = load_eval_cases()
    logger.info(f"Loaded {len(cases)} eval cases from {EVAL_DATASET_PATH}")

    retriever = HybridRetriever()
    eval_data = run_pipeline_on_eval_set(retriever, cases)
    dataset = Dataset.from_dict(eval_data)

    logger.info("Running Ragas evaluation (faithfulness, answer_relevancy, context_precision, context_recall)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "scores": scores,
        "thresholds": {
            "faithfulness": config.MIN_FAITHFULNESS,
            "answer_relevancy": config.MIN_ANSWER_RELEVANCY,
            "context_precision": config.MIN_CONTEXT_PRECISION,
            "context_recall": config.MIN_CONTEXT_RECALL,
        },
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to {REPORT_PATH}")

    # -- CI gate: fail the build if any metric misses its threshold --------
    failures = []
    checks = [
        ("faithfulness", config.MIN_FAITHFULNESS),
        ("answer_relevancy", config.MIN_ANSWER_RELEVANCY),
        ("context_precision", config.MIN_CONTEXT_PRECISION),
        ("context_recall", config.MIN_CONTEXT_RECALL),
    ]
    for metric_name, threshold in checks:
        value = scores.get(metric_name)
        status = "PASS" if value is not None and value >= threshold else "FAIL"
        logger.info(f"{metric_name:20s} = {value:.3f}  (threshold {threshold})  [{status}]")
        if status == "FAIL":
            failures.append(metric_name)

    if failures:
        logger.error(f"Eval gate FAILED on: {', '.join(failures)}")
        return 1

    logger.info("All eval metrics passed threshold. Eval gate PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
