"""
RAGAS-based evaluation harness.

RAGAS metrics used
------------------
* **Faithfulness**        — What fraction of claims in the answer are
                            supported by the retrieved contexts?
                            (LLM-judge; no ground-truth needed)

* **AnswerRelevancy**     — How well does the answer address the question?
                            (Embedding-similarity of generated question ↔ original)

* **ContextPrecision**    — Are the retrieved chunks ranked with the most
                            useful ones first?
                            (LLM-judge; needs ground-truth answer)

* **ContextRecall**       — How many claims in the ground-truth answer are
                            covered by the retrieved context?
                            (LLM-judge; needs ground-truth answer)

Dataset format (JSON)
---------------------
[
  {
    "question":     "What is RAG?",
    "ground_truth": "RAG stands for Retrieval Augmented Generation …"
  },
  …
]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import EvalConfig, LLMConfig, RAGConfig
from src.rag_pipeline import RAGPipeline, RAGResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_eval_dataset(path: str | Path) -> List[Dict]:
    """
    Load and validate the evaluation Q&A dataset from a JSON file.

    Expected schema per item::

        {
            "question":     "<natural-language question>",
            "ground_truth": "<reference answer>"
        }

    Parameters
    ----------
    path:
        Path to the JSON evaluation dataset.

    Returns
    -------
    List[Dict]
        Validated list of evaluation items.

    Raises
    ------
    ValueError
        If any item is missing required fields.
    FileNotFoundError
        If the dataset file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}\n"
            "Create data/eval/eval_dataset.json or specify --eval-dataset."
        )

    data: List[Dict] = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {"question", "ground_truth"}

    for i, item in enumerate(data):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"Eval item {i} is missing required field(s): {missing}"
            )

    logger.info("Loaded %d evaluation item(s) from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# RAG execution over evaluation set
# ---------------------------------------------------------------------------


def run_rag_on_dataset(
    pipeline: RAGPipeline,
    eval_data: List[Dict],
) -> List[Dict]:
    """
    Run *pipeline* on every question in *eval_data*.

    Collects the four fields RAGAS requires:
    ``question``, ``answer``, ``contexts``, ``ground_truth``.

    Parameters
    ----------
    pipeline:
        A built :class:`~src.rag_pipeline.RAGPipeline`.
    eval_data:
        Evaluation items from :func:`load_eval_dataset`.

    Returns
    -------
    List[Dict]
        One record per evaluation item, ready for :func:`evaluate_with_ragas`.
    """
    records = []
    n = len(eval_data)

    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        logger.info("RAG query %d/%d: %.60s …", i, n, question)

        result: RAGResult = pipeline.query(question)

        records.append(
            {
                "question": result.question,
                "answer": result.answer,
                "contexts": result.contexts,
                "ground_truth": item["ground_truth"],
            }
        )

    return records


# ---------------------------------------------------------------------------
# RAGAS LLM / embeddings setup
# ---------------------------------------------------------------------------


def _get_ragas_llm(config: LLMConfig):
    """Build a RAGAS-wrapped LLM from the pipeline's LLM config."""
    from ragas.llms import LangchainLLMWrapper

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=config.model_name,
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider for RAGAS: {config.provider}")

    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings(config: RAGConfig):
    """Build RAGAS-wrapped embeddings for AnswerRelevancy."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if config.embedding.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(
            model=config.embedding.model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        # Fall back to a small OpenAI model for RAGAS embeddings if available,
        # otherwise use HuggingFace (slower but works without an API key).
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------


def evaluate_with_ragas(
    records: List[Dict],
    config: RAGConfig,
    metrics: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute RAGAS metrics on *records* and return a per-question DataFrame
    plus aggregate statistics.

    Parameters
    ----------
    records:
        Output of :func:`run_rag_on_dataset`.
    config:
        Full pipeline config (used to build the RAGAS judge LLM/embeddings).
    metrics:
        Which RAGAS metrics to compute.  Defaults to
        ``config.eval.metrics``.

    Returns
    -------
    df: pd.DataFrame
        Per-question scores for each metric.
    aggregates: Dict[str, float]
        Mean, std, min, max per metric (keys like ``faithfulness_mean``).
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    if metrics is None:
        metrics = list(config.eval.metrics)

    # Build judge LLM and embeddings
    ragas_llm = _get_ragas_llm(config.llm)
    ragas_emb = _get_ragas_embeddings(config)

    # Metric registry — each metric is configured with the judge LLM/embeddings
    metric_registry = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }

    active_metrics = []
    for name in metrics:
        if name in metric_registry:
            active_metrics.append(metric_registry[name])
        else:
            logger.warning("Unknown RAGAS metric '%s' — skipping.", name)

    # Build RAGAS evaluation dataset
    samples = [
        SingleTurnSample(
            user_input=rec["question"],
            response=rec["answer"],
            retrieved_contexts=rec["contexts"],
            reference=rec["ground_truth"],
        )
        for rec in records
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info(
        "Running RAGAS evaluation on %d sample(s) with metrics: %s",
        len(samples),
        [type(m).__name__ for m in active_metrics],
    )

    result = ragas_evaluate(dataset=dataset, metrics=active_metrics)
    df: pd.DataFrame = result.to_pandas()

    # Compute aggregate statistics per metric column
    metric_cols = [c for c in df.columns if c in metric_registry]
    aggregates: Dict[str, float] = {}
    for col in metric_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        aggregates[f"{col}_mean"] = float(numeric.mean())
        aggregates[f"{col}_std"] = float(numeric.std())
        aggregates[f"{col}_min"] = float(numeric.min())
        aggregates[f"{col}_max"] = float(numeric.max())

    return df, aggregates


# ---------------------------------------------------------------------------
# Persistence and reporting
# ---------------------------------------------------------------------------


def save_evaluation_results(
    df: pd.DataFrame,
    aggregates: Dict[str, float],
    config: EvalConfig,
    experiment_name: str = "eval",
) -> Path:
    """
    Save per-question results to CSV and print a human-readable summary.

    Parameters
    ----------
    df:
        Per-question scores DataFrame from :func:`evaluate_with_ragas`.
    aggregates:
        Aggregate statistics dict from :func:`evaluate_with_ragas`.
    config:
        Eval config (used to determine the output directory).
    experiment_name:
        Label embedded in the output filename.

    Returns
    -------
    Path
        Absolute path of the saved CSV file.
    """
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"ragas_{experiment_name}_{ts}.csv"

    df.to_csv(output_path, index=False)
    logger.info("Evaluation results saved → %s", output_path)

    # Console summary
    separator = "=" * 62
    print(f"\n{separator}")
    print("  RAGAS EVALUATION SUMMARY")
    print(separator)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    for metric in metric_names:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in aggregates:
            mean_val = aggregates[mean_key]
            std_val = aggregates.get(std_key, 0.0)
            bar_len = int(mean_val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(
                f"  {metric:<22}  [{bar}]  {mean_val:.4f} ± {std_val:.4f}"
            )

    print(f"{separator}\n")
    print(f"  Full results saved to: {output_path}")
    print(f"{separator}\n")

    return output_path
