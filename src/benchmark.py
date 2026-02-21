"""
Benchmark system for systematic RAG configuration comparison.

``BenchmarkRunner`` runs multiple experiments where exactly one parameter
varies at a time (chunk size, retrieval k, embedding model, …), collects
RAGAS metrics for each configuration, and generates comparison plots.

Design principles
-----------------
* Each experiment gets its **own vector store directory** so indexes from
  different configurations never collide.
* Raw documents are loaded once and passed into every experiment, avoiding
  redundant I/O.
* Results are accumulated in memory and written to a single summary CSV
  at the end, so partial failures don't lose completed experiment data.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.documents import Document

from src.chunking import chunk_documents
from src.config import RAGConfig
from src.evaluator import (
    evaluate_with_ragas,
    load_eval_dataset,
    run_rag_on_dataset,
    save_evaluation_results,
)
from src.ingestion import ingest_directory
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Metric names we expect from RAGAS
_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Outcome of a single benchmark experiment."""

    name: str
    """Human-readable experiment identifier."""

    param_name: str
    """Name of the parameter that was varied (e.g. ``'chunk_size'``)."""

    param_value: Any
    """Value used for this run (e.g. ``512``)."""

    aggregates: Dict[str, float]
    """Aggregate statistics keyed as ``<metric>_<stat>``."""

    per_question_df: pd.DataFrame
    """Per-question RAGAS scores."""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Orchestrates multi-experiment benchmarks over a fixed corpus.

    Parameters
    ----------
    base_config:
        The baseline :class:`~src.config.RAGConfig` that each experiment
        starts from.  Individual experiments override specific parameters.
    """

    def __init__(self, base_config: RAGConfig) -> None:
        self.base_config = base_config
        self.results: List[ExperimentResult] = []

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _make_variant(self, overrides: Dict[str, Any]) -> RAGConfig:
        """
        Deep-copy the base config and apply *overrides*.

        Override keys use dot-notation to address nested attributes, e.g.
        ``"chunking.chunk_size"``, ``"retriever.k"``.
        """
        config = copy.deepcopy(self.base_config)
        for key_path, value in overrides.items():
            parts = key_path.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return config

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        name: str,
        param_name: str,
        param_value: Any,
        config_overrides: Dict[str, Any],
        raw_documents: Optional[List[Document]] = None,
    ) -> ExperimentResult:
        """
        Run one experiment with the given parameter overrides.

        Parameters
        ----------
        name:
            Unique identifier for this experiment run.
        param_name:
            Which parameter is being varied (for plot axis labels).
        param_value:
            The concrete value used in this run.
        config_overrides:
            Dot-notation overrides applied on top of the base config.
        raw_documents:
            Pre-loaded documents.  If ``None``, they are loaded from disk.

        Returns
        -------
        ExperimentResult
        """
        logger.info(
            "\n%s\nExperiment: %s  |  %s = %s\n%s",
            "=" * 56,
            name,
            param_name,
            param_value,
            "=" * 56,
        )

        config = self._make_variant(config_overrides)

        # Load corpus if not provided
        if raw_documents is None:
            raw_documents = ingest_directory(config.raw_data_dir)

        # Chunk with this experiment's config
        chunks = chunk_documents(raw_documents, config.chunking)

        # Build a fresh pipeline (force_rebuild ensures a clean index)
        pipeline = RAGPipeline(config)
        pipeline.build(chunks, force_rebuild=True)

        # Load evaluation dataset
        eval_data = load_eval_dataset(config.eval.dataset_path)

        # Run RAG inference on every eval question
        records = run_rag_on_dataset(pipeline, eval_data)

        # Compute RAGAS metrics
        per_q_df, aggregates = evaluate_with_ragas(records, config)

        result = ExperimentResult(
            name=name,
            param_name=param_name,
            param_value=param_value,
            aggregates=aggregates,
            per_question_df=per_q_df,
        )
        self.results.append(result)

        # Save per-experiment CSV immediately (so partial runs aren't lost)
        save_evaluation_results(per_q_df, aggregates, config.eval, name)

        return result

    # ------------------------------------------------------------------
    # Pre-built experiment suites
    # ------------------------------------------------------------------

    def run_chunk_size_comparison(
        self,
        chunk_sizes: List[int] = (256, 512, 1024),
        raw_documents: Optional[List[Document]] = None,
    ) -> List[ExperimentResult]:
        """
        Compare RAGAS metrics across different chunk sizes.

        A separate vector store is created for each chunk size so the
        embeddings are always consistent with the chunk content.

        Parameters
        ----------
        chunk_sizes:
            Chunk sizes to evaluate (in characters).
        raw_documents:
            Shared corpus; loaded once to avoid repeated disk reads.
        """
        results = []
        for size in chunk_sizes:
            vs_path = str(
                Path(self.base_config.vectorstore.persist_path).parent
                / f"vs_chunk{size}"
            )
            result = self.run_experiment(
                name=f"chunk_size_{size}",
                param_name="chunk_size",
                param_value=size,
                config_overrides={
                    "chunking.chunk_size": size,
                    "chunking.chunk_overlap": max(32, size // 8),
                    "vectorstore.persist_path": vs_path,
                },
                raw_documents=raw_documents,
            )
            results.append(result)
        return results

    def run_k_comparison(
        self,
        k_values: List[int] = (2, 4, 6, 8),
        raw_documents: Optional[List[Document]] = None,
    ) -> List[ExperimentResult]:
        """
        Compare RAGAS metrics across different retrieval *k* values.

        The vector store is shared across runs (same corpus, same embeddings)
        so we only rebuild the index once.  We override the persist path so
        load_vectorstore can find it.
        """
        # Build the corpus index once to share across k values
        results = []
        for k in k_values:
            result = self.run_experiment(
                name=f"retrieval_k_{k}",
                param_name="retrieval_k",
                param_value=k,
                config_overrides={"retriever.k": k},
                raw_documents=raw_documents,
            )
            results.append(result)
        return results

    def run_embedding_comparison(
        self,
        embedding_configs: List[Dict[str, Any]],
        raw_documents: Optional[List[Document]] = None,
    ) -> List[ExperimentResult]:
        """
        Compare RAGAS metrics across different embedding models.

        Parameters
        ----------
        embedding_configs:
            List of dicts, each with keys ``'name'``, ``'provider'``,
            ``'model_name'`` describing one embedding configuration.

        Example
        -------
        >>> runner.run_embedding_comparison([
        ...     {"name": "minilm", "provider": "huggingface",
        ...      "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        ...     {"name": "mpnet", "provider": "huggingface",
        ...      "model_name": "sentence-transformers/all-mpnet-base-v2"},
        ... ])
        """
        results = []
        for cfg in embedding_configs:
            emb_name = cfg["name"]
            vs_path = str(
                Path(self.base_config.vectorstore.persist_path).parent
                / f"vs_emb_{emb_name}"
            )
            result = self.run_experiment(
                name=f"embedding_{emb_name}",
                param_name="embedding_model",
                param_value=cfg["model_name"],
                config_overrides={
                    "embedding.provider": cfg["provider"],
                    "embedding.model_name": cfg["model_name"],
                    "vectorstore.persist_path": vs_path,
                },
                raw_documents=raw_documents,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Persistence and visualisation
    # ------------------------------------------------------------------

    def save_benchmark_results(self) -> Optional[Path]:
        """
        Write a summary CSV with one row per experiment.

        Columns: ``experiment``, ``param_name``, ``param_value``, and one
        column per aggregate statistic (``<metric>_mean``, ``<metric>_std``, …).
        """
        if not self.results:
            logger.warning("No benchmark results to save.")
            return None

        rows = []
        for r in self.results:
            row: Dict[str, Any] = {
                "experiment": r.name,
                "param_name": r.param_name,
                "param_value": r.param_value,
            }
            row.update(r.aggregates)
            rows.append(row)

        df = pd.DataFrame(rows)

        results_dir = Path(self.base_config.eval.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"experiment_{ts}.csv"

        df.to_csv(output_path, index=False)
        logger.info("Benchmark summary saved → %s", output_path)

        # Pretty-print the summary table
        print("\nBENCHMARK SUMMARY")
        print(df.to_string(index=False))
        print()

        return output_path

    def generate_plots(self) -> None:
        """
        Generate and save one comparison plot per varied parameter.

        Each plot is a 2×2 grid of bar charts — one panel per RAGAS metric —
        with the parameter value on the x-axis and the mean score on the y-axis.
        Error bars represent ±1 standard deviation.
        """
        if not self.results:
            return

        plots_dir = Path(self.base_config.eval.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Group results by the parameter that was varied
        by_param: Dict[str, List[ExperimentResult]] = {}
        for r in self.results:
            by_param.setdefault(r.param_name, []).append(r)

        sns.set_theme(style="whitegrid", palette="husl")

        for param_name, exp_results in by_param.items():
            param_values = [str(r.param_value) for r in exp_results]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f"RAGAS Metrics vs {param_name}",
                fontsize=16,
                fontweight="bold",
                y=1.01,
            )

            colours = sns.color_palette("husl", len(exp_results))

            for ax, metric in zip(axes.flatten(), _RAGAS_METRICS):
                means = [
                    r.aggregates.get(f"{metric}_mean", 0.0) for r in exp_results
                ]
                stds = [
                    r.aggregates.get(f"{metric}_std", 0.0) for r in exp_results
                ]

                bars = ax.bar(
                    param_values,
                    means,
                    yerr=stds,
                    capsize=6,
                    color=colours,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Annotate each bar with its mean value
                for bar, mean in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

                ax.set_title(
                    metric.replace("_", " ").title(), fontsize=12, fontweight="bold"
                )
                ax.set_xlabel(param_name, fontsize=10)
                ax.set_ylabel("Score (0–1)", fontsize=10)
                ax.set_ylim(0, 1.15)
                ax.tick_params(axis="x", rotation=15)

            plt.tight_layout()
            plot_path = plots_dir / f"comparison_{param_name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Plot saved → %s", plot_path)

    def generate_heatmap(self) -> None:
        """
        Generate a heatmap showing all experiments × all RAGAS metrics.

        Useful when multiple experiment suites have been run and you want
        a high-level overview of which configuration works best.
        """
        if not self.results:
            return

        plots_dir = Path(self.base_config.eval.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in self.results:
            row = {"experiment": r.name}
            for metric in _RAGAS_METRICS:
                row[metric] = r.aggregates.get(f"{metric}_mean", float("nan"))
            rows.append(row)

        df = pd.DataFrame(rows).set_index("experiment")

        fig, ax = plt.subplots(figsize=(10, max(4, len(self.results) * 0.6 + 2)))
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "RAGAS Score"},
        )
        ax.set_title("Benchmark Heatmap — RAGAS Metrics per Experiment", fontsize=13)
        plt.tight_layout()

        heatmap_path = plots_dir / "benchmark_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Heatmap saved → %s", heatmap_path)
