"""
main.py — CLI entry point for the RAG Evaluation Pipeline.

Commands
--------
  ingest     Load documents → chunk → build vector store
  evaluate   Run RAGAS evaluation on the eval dataset
  benchmark  Run multi-config benchmark experiments
  query      Answer a single question interactively

Examples
--------
  # Ingest documents and build the vector store
  python main.py ingest

  # Evaluate with default config
  python main.py evaluate

  # Evaluate with a specific chunk size and k
  python main.py evaluate --chunk-size 256 --k 6

  # Run benchmark comparing chunk sizes
  python main.py benchmark --experiment chunk_size --chunk-sizes 256 512 1024

  # Ask a single question
  python main.py query "What is Retrieval Augmented Generation?"

  # Run the full pipeline end-to-end (no sub-command)
  python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any module that might need API keys
load_dotenv()

from src.benchmark import BenchmarkRunner
from src.chunking import chunk_documents, save_chunks
from src.config import RAGConfig
from src.evaluator import (
    evaluate_with_ragas,
    load_eval_dataset,
    run_rag_on_dataset,
    save_evaluation_results,
)
from src.ingestion import ingest_directory
from src.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="Production-grade RAG pipeline with RAGAS evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["openai", "anthropic"],
        help="Override the LLM provider.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the LLM model name.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["huggingface", "openai"],
        help="Override the embedding provider.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- ingest ----
    ingest_p = sub.add_parser("ingest", help="Ingest documents and build vector store.")
    ingest_p.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing raw documents (default: data/raw).",
    )
    ingest_p.add_argument(
        "--chunk-size", type=int, default=None, help="Override chunk size."
    )
    ingest_p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if an index already exists.",
    )

    # ---- evaluate ----
    eval_p = sub.add_parser("evaluate", help="Run RAGAS evaluation.")
    eval_p.add_argument("--eval-dataset", default=None, help="Path to eval JSON file.")
    eval_p.add_argument("--chunk-size", type=int, default=None)
    eval_p.add_argument("--k", type=int, default=None, help="Retrieval top-k.")
    eval_p.add_argument("--name", default="eval", help="Experiment name label.")
    eval_p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild vector store before evaluating.",
    )

    # ---- benchmark ----
    bench_p = sub.add_parser("benchmark", help="Run benchmark experiments.")
    bench_p.add_argument(
        "--experiment",
        choices=["chunk_size", "k_values", "embeddings", "all"],
        default="all",
        help="Which experiment suite to run.",
    )
    bench_p.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        metavar="N",
    )
    bench_p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        metavar="K",
    )

    # ---- query ----
    query_p = sub.add_parser("query", help="Answer a single question.")
    query_p.add_argument("question", help="The question to answer.")
    query_p.add_argument("--chunk-size", type=int, default=None)
    query_p.add_argument("--k", type=int, default=None)

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _apply_global_overrides(args: argparse.Namespace, config: RAGConfig) -> None:
    """Apply top-level provider / model overrides from CLI args."""
    if getattr(args, "llm_provider", None):
        config.llm.provider = args.llm_provider
    if getattr(args, "llm_model", None):
        config.llm.model_name = args.llm_model
    if getattr(args, "embedding_provider", None):
        config.embedding.provider = args.embedding_provider


def cmd_ingest(args: argparse.Namespace, config: RAGConfig) -> None:
    """Load documents, chunk, and index them."""
    if args.data_dir:
        config.raw_data_dir = args.data_dir
    if args.chunk_size:
        config.chunking.chunk_size = args.chunk_size

    documents = ingest_directory(config.raw_data_dir)
    if not documents:
        logger.error(
            "No documents found in '%s'.  Add .txt / .md / .pdf files.",
            config.raw_data_dir,
        )
        sys.exit(1)

    chunks = chunk_documents(documents, config.chunking)
    save_chunks(chunks, Path(config.processed_data_dir) / "chunks.json")

    pipeline = RAGPipeline(config)
    pipeline.build(chunks, force_rebuild=args.rebuild)

    logger.info(
        "Ingestion complete — %d document(s), %d chunk(s).",
        len(documents),
        len(chunks),
    )


def cmd_evaluate(args: argparse.Namespace, config: RAGConfig) -> None:
    """Run the RAGAS evaluation harness."""
    if args.eval_dataset:
        config.eval.dataset_path = args.eval_dataset
    if args.chunk_size:
        config.chunking.chunk_size = args.chunk_size
    if args.k:
        config.retriever.k = args.k

    documents = ingest_directory(config.raw_data_dir)
    if not documents:
        logger.error("No documents found.  Run 'python main.py ingest' first.")
        sys.exit(1)

    chunks = chunk_documents(documents, config.chunking)
    pipeline = RAGPipeline(config)
    pipeline.build(chunks, force_rebuild=getattr(args, "rebuild", False))

    eval_data = load_eval_dataset(config.eval.dataset_path)
    records = run_rag_on_dataset(pipeline, eval_data)
    df, aggregates = evaluate_with_ragas(records, config)
    save_evaluation_results(df, aggregates, config.eval, args.name)


def cmd_benchmark(args: argparse.Namespace, config: RAGConfig) -> None:
    """Run multi-experiment benchmark suite."""
    documents = ingest_directory(config.raw_data_dir)
    if not documents:
        logger.error("No documents found.  Add files to data/raw/ first.")
        sys.exit(1)

    runner = BenchmarkRunner(config)

    if args.experiment in ("chunk_size", "all"):
        runner.run_chunk_size_comparison(
            chunk_sizes=args.chunk_sizes,
            raw_documents=documents,
        )

    if args.experiment in ("k_values", "all"):
        runner.run_k_comparison(
            k_values=args.k_values,
            raw_documents=documents,
        )

    runner.save_benchmark_results()
    runner.generate_plots()
    runner.generate_heatmap()


def cmd_query(args: argparse.Namespace, config: RAGConfig) -> None:
    """Answer a single interactive question."""
    if args.chunk_size:
        config.chunking.chunk_size = args.chunk_size
    if args.k:
        config.retriever.k = args.k

    documents = ingest_directory(config.raw_data_dir)
    chunks = chunk_documents(documents, config.chunking)
    pipeline = RAGPipeline(config)
    pipeline.build(chunks)

    result = pipeline.query(args.question)

    separator = "=" * 64
    print(f"\n{separator}")
    print(f"  QUESTION : {result.question}")
    print(separator)
    print(f"\n  ANSWER :\n\n{result.answer}\n")
    print(f"  SOURCES ({len(result.source_documents)} retrieved chunks):")
    for i, doc in enumerate(result.source_documents, 1):
        fname = doc.metadata.get("filename", "unknown")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] {fname} — {preview} …")
    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Full end-to-end pipeline (default when no sub-command is given)
# ---------------------------------------------------------------------------


def run_full_pipeline(config: RAGConfig) -> None:
    """
    Ingest → chunk → build index → evaluate with RAGAS.

    This is the "batteries-included" entry point that runs the entire
    pipeline in one shot.  Useful for CI / automated experiments.
    """
    logger.info("=" * 56)
    logger.info("  Starting full RAG evaluation pipeline")
    logger.info("=" * 56)

    # 1. Ingest
    logger.info("Step 1/4 — Ingesting documents …")
    documents = ingest_directory(config.raw_data_dir)
    if not documents:
        logger.error(
            "No documents found in '%s'.  "
            "Add .txt / .md / .pdf files and try again.",
            config.raw_data_dir,
        )
        sys.exit(1)

    # 2. Chunk
    logger.info("Step 2/4 — Chunking documents …")
    chunks = chunk_documents(documents, config.chunking)
    save_chunks(chunks, Path(config.processed_data_dir) / "chunks.json")

    # 3. Build pipeline
    logger.info("Step 3/4 — Building vector store and pipeline …")
    pipeline = RAGPipeline(config)
    pipeline.build(chunks)

    # 4. Evaluate
    logger.info("Step 4/4 — Running RAGAS evaluation …")
    eval_data = load_eval_dataset(config.eval.dataset_path)
    records = run_rag_on_dataset(pipeline, eval_data)
    df, aggregates = evaluate_with_ragas(records, config)
    save_evaluation_results(df, aggregates, config.eval)

    logger.info("Pipeline complete.  Check results/ for output files.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    config = RAGConfig()
    parser = build_parser()
    args = parser.parse_args()

    _apply_global_overrides(args, config)

    dispatch = {
        "ingest": cmd_ingest,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
        "query": cmd_query,
    }

    if args.command in dispatch:
        dispatch[args.command](args, config)
    else:
        # No sub-command — run the full pipeline
        run_full_pipeline(config)
