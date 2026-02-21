"""
Central configuration for the RAG evaluation pipeline.

All pipeline parameters live here as typed dataclasses so every module
receives explicit, validated settings rather than relying on globals.
This also makes systematic benchmarking trivial: clone the default config
and override the single parameter under test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

# ---------------------------------------------------------------------------
# Resolve project root so paths work regardless of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Sub-system configs
# ---------------------------------------------------------------------------


@dataclass
class ChunkingConfig:
    """Controls how raw documents are split into retrieval chunks."""

    chunk_size: int = 512
    """Maximum number of characters per chunk."""

    chunk_overlap: int = 64
    """Overlap between consecutive chunks to preserve cross-boundary context."""

    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    """Ordered list of separators used by RecursiveCharacterTextSplitter."""


@dataclass
class EmbeddingConfig:
    """Embedding model selection and caching."""

    provider: Literal["huggingface", "openai"] = "huggingface"
    """
    'huggingface' uses sentence-transformers (no API key needed).
    'openai' uses text-embedding-3-* (requires OPENAI_API_KEY).
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """
    HuggingFace: any sentence-transformers model name.
    OpenAI: 'text-embedding-3-small' or 'text-embedding-3-large'.
    """

    cache_dir: str = str(PROJECT_ROOT / "data" / "embedding_cache")
    """Disk path where computed embeddings are pickled for reuse."""


@dataclass
class VectorStoreConfig:
    """Vector database backend configuration."""

    backend: Literal["faiss", "chroma"] = "faiss"
    """
    'faiss'  — in-process, CPU-only, ultra-fast, no server required.
    'chroma' — embedded SQLite-backed store, richer metadata filtering.
    """

    persist_path: str = str(PROJECT_ROOT / "data" / "vectorstore")
    """Root directory where the index files are serialised."""

    collection_name: str = "rag_documents"
    """Chroma collection name (ignored for FAISS)."""


@dataclass
class RetrieverConfig:
    """Retrieval strategy parameters."""

    k: int = 4
    """Number of chunks to return per query."""

    search_type: Literal["similarity", "mmr"] = "similarity"
    """
    'similarity' — standard cosine/dot-product nearest neighbour.
    'mmr'        — Maximal Marginal Relevance for diversity-aware retrieval.
    """

    fetch_k: int = 20
    """For MMR: candidate pool size before diversity re-ranking."""

    lambda_mult: float = 0.5
    """For MMR: trade-off between relevance (1.0) and diversity (0.0)."""


@dataclass
class LLMConfig:
    """LLM provider and generation parameters."""

    provider: Literal["openai", "anthropic"] = "openai"
    """
    'openai'    — requires OPENAI_API_KEY.
    'anthropic' — requires ANTHROPIC_API_KEY.
    """

    model_name: str = "gpt-4o-mini"
    """
    OpenAI:    'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', …
    Anthropic: 'claude-3-5-haiku-20241022', 'claude-sonnet-4-6', …
    """

    temperature: float = 0.0
    """0.0 = deterministic / faithful; increase for more creative answers."""

    max_tokens: int = 1024
    """Maximum tokens in the generated answer."""


@dataclass
class EvalConfig:
    """RAGAS evaluation harness configuration."""

    dataset_path: str = str(PROJECT_ROOT / "data" / "eval" / "eval_dataset.json")
    """Path to JSON file containing evaluation Q&A pairs."""

    results_dir: str = str(PROJECT_ROOT / "results")
    """Directory where CSV results are written."""

    plots_dir: str = str(PROJECT_ROOT / "results" / "plots")
    """Directory where benchmark visualisations are saved."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    """
    RAGAS metrics to compute.  Remove 'context_recall' if you don't have
    high-quality ground-truth answers (it is the most expensive to compute).
    """


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """
    Master configuration that bundles all sub-system configs.

    Usage
    -----
    >>> cfg = RAGConfig()                         # sensible defaults
    >>> cfg.chunking.chunk_size = 256             # override one param
    >>> cfg_openai = RAGConfig(                   # full OpenAI stack
    ...     embedding=EmbeddingConfig(
    ...         provider="openai",
    ...         model_name="text-embedding-3-small",
    ...     ),
    ...     llm=LLMConfig(provider="openai", model_name="gpt-4o"),
    ... )
    """

    # Data directories
    raw_data_dir: str = str(PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")
    experiments_dir: str = str(PROJECT_ROOT / "experiments")

    # Sub-system configs (use field() to avoid mutable default pitfalls)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        """Ensure all required directories exist on disk."""
        dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.experiments_dir,
            self.embedding.cache_dir,
            self.vectorstore.persist_path,
            self.eval.results_dir,
            self.eval.plots_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level default — import this rather than instantiating repeatedly.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = RAGConfig()
