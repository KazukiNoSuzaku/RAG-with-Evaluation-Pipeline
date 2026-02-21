"""
Pytest fixtures shared across all test modules.

Fixtures are designed to avoid external dependencies (no API keys required)
by using small in-memory datasets and LangChain's FakeListChatModel.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from src.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EvalConfig,
    LLMConfig,
    RAGConfig,
    RetrieverConfig,
    VectorStoreConfig,
)


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    (
        "rag_overview.txt",
        """Retrieval Augmented Generation (RAG) is an AI architecture that enhances
large language model responses by retrieving relevant information from an
external knowledge base at inference time.

RAG consists of two main components: a retrieval component that finds relevant
passages from a document collection, and a generation component that produces
the final answer grounded in retrieved context.

The main benefits of RAG include reduced hallucination, knowledge currency,
and source attribution.""",
    ),
    (
        "vector_databases.txt",
        """A vector database stores high-dimensional vector embeddings and enables
fast similarity search using approximate nearest neighbor algorithms.

FAISS (Facebook AI Similarity Search) is an open-source in-process library
that provides highly optimised ANN search with GPU acceleration support.

Chroma is an open-source embedded vector database with metadata filtering
support, built on SQLite, ideal for development and medium-scale production.""",
    ),
    (
        "evaluation.txt",
        """RAGAS is an open-source framework for evaluating RAG pipelines.

Faithfulness measures what fraction of claims in the answer are supported
by the retrieved context.

Answer relevancy measures how well the generated answer addresses the
original question.

Context precision measures whether the most relevant chunks are ranked
higher than less relevant ones.""",
    ),
]


@pytest.fixture
def sample_documents() -> List[Document]:
    """Three small Documents simulating an ingested corpus."""
    return [
        Document(
            page_content=text,
            metadata={"source": fname, "filename": fname, "file_type": "txt"},
        )
        for fname, text in SAMPLE_TEXTS
    ]


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory that is cleaned up after each test."""
    return tmp_path


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    return ChunkingConfig(chunk_size=200, chunk_overlap=20)


@pytest.fixture
def eval_config(tmp_path: Path) -> EvalConfig:
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "results" / "plots")
    return EvalConfig(
        dataset_path=str(tmp_path / "eval_dataset.json"),
        results_dir=results_dir,
        plots_dir=plots_dir,
        metrics=["faithfulness", "answer_relevancy"],
    )


@pytest.fixture
def sample_eval_dataset(tmp_path: Path) -> Path:
    """Write a minimal eval dataset JSON and return its path."""
    dataset = [
        {
            "question": "What is RAG?",
            "ground_truth": "RAG is a retrieval augmented generation architecture.",
        },
        {
            "question": "What is faithfulness?",
            "ground_truth": "Faithfulness measures if claims in the answer are supported by context.",
        },
    ]
    path = tmp_path / "eval_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fake LLM for offline testing
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm():
    """
    A LangChain FakeListChatModel that cycles through canned responses.
    Used to test generator and pipeline logic without API calls.
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(
        responses=[
            "RAG stands for Retrieval Augmented Generation.",
            "Faithfulness measures if the answer is grounded in context.",
            "FAISS is a fast vector similarity search library.",
            "Context precision measures retrieval ranking quality.",
            "Vector databases store high-dimensional embeddings.",
        ]
    )
