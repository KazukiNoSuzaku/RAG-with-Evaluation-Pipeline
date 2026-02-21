"""
Retrieval system with configurable top-k and search strategies.

The ``Retriever`` class wraps a LangChain VectorStore and returns a
structured ``RetrievalResult`` that carries both the raw Document objects
(for inspection) and the plain-text context strings that downstream
modules (generator and RAGAS evaluator) consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from src.config import RetrieverConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Structured output from a single retrieval call."""

    query: str
    """The original user question."""

    documents: List[Document]
    """Full LangChain Documents with metadata (source, filename, …)."""

    contexts: List[str]
    """Plain-text content strings extracted from *documents*.
    This is the form consumed by the generator and by RAGAS metrics."""


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Top-k retriever wrapping a :class:`~langchain_core.vectorstores.VectorStore`.

    Supports two search strategies:

    * **similarity** — standard cosine-nearest-neighbour; deterministic and
      fast.  Best default for most RAG use-cases.
    * **mmr** (Maximal Marginal Relevance) — fetches a larger candidate set
      then re-ranks for diversity.  Reduces redundant context when chunks are
      semantically near-duplicate.

    Parameters
    ----------
    vectorstore:
        An indexed vector store (FAISS or Chroma).
    config:
        Retrieval strategy, k, and MMR tuning parameters.
    """

    def __init__(self, vectorstore: VectorStore, config: RetrieverConfig) -> None:
        self.vectorstore = vectorstore
        self.config = config

    # ------------------------------------------------------------------
    # Core retrieve method
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language question or search string.

        Returns
        -------
        RetrievalResult
            Structured result containing source Documents and plain-text
            context strings.
        """
        if self.config.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, k=self.config.k
            )

        elif self.config.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.config.k,
                fetch_k=self.config.fetch_k,
                lambda_mult=self.config.lambda_mult,
            )

        else:
            raise ValueError(
                f"Unknown search_type: '{self.config.search_type}'. "
                "Expected 'similarity' or 'mmr'."
            )

        contexts = [doc.page_content for doc in docs]

        logger.debug(
            "Retrieved %d chunk(s) for query: '%.60s …'", len(docs), query
        )
        return RetrievalResult(query=query, documents=docs, contexts=contexts)

    # ------------------------------------------------------------------
    # LangChain-compatible retriever (for use in chains/agents)
    # ------------------------------------------------------------------

    def as_langchain_retriever(self) -> BaseRetriever:
        """
        Return a LangChain ``BaseRetriever`` compatible with LCEL chains.

        Useful when composing the retriever into a LangChain Expression
        Language pipeline or using it with other LangChain utilities.
        """
        if self.config.search_type == "similarity":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.k},
            )

        # MMR
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.k,
                "fetch_k": self.config.fetch_k,
                "lambda_mult": self.config.lambda_mult,
            },
        )
