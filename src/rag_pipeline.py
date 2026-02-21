"""
Unified RAG pipeline.

Combines document chunking, embedding, vector store, retrieval, and
generation into a single coherent interface.  Callers only need to call
``build()`` once (or skip it if an existing index is present), then call
``query()`` / ``batch_query()`` as needed.

Separation of concerns
-----------------------
* The pipeline **orchestrates** but does not implement any ML logic itself.
* Each sub-module (embeddings, vectorstore, retriever, generator) can be
  swapped independently without touching this file.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from src.config import RAGConfig
from src.embeddings import CachedEmbeddings, get_embedding_model
from src.generator import Generator
from src.retriever import Retriever, RetrievalResult
from src.vectorstore import build_vectorstore, load_vectorstore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class RAGResult:
    """Complete output from a single end-to-end RAG query."""

    question: str
    """The original user question."""

    answer: str
    """The LLM-generated, context-grounded answer."""

    contexts: List[str]
    """Plain-text context strings used to generate the answer.
    Preserved for RAGAS evaluation (``retrieved_contexts`` field)."""

    source_documents: List[Document] = field(default_factory=list)
    """Full LangChain Documents including metadata (source file, chunk index, …)."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Lifecycle
    ---------
    1. Instantiate with a :class:`~src.config.RAGConfig`.
    2. Call :meth:`build` with pre-chunked documents to index them
       (or load a persisted index automatically).
    3. Call :meth:`query` for single questions or :meth:`batch_query`
       for a list of questions.

    Parameters
    ----------
    config:
        Full pipeline configuration.
    """

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._retriever: Optional[Retriever] = None
        self._generator: Optional[Generator] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: List[Document],
        force_rebuild: bool = False,
    ) -> None:
        """
        Index *chunks* into the vector store and initialise all components.

        If a persisted vector store already exists at the configured path and
        ``force_rebuild`` is ``False``, the existing index is loaded directly
        (skipping embedding computation for the corpus).

        Parameters
        ----------
        chunks:
            Document chunks from :func:`~src.chunking.chunk_documents`.
        force_rebuild:
            If ``True``, always rebuild the index even if one exists.
            Use this when the corpus changes or the embedding model changes.
        """
        # Wrap the base embedding model with disk caching
        base_emb = get_embedding_model(self.config.embedding)
        embeddings = CachedEmbeddings(base_emb, self.config.embedding.cache_dir)

        # Try loading an existing index first
        vectorstore = None
        if not force_rebuild:
            vectorstore = load_vectorstore(embeddings, self.config.vectorstore)
            if vectorstore is not None:
                logger.info("Loaded existing vector store — skipping rebuild.")

        # Build new index if nothing was loaded
        if vectorstore is None:
            logger.info("Building new vector store …")
            vectorstore = build_vectorstore(chunks, embeddings, self.config.vectorstore)

        self._retriever = Retriever(vectorstore, self.config.retriever)
        self._generator = Generator(self.config.llm)
        logger.info("RAG pipeline is ready.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """
        Answer *question* using retrieval-augmented generation.

        Parameters
        ----------
        question:
            Natural-language question from the user or evaluation dataset.

        Returns
        -------
        RAGResult
            Structured result containing the answer, contexts, and source docs.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        """
        self._ensure_built()

        retrieval: RetrievalResult = self._retriever.retrieve(question)
        answer = self._generator.generate(question, retrieval.contexts)

        return RAGResult(
            question=question,
            answer=answer,
            contexts=retrieval.contexts,
            source_documents=retrieval.documents,
        )

    def batch_query(self, questions: List[str]) -> List[RAGResult]:
        """
        Run :meth:`query` on each question in *questions*.

        Processes questions sequentially.  For async parallel processing see
        :meth:`abatch_query`.

        Parameters
        ----------
        questions:
            List of natural-language questions.

        Returns
        -------
        List[RAGResult]
            One result per question, in the same order.
        """
        results = []
        for i, q in enumerate(questions, 1):
            logger.info("Query %d/%d: %.60s …", i, len(questions), q)
            results.append(self.query(q))
        return results

    async def aquery(self, question: str) -> RAGResult:
        """Async variant of :meth:`query`."""
        self._ensure_built()

        # Retrieval is sync (FAISS / Chroma are not async-native)
        retrieval = await asyncio.get_event_loop().run_in_executor(
            None, self._retriever.retrieve, question
        )
        answer = await self._generator.agenerate(question, retrieval.contexts)

        return RAGResult(
            question=question,
            answer=answer,
            contexts=retrieval.contexts,
            source_documents=retrieval.documents,
        )

    async def abatch_query(self, questions: List[str]) -> List[RAGResult]:
        """Async batch processing — runs all queries concurrently."""
        tasks = [self.aquery(q) for q in questions]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if self._retriever is None or self._generator is None:
            raise RuntimeError(
                "Pipeline is not initialised.  Call build() before querying."
            )
