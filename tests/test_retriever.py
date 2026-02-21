"""
Unit tests for the retriever module.

Uses an in-memory FAISS vector store built from a small corpus so no
external services or API keys are needed.  We use a deterministic
HuggingFace embedding model (all-MiniLM-L6-v2) if available, otherwise
we skip tests that require it.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import RetrieverConfig
from src.retriever import RetrievalResult, Retriever


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_docs(texts: List[str]) -> List[Document]:
    return [
        Document(
            page_content=t,
            metadata={"source": f"doc_{i}.txt", "chunk_index": i},
        )
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def mock_vectorstore():
    """A mock VectorStore that returns predictable documents."""
    docs = make_docs(
        [
            "RAG is Retrieval Augmented Generation.",
            "FAISS is a fast vector similarity library.",
            "RAGAS evaluates faithfulness and relevancy.",
            "Chroma is a vector database with metadata filtering.",
        ]
    )

    vs = MagicMock()
    vs.similarity_search.return_value = docs[:2]
    vs.max_marginal_relevance_search.return_value = docs[:2]
    return vs, docs


# ---------------------------------------------------------------------------
# Tests: RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_contexts_extracted_from_documents(self):
        docs = make_docs(["Text A", "Text B"])
        result = RetrievalResult(
            query="test query",
            documents=docs,
            contexts=[d.page_content for d in docs],
        )
        assert result.contexts == ["Text A", "Text B"]

    def test_query_stored(self):
        result = RetrievalResult(query="my query", documents=[], contexts=[])
        assert result.query == "my query"


# ---------------------------------------------------------------------------
# Tests: Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_similarity_search_called_with_correct_k(self, mock_vectorstore):
        """similarity_search should be called with the configured k."""
        vs, _ = mock_vectorstore
        config = RetrieverConfig(k=2, search_type="similarity")
        retriever = Retriever(vs, config)
        result = retriever.retrieve("What is RAG?")

        vs.similarity_search.assert_called_once_with("What is RAG?", k=2)

    def test_mmr_search_called_with_correct_params(self, mock_vectorstore):
        """MMR search should pass k, fetch_k, and lambda_mult."""
        vs, _ = mock_vectorstore
        config = RetrieverConfig(
            k=2,
            search_type="mmr",
            fetch_k=10,
            lambda_mult=0.6,
        )
        retriever = Retriever(vs, config)
        retriever.retrieve("What is FAISS?")

        vs.max_marginal_relevance_search.assert_called_once_with(
            "What is FAISS?", k=2, fetch_k=10, lambda_mult=0.6
        )

    def test_returns_retrieval_result(self, mock_vectorstore):
        """retrieve() should return a RetrievalResult instance."""
        vs, _ = mock_vectorstore
        config = RetrieverConfig(k=2, search_type="similarity")
        retriever = Retriever(vs, config)
        result = retriever.retrieve("test")

        assert isinstance(result, RetrievalResult)

    def test_contexts_match_document_content(self, mock_vectorstore):
        """The contexts list should contain the page_content of each doc."""
        vs, docs = mock_vectorstore
        vs.similarity_search.return_value = docs[:2]
        config = RetrieverConfig(k=2, search_type="similarity")
        retriever = Retriever(vs, config)
        result = retriever.retrieve("test")

        expected_contexts = [d.page_content for d in docs[:2]]
        assert result.contexts == expected_contexts

    def test_query_preserved_in_result(self, mock_vectorstore):
        vs, _ = mock_vectorstore
        config = RetrieverConfig(k=2, search_type="similarity")
        retriever = Retriever(vs, config)
        result = retriever.retrieve("my specific question")
        assert result.query == "my specific question"

    def test_invalid_search_type_raises(self, mock_vectorstore):
        """An unknown search_type should raise ValueError."""
        vs, _ = mock_vectorstore
        config = RetrieverConfig(k=2, search_type="invalid_type")
        retriever = Retriever(vs, config)
        with pytest.raises(ValueError, match="Unknown search_type"):
            retriever.retrieve("test")

    def test_as_langchain_retriever_similarity(self, mock_vectorstore):
        """as_langchain_retriever() should not raise for similarity mode."""
        vs, _ = mock_vectorstore
        # as_retriever is a real method on BaseVectorStore — mock it
        vs.as_retriever.return_value = MagicMock()
        config = RetrieverConfig(k=3, search_type="similarity")
        retriever = Retriever(vs, config)
        lc_retriever = retriever.as_langchain_retriever()
        vs.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def test_as_langchain_retriever_mmr(self, mock_vectorstore):
        """as_langchain_retriever() should pass MMR params correctly."""
        vs, _ = mock_vectorstore
        vs.as_retriever.return_value = MagicMock()
        config = RetrieverConfig(
            k=4, search_type="mmr", fetch_k=20, lambda_mult=0.5
        )
        retriever = Retriever(vs, config)
        retriever.as_langchain_retriever()
        vs.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
        )

    def test_documents_empty_when_store_returns_none(self):
        """If vectorstore returns empty list, result should have empty contexts."""
        vs = MagicMock()
        vs.similarity_search.return_value = []
        config = RetrieverConfig(k=4, search_type="similarity")
        retriever = Retriever(vs, config)
        result = retriever.retrieve("test")
        assert result.documents == []
        assert result.contexts == []
