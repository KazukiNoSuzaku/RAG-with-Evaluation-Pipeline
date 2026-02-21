"""
Vector store abstraction supporting FAISS and Chroma backends.

Design decisions
----------------
* **FAISS** is used as the default because it has no external service
  dependency, is purely in-process, and scales to millions of vectors on a
  single machine.  The index is serialised to disk so it survives restarts.

* **Chroma** provides richer metadata filtering and a more feature-complete
  API at the cost of slightly higher initialisation overhead.  Prefer it when
  you need structured filtering on document metadata.

Both backends are wrapped behind the same ``build_vectorstore`` /
``load_vectorstore`` interface so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.config import VectorStoreConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_vectorstore(
    chunks: List[Document],
    embeddings: Embeddings,
    config: VectorStoreConfig,
) -> VectorStore:
    """
    Embed *chunks* and persist the resulting vector store to disk.

    Parameters
    ----------
    chunks:
        Pre-processed document chunks from :func:`~src.chunking.chunk_documents`.
    embeddings:
        Embedding model (typically wrapped with CachedEmbeddings).
    config:
        Vector store backend and path settings.

    Returns
    -------
    VectorStore
        A fully-indexed, ready-to-query vector store.
    """
    persist_path = Path(config.persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building %s vector store with %d chunks …", config.backend, len(chunks)
    )

    if config.backend == "faiss":
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(persist_path))
        logger.info("FAISS index saved → %s", persist_path)
        return store

    if config.backend == "chroma":
        from langchain_chroma import Chroma

        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=config.collection_name,
            persist_directory=str(persist_path),
        )
        logger.info("Chroma collection '%s' saved → %s", config.collection_name, persist_path)
        return store

    raise ValueError(f"Unsupported vector store backend: '{config.backend}'")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_vectorstore(
    embeddings: Embeddings,
    config: VectorStoreConfig,
) -> Optional[VectorStore]:
    """
    Load a previously persisted vector store from disk.

    Returns ``None`` if no persisted index is found so callers can decide
    whether to rebuild without catching exceptions.

    Parameters
    ----------
    embeddings:
        Must be the same model used during :func:`build_vectorstore`.
    config:
        Vector store backend and path settings.

    Returns
    -------
    VectorStore or None
    """
    persist_path = Path(config.persist_path)

    if config.backend == "faiss":
        from langchain_community.vectorstores import FAISS

        index_file = persist_path / "index.faiss"
        if index_file.exists():
            logger.info("Loading existing FAISS index from %s", persist_path)
            return FAISS.load_local(
                str(persist_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    if config.backend == "chroma":
        from langchain_chroma import Chroma

        db_file = persist_path / "chroma.sqlite3"
        if db_file.exists():
            logger.info(
                "Loading existing Chroma collection '%s' from %s",
                config.collection_name,
                persist_path,
            )
            return Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
        return None

    raise ValueError(f"Unsupported vector store backend: '{config.backend}'")
