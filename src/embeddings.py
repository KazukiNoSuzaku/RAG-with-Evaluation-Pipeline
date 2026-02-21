"""
Embedding model abstraction with disk-based caching.

Design rationale
----------------
* **Factory pattern** — ``get_embedding_model`` returns the right
  LangChain Embeddings object based on config without callers knowing
  provider details.
* **CachedEmbeddings** — wraps any Embeddings with a SHA-256-keyed
  pickle cache so identical texts are never re-embedded across runs.
  This is critical when running benchmark sweeps where the same corpus
  is embedded with different configs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

from langchain_core.embeddings import Embeddings

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_model(config: EmbeddingConfig) -> Embeddings:
    """
    Instantiate and return a LangChain-compatible Embeddings object.

    HuggingFace embeddings run locally (CPU by default) with no API key.
    OpenAI embeddings require ``OPENAI_API_KEY`` to be set in the environment.

    Parameters
    ----------
    config:
        Embedding configuration specifying provider and model name.

    Returns
    -------
    Embeddings
        Ready-to-use LangChain Embeddings instance.
    """
    if config.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(
            "Loading HuggingFace embedding model: %s", config.model_name
        )
        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            cache_folder=config.cache_dir,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it or add it to your .env file."
            )
        logger.info("Using OpenAI embedding model: %s", config.model_name)
        return OpenAIEmbeddings(
            model=config.model_name,
            openai_api_key=api_key,
        )

    raise ValueError(f"Unsupported embedding provider: '{config.provider}'")


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


class CachedEmbeddings(Embeddings):
    """
    Wraps any LangChain Embeddings with a persistent disk cache.

    Cache entries are keyed by the SHA-256 hash of the text string.
    On each call to ``embed_documents`` or ``embed_query``:

    1. Check which texts already have a cached embedding.
    2. Batch-embed only the uncached texts.
    3. Merge and return in the original order.
    4. Persist the updated cache to disk.

    This gives near-instant repeat evaluations for benchmark sweeps.
    """

    def __init__(self, base_embeddings: Embeddings, cache_dir: str) -> None:
        self._base = base_embeddings
        self._cache_path = Path(cache_dir) / "embedding_cache.pkl"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, List[float]]:
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as fh:
                data = pickle.load(fh)
            logger.debug("Loaded %d cached embeddings from disk", len(data))
            return data
        return {}

    def _save(self) -> None:
        with open(self._cache_path, "wb") as fh:
            pickle.dump(self._cache, fh)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, using the cache for texts seen before."""
        results: List[tuple[int, List[float]]] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            logger.info(
                "Embedding %d new text(s)  (%d already cached)",
                len(uncached_texts),
                len(texts) - len(uncached_texts),
            )
            new_embeddings = self._base.embed_documents(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                key = self._hash(text)
                self._cache[key] = emb
                results.append((idx, emb))
            self._save()

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string, using the cache if available."""
        key = self._hash(text)
        if key not in self._cache:
            self._cache[key] = self._base.embed_query(text)
            self._save()
        return self._cache[key]
