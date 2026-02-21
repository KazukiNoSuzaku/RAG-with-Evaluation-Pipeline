"""
Text chunking with configurable strategies.

Uses LangChain's RecursiveCharacterTextSplitter which tries successively
smaller separators until chunks fit within chunk_size.  This produces more
semantically coherent chunks than fixed-character splitting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------


def chunk_documents(
    documents: List[Document],
    config: ChunkingConfig,
) -> List[Document]:
    """
    Split *documents* into retrieval-sized chunks.

    Each output chunk inherits the metadata of its parent document and gains
    two extra fields:

    * ``chunk_index`` — global position in the full chunk list.
    * ``char_count``  — character length of this specific chunk.

    Parameters
    ----------
    documents:
        Raw documents as returned by :func:`~src.ingestion.ingest_directory`.
    config:
        Chunking parameters (size, overlap, separators).

    Returns
    -------
    List[Document]
        Flat list of chunks ready for embedding and vector store ingestion.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)

    logger.info(
        "Chunked %d document(s) → %d chunk(s)  "
        "(size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        config.chunk_size,
        config.chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Persistence helpers — useful for inspecting chunks and skipping re-chunking
# ---------------------------------------------------------------------------


def save_chunks(chunks: List[Document], output_path: Path) -> None:
    """
    Serialise *chunks* to a JSON file for caching and offline inspection.

    Parameters
    ----------
    chunks:
        List of chunk Documents to save.
    output_path:
        Destination file path (will create parent directories if needed).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {"content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %d chunks → %s", len(chunks), output_path)


def load_chunks(input_path: Path) -> List[Document]:
    """
    Deserialise chunks previously saved by :func:`save_chunks`.

    Parameters
    ----------
    input_path:
        Path to the JSON file written by :func:`save_chunks`.

    Returns
    -------
    List[Document]
        Reconstructed chunk Documents with their original metadata.
    """
    input_path = Path(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in payload
    ]
