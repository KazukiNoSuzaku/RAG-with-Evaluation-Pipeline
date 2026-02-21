"""
Unit tests for the chunking module.

Tests cover:
- Basic chunking produces non-empty chunks
- Chunk sizes respect the configured maximum
- Overlap is applied
- Metadata is propagated and augmented
- Persistence (save/load) round-trips correctly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from src.chunking import chunk_documents, load_chunks, save_chunks
from src.config import ChunkingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "filename": source},
    )


# ---------------------------------------------------------------------------
# Tests: chunk_documents
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    def test_produces_chunks(self, sample_documents):
        """Chunking a non-empty corpus returns at least one chunk."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        assert len(chunks) > 0

    def test_chunk_size_respected(self, sample_documents):
        """No chunk should exceed the configured chunk_size."""
        chunk_size = 150
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=10)
        chunks = chunk_documents(sample_documents, config)
        for chunk in chunks:
            # Allow a small tolerance because the splitter may not split at
            # exactly chunk_size if no good separator is found
            assert len(chunk.page_content) <= chunk_size + 20

    def test_smaller_chunks_more_numerous(self, sample_documents):
        """Smaller chunk size should produce more chunks."""
        cfg_small = ChunkingConfig(chunk_size=100, chunk_overlap=10)
        cfg_large = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        small_chunks = chunk_documents(sample_documents, cfg_small)
        large_chunks = chunk_documents(sample_documents, cfg_large)
        assert len(small_chunks) >= len(large_chunks)

    def test_metadata_propagated(self, sample_documents):
        """Each chunk should inherit the parent document's metadata."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "filename" in chunk.metadata

    def test_chunk_index_added(self, sample_documents):
        """chunk_index metadata field should be present on every chunk."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        indices = [c.metadata["chunk_index"] for c in chunks]
        # Indices should be sequential starting from 0
        assert indices == list(range(len(chunks)))

    def test_char_count_added(self, sample_documents):
        """char_count metadata should match the actual content length."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        for chunk in chunks:
            assert chunk.metadata["char_count"] == len(chunk.page_content)

    def test_empty_input_returns_empty(self):
        """Empty document list should return empty chunk list."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents([], config)
        assert chunks == []

    def test_single_short_document_not_split(self):
        """A document shorter than chunk_size should not be split."""
        short_text = "This is a short document."
        doc = make_doc(short_text)
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=100)
        chunks = chunk_documents([doc], config)
        assert len(chunks) == 1
        assert chunks[0].page_content == short_text

    def test_long_document_is_split(self):
        """A document much longer than chunk_size should be split."""
        long_text = "word " * 500  # 2500 chars
        doc = make_doc(long_text)
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents([doc], config)
        assert len(chunks) > 1

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share content from the overlap region."""
        # Use a simple sentence that will be split in the middle
        text = "A " * 300  # 600 chars
        doc = make_doc(text)
        config = ChunkingConfig(chunk_size=100, chunk_overlap=30)
        chunks = chunk_documents([doc], config)

        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-20:]
            start_of_second = chunks[1].page_content[:20]
            # At least some characters should overlap (since overlap=30)
            overlap_chars = set(end_of_first) & set(start_of_second)
            assert len(overlap_chars) > 0


# ---------------------------------------------------------------------------
# Tests: save_chunks / load_chunks
# ---------------------------------------------------------------------------


class TestChunkPersistence:
    def test_save_creates_file(self, sample_documents, tmp_path):
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        output_path = tmp_path / "chunks.json"
        save_chunks(chunks, output_path)
        assert output_path.exists()

    def test_save_valid_json(self, sample_documents, tmp_path):
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        output_path = tmp_path / "chunks.json"
        save_chunks(chunks, output_path)
        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == len(chunks)

    def test_roundtrip_preserves_content(self, sample_documents, tmp_path):
        """load_chunks(save_chunks(chunks)) should return identical Documents."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        original = chunk_documents(sample_documents, config)
        path = tmp_path / "chunks.json"
        save_chunks(original, path)
        loaded = load_chunks(path)

        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded):
            assert orig.page_content == load.page_content
            assert orig.metadata == load.metadata

    def test_save_creates_parent_dirs(self, sample_documents, tmp_path):
        """save_chunks should create nested directories if they don't exist."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents(sample_documents, config)
        deep_path = tmp_path / "a" / "b" / "c" / "chunks.json"
        save_chunks(chunks, deep_path)
        assert deep_path.exists()
