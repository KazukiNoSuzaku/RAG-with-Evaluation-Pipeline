"""
Document ingestion pipeline.

Supports PDF, TXT, and Markdown files.  Each loader normalises its output
to plain text before returning a LangChain Document so every downstream
module works with the same type regardless of source format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> str:
    """Read a plain-text file, replacing undecodable bytes gracefully."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_markdown(path: Path) -> str:
    """
    Read a Markdown file and strip the most common markup tokens so the
    text fed into the chunker is prose rather than raw markdown syntax.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # ATX headings (# H1, ## H2, …)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold / italic (**bold**, *italic*, __bold__, _italic_)
    text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]+)_{1,2}", r"\1", text)
    # Inline & fenced code
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`\n]+`", "", text)
    # Hyperlinks  [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Images  ![alt](url) → alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    return text


def _load_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pypdf.

    Each page is joined with a double newline so paragraph boundaries are
    preserved across page breaks.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF support.  Install it with:  pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)

    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_LOADER_MAP = {
    ".txt": _load_txt,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".pdf": _load_pdf,
}


def clean_text(text: str) -> str:
    """
    Normalise whitespace and strip artefacts common in extracted text.

    Operations (in order):
    1. Remove null bytes (common in some PDF extractions).
    2. Normalise Windows / old Mac line endings to LF.
    3. Collapse runs of 3+ blank lines to exactly two (one blank paragraph).
    4. Collapse runs of spaces/tabs within a line to a single space.
    5. Strip leading/trailing whitespace.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_document(file_path: Path) -> Optional[Document]:
    """
    Load and clean a single document from *file_path*.

    Returns a :class:`~langchain_core.documents.Document` with page content
    and source metadata, or ``None`` if the file type is unsupported or the
    content is empty after cleaning.
    """
    suffix = file_path.suffix.lower()
    loader = _LOADER_MAP.get(suffix)

    if loader is None:
        logger.warning("Unsupported file type '%s' — skipping %s", suffix, file_path.name)
        return None

    try:
        raw = loader(file_path)
    except Exception as exc:
        logger.error("Failed to load '%s': %s", file_path, exc)
        return None

    cleaned = clean_text(raw)
    if not cleaned:
        logger.warning("Empty content after cleaning — skipping %s", file_path.name)
        return None

    return Document(
        page_content=cleaned,
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": suffix.lstrip("."),
            "char_count": len(cleaned),
        },
    )


def ingest_directory(directory: str | Path) -> List[Document]:
    """
    Recursively load all supported documents from *directory*.

    Supported extensions: ``.txt``, ``.md``, ``.markdown``, ``.pdf``.

    Parameters
    ----------
    directory:
        Path to the folder containing raw source documents.

    Returns
    -------
    List[Document]
        One Document per successfully loaded file, sorted by filename for
        deterministic ordering across runs.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    supported = set(_LOADER_MAP.keys())
    files = sorted(
        f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in supported
    )

    if not files:
        logger.warning("No supported documents found in %s", directory)
        return []

    documents: List[Document] = []
    for fp in files:
        doc = load_document(fp)
        if doc:
            documents.append(doc)
            logger.info(
                "Loaded %-40s  (%d chars)", fp.name, len(doc.page_content)
            )

    logger.info(
        "Ingested %d document(s) from %s", len(documents), directory
    )
    return documents
