"""
Hybrid Smart Chunker for Scientific Articles

Tier 1: MarkdownHeaderTextSplitter — split by section headers
Tier 2: RecursiveCharacterTextSplitter — fallback for large sections

Each chunk gets enriched metadata for precise retrieval.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.metadata_extractor import ArticleMetadata

logger = logging.getLogger(__name__)

# === Configuration ===
CHUNK_SIZE = 1000       # chars (~250 tokens)
CHUNK_OVERLAP = 200     # chars (~50 tokens), ~20% overlap
MIN_CHUNK_SIZE = 100    # Ignore chunks smaller than this

# Headers to split on
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@dataclass
class Chunk:
    """A single chunk with content and metadata."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""

    # === Article reference ===
    article_id: str = ""
    title: str = ""
    authors: str = ""  # Comma-separated for Qdrant string storage
    doi: str = ""

    # === Chunk position ===
    chunk_index: int = 0
    total_chunks: int = 0
    section_header: str = ""
    section_hierarchy: str = ""

    # === Content flags ===
    content_type: str = "text"  # text, table, references, figure_caption
    has_table: bool = False
    has_citation: bool = False

    def to_metadata_dict(self) -> dict:
        """Convert to flat dict for Qdrant payload."""
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi or "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "section_header": self.section_header,
            "section_hierarchy": self.section_hierarchy,
            "content_type": self.content_type,
            "has_table": self.has_table,
            "has_citation": self.has_citation,
        }


def _detect_content_type(text: str, section_header: str = "") -> str:
    """Detect what kind of content this chunk contains."""
    if section_header and re.match(r'(?i)^[\s\*\-_]*(references|bibliography|daftar pustaka)', section_header):
        return "references"
        
    if re.search(r'\|.+\|.+\|', text) and text.count('|') > 5:
        return "table"
    if re.match(r'(?i)^[\s\*\-_]*(references|bibliography|daftar pustaka)', text):
        return "references"
    if re.match(r'(?i)^\s*(figure|fig\.?)\s+\d', text):
        return "figure_caption"
    return "text"


def _has_table(text: str) -> bool:
    """Check if text contains a markdown table."""
    return bool(re.search(r'\|.+\|.+\|', text) and text.count('|') > 5)


def _has_citation(text: str) -> bool:
    """Check if text contains academic citations."""
    return bool(re.search(
        r'\[\d+\]|\[\d+[,;\s]+\d+\]|\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)',
        text
    ))


def _build_section_hierarchy(header_metadata: dict) -> str:
    """Build section hierarchy string from header metadata."""
    parts = []
    for key in ["h1", "h2", "h3"]:
        if key in header_metadata and header_metadata[key]:
            parts.append(header_metadata[key])
    return " > ".join(parts)


def _get_deepest_header(header_metadata: dict) -> str:
    """Get the most specific (deepest) section header."""
    for key in reversed(["h3", "h2", "h1"]):
        if key in header_metadata and header_metadata[key]:
            return header_metadata[key]
    return "Unknown"


def chunk_markdown(
    markdown_text: str,
    article_meta: ArticleMetadata,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[Chunk]:
    """
    Hybrid 2-tier chunking:

    Tier 1: Split by markdown headers (##, ###) to respect document structure
    Tier 2: Further split large sections using recursive character splitting

    Args:
        markdown_text: Full markdown text of the article
        article_meta: Metadata of the parent article
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size (smaller chunks are discarded)

    Returns:
        List of Chunk objects with enriched metadata
    """
    logger.info(f"Chunking article: {article_meta.title}")

    # === Tier 1: Split by markdown headers ===
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(markdown_text)

    # === Tier 2: Split large sections ===
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []

    for doc in header_docs:
        text = doc.page_content.strip()
        header_meta = doc.metadata

        if len(text) < min_chunk_size:
            continue

        section_header = _get_deepest_header(header_meta)
        section_hierarchy = _build_section_hierarchy(header_meta)

        if len(text) <= chunk_size:
            chunk = Chunk(
                content=text,
                article_id=article_meta.article_id,
                title=article_meta.title,
                authors=", ".join(article_meta.authors),
                doi=article_meta.doi or "",
                section_header=section_header,
                section_hierarchy=section_hierarchy,
                content_type=_detect_content_type(text, section_header),
                has_table=_has_table(text),
                has_citation=_has_citation(text),
            )
            chunks.append(chunk)
        else:
            sub_docs = recursive_splitter.split_text(text)
            for sub_text in sub_docs:
                sub_text = sub_text.strip()
                if len(sub_text) < min_chunk_size:
                    continue

                chunk = Chunk(
                    content=sub_text,
                    article_id=article_meta.article_id,
                    title=article_meta.title,
                    authors=", ".join(article_meta.authors),
                    doi=article_meta.doi or "",
                    section_header=section_header,
                    section_hierarchy=section_hierarchy,
                    content_type=_detect_content_type(sub_text, section_header),
                    has_table=_has_table(sub_text),
                    has_citation=_has_citation(sub_text),
                )
                chunks.append(chunk)

    # Set chunk indices
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.total_chunks = len(chunks)

    # Update article metadata with chunk count
    article_meta.total_chunks = len(chunks)

    logger.info(
        f"Chunking complete: {len(chunks)} chunks created "
        f"(avg {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} chars/chunk)"
    )

    return chunks
