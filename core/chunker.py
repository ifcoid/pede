"""
Hybrid Smart Chunker for Scientific Articles

Tier 1: deterministic markdown header split — split by section headers
Tier 2: RecursiveCharacterTextSplitter — fallback for large sections

Each chunk gets enriched metadata for precise retrieval.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.metadata_extractor import ArticleMetadata

logger = logging.getLogger(__name__)

# === Configuration ===
CHUNK_SIZE = 2500       # chars (~600 tokens)
CHUNK_OVERLAP = 400     # chars (~100 tokens), ~15% overlap
MIN_CHUNK_SIZE = 100    # Ignore chunks smaller than this

# Level keys for the (up to 3-deep) header hierarchy, matching the old
# MarkdownHeaderTextSplitter metadata shape ({"h1": ..., "h2": ..., "h3": ...}).
_LEVEL_KEY = {1: "h1", 2: "h2", 3: "h3"}

# A markdown header line: 1-3 leading '#', a space, then the title text.
_HEADER_RE = re.compile(r'^(#{1,3})\s+(.*\S)\s*$')

# A whole-line bold span, e.g. "**1 Introduction**" or "**Conclusion**".
_BOLD_LINE_RE = re.compile(r'^\*\*\s*(.+?)\s*\*\*$')

# A numbered heading like "1 Introduction", "2.1 Data", "3.4.2 Foo".
_NUMBERED_HEADING_RE = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+\S')

# Common scientific-article section titles (used to promote un-prefixed
# bold pseudo-headings into real markdown headers).
_SECTION_KEYWORDS_RE = re.compile(
    r'(?i)^(abstract|keywords?|introduction|related\s+works?|background|'
    r'preliminaries|materials?(\s+and\s+methods?)?|methods?|methodology|'
    r'approach|experiments?|experimental\s+(setup|details?)|results?'
    r'(\s+and\s+discussions?)?|evaluation|discussions?|ablation(\s+study)?|'
    r'analysis|conclusions?(\s+and\s+(discussions?|future\s+work))?|'
    r'future\s+work|acknowledge?ments?|author\s+contributions?|'
    r'references|bibliography|appendix)\b'
)

# Markers that reveal a "header" is actually an author byline / affiliation
# line (noise from the PDF→markdown converter), not a real section.
_AFFILIATION_MARKER_RE = re.compile(r'\[\s*\d')  # superscript affiliation: [1], [1, 2], [1,2,*]
_AFFILIATION_KEYWORDS_RE = re.compile(
    r'(?i)\b(universit|departe?ment|institut|laborator|facult|fakultas|'
    r'school\s+of|college|academy|corresponding\s+author)\b'
)


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


def _clean_header(text: str) -> str:
    """Strip markdown decoration (#, *, _, whitespace) from a header title."""
    if not text:
        return text
    return re.sub(r'^[#\s*_]+', '', re.sub(r'[\s*_]+$', '', text)).strip()


def _is_noise_header(title: str) -> bool:
    """True jika 'header' sebenarnya baris penulis/afiliasi, bukan section.

    Converter PDF→markdown kadang menandai baris penulis/afiliasi sebagai
    heading (mis. ``Weining Weng[1, 2], Yang Gu[1, 2,*] ...``). Baris seperti
    ini diperlakukan sebagai konten biasa, bukan section header.
    """
    if not title:
        return False
    if "@" in title:                              # email
        return True
    if _AFFILIATION_MARKER_RE.search(title):      # [1], [1, 2], [1,2,*]
        return True
    if _AFFILIATION_KEYWORDS_RE.search(title):    # university/department/...
        return True
    return False


def _build_section_hierarchy(header_metadata: dict) -> str:
    """Build section hierarchy string from header metadata."""
    parts = []
    for key in ["h1", "h2", "h3"]:
        if key in header_metadata and header_metadata[key]:
            parts.append(header_metadata[key])
    return " > ".join(parts)


def _get_deepest_header(header_metadata: dict) -> str:
    """Get the most specific (deepest) section header (h3 > h2 > h1)."""
    for key in ["h3", "h2", "h1"]:
        if header_metadata.get(key):
            return header_metadata[key]
    return "Unknown"


def _promote_bold_headings(markdown_text: str) -> str:
    """Promote standalone bold lines that look like section headings into real
    markdown headers.

    Some OCR/PDF->markdown conversions render section titles as bold text on
    their own line (e.g. ``**1 Introduction**`` or ``**Conclusion**``) instead
    of ``## 1 Introduction``. Without a ``#`` prefix the document has no
    structure to split on. This pre-pass rewrites those lines so header
    splitting can recover the section layout. Lines that already start with
    ``#`` are left untouched.
    """
    out_lines = []
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        m = _BOLD_LINE_RE.match(stripped)
        if m:
            inner = m.group(1).strip()
            # Section titles are short; skip fully-bolded sentences/captions.
            if inner and len(inner) <= 80 and "|" not in inner:
                num = _NUMBERED_HEADING_RE.match(inner)
                if num:
                    depth = num.group(1).count(".") + 1
                    level = "#" * min(max(depth + 1, 2), 3)  # top-level -> h2
                    out_lines.append(f"{level} {inner}")
                    continue
                if _SECTION_KEYWORDS_RE.match(inner):
                    out_lines.append(f"## {inner}")
                    continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _split_markdown_by_headers(markdown_text: str) -> list[tuple[str, dict]]:
    """Deterministically split markdown into sections on ``#``/``##``/``###``
    header lines, returning ``(section_text, {"h1":..., "h2":..., "h3":...})``.

    This replaces langchain's ``MarkdownHeaderTextSplitter`` because its
    section detection varies across library versions (e.g. code-fence
    tracking that can swallow all later headers into a single section). A
    plain line-by-line scan is version-independent and predictable.

    Header lines are kept at the start of their section (equivalent to
    ``strip_headers=False``); header titles in the metadata are cleaned of
    markdown decoration.
    """
    sections: list[tuple[str, dict]] = []
    meta: dict = {}
    buf: list[str] = []

    def flush():
        content = "\n".join(buf).strip()
        if content:
            sections.append((content, dict(meta)))

    for line in markdown_text.split("\n"):
        m = _HEADER_RE.match(line)
        title = _clean_header(m.group(2)) if m else ""
        # Treat author/affiliation byline as body, not a section boundary.
        if m and not _is_noise_header(title):
            flush()
            buf = [line]  # keep the header line in the section body
            level = len(m.group(1))
            # Replace this level and clear any deeper levels.
            for lv in (1, 2, 3):
                if lv >= level:
                    meta.pop(_LEVEL_KEY[lv], None)
            if title:
                meta[_LEVEL_KEY[level]] = title
        else:
            buf.append(line)
    flush()

    # No headers at all -> one big section so Tier-2 still chunks the text.
    if not sections and markdown_text.strip():
        sections.append((markdown_text.strip(), {}))

    return sections


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

    # === Tier 0: Normalize headings ===
    # Recover section structure when the converter emitted bold pseudo-headings
    # (e.g. "**1 Introduction**") instead of real "## " markdown headers.
    markdown_text = _promote_bold_headings(markdown_text)

    # === Tier 1: Split by markdown headers (deterministic, version-independent) ===
    header_docs = _split_markdown_by_headers(markdown_text)

    # === Tier 2: Split large sections ===
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []

    for text, header_meta in header_docs:
        text = text.strip()

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
