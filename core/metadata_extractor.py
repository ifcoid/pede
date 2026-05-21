"""
Metadata Extractor — 3-Layer Strategy

Layer 1: PDF native metadata (via PyMuPDF)
Layer 2: Regex/heuristic dari markdown text
Layer 3: CrossRef API via DOI lookup (paling akurat)
"""

import re
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ArticleMetadata:
    """Schema metadata untuk satu artikel ilmiah."""
    # === Identifiers ===
    article_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""

    # === Bibliographic ===
    title: str = "Untitled"
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    doi: str | None = None
    publication_date: str | None = None
    journal: str | None = None
    keywords: list[str] = field(default_factory=list)

    # === Processing Info ===
    upload_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_pages: int = 0
    total_chunks: int = 0  # Filled after chunking

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = asdict(self)
        d["authors"] = ", ".join(self.authors) if self.authors else ""
        d["keywords"] = ", ".join(self.keywords) if self.keywords else ""
        return d


def extract_doi(text: str) -> str | None:
    """Extract DOI from text using regex."""
    pattern = r'10\.\d{4,9}/[^\s,;\]\)]+'
    match = re.search(pattern, text)
    if match:
        doi = match.group(0).rstrip('.')
        logger.info(f"DOI found: {doi}")
        return doi
    return None


def extract_abstract(markdown_text: str) -> str:
    """Extract abstract from markdown text using heuristics."""
    patterns = [
        r'(?:^|\n)#+\s*Abstract\s*\n(.*?)(?=\n#+\s|\n\*\*|$)',
        r'(?:^|\n)\*\*Abstract\*\*[:\s]*(.*?)(?=\n#+\s|\n\*\*|$)',
        r'(?i)abstract[:\s\-]*\n?(.*?)(?=\n(?:#{1,3}\s|\d+\.?\s+Introduction|Keywords|\*\*1))',
    ]

    for pattern in patterns:
        match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]

    return ""


def extract_title_from_markdown(markdown_text: str) -> str:
    """Extract title from first heading in markdown."""
    # Try H1 first
    match = re.search(r'^#\s+(.+)$', markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Try first bold text
    match = re.search(r'^\*\*(.+?)\*\*', markdown_text, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        if len(title) > 10:
            return title

    # First non-empty line
    for line in markdown_text.split('\n'):
        line = line.strip()
        if line and len(line) > 10:
            return line[:200]

    return "Untitled"


def extract_authors_heuristic(first_page_text: str, title: str) -> list[str]:
    """Try to extract authors from the text between title and abstract."""
    lines = first_page_text.split('\n')
    author_candidates = []
    found_title = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if title and title[:30].lower() in line.lower():
            found_title = True
            continue
        if found_title:
            if re.match(r'(?i)(abstract|introduction|\d+\.)', line):
                break
            if len(line) < 200 and not line.startswith('#'):
                author_candidates.append(line)
            if len(author_candidates) >= 5:
                break

    authors = []
    for candidate in author_candidates[:3]:
        if re.search(r'[@\d]|university|department|institute', candidate, re.IGNORECASE):
            continue
        names = re.split(r',|\band\b', candidate)
        for name in names:
            name = name.strip().strip('*').strip('\u2020').strip('\u2021')
            if name and 2 <= len(name.split()) <= 5 and not re.search(r'[{}@#$%^&()\[\]]', name):
                authors.append(name)

    return authors


def fetch_crossref_metadata(doi: str) -> dict | None:
    """
    Fetch verified metadata from CrossRef API using DOI.
    This is the most accurate metadata source.
    """
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            "User-Agent": "PEDE/1.0 (PDF Embedding Pipeline; mailto:noreply@example.com)"
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)

        if response.status_code != 200:
            logger.warning(f"CrossRef API returned {response.status_code} for DOI {doi}")
            return None

        data = response.json().get("message", {})

        # Parse authors
        authors = []
        for author in data.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)

        # Parse publication date
        pub_date = None
        date_parts = (
            data.get("published-print", {})
            .get("date-parts", [[]])[0]
        )
        if not date_parts:
            date_parts = (
                data.get("published-online", {})
                .get("date-parts", [[]])[0]
            )
        if date_parts:
            pub_date = "-".join(str(p) for p in date_parts)

        result = {
            "title": (data.get("title") or [None])[0],
            "authors": authors,
            "journal": (data.get("container-title") or [None])[0],
            "publication_date": pub_date,
            "doi": doi,
            "keywords": data.get("subject", []),
        }

        logger.info(f"CrossRef metadata fetched for DOI {doi}: {result['title']}")
        return result

    except Exception as e:
        logger.warning(f"CrossRef API error for DOI {doi}: {e}")
        return None


def extract_metadata(
    pdf_path: str,
    markdown_text: str,
    pdf_native_meta: dict,
) -> ArticleMetadata:
    """
    3-layer metadata extraction:
    1. PDF native metadata
    2. Regex/heuristic from markdown
    3. CrossRef API (if DOI found)
    """
    import os
    filename = os.path.basename(pdf_path)

    meta = ArticleMetadata(
        filename=filename,
        total_pages=pdf_native_meta.get("page_count", 0),
    )

    # === Layer 1: PDF Native Metadata ===
    if pdf_native_meta.get("title"):
        meta.title = pdf_native_meta["title"]
    if pdf_native_meta.get("author"):
        meta.authors = [a.strip() for a in pdf_native_meta["author"].split(",")]
    if pdf_native_meta.get("keywords"):
        meta.keywords = [k.strip() for k in pdf_native_meta["keywords"].split(",")]

    # === Layer 2: Regex/Heuristic from Markdown ===
    if meta.title == "Untitled" or not meta.title:
        meta.title = extract_title_from_markdown(markdown_text)

    meta.abstract = extract_abstract(markdown_text)

    full_text = pdf_native_meta.get("first_page_text", "") + "\n" + markdown_text[:5000]
    meta.doi = extract_doi(full_text)

    if not meta.authors:
        first_page = pdf_native_meta.get("first_page_text", "")
        meta.authors = extract_authors_heuristic(first_page, meta.title)

    # === Layer 3: CrossRef API (DOI-first, most accurate) ===
    if meta.doi:
        crossref = fetch_crossref_metadata(meta.doi)
        if crossref:
            if crossref.get("title"):
                meta.title = crossref["title"]
            if crossref.get("authors"):
                meta.authors = crossref["authors"]
            if crossref.get("journal"):
                meta.journal = crossref["journal"]
            if crossref.get("publication_date"):
                meta.publication_date = crossref["publication_date"]
            if crossref.get("keywords"):
                meta.keywords = crossref["keywords"]

    logger.info(
        f"Metadata extracted for '{meta.title}': "
        f"{len(meta.authors)} authors, DOI={meta.doi}"
    )
    return meta
