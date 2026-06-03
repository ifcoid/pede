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
        
        # Clean common PDF ligatures that break DOI matching
        ligatures = {
            '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
            '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'ft', '\ufb06': 'st'
        }
        for lig, char in ligatures.items():
            doi = doi.replace(lig, char)
            
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
    """Extract title from first heading in markdown, skipping common headers."""
    lines = markdown_text.split('\n')
    
    # Generic patterns to skip
    skip_patterns = [
        r'(?i)^conference\s+on',
        r'(?i)^proceedings\s+of',
        r'(?i)^journal\s+of',
        r'(?i)^transactions\s+on',
        r'(?i)^ieee\s+',
        r'(?i)^acm\s+',
        r'(?i)^vol\.\s*\d+',
        r'(?i)^no\.\s*\d+',
        r'(?i)arxiv',
        r'(?i)^accepted\s+to',
        r'(?i)^submitted\s+to',
        r'(?i)^preprint',
        r'(?i)^copyright',
        r'(?i)^published\s+in',
        r'(?i)^\d{4}\s*$',
        r'(?i)^(data\s+and\s+)?code\s+availability$',
        r'(?i)^abstract$',
        r'(?i)^introduction$',
        r'(?i)^acknowledgments?$',
        r'(?i)^acknowledgements?$',
        r'(?i)^references?$',
        r'(?i)^conclusions?$',
        r'(?i)^appendix\s+[a-z]$',
        r'(?i)institutional\s+review\s+board',
        r'(?i)^ethics\s+statement',
        r'(?i)^funding',
        r'(?i)^declaration',
        r'(?i)competing\s+interest',
        r'(?i)^author\s+contributions?',
        r'(?i)^(data|code)\s+availability\s+statement',
        r'(?i)^consent\s+for\s+publication',
        r'(?i)^supplementary\s+material',
        r'(?i)^abbreviations?',
        r'(?i)^keywords?',
        r'(?i)^\-*\s*start\s+of\s+(picture|image)\s+text\s*\-*$',
        r'(?i)^\-*\s*end\s+of\s+(picture|image)\s+text\s*\-*$'
    ]
    
    def is_skip_header(line: str) -> bool:
        for p in skip_patterns:
            if re.search(p, line.strip()):
                return True
        return False

    def is_valid_title(candidate: str) -> bool:
        # Should be longer than 15 chars and not a known header
        if len(candidate) <= 15 or is_skip_header(candidate):
            return False
        # Should have at least 3 words (avoids just author names like 'Ali Behrouz')
        if len(candidate.split()) < 3:
            return False
        return True

    # Try H1 first
    for line in lines:
        match = re.search(r'^#\s+(.+)$', line)
        if match:
            candidate = match.group(1).strip()
            if is_valid_title(candidate):
                return candidate

    # Try first bold text
    for line in lines:
        match = re.search(r'^\*\*(.+?)\*\*', line)
        if match:
            candidate = match.group(1).strip()
            if is_valid_title(candidate):
                return candidate

    # First non-empty line
    for line in lines:
        line = line.strip()
        # Clean markdown characters for the generic fallback check
        cleaned = re.sub(r'^[\#\*\-]+\s*', '', line).strip()
        if cleaned and is_valid_title(cleaned):
            return cleaned[:200]

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


import difflib

def check_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def fetch_crossref_metadata(doi: str, expected_title: str | None = None) -> dict | None:
    """
    Fetch verified metadata from CrossRef API using DOI.
    If expected_title is provided, it validates the match.
    If the match fails (e.g., truncated DOI returning a book), it falls back to a title search.
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

        crossref_title = (data.get("title") or [None])[0]
        
        # Validasi Kecocokan Judul
        if expected_title and crossref_title:
            sim = check_similarity(expected_title, crossref_title)
            if sim < 0.7:
                logger.warning(f"CrossRef title mismatch for DOI {doi}. Expected: '{expected_title[:50]}...', Got: '{crossref_title[:50]}...' (Sim: {sim:.2f})")
                # DOI kemungkinan terpotong/salah, lanjutkan ke fallback pencarian judul
                return fallback_search_crossref_by_title(expected_title)

        result = {
            "title": crossref_title,
            "authors": authors,
            "journal": (data.get("container-title") or [None])[0],
            "publication_date": pub_date,
            "doi": data.get("DOI", doi), # Pastikan gunakan DOI resmi dari CrossRef
            "keywords": data.get("subject", []),
        }

        logger.info(f"CrossRef metadata fetched for DOI {doi}: {result['title']}")
        return result

    except Exception as e:
        logger.warning(f"CrossRef API error for DOI {doi}: {e}")
        if expected_title:
            return fallback_search_crossref_by_title(expected_title)
        return None

def fallback_search_crossref_by_title(title: str) -> dict | None:
    """
    Fallback mechanism to search CrossRef by title if DOI fails or is truncated.
    """
    try:
        logger.info(f"Attempting CrossRef fallback search for title: {title[:50]}...")
        url = f"https://api.crossref.org/works?query.title={title}&select=DOI,title,author,container-title,published-print,published-online,subject&rows=3"
        headers = {"User-Agent": "PEDE/1.0 (mailto:noreply@example.com)"}
        
        with httpx.Client(timeout=10.0) as client:
            res = client.get(url, headers=headers)
            
        if res.status_code == 200:
            items = res.json().get("message", {}).get("items", [])
            for data in items:
                crossref_title = (data.get("title") or [None])[0]
                if crossref_title and check_similarity(title, crossref_title) > 0.7:
                    # Match found! Parse details
                    authors = []
                    for author in data.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        if given and family:
                            authors.append(f"{given} {family}")
                        elif family:
                            authors.append(family)

                    pub_date = None
                    date_parts = data.get("published-print", {}).get("date-parts", [[]])[0]
                    if not date_parts:
                        date_parts = data.get("published-online", {}).get("date-parts", [[]])[0]
                    if date_parts:
                        pub_date = "-".join(str(p) for p in date_parts)

                    result = {
                        "title": crossref_title,
                        "authors": authors,
                        "journal": (data.get("container-title") or [None])[0],
                        "publication_date": pub_date,
                        "doi": data.get("DOI"),
                        "keywords": data.get("subject", []),
                    }
                    logger.info(f"Fallback search successful. Found DOI: {result['doi']}")
                    return result
        return None
    except Exception as e:
        logger.warning(f"Fallback search error: {e}")
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
    import hashlib
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

    # Provide enough text context to find DOI (up to 20,000 chars)
    full_text = pdf_native_meta.get("first_page_text", "") + "\n" + markdown_text[:20000]
    meta.doi = extract_doi(full_text)

    if not meta.authors:
        first_page = pdf_native_meta.get("first_page_text", "")
        meta.authors = extract_authors_heuristic(first_page, meta.title)

    # === Layer 3: CrossRef API (DOI-first, most accurate) ===
    if meta.doi or meta.title:
        crossref = None
        if meta.doi:
            crossref = fetch_crossref_metadata(meta.doi, expected_title=meta.title)
            
        # Jika DOI terpotong/salah dan lookup gagal, fallback ke pencarian judul
        if not crossref and meta.title:
            crossref = fallback_search_crossref_by_title(meta.title)
            
        if crossref:
            if crossref.get("doi"):
                meta.doi = crossref["doi"] # Perbarui DOI jika terpotong/ada ligature
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

    # === Generate Robust Deterministic ID ===
    # Best practice: use DOI if available. If not, use the SHA-256 hash of the file contents.
    if meta.doi:
        meta.article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, meta.doi))
    else:
        # Read file bytes to generate a unique hash for identical files with different names
        hasher = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        meta.article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hasher.hexdigest()))

    logger.info(
        f"Metadata extracted for '{meta.title}': "
        f"{len(meta.authors)} authors, DOI={meta.doi}, ID={meta.article_id}"
    )
    return meta
