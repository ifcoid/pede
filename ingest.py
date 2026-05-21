#!/usr/bin/env python3
"""
PEDE — PDF to Model Embedding

Batch ingestion script: converts PDF scientific articles to markdown,
extracts metadata, chunks intelligently, and stores in Qdrant.

Usage:
    python ingest.py <pdf_file_or_directory>
    
Examples:
    python ingest.py paper.pdf
    python ingest.py ./papers/
    python ingest.py paper1.pdf paper2.pdf paper3.pdf
"""

import os
import sys

# === Dynamic Offline Mode for Hugging Face ===
# If the Nomic model is already cached locally, force offline mode to avoid network checks and start up instantly.
cache_dir = os.path.join(
    os.path.expanduser("~"), 
    ".cache", 
    "huggingface", 
    "hub", 
    "models--nomic-ai--nomic-embed-text-v1.5"
)
if os.path.exists(cache_dir):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import json
import logging
import argparse
from pathlib import Path

from core.pdf_converter import convert_pdf_to_markdown, get_pdf_native_metadata
from core.metadata_extractor import extract_metadata, ArticleMetadata
from core.chunker import chunk_markdown, Chunk
from core.vector_store import VectorStore

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Directories ===
DATA_DIR = Path("./data")
MARKDOWN_DIR = DATA_DIR / "markdown"
IMAGE_DIR = DATA_DIR / "images"
META_DIR = DATA_DIR / "metadata"


def setup_directories():
    """Create necessary directories."""
    for d in [DATA_DIR, MARKDOWN_DIR, IMAGE_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def process_single_pdf(
    pdf_path: str, vector_store: VectorStore, skip_references: bool = False
) -> ArticleMetadata | None:
    """
    Process a single PDF through the full pipeline:
    PDF -> Markdown -> Metadata -> Chunks -> Qdrant
    
    Returns ArticleMetadata on success, None on failure.
    """
    filename = os.path.basename(pdf_path)
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Processing: {filename}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # === Step 1: PDF -> Markdown ===
        logger.info("[1/4] Converting PDF to Markdown...")
        markdown_text = convert_pdf_to_markdown(
            pdf_path,
            image_dir=str(IMAGE_DIR),
            write_images=True,
        )
        
        # Save markdown for reference
        md_filename = Path(filename).stem + ".md"
        md_path = MARKDOWN_DIR / md_filename
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info(f"  -> Markdown saved: {md_path} ({len(markdown_text):,} chars)")
        
        # === Step 2: Extract Metadata ===
        logger.info("[2/4] Extracting metadata...")
        pdf_native_meta = get_pdf_native_metadata(pdf_path)
        article_meta = extract_metadata(pdf_path, markdown_text, pdf_native_meta)
        
        logger.info(f"  -> Title:   {article_meta.title}")
        logger.info(f"  -> Authors: {', '.join(article_meta.authors) if article_meta.authors else 'Unknown'}")
        logger.info(f"  -> DOI:     {article_meta.doi or 'Not found'}")
        logger.info(f"  -> Pages:   {article_meta.total_pages}")
        
        # Save metadata JSON for reference
        meta_filename = Path(filename).stem + ".json"
        meta_path = META_DIR / meta_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(article_meta.to_dict(), f, indent=2, ensure_ascii=False)
        
        # === Step 3: Chunking ===
        logger.info("[3/4] Chunking markdown...")
        chunks = chunk_markdown(markdown_text, article_meta)
        
        if skip_references:
            original_count = len(chunks)
            chunks = [c for c in chunks if c.content_type != "references"]
            logger.info(f"  -> Skipped {original_count - len(chunks)} reference chunks.")
        
        
        # Log chunk stats
        sections = set(c.section_header for c in chunks)
        types = {}
        for c in chunks:
            types[c.content_type] = types.get(c.content_type, 0) + 1
        
        logger.info(f"  -> {len(chunks)} chunks created")
        logger.info(f"  -> Sections: {', '.join(sorted(sections))}")
        logger.info(f"  -> Types: {types}")
        
        # === Step 4: Store in Qdrant ===
        logger.info("[4/4] Embedding & storing in Qdrant...")
        stored = vector_store.add_chunks(chunks)
        
        elapsed = time.time() - start_time
        logger.info(f"  -> {stored} chunks stored in Qdrant")
        logger.info(f"  -> Total time: {elapsed:.1f}s")
        logger.info(f"  -> Article ID: {article_meta.article_id}")
        
        return article_meta
    
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}", exc_info=True)
        return None


def collect_pdf_paths(paths: list[str]) -> list[str]:
    """Collect all PDF paths from files and directories."""
    pdf_paths = []
    
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdf_paths.append(str(p.resolve()))
        elif p.is_dir():
            for pdf_file in sorted(p.glob("**/*.pdf")):
                pdf_paths.append(str(pdf_file.resolve()))
        else:
            logger.warning(f"Skipping: {path} (not a PDF file or directory)")
    
    return pdf_paths


def main():
    parser = argparse.ArgumentParser(
        description="PEDE — PDF to Model Embedding. Ingest scientific PDFs into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py paper.pdf                   # Single file
  python ingest.py ./papers/                    # All PDFs in directory
  python ingest.py paper1.pdf paper2.pdf        # Multiple files
  python ingest.py ./papers/ --list             # List articles in Qdrant
"""
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF file(s) or directory containing PDFs",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant server host (default: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)",
    )
    parser.add_argument(
        "--collection",
        default="scientific_articles",
        help="Qdrant collection name (default: scientific_articles)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max chunk size in chars (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in chars (default: 200)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all articles currently in Qdrant",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection statistics",
    )
    parser.add_argument(
        "--skip-references",
        action="store_true",
        help="Exclude bibliography/reference chunks from being stored in the database",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for chunks matching a query (for testing)",
    )
    
    args = parser.parse_args()
    
    # Initialize vector store
    logger.info("Initializing PEDE...")
    vector_store = VectorStore(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
    )
    vector_store.ensure_collection()
    
    # === List mode ===
    if args.list:
        articles = vector_store.list_articles()
        if not articles:
            print("\nNo articles in collection.")
            return
        
        print(f"\n{'='*70}")
        print(f"Articles in Qdrant ({len(articles)} total)")
        print(f"{'='*70}")
        for i, article in enumerate(articles, 1):
            print(f"\n  [{i}] {article['title']}")
            print(f"      Authors: {article['authors']}")
            print(f"      DOI:     {article['doi'] or 'N/A'}")
            print(f"      Chunks:  {article['total_chunks']}")
            print(f"      ID:      {article['article_id']}")
        print()
        return
    
    # === Info mode ===
    if args.info:
        info = vector_store.get_collection_info()
        print(f"\nCollection: {info['name']}")
        print(f"  Points:  {info['points_count']}")
        print(f"  Vectors: {info['vectors_count']}")
        print(f"  Status:  {info['status']}")
        return
    
    # === Search mode (for testing) ===
    if args.search:
        results = vector_store.search(args.search, n_results=5)
        print(f"\nSearch results for: '{args.search}'")
        print(f"{'='*70}")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n  [{i}] Score: {r['score']:.4f}")
            print(f"      Article: {meta.get('title', 'N/A')}")
            print(f"      Section: {meta.get('section_header', 'N/A')}")
            print(f"      Content: {r['content'][:200]}...")
        return
    
    # === Ingest mode ===
    if not args.paths:
        parser.print_help()
        sys.exit(1)
    
    setup_directories()
    
    pdf_paths = collect_pdf_paths(args.paths)
    if not pdf_paths:
        logger.error("No PDF files found in the specified paths")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_paths)} PDF(s) to process")
    
    # Process each PDF
    results = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"\n[{i}/{len(pdf_paths)}]")
        meta = process_single_pdf(pdf_path, vector_store, args.skip_references)
        results.append((pdf_path, meta))
    
    # === Summary ===
    success = sum(1 for _, m in results if m is not None)
    failed = sum(1 for _, m in results if m is None)
    
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {len(results)}")
    print(f"  Success:   {success}")
    print(f"  Failed:    {failed}")
    
    if success > 0:
        print(f"\n  Articles ingested:")
        for path, meta in results:
            if meta:
                print(f"    [OK] {meta.title}")
                print(f"      -> {meta.total_chunks} chunks, ID: {meta.article_id}")
        
        # Show collection info
        info = vector_store.get_collection_info()
        print(f"\n  Collection total: {info['points_count']} chunks")
    
    if failed > 0:
        print(f"\n  Failed files:")
        for path, meta in results:
            if meta is None:
                print(f"    [FAIL] {os.path.basename(path)}")
    
    print()


if __name__ == "__main__":
    main()
