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
# If the model is already cached locally, force offline mode to avoid network checks and start up instantly.
cache_dir = os.path.join(
    os.path.expanduser("~"), 
    ".cache", 
    "huggingface", 
    "hub", 
    "models--BAAI--bge-m3"
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
from core.chunker import chunk_markdown, Chunk, CHUNK_SIZE, CHUNK_OVERLAP
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
    pdf_path: str,
    vector_store: VectorStore,
    include_references: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
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
        # === Step 0: Early Deduplication Check ===
        from core.metadata_extractor import extract_doi
        import hashlib
        import uuid
        
        logger.info("[0/4] Checking early deduplication...")
        pdf_native_meta = get_pdf_native_metadata(pdf_path)
        first_page = pdf_native_meta.get("first_page_text", "")
        doi_cand = extract_doi(first_page)
        
        possible_ids = []
        if doi_cand:
            possible_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, doi_cand)))
            
        hasher = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for byte_chunk in iter(lambda: f.read(4096), b""):
                hasher.update(byte_chunk)
        file_hash_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hasher.hexdigest()))
        possible_ids.append(file_hash_id)
        
        for cand_id in possible_ids:
            existing_meta = vector_store.article_exists(cand_id)
            if existing_meta:
                title = existing_meta.get("title", "Unknown") if isinstance(existing_meta, dict) else "Unknown"
                doi = existing_meta.get("doi", "Unknown") if isinstance(existing_meta, dict) else "Unknown"
                
                logger.info(f"  -> Article already exists in DB (ID: {cand_id}).")
                logger.info(f"  -> Existing Title : {title}")
                logger.info(f"  -> Existing DOI   : {doi}")
                logger.info("  -> SKIPPING conversion to save compute! ✅")
                return ArticleMetadata(
                    article_id=cand_id, 
                    title=title if title != "Unknown" else (pdf_native_meta.get("title") or "Existing Article (Cached)"),
                    doi=doi if doi != "Unknown" else None,
                    filename=filename
                )
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

        # === Step 2.5: Post-metadata Deduplication Check ===
        # The early check (Step 0) only catches embedded DOIs / identical file hashes.
        # Some PDFs (e.g. arXiv preprint + conference version) only resolve their DOI
        # via CrossRef here, producing a shared article_id. Re-check now so we don't
        # re-embed and store a duplicate under an article_id that already exists.
        existing_meta = vector_store.article_exists(article_meta.article_id)
        if existing_meta:
            existing_title = existing_meta.get("title", article_meta.title) if isinstance(existing_meta, dict) else article_meta.title
            logger.info(f"  -> Article already exists in DB (ID: {article_meta.article_id}).")
            logger.info(f"  -> Existing Title : {existing_title}")
            logger.info("  -> SKIPPING embedding to avoid duplicate. ✅")
            return article_meta

        # Save metadata JSON for reference
        meta_filename = Path(filename).stem + ".json"
        meta_path = META_DIR / meta_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(article_meta.to_dict(), f, indent=2, ensure_ascii=False)
        
        # === Step 3: Chunking ===
        logger.info("[3/4] Chunking markdown...")
        chunks = chunk_markdown(
            markdown_text, article_meta,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        
        if not include_references:
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


def free_memory():
    """Bebaskan memori antar-PDF agar tidak menumpuk lalu di-OOM-kill (exit=-9) di Colab.

    Ingesti 60+ PDF dalam satu proses membuat cache CUDA bge-m3 terfragmentasi + objek
    Python (markdown/chunks/images) menumpuk sampai kernel kehabisan RAM/VRAM dan proses
    di-SIGKILL sebelum sempat mencetak ringkasan "Failed: N" (itu sebabnya notebook memunculkan
    "Failed=?, exit=-9"). Panggil ini setelah tiap PDF. Semua di-guard: aman tanpa torch/CUDA.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


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
        "--qdrant-path",
        default="./qdrant_db",
        help="Qdrant local DB path (default: ./qdrant_db)",
    )
    parser.add_argument(
        "--collection",
        default="scientific_articles",
        help="Qdrant collection name (default: scientific_articles)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Max chunk size in chars (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap in chars (default: {CHUNK_OVERLAP})",
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
        "--include-references",
        action="store_true",
        help="Include bibliography/reference chunks in the database (default is to skip them)",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for chunks matching a query (for testing)",
    )
    parser.add_argument(
        "--delete",
        type=str,
        default=None,
        help="Delete all chunks of an article by its article_id",
    )
    parser.add_argument(
        "--doi",
        type=str,
        default=None,
        help="Filter search results by a specific DOI (used with --search)",
    )
    
    args = parser.parse_args()
    
    # Initialize vector store
    logger.info("Initializing PEDE...")
    vector_store = VectorStore(
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
    )
    vector_store.ensure_collection()
    
    # === List mode ===
    if args.list:
        articles = vector_store.list_articles()
        if not articles:
            print("\nNo articles in collection.")
            return
        
        # Sort by total_chunks (ascending) so the smallest/broken ones appear first
        articles.sort(key=lambda x: x.get('total_chunks', 0))
        
        print(f"\n{'='*70}")
        print(f"Articles in Qdrant ({len(articles)} total) - Sorted by size (chunks)")
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
        results = vector_store.search(args.search, n_results=5, doi_filter=args.doi)
        print(f"\nSearch results for: '{args.search}'")
        if args.doi:
            print(f"Filter applied  : DOI = {args.doi}")
        print(f"{'='*70}")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n  [{i}] Score: {r['score']:.4f}")
            print(f"      Article: {meta.get('title', 'N/A')}")
            print(f"      Section: {meta.get('section_header', 'N/A')}")
            print(f"      Content: {r['content'][:200]}...")
        return
    
    # === Delete mode ===
    if args.delete:
        print(f"\nAttempting to delete article ID: {args.delete}")
        vector_store.delete_article(args.delete)
        print(f"✅ Successfully deleted article '{args.delete}' from Qdrant.")
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
        meta = process_single_pdf(
            pdf_path, vector_store, args.include_references,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        )
        results.append((pdf_path, meta))
        # Bebaskan memori antar-PDF (cegah OOM/SIGKILL exit=-9 di Colab pada batch besar).
        free_memory()
        # Ringkasan berjalan: bila proses nanti di-OOM-kill, baris terakhir ini tetap
        # memperlihatkan seberapa jauh progres (bukan "Failed=?").
        ok = sum(1 for _, m in results if m is not None)
        logger.info(f"  [progress] {i}/{len(pdf_paths)} selesai | sukses={ok} gagal={i - ok}")
    
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
