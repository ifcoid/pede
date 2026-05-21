# 📄 PEDE — PDF to Model Embedding

Pipeline CLI untuk mengkonversi artikel ilmiah PDF ke vector embeddings di Qdrant.

```
PDF → Markdown → Smart Chunking + Metadata → Embedding → Qdrant Vector DB
```

## Quick Start

### 1. Start Qdrant (Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest PDFs

```bash
# Single file
python ingest.py paper.pdf

# Entire directory
python ingest.py ./papers/

# Multiple files
python ingest.py paper1.pdf paper2.pdf paper3.pdf
```

### 4. Check Results

```bash
# List all ingested articles
python ingest.py --list

# Collection statistics
python ingest.py --info

# Test search
python ingest.py --search "neurosymbolic AI"
```

## Architecture

| Stage | Tool | Output |
|-------|------|--------|
| PDF → Markdown | `pymupdf4llm` | Structured markdown with headings |
| Metadata Extraction | 3-layer (PDF + Regex + CrossRef API) | Title, authors, DOI, abstract, etc. |
| Chunking | Hybrid (Header + Recursive) | ~1000 char chunks with section metadata |
| Embedding | `sentence-transformers` (nomic-embed-text-v1.5) | 768-dim vectors (8192 context) |
| Storage | Qdrant | Vectors + rich payload metadata |

## Chunk Metadata

Each chunk stored in Qdrant carries:

- `article_id` — UUID per artikel (untuk filter retrieval)
- `title`, `authors`, `doi` — identitas artikel
- `section_header` — "Introduction", "Methods", "Results", dll
- `section_hierarchy` — "Methods > Data Collection > Survey"
- `content_type` — "text", "table", "references", "figure_caption"
- `chunk_index` / `total_chunks` — posisi dalam dokumen

## CLI Options

```
python ingest.py [paths] [options]

positional:
  paths                  PDF file(s) or directory

options:
  --qdrant-host HOST     Qdrant host (default: localhost)
  --qdrant-port PORT     Qdrant port (default: 6333)
  --collection NAME      Collection name (default: scientific_articles)
  --chunk-size N         Max chunk size in chars (default: 1000)
  --chunk-overlap N      Chunk overlap in chars (default: 200)
  --list                 List articles in Qdrant
  --info                 Show collection stats
  --search QUERY         Test search
```

## License

GNU GPL v3
