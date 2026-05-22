# 📄 PEDE — PDF to Model Embedding

Pipeline CLI untuk mengkonversi artikel ilmiah PDF ke vector embeddings di Qdrant.

```
PDF → Markdown → Smart Chunking + Metadata → Embedding → Qdrant Vector DB
```

cek hasil chunking:

```sh
python dump_chunks.py --doi "10.1016/j.inpa.2026.02.006"
```

> **📖 BACA DOKUMENTASI LENGKAP API:** Silakan cek file [API_REFERENCE.md](API_REFERENCE.md) untuk melihat daftar lengkap *endpoint* dan cara melakukan RAG via HTTP!

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Qdrant (Lokal vs Cloud)

Secara bawaan (*default*), *database* akan disimpan di folder lokal `./qdrant_db`. 
Namun, jika Anda ingin menggunakan **Qdrant Cloud** untuk skalabilitas (agar Colab dan Lokal terhubung ke *database* yang sama), Anda cukup menyalin file `.env`:

```bash
cp .env.example .env
```
Kemudian isi file `.env` tersebut dengan *Endpoint URL* dan *API Key* Anda:
```ini
QDRANT_URL="https://xxx.cloud.qdrant.io"
QDRANT_API_KEY="api_key_anda"
```

### 3. Ingesting PDFs

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
| Embedding | `sentence-transformers` (BAAI/bge-m3) | 1024-dim vectors (8192 context, Multi-lingual) |
| Storage | Qdrant | Vectors + rich payload metadata |

## 🌟 Advanced SOTA Features (Baru)
1. **Content-Based Deduplication**: Mencegah duplikasi artikel walaupun nama file PDF diubah-ubah. ID artikel dihasilkan secara deterministik menggunakan kombinasi DOI artikel atau _SHA-256 Byte Hash_ dari file.
2. **Page Boundary Stitching**: Otomatis menghapus nomor halaman dan _header/footer_ yang menyela kalimat di tengah perpindahan halaman PDF, lalu menyambungkan kalimat yang terputus.
3. **Reference Dropping**: Otomatis melewati (skip) bagian Daftar Pustaka untuk mencegah polusi _Semantic Search_ (kecuali flag `--include-references` diaktifkan).
4. **Table Cleanup**: Membersihkan artefak ekstraksi tabel untuk membantu LLM bernalar pada data sel.

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
  --qdrant-path PATH     Qdrant local DB path (default: ./qdrant_db)
  --collection NAME      Collection name (default: scientific_articles)
  --chunk-size N         Max chunk size in chars (default: 1000)
  --chunk-overlap N      Chunk overlap in chars (default: 200)
  --list                 List articles in Qdrant
  --info                 Show collection stats
  --search QUERY         Test search
  --doi DOI              Filter search results by DOI
  --include-references   Include references (default is to SKIP them)
```

**Contoh Pencarian via CLI:**
```bash
# Pencarian global (semua jurnal)
python ingest.py --search "Apa itu neurosymbolic?"

# Pencarian spesifik ke 1 jurnal menggunakan DOI
python ingest.py --search "Apa hasil eksperimennya?" --doi "10.1016/j.inpa.2026.02.006"
```

## 🤖 Integration with Golang Agentic AI

Proyek ini telah dilengkapi dengan purwarupa **Agentic AI** berbasis Golang di dalam folder `agent-go/`. 

Agen Golang ini menggunakan SDK `github.com/google/generative-ai-go` dan dilengkapi kemampuan **Function Calling** (Tools). Ia tidak memanggil Qdrant secara langsung, melainkan menggunakan API Server Python (`api.py`) sebagai jembatan.

### Arsitektur Agentic RAG
1. **User Prompt:** Anda bertanya *"Apa hasil eksperimen jurnal X?"* di terminal Golang.
2. **Gemini Reasoning:** LLM Gemini menyadari bahwa itu adalah pertanyaan akademis, lalu ia memutuskan untuk menggunakan fungsi `query_scientific_database`.
3. **Golang Action:** Golang menangkap permintaan fungsi tersebut, lalu mengirim HTTP POST `{"query": "...", "doi": "..."}` ke `http://localhost:8000/search`.
4. **Python RAG:** FastAPI meng-embed *query* via BGE-M3, mencari 5 *chunks* terdekat di Qdrant, dan mengembalikannya ke Golang.
5. **Synthesis:** Golang menyodorkan 5 *chunks* tersebut ke Gemini, dan Gemini merangkumnya menjadi jawaban akhir yang sangat akurat.

### Cara Menjalankan Agen Golang
1. Pastikan server API Python berjalan:
   ```bash
   uvicorn api:app --port 8000
   ```
2. Buka terminal baru, masuk ke folder `agent-go`:
   ```bash
   cd agent-go
   ```
3. Set *environment variable* untuk Gemini API Key Anda:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="AIzaSy..."
   ```
4. Jalankan agen:
   ```bash
   go run .
   ```

Selamat bereksperimen dengan Agentic RAG Anda!

## License

GNU GPL v3
