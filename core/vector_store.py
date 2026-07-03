"""
Qdrant Vector Store Operations — Hybrid (Dense + Sparse) dengan BGE-M3.

Memakai FlagEmbedding `BGEM3FlagModel` untuk menghasilkan vektor **dense** (1024-d)
dan **sparse/lexical** sekaligus dalam satu kali encode, menyimpan keduanya di
Qdrant sebagai *named vectors* ("dense" + "sparse"), lalu melakukan **hybrid
search** dengan RRF fusion — memanfaatkan kekuatan M3 untuk istilah eksak
(nama model, kode dataset, DOI) sekaligus kemiripan semantik.

Setiap chunk di-embed dengan **konteks** (judul + section header) diprepend ke
isi, agar sub-chunk yang panjang tidak kehilangan konteks section.

Fallback: bila FlagEmbedding tidak tersedia, otomatis memakai SentenceTransformer
dense-only (tetap named vector "dense"), sehingga pipeline tetap jalan.
"""

import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load env variables (for Qdrant Cloud)
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    FilterSelector,
    Prefetch,
    FusionQuery,
    Fusion,
)

from core.chunker import Chunk
from core.retry import retry_call

logger = logging.getLogger(__name__)

# === Configuration ===
COLLECTION_NAME = "scientific_articles"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 8192 context, 1024-d, multilingual, hybrid-capable
DENSE_DIM = 1024
MAX_LENGTH = 8192  # full BGE-M3 context window

# Named vectors in Qdrant
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Qdrant Database Settings (Local or Cloud)
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_db")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def _to_list(vec) -> list:
    """Konversi vektor (numpy / list) ke list float."""
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


class VectorStore:
    """
    Mengelola embedding (hybrid dense+sparse) dan penyimpanan di Qdrant.

    Usage:
        store = VectorStore()
        store.ensure_collection()
        store.add_chunks(chunks)
    """

    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        logger.info("Initializing VectorStore...")
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.hybrid = False          # True jika sparse tersedia (FlagEmbedding)
        self.backend = "unknown"

        self._load_model(embedding_model)

        # Connect to Cloud if URL is provided, otherwise use Local DB
        if QDRANT_URL and QDRANT_API_KEY:
            # Guard: kesalahan umum adalah menukar QDRANT_URL <-> QDRANT_API_KEY.
            # Validasi sebelum konek, dan JANGAN cetak nilai URL/key ke log
            # (cegah kebocoran key jika tertukar).
            if not QDRANT_URL.lower().startswith(("http://", "https://")):
                raise ValueError(
                    "QDRANT_URL tidak valid: harus diawali 'http://' atau 'https://' "
                    "(mis. https://xxx.qdrant.io). Kemungkinan tertukar dengan "
                    "QDRANT_API_KEY - periksa kembali Colab Secrets / environment."
                )
            if QDRANT_API_KEY.lower().startswith(("http://", "https://")):
                raise ValueError(
                    "QDRANT_API_KEY tampak berisi URL - kemungkinan tertukar dengan "
                    "QDRANT_URL. Periksa kembali Colab Secrets / environment."
                )
            host = urlparse(QDRANT_URL).hostname or "?"
            self._qdrant_host = host
            logger.info(f"Connecting to Qdrant Cloud at {host}")  # host saja, bukan key
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self._qdrant_host = qdrant_path
            logger.info(f"Connecting to Qdrant Local DB at {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)

        logger.info(
            f"VectorStore ready. Backend={self.backend}, hybrid={self.hybrid}, "
            f"dim={self.vector_size}"
        )

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self, embedding_model: str):
        """Coba FlagEmbedding (hybrid); jika gagal, fallback SentenceTransformer."""
        is_m3 = "m3" in embedding_model.lower()

        if is_m3:
            try:
                import torch
                use_fp16 = torch.cuda.is_available()
                from FlagEmbedding import BGEM3FlagModel

                self.model = BGEM3FlagModel(embedding_model, use_fp16=use_fp16)
                self.backend = "FlagEmbedding"
                self.hybrid = True
                self.vector_size = DENSE_DIM
                logger.info(
                    f"Loaded BGE-M3 via FlagEmbedding (hybrid dense+sparse, fp16={use_fp16})."
                )
                return
            except Exception as e:
                logger.warning(
                    f"FlagEmbedding tidak tersedia ({e}). "
                    f"Fallback ke SentenceTransformer dense-only."
                )

        self._load_sentence_transformer(embedding_model, is_m3)

    def _load_sentence_transformer(self, embedding_model: str, is_m3: bool):
        """Fallback dense-only via SentenceTransformer (dengan penanganan offline)."""
        orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        orig_trans_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                embedding_model, trust_remote_code=True, local_files_only=True
            )
            logger.info("Loaded model from local cache (Offline Mode).")
        except Exception as e:
            logger.debug(f"Offline load attempt failed: {e}")
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if orig_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

            logger.info("Model not found in local cache. Connecting to Hugging Face...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                embedding_model, trust_remote_code=True, local_files_only=False
            )
        finally:
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            if orig_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

        self.backend = "SentenceTransformer"
        self.hybrid = False
        self.vector_size = self.model.get_embedding_dimension()

        # Pastikan context window penuh dipakai (BGE-M3 mendukung 8192;
        # SentenceTransformer kadang diam-diam membatasi ke 512).
        try:
            current = getattr(self.model, "max_seq_length", None)
            logger.info(f"SentenceTransformer max_seq_length: {current}")
            if is_m3 and (current is None or current < MAX_LENGTH):
                self.model.max_seq_length = MAX_LENGTH
                logger.info(f"Set max_seq_length to {MAX_LENGTH} for BGE-M3.")
        except Exception as e:
            logger.debug(f"Tidak bisa inspeksi/atur max_seq_length: {e}")

    # ------------------------------------------------------------------ #
    #  Embedding helpers
    # ------------------------------------------------------------------ #
    def _doc_text(self, chunk: Chunk) -> str:
        """Prepend konteks (judul + section header) ke isi chunk sebelum embedding."""
        ctx = []
        title = (chunk.title or "").strip()
        section = (chunk.section_header or "").strip()
        if title:
            ctx.append(title)
        if section and section.lower() != "unknown" and section != title:
            ctx.append(section)
        head = " | ".join(ctx)
        return f"{head}\n\n{chunk.content}" if head else chunk.content

    @staticmethod
    def _to_sparse_vector(lexical_weights: dict) -> SparseVector:
        """Konversi lexical_weights BGE-M3 (token_id -> weight) ke SparseVector Qdrant."""
        indices, values = [], []
        for k, v in lexical_weights.items():
            fv = float(v)
            if fv > 0:
                indices.append(int(k))
                values.append(fv)
        return SparseVector(indices=indices, values=values)

    def _embed_documents(self, texts: list[str]):
        """Return (dense_list, sparse_list_or_None) untuk daftar teks dokumen."""
        if self.hybrid:
            out = self.model.encode(
                texts,
                batch_size=12,
                max_length=MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            return out["dense_vecs"], out["lexical_weights"]

        # Dense-only fallback
        model_lower = self.embedding_model.lower()
        if "nomic" in model_lower:
            texts = [f"search_document: {t}" for t in texts]
        dense = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return dense, None

    def _embed_query(self, query: str):
        """Return (dense_vec, sparse_vec_or_None) untuk satu query."""
        if self.hybrid:
            out = self.model.encode(
                [query],
                max_length=MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            return _to_list(out["dense_vecs"][0]), self._to_sparse_vector(out["lexical_weights"][0])

        model_lower = self.embedding_model.lower()
        q = query
        if "nomic" in model_lower:
            q = f"search_query: {query}"
        elif "bge" in model_lower and "m3" not in model_lower:
            q = f"Represent this sentence for searching relevant passages: {query}"
        return _to_list(self.model.encode(q, normalize_embeddings=True)), None

    # ------------------------------------------------------------------ #
    #  Collection management
    # ------------------------------------------------------------------ #
    def ensure_collection(self):
        """Create collection (named dense + sparse) jika belum ada."""
        # xAI/self-explanatory: bila cluster Qdrant tak bisa dihubungi, jangan biarkan
        # traceback mentah 404/koneksi lolos begitu saja — beri pesan yang menyebut
        # host + kemungkinan akar + langkah perbaikan, agar user bisa bertindak sendiri.
        host = getattr(self, "_qdrant_host", "?")
        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            msg = str(e)
            is404 = "404" in msg or "not found" in msg.lower()
            hint = (
                "Cluster Qdrant tidak ditemukan (HTTP 404). Kemungkinan penyebab:\n"
                "  1) Cluster Qdrant Cloud sudah DIHAPUS atau di-SUSPEND (free-tier "
                "otomatis di-hibernasi/dihapus setelah lama tidak aktif) — buka dashboard "
                "cloud.qdrant.io, pastikan cluster berstatus aktif/running.\n"
                "  2) QDRANT_URL salah (ID cluster/region/port keliru). URL harus persis "
                "seperti di dashboard, mis. https://<id>.<region>.gcp.cloud.qdrant.io:6333\n"
                "  3) API key tidak cocok dengan cluster tersebut.\n"
                "Perbaiki QDRANT_URL / QDRANT_API_KEY di Colab Secrets lalu jalankan ulang."
            ) if is404 else (
                "Gagal terhubung ke Qdrant. Periksa koneksi internet, QDRANT_URL, dan "
                "QDRANT_API_KEY di Colab Secrets, serta status cluster di cloud.qdrant.io."
            )
            # Tidak mencetak URL/key penuh (cegah kebocoran) — cukup host.
            raise RuntimeError(
                f"Tidak bisa mengakses Qdrant di host '{host}'. {hint}\n"
                f"(detail teknis: {msg})"
            ) from e

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR: VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    )
                },
                # Sparse config selalu dibuat agar skema konsisten lintas mode.
                # Tanpa modifier IDF: bobot lexical BGE-M3 sudah hasil pembelajaran.
                sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams()},
            )
            logger.info(f"Created collection (hybrid schema): {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")

        # Ensure payload index for article_id (required for count/scroll filters)
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="article_id",
                field_schema="keyword",
            )
            logger.info("Ensured payload index on 'article_id'.")
        except Exception as e:
            logger.debug(f"Payload index creation note (might already exist): {e}")

        # Index 'doi' too — needed for the stale-duplicate self-heal filter (delete by doi).
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doi",
                field_schema="keyword",
            )
            logger.info("Ensured payload index on 'doi'.")
        except Exception as e:
            logger.debug(f"Payload index 'doi' note (might already exist): {e}")

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed (hybrid) & simpan chunks ke Qdrant. Return jumlah chunk tersimpan."""
        if not chunks:
            logger.warning("No chunks to store")
            return 0

        logger.info(f"Embedding {len(chunks)} chunks...")

        all_points = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [self._doc_text(c) for c in batch]  # konteks diprepend
            dense, sparse = self._embed_documents(texts)

            for j, chunk in enumerate(batch):
                vector = {DENSE_VECTOR: _to_list(dense[j])}
                if self.hybrid and sparse is not None:
                    vector[SPARSE_VECTOR] = self._to_sparse_vector(sparse[j])

                all_points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload={
                            "content": chunk.content,  # isi asli tanpa prefix konteks
                            **chunk.to_metadata_dict(),
                        },
                    )
                )

        retry_call(
            lambda: self.client.upsert(
                collection_name=self.collection_name,
                points=all_points,
            ),
            label="Qdrant upsert",
        )

        logger.info(f"Stored {len(all_points)} chunks in Qdrant")

        # Self-heal: setelah entri baru tersimpan, buang entri LAMA dari paper yang sama
        # (DOI identik) yang ter-ingest di bawah article_id BERBEDA. Ini terjadi ketika run
        # sebelumnya gagal menemukan DOI (article_id berbasis hash-file) lalu run ini berhasil
        # me-resolve DOI (article_id berbasis DOI) — tanpa pembersihan ini entri lama jadi
        # duplikat yatim di Qdrant. Dijalankan SETELAH upsert agar paper tak pernah hilang.
        art_id = getattr(chunks[0], "article_id", None)
        doi = (getattr(chunks[0], "doi", None) or "").strip()
        if doi and art_id:
            self._cleanup_stale_doi_duplicates(doi, keep_article_id=art_id)

        return len(all_points)

    def _cleanup_stale_doi_duplicates(self, doi: str, keep_article_id: str) -> None:
        """Hapus chunk dengan DOI sama tetapi article_id berbeda (duplikat yatim dari run
        sebelum DOI ter-resolve). Aman lintas-tenant: DOI sama = paper yang sama untuk semua
        pengguna (collection di-share by-design); kita hanya menyatukan ke article_id baru."""
        try:
            flt = Filter(
                must=[FieldCondition(key="doi", match=MatchValue(value=doi))],
                must_not=[FieldCondition(key="article_id", match=MatchValue(value=keep_article_id))],
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=flt),
                wait=True,
            )
            logger.info(f"Self-heal: cleared stale duplicates for doi={doi} under other article_ids.")
        except Exception as e:
            logger.warning(f"Self-heal stale-doi cleanup skipped: {e}")

    def delete_article(self, article_id: str) -> bool:
        """Hapus semua chunk milik satu artikel."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
            ),
        )
        logger.info(f"Deleted all chunks for article {article_id}")
        return True

    def article_exists(self, article_id: str) -> dict | bool:
        """Cek apakah artikel sudah ter-ingest penuh; bersihkan jika parsial.
        Return dict metadata bila ada & lengkap, selain itu False."""
        result = retry_call(
            lambda: self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
                ),
                exact=True,
            ),
            label="Qdrant count",
        )

        count = result.count
        if count == 0:
            return False

        points, _ = retry_call(
            lambda: self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
                ),
                limit=1,
                with_payload=["total_chunks", "title", "doi"],
                with_vectors=False,
            ),
            label="Qdrant scroll",
        )

        if points:
            expected_chunks = points[0].payload.get("total_chunks", 0)
            if expected_chunks > 0 and count >= expected_chunks:
                return {
                    "title": points[0].payload.get("title", "Unknown"),
                    "doi": points[0].payload.get("doi", "Unknown"),
                }

            logger.warning(
                f"Detected partial insertion for article {article_id} "
                f"({count}/{expected_chunks} chunks). Cleaning up for re-processing..."
            )
            self.delete_article(article_id)

        return False

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "points_count": info.points_count,
            "status": info.status.value,
        }

    def list_articles(self) -> list[dict]:
        """List semua artikel unik di collection."""
        articles = {}
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["article_id", "title", "authors", "doi", "total_chunks"],
                with_vectors=False,
            )

            for point in results:
                aid = point.payload.get("article_id")
                if aid and aid not in articles:
                    articles[aid] = {
                        "article_id": aid,
                        "title": point.payload.get("title", "Untitled"),
                        "authors": point.payload.get("authors", ""),
                        "doi": point.payload.get("doi", ""),
                        "total_chunks": point.payload.get("total_chunks", 0),
                    }

            if offset is None:
                break

        return list(articles.values())

    # ------------------------------------------------------------------ #
    #  Search (hybrid dense+sparse with RRF, or dense-only fallback)
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        n_results: int = 10,
        article_ids: list[str] | None = None,
        section_filter: str | None = None,
        doi_filter: str | None = None,
    ) -> list[dict]:
        """
        Cari chunk relevan. Hybrid (dense+sparse, RRF fusion) bila tersedia,
        selain itu dense-only. Return list dict {content, score, metadata}.
        """
        must_conditions = []

        if article_ids:
            if len(article_ids) == 1:
                must_conditions.append(
                    FieldCondition(key="article_id", match=MatchValue(value=article_ids[0]))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="article_id", match=MatchAny(any=article_ids))
                )

        if section_filter:
            must_conditions.append(
                FieldCondition(key="section_header", match=MatchValue(value=section_filter))
            )

        if doi_filter:
            must_conditions.append(
                FieldCondition(key="doi", match=MatchValue(value=doi_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        dense_vec, sparse_vec = self._embed_query(query)

        if self.hybrid and sparse_vec is not None:
            prefetch_limit = max(n_results * 4, 20)
            prefetch = [
                Prefetch(
                    query=dense_vec, using=DENSE_VECTOR,
                    limit=prefetch_limit, filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vec, using=SPARSE_VECTOR,
                    limit=prefetch_limit, filter=query_filter,
                ),
            ]
            results = retry_call(
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=n_results,
                    with_payload=True,
                ),
                label="Qdrant hybrid search",
            )
        else:
            results = retry_call(
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_vec,
                    using=DENSE_VECTOR,
                    query_filter=query_filter,
                    limit=n_results,
                    with_payload=True,
                ),
                label="Qdrant search",
            )

        return [
            {
                "content": point.payload.get("content", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "content"},
            }
            for point in results.points
        ]
