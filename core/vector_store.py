"""
Qdrant Vector Store Operations

Handles embedding generation (sentence-transformers) and 
storing/retrieving chunks from Qdrant vector database.
"""

import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from core.chunker import Chunk

logger = logging.getLogger(__name__)

# === Configuration ===
COLLECTION_NAME = "scientific_articles"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # 8192 context window, 768 dimensions
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


class VectorStore:
    """
    Manages embedding generation and Qdrant storage.
    
    Usage:
        store = VectorStore()
        store.ensure_collection()
        store.add_chunks(chunks)
    """
    
    def __init__(
        self,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        logger.info(f"Initializing VectorStore...")
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        
        # Save original offline settings
        orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        orig_trans_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        
        try:
            # Force library to run offline to block checking Hugging Face server updates
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model, trust_remote_code=True, local_files_only=True)
            logger.info("Loaded model from local cache (Offline Mode).")
        except Exception as e:
            logger.debug(f"Offline load attempt failed: {e}")
            # Restore settings if loading fails (e.g. model needs downloading)
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
            self.model = SentenceTransformer(embedding_model, trust_remote_code=True, local_files_only=False)
        finally:
            # Always restore environment variables
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
                
            if orig_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            
        self.vector_size = self.model.get_embedding_dimension()
        
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        logger.info(f"VectorStore ready. Embedding dim: {self.vector_size}")
    
    def ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection already exists: {self.collection_name}")
    
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """
        Embed and store chunks in Qdrant.
        
        Args:
            chunks: List of Chunk objects to store
            batch_size: Number of chunks to embed at once
        
        Returns:
            Number of chunks stored
        """
        if not chunks:
            logger.warning("No chunks to store")
            return 0
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts for batch embedding (with prefix if Nomic)
        model_lower = self.embedding_model.lower()
        if "nomic" in model_lower:
            texts_to_embed = [f"search_document: {chunk.content}" for chunk in chunks]
        else:
            texts_to_embed = [chunk.content for chunk in chunks]
        
        # Batch embed
        all_points = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts_to_embed = texts_to_embed[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            
            embeddings = self.model.encode(
                batch_texts_to_embed,
                show_progress_bar=len(texts_to_embed) > batch_size,
                normalize_embeddings=True,
            )
            
            for chunk, embedding in zip(batch_chunks, embeddings):
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk.content,  # Original un-prefixed content
                        **chunk.to_metadata_dict(),
                    },
                )
                all_points.append(point)
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=all_points,
        )
        
        logger.info(f"Stored {len(all_points)} chunks in Qdrant")
        return len(all_points)
    
    def delete_article(self, article_id: str) -> bool:
        """
        Delete all chunks belonging to a specific article.
        
        Args:
            article_id: UUID of the article to delete
        
        Returns:
            True if deletion was successful
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="article_id",
                        match=MatchValue(value=article_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted all chunks for article {article_id}")
        return True
    
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
        """
        List all unique articles in the collection.
        Returns list of {article_id, title, authors, doi, total_chunks}
        """
        # Scroll through all points and collect unique article_ids
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
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        article_ids: list[str] | None = None,
        section_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            article_ids: Filter by specific article(s). None = all.
            section_filter: Filter by section header (e.g., "Methods")
        
        Returns:
            List of dicts with content, metadata, and score
        """
        # Build filter
        must_conditions = []
        
        if article_ids:
            if len(article_ids) == 1:
                must_conditions.append(
                    FieldCondition(
                        key="article_id",
                        match=MatchValue(value=article_ids[0]),
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key="article_id",
                        match=MatchAny(any=article_ids),
                    )
                )
        
        if section_filter:
            must_conditions.append(
                FieldCondition(
                    key="section_header",
                    match=MatchValue(value=section_filter),
                )
            )
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        # Embed query (with prefix if Nomic or BGE)
        model_lower = self.embedding_model.lower()
        query_to_embed = query
        if "nomic" in model_lower:
            query_to_embed = f"search_query: {query}"
        elif "bge" in model_lower and "m3" not in model_lower:
            query_to_embed = f"Represent this sentence for searching relevant passages: {query}"
            
        # Embed query
        query_vector = self.model.encode(
            query_to_embed, normalize_embeddings=True
        ).tolist()
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=n_results,
            with_payload=True,
        )
        
        return [
            {
                "content": point.payload.get("content", ""),
                "score": point.score,
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "content"
                },
            }
            for point in results.points
        ]
