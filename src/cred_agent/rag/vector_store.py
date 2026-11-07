"""Persistent ChromaDB Vector Store for Cred Policy Collections.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag.vector_store

This module manages two strictly separated ChromaDB collections:
1. 'cred_fixed_size' (Character-based fixed-size chunks with overlap)
2. 'cred_sentence_based' (Sentence-based grammatical chunks)

Both collections use persistent local storage at ./chroma_db and share
the local SentenceTransformers embedding model. Indexing uses collection.upsert()
to prevent duplicate records on repeated execution. Collections are never combined.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import chromadb
from chromadb.api.types import Metadata, PyEmbedding
from chromadb.config import Settings

try:
    from src.cred_agent.config import (
        COLLECTION_FIXED_SIZE,
        COLLECTION_SENTENCE_BASED,
        DEFAULT_CHROMA_PATH,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_SENTENCES_PER_CHUNK,
    )
except ImportError:
    DEFAULT_CHROMA_PATH = "./chroma_db"
    COLLECTION_FIXED_SIZE = "cred_fixed_size"
    COLLECTION_SENTENCE_BASED = "cred_sentence_based"
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 100
    DEFAULT_SENTENCES_PER_CHUNK = 3

from src.cred_agent.rag.chunking import (
    Chunk,
    chunk_documents_fixed,
    chunk_documents_sentence,
)
from src.cred_agent.rag.document_loader import load_knowledge_base_documents
from src.cred_agent.rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# Constants and backward-compatibility aliases
COLLECTION_FIXED: str = COLLECTION_FIXED_SIZE
COLLECTION_SENTENCE: str = COLLECTION_SENTENCE_BASED
DEFAULT_CHROMA_PERSIST_DIR: str = DEFAULT_CHROMA_PATH


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage across dual distinct collections."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH,
        embedder: Optional[EmbeddingManager] = None,
    ):
        """Initialize ChromaDB persistent client and dual vector collections.

        Args:
            persist_directory: Filesystem directory for local ChromaDB files.
            embedder: Shared EmbeddingManager instance (reuses singleton model).
        """
        self.persist_directory = str(Path(persist_directory).resolve())
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = embedder or EmbeddingManager()

        # Initialize the two separate collections with cosine distance space
        self.collection_fixed = self._client.get_or_create_collection(
            name=COLLECTION_FIXED_SIZE,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_sentence = self._client.get_or_create_collection(
            name=COLLECTION_SENTENCE_BASED,
            metadata={"hnsw:space": "cosine"},
        )
        self._require_cosine_metric(self.collection_fixed)
        self._require_cosine_metric(self.collection_sentence)

    @staticmethod
    def _require_cosine_metric(collection: chromadb.Collection) -> None:
        """Reject collections whose persisted metric would make cosine scores invalid."""
        metric = (collection.metadata or {}).get("hnsw:space")
        if metric != "cosine":
            raise RuntimeError(
                f"Collection '{collection.name}' uses {metric!r}, not cosine distance. "
                "Do not query it as cosine. Build and verify a cosine index in a new "
                "collection or persistence directory, then switch over without deleting "
                "the existing index."
            )

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """Retrieve a collection instance by its unique collection name.

        Args:
            collection_name: 'cred_fixed_size' or 'cred_sentence_based'.

        Returns:
            chromadb.Collection instance.
        """
        if collection_name == COLLECTION_FIXED_SIZE:
            return self.collection_fixed
        elif collection_name == COLLECTION_SENTENCE_BASED:
            return self.collection_sentence
        else:
            return self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def index_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str,
        batch_size: int = 64,
    ) -> int:
        """Upsert a list of Chunk objects into the specified ChromaDB collection.

        Uses collection.upsert() to ensure idempotent indexing and prevent duplicate
        records on repeated runs with identical chunk_ids.

        Args:
            chunks: List of Chunk objects to store.
            collection_name: Target collection name.
            batch_size: Batch size for embedding and upsert operations.

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            return 0

        target_collection = self.get_collection(collection_name)
        texts = [c.text for c in chunks]

        # Generate normalized 384-dimensional embeddings
        embeddings = cast(
            List[PyEmbedding],
            self.embedder.encode_batch(texts, batch_size=batch_size),
        )

        ids = [c.chunk_id for c in chunks]
        metadatas: List[Metadata] = [
            {
                "parent_document_id": c.parent_document_id,
                "source": c.source,
                "chunking_strategy": c.chunking_strategy,
                "chunk_index": c.chunk_index,
                "char_count": len(c.text),
                **c.metadata,
            }
            for c in chunks
        ]
        documents = texts

        # Upsert in batches to prevent duplicate records
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            target_collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=documents[i:end_idx],
            )

        return len(chunks)

    # ------------------------------------------------------------------------
    # Dedicated Strategy-Specific Indexing Functions
    # ------------------------------------------------------------------------

    def index_fixed_size_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Index fixed-size chunks strictly into the cred_fixed_size collection.

        Args:
            chunks: List of Chunk objects generated via fixed-size chunking.
            batch_size: Batch size for upserting.

        Returns:
            Number of chunks upserted.
        """
        return self.index_chunks(chunks, collection_name=COLLECTION_FIXED_SIZE, batch_size=batch_size)

    def index_sentence_based_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Index sentence-based chunks strictly into the cred_sentence_based collection.

        Args:
            chunks: List of Chunk objects generated via sentence-based chunking.
            batch_size: Batch size for upserting.

        Returns:
            Number of chunks upserted.
        """
        return self.index_chunks(chunks, collection_name=COLLECTION_SENTENCE_BASED, batch_size=batch_size)

    # ------------------------------------------------------------------------
    # Retrieval Functions
    # ------------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query a specific collection and return ranked matches with cosine similarities.

        Args:
            query_text: Natural language query string.
            collection_name: Collection to search ('cred_fixed_size' or 'cred_sentence_based').
            top_k: Maximum number of chunks to retrieve.

        Returns:
            List of result dictionaries sorted by descending cosine similarity.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_emb = self.embedder.encode_text(query_text)
        target_collection = self.get_collection(collection_name)
        self._require_cosine_metric(target_collection)

        total_count = target_collection.count()
        actual_k = min(top_k, total_count) if total_count > 0 else 0

        if actual_k == 0:
            return []

        results = target_collection.query(
            query_embeddings=[query_emb],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted: List[Dict[str, Any]] = []
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            cosine_distance = float(dist)
            if not math.isfinite(cosine_distance) or not 0.0 <= cosine_distance <= 2.0:
                raise RuntimeError(
                    f"Collection '{collection_name}' returned invalid cosine distance "
                    f"{cosine_distance!r} for chunk '{cid}'"
                )
            cos_sim = 1.0 - cosine_distance
            formatted.append({
                "chunk_id": cid,
                "text": doc,
                "cosine_similarity": cos_sim,
                "similarity_score": cos_sim,
                "distance": cosine_distance,
                "metadata": meta,
                "parent_document_id": meta.get("parent_document_id", ""),
                "doc_id": meta.get("parent_document_id", ""),
                "source": meta.get("source", ""),
                "chunking_strategy": meta.get("chunking_strategy", ""),
            })

        formatted.sort(key=lambda hit: hit["cosine_similarity"], reverse=True)
        return formatted

    def query_fixed_size(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k matches exclusively from the cred_fixed_size collection.

        Args:
            query_text: Natural language user question.
            top_k: Number of chunks to retrieve.

        Returns:
            Ranked list of chunk dictionaries with similarity scores.
        """
        return self.query(query_text, collection_name=COLLECTION_FIXED_SIZE, top_k=top_k)

    def query_sentence_based(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k matches exclusively from the cred_sentence_based collection.

        Args:
            query_text: Natural language user question.
            top_k: Number of chunks to retrieve.

        Returns:
            Ranked list of chunk dictionaries with similarity scores.
        """
        return self.query(query_text, collection_name=COLLECTION_SENTENCE_BASED, top_k=top_k)

    # ------------------------------------------------------------------------
    # Pipeline Orchestration
    # ------------------------------------------------------------------------

    def build_and_index_all(
        self,
        kb_dir: Union[str, Path] = "data/knowledge_base",
    ) -> Tuple[int, int]:
        """Load knowledge base documents and index each strategy in its separate collection.

        Args:
            kb_dir: Path to directory containing knowledge-base markdown files.

        Returns:
            Tuple of (fixed_chunk_count, sentence_chunk_count).
        """
        docs = load_knowledge_base_documents(kb_dir=kb_dir)

        # 1. Fixed-size chunking & indexing
        fixed_chunks = chunk_documents_fixed(
            docs,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )
        n_fixed = self.index_fixed_size_chunks(fixed_chunks)

        # 2. Sentence-based chunking & indexing
        sentence_chunks = chunk_documents_sentence(
            docs,
            sentences_per_chunk=DEFAULT_SENTENCES_PER_CHUNK,
        )
        n_sent = self.index_sentence_based_chunks(sentence_chunks)

        return n_fixed, n_sent


# Module-level convenience functions
def get_vector_store(persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH) -> ChromaVectorStore:
    """Factory function to get or create a ChromaVectorStore."""
    return ChromaVectorStore(persist_directory=persist_directory)


def index_fixed_size_chunks(
    chunks: List[Chunk],
    persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH,
) -> int:
    """Module-level function to index fixed-size chunks into cred_fixed_size."""
    store = ChromaVectorStore(persist_directory=persist_directory)
    return store.index_fixed_size_chunks(chunks)


def index_sentence_based_chunks(
    chunks: List[Chunk],
    persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH,
) -> int:
    """Module-level function to index sentence-based chunks into cred_sentence_based."""
    store = ChromaVectorStore(persist_directory=persist_directory)
    return store.index_sentence_based_chunks(chunks)


def query_fixed_size(
    query_text: str,
    top_k: int = 3,
    persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH,
) -> List[Dict[str, Any]]:
    """Module-level function to query cred_fixed_size."""
    store = ChromaVectorStore(persist_directory=persist_directory)
    return store.query_fixed_size(query_text, top_k=top_k)


def query_sentence_based(
    query_text: str,
    top_k: int = 3,
    persist_directory: Union[str, Path] = DEFAULT_CHROMA_PATH,
) -> List[Dict[str, Any]]:
    """Module-level function to query cred_sentence_based."""
    store = ChromaVectorStore(persist_directory=persist_directory)
    return store.query_sentence_based(query_text, top_k=top_k)


if __name__ == "__main__":
    store = ChromaVectorStore()
    n_fixed, n_sent = store.build_and_index_all()

    print("\n" + "=" * 76)
    print(" CHROMA DB VECTOR STORE — DUAL COLLECTION INDEXING REPORT")
    print("=" * 76)
    print(f" Storage Location            : {store.persist_directory}")
    print(f" Collection 1 (Fixed-Size)   : '{COLLECTION_FIXED_SIZE}' -> {n_fixed} chunks (Count: {store.collection_fixed.count()})")
    print(f" Collection 2 (Sentence-Based): '{COLLECTION_SENTENCE_BASED}' -> {n_sent} chunks (Count: {store.collection_sentence.count()})")
    print("-" * 76)

    # Test query
    sample_q = "What is the penalty for early loan prepayment?"
    print(f"Query: '{sample_q}'\n")

    hits_fixed = store.query_fixed_size(sample_q, top_k=2)
    print(f"Top-2 hits from [{COLLECTION_FIXED_SIZE}]:")
    for h in hits_fixed:
        print(f"  [{h['chunk_id']}] sim={h['cosine_similarity']:.4f} | parent={h['parent_document_id']} | {h['text'][:80]}...")

    hits_sent = store.query_sentence_based(sample_q, top_k=2)
    print(f"\nTop-2 hits from [{COLLECTION_SENTENCE_BASED}]:")
    for h in hits_sent:
        print(f"  [{h['chunk_id']}] sim={h['cosine_similarity']:.4f} | parent={h['parent_document_id']} | {h['text'][:80]}...")
    print("=" * 76)
