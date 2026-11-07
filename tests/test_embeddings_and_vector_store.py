"""Unit Tests for Local Embeddings & ChromaDB Vector Store.

Track: Banking & FinTech (Cred)
Target Modules:
- src.cred_agent.rag.embeddings
- src.cred_agent.rag.vector_store
"""

from pathlib import Path
import pytest

from src.cred_agent.rag.chunking import Chunk
from src.cred_agent.rag.embeddings import (
    EmbeddingManager,
    get_embedding_model,
)
from src.cred_agent.rag.vector_store import (
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE_BASED,
    ChromaVectorStore,
)


# ============================================================================
# Embedding Generation & Validation Tests
# ============================================================================

def test_embedding_generation_and_dimensions():
    """Verify SentenceTransformers embedding returns a 384-dimensional numerical vector."""
    manager = EmbeddingManager()
    text = "What is the minimum balance required for Cred savings accounts?"
    vector = manager.encode_text(text)

    assert isinstance(vector, list), "Embedding should be a list of floats"
    assert len(vector) == 384, f"Expected 384 dimensions, got {len(vector)}"
    assert all(isinstance(x, float) for x in vector), "All elements must be floats"
    # Ensure non-trivial embedding
    assert sum(abs(x) for x in vector) > 0.0, "Vector should not be all zeros for valid text"


def test_embedding_empty_and_invalid_input():
    """Verify empty, whitespace, and non-string inputs are handled gracefully with 384-dim zero vectors."""
    manager = EmbeddingManager()

    # Empty string
    v_empty = manager.encode_text("")
    assert len(v_empty) == 384
    assert sum(v_empty) == 0.0

    # Whitespace only
    v_space = manager.encode_text("   \n\t  ")
    assert len(v_space) == 384
    assert sum(v_space) == 0.0

    # None input
    v_none = manager.encode_text(None)
    assert len(v_none) == 384
    assert sum(v_none) == 0.0

    # Non-string input
    v_num = manager.encode_text(12345)  # type: ignore
    assert len(v_num) == 384
    assert sum(v_num) == 0.0


def test_embedding_batch_processing():
    """Verify batch encoding handles empty batches and mixed valid/invalid strings."""
    manager = EmbeddingManager()

    # Empty batch
    assert manager.encode_batch([]) == []

    # Mixed batch
    batch = [
        "First valid policy sentence.",
        "",
        "Second valid policy sentence.",
        None,
    ]
    vecs = manager.encode_batch(batch)
    assert len(vecs) == 4
    assert all(len(v) == 384 for v in vecs)

    # Valid items have non-zero embeddings
    assert sum(abs(x) for x in vecs[0]) > 0.0
    assert sum(abs(x) for x in vecs[2]) > 0.0

    # Invalid/empty items have zero vectors
    assert sum(vecs[1]) == 0.0
    assert sum(vecs[3]) == 0.0


def test_embedding_model_singleton_reuse():
    """Verify the embedding model is loaded once and reused across instances."""
    model_a = get_embedding_model()
    model_b = get_embedding_model()
    assert model_a is model_b, "get_embedding_model must return the identical singleton instance"

    manager_1 = EmbeddingManager()
    manager_2 = EmbeddingManager()
    assert manager_1._model is manager_2._model, "EmbeddingManager instances must share the underlying model"


# ============================================================================
# Vector Store & ChromaDB Tests
# ============================================================================

def test_collection_creation_independent(tmp_path: Path):
    """Verify ChromaVectorStore creates two separate, independent collections."""
    store = ChromaVectorStore(persist_directory=tmp_path / "chroma_db")

    assert store.collection_fixed.name == COLLECTION_FIXED_SIZE
    assert store.collection_sentence.name == COLLECTION_SENTENCE_BASED
    assert store.collection_fixed.id != store.collection_sentence.id, "Collections must be distinct"

    assert store.collection_fixed.count() == 0
    assert store.collection_sentence.count() == 0


def test_independent_indexing_and_metadata(tmp_path: Path):
    """Verify fixed-size and sentence-based chunks index independently without cross-contamination."""
    store = ChromaVectorStore(persist_directory=tmp_path / "chroma_db")

    fixed_chunks = [
        Chunk(
            chunk_id="fixed_doc1_0",
            parent_document_id="doc1",
            source="doc1.md",
            chunking_strategy="fixed_size",
            chunk_index=0,
            text="Personal loans require minimum monthly income of INR 35,000.",
        ),
        Chunk(
            chunk_id="fixed_doc1_1",
            parent_document_id="doc1",
            source="doc1.md",
            chunking_strategy="fixed_size",
            chunk_index=1,
            text="Home loan applicants require minimum CIBIL score of 720.",
        ),
    ]

    sentence_chunks = [
        Chunk(
            chunk_id="sent_doc2_0",
            parent_document_id="doc2",
            source="doc2.md",
            chunking_strategy="sentence_based",
            chunk_index=0,
            text="Cred Credit Cards carry an annual membership fee of INR 1,500.",
        ),
    ]

    # Index fixed-size only
    count_fixed = store.index_fixed_size_chunks(fixed_chunks)
    assert count_fixed == 2
    assert store.collection_fixed.count() == 2
    assert store.collection_sentence.count() == 0, "Sentence collection must remain empty"

    # Index sentence-based
    count_sent = store.index_sentence_based_chunks(sentence_chunks)
    assert count_sent == 1
    assert store.collection_fixed.count() == 2
    assert store.collection_sentence.count() == 1

    # Verify metadata preservation on retrieval
    hits = store.query_fixed_size("personal loan income", top_k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit["parent_document_id"] == "doc1"
    assert hit["source"] == "doc1.md"
    assert hit["chunking_strategy"] == "fixed_size"
    assert hit["metadata"]["chunk_index"] == 0
    assert "cosine_similarity" in hit
    assert -1.0 <= hit["cosine_similarity"] <= 1.0


def test_repeated_upserts_prevent_duplicates(tmp_path: Path):
    """Verify repeated indexing with the same chunk IDs updates in-place and prevents duplicate records."""
    store = ChromaVectorStore(persist_directory=tmp_path / "chroma_db")

    chunks = [
        Chunk(
            chunk_id="fixed_sample_0",
            parent_document_id="sample_doc",
            source="sample.md",
            chunking_strategy="fixed_size",
            chunk_index=0,
            text="Initial text for the sample policy chunk.",
        ),
    ]

    # First indexing
    store.index_fixed_size_chunks(chunks)
    assert store.collection_fixed.count() == 1

    # Second indexing with identical ID
    store.index_fixed_size_chunks(chunks)
    assert store.collection_fixed.count() == 1, "Count must remain 1 after repeated upsert"

    # Repeated indexing with updated text
    updated_chunks = [
        Chunk(
            chunk_id="fixed_sample_0",
            parent_document_id="sample_doc",
            source="sample.md",
            chunking_strategy="fixed_size",
            chunk_index=0,
            text="Updated text for the sample policy chunk.",
        ),
    ]
    store.index_fixed_size_chunks(updated_chunks)
    assert store.collection_fixed.count() == 1

    hits = store.query_fixed_size("sample policy chunk", top_k=1)
    assert "Updated text" in hits[0]["text"]


def test_persistence_across_sessions(tmp_path: Path):
    """Verify data written to persistent directory survives client re-initialization."""
    persist_dir = tmp_path / "persistent_chroma"

    # Session 1: Create and index
    store1 = ChromaVectorStore(persist_directory=persist_dir)
    chunks = [
        Chunk(
            chunk_id="sent_persist_0",
            parent_document_id="persist_doc",
            source="persist.md",
            chunking_strategy="sentence_based",
            chunk_index=0,
            text="Data that must persist across process restarts.",
        ),
    ]
    store1.index_sentence_based_chunks(chunks)
    assert store1.collection_sentence.count() == 1
    del store1

    # Session 2: Re-open existing directory
    store2 = ChromaVectorStore(persist_directory=persist_dir)
    assert store2.collection_sentence.count() == 1

    hits = store2.query_sentence_based("persist across process restarts", top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "sent_persist_0"
    assert hits[0]["parent_document_id"] == "persist_doc"

