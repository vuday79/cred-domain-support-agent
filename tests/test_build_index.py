"""Unit Tests for Knowledge Base Indexing Pipeline (scripts/build_index.py).

Track: Banking & FinTech (Cred)
Target Script: scripts.build_index
"""

from pathlib import Path
import pytest

from scripts.build_index import build_knowledge_base_index
from src.cred_agent.config import (
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE_BASED,
)
from src.cred_agent.rag.vector_store import ChromaVectorStore


def test_successful_indexing_and_collections_populated(tmp_path: Path):
    """Verify end-to-end indexing runs cleanly and populates both collections."""
    chroma_dir = tmp_path / "test_chroma_success"

    report = build_knowledge_base_index(chroma_dir=chroma_dir)

    assert report["status"] == "SUCCESS"
    assert report["documents_loaded"] == 12, "All 12 policy documents must be loaded"
    assert report["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert report["embedding_dimensions"] == 384
    assert report["collections_independent"] is True

    counts = report["final_collection_counts"]
    assert counts[COLLECTION_FIXED_SIZE] > 0, "Fixed-size collection must be populated"
    assert counts[COLLECTION_SENTENCE_BASED] > 0, "Sentence-based collection must be populated"
    assert report["total_records_indexed"] == counts[COLLECTION_FIXED_SIZE] + counts[COLLECTION_SENTENCE_BASED]


def test_metadata_preservation(tmp_path: Path):
    """Verify that parent_document_id and all required metadata fields are properly stored."""
    chroma_dir = tmp_path / "test_chroma_meta"
    build_knowledge_base_index(chroma_dir=chroma_dir)

    store = ChromaVectorStore(persist_directory=chroma_dir)

    # 1. Verify fixed collection metadata
    fixed_res = store.collection_fixed.get(include=["metadatas", "documents"])
    assert len(fixed_res["ids"]) > 0
    for meta, doc in zip(fixed_res["metadatas"], fixed_res["documents"]):
        assert "parent_document_id" in meta and meta["parent_document_id"], "parent_document_id must be preserved"
        assert "source" in meta and meta["source"].endswith(".md"), "source must be preserved"
        assert meta["chunking_strategy"] == "fixed_size"
        assert isinstance(meta["chunk_index"], int)
        assert meta["char_count"] == len(doc)

    # 2. Verify sentence collection metadata
    sent_res = store.collection_sentence.get(include=["metadatas", "documents"])
    assert len(sent_res["ids"]) > 0
    for meta, doc in zip(sent_res["metadatas"], sent_res["documents"]):
        assert "parent_document_id" in meta and meta["parent_document_id"], "parent_document_id must be preserved"
        assert "source" in meta and meta["source"].endswith(".md"), "source must be preserved"
        assert meta["chunking_strategy"] == "sentence_based"
        assert isinstance(meta["chunk_index"], int)
        assert meta["char_count"] == len(doc)


def test_repeated_execution_no_duplicates(tmp_path: Path):
    """Verify that executing the indexing script repeatedly does not create duplicate records."""
    chroma_dir = tmp_path / "test_chroma_dedup"

    # Run 1
    report1 = build_knowledge_base_index(chroma_dir=chroma_dir)
    count1_fixed = report1["final_collection_counts"][COLLECTION_FIXED_SIZE]
    count1_sent = report1["final_collection_counts"][COLLECTION_SENTENCE_BASED]

    # Run 2 on the exact same storage directory
    report2 = build_knowledge_base_index(chroma_dir=chroma_dir)
    count2_fixed = report2["final_collection_counts"][COLLECTION_FIXED_SIZE]
    count2_sent = report2["final_collection_counts"][COLLECTION_SENTENCE_BASED]

    assert count2_fixed == count1_fixed, "Fixed-size collection count must remain identical after repeated upsert"
    assert count2_sent == count1_sent, "Sentence-based collection count must remain identical after repeated upsert"


def test_independent_collection_contents(tmp_path: Path):
    """Verify that both collections remain strictly independent with disjoint chunk IDs and distinct schemas."""
    chroma_dir = tmp_path / "test_chroma_independent"
    build_knowledge_base_index(chroma_dir=chroma_dir)

    store = ChromaVectorStore(persist_directory=chroma_dir)

    # Collections must have different internal IDs and distinct collection names
    assert store.collection_fixed.id != store.collection_sentence.id
    assert store.collection_fixed.name == COLLECTION_FIXED_SIZE
    assert store.collection_sentence.name == COLLECTION_SENTENCE_BASED

    fixed_ids = set(store.collection_fixed.get()["ids"])
    sent_ids = set(store.collection_sentence.get()["ids"])

    # Chunk IDs must have distinct strategy prefixes
    assert all(cid.startswith("fixed_") for cid in fixed_ids)
    assert all(cid.startswith("sent_") for cid in sent_ids)

    # Collections must be completely disjoint (no shared IDs)
    assert fixed_ids.isdisjoint(sent_ids), "Collections must not share chunk IDs"

    # Collections must maintain strictly separate chunk strategies in metadata
    fixed_metas = store.collection_fixed.get(include=["metadatas"])["metadatas"]
    sent_metas = store.collection_sentence.get(include=["metadatas"])["metadatas"]

    assert all(m["chunking_strategy"] == "fixed_size" for m in fixed_metas)
    assert all(m["chunking_strategy"] == "sentence_based" for m in sent_metas)

