"""Unit Tests for Knowledge Base Document Loader & Chunking Pipeline.

Track: Banking & FinTech (Cred)
Target Modules:
- src.cred_agent.rag.document_loader
- src.cred_agent.rag.chunking
"""

from pathlib import Path
import pytest

from src.cred_agent.rag.document_loader import (
    Document,
    clean_markdown_text,
    extract_document_name,
    load_knowledge_base_documents,
    load_single_document,
)
from src.cred_agent.rag.chunking import (
    Chunk,
    chunk_documents_fixed,
    chunk_documents_sentence,
    chunk_fixed_size,
    chunk_sentence_based,
    split_text_into_sentences,
)


# ============================================================================
# Document Loader Tests
# ============================================================================

def test_load_knowledge_base_documents_live():
    """Verify loading real markdown files from data/knowledge_base."""
    docs = load_knowledge_base_documents("data/knowledge_base")
    assert len(docs) == 12, f"Expected 12 documents, got {len(docs)}"

    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.doc_id, "doc_id must not be empty"
        assert doc.doc_name, "doc_name must not be empty"
        assert doc.filename.endswith(".md"), "filename must end with .md"
        assert doc.source, "source must not be empty"
        assert len(doc.text) > 0, "text must not be empty"
        assert len(doc.raw_content) > 0, "raw_content must not be empty"
        assert "readme" not in doc.filename.lower(), "README.md must not be loaded"


def test_readme_exclusion(tmp_path: Path):
    """Ensure that README.md (case-insensitive) is explicitly excluded from loading."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()

    # Create a valid policy doc and a README file
    (kb_dir / "policy_one.md").write_text("# Policy One\n\nSubstantive policy text here.", encoding="utf-8")
    (kb_dir / "README.md").write_text("# Knowledge Base README\n\nOverview documentation.", encoding="utf-8")
    (kb_dir / "readme.md").write_text("# Lowercase readme\n\nAnother readme file.", encoding="utf-8")

    docs = load_knowledge_base_documents(kb_dir)
    assert len(docs) == 1
    assert docs[0].doc_id == "policy_one"
    assert docs[0].filename == "policy_one.md"


def test_missing_directory_handling():
    """Verify that a non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_knowledge_base_documents("non_existent_folder_path_xyz")


def test_empty_document_handling(tmp_path: Path):
    """Verify that empty or whitespace-only files are handled gracefully without raising crashes."""
    kb_dir = tmp_path / "kb_empty"
    kb_dir.mkdir()

    (kb_dir / "empty.md").write_text("", encoding="utf-8")
    (kb_dir / "whitespace.md").write_text("   \n\n\t  \n", encoding="utf-8")
    (kb_dir / "valid.md").write_text("# Title\n\nActual substantive policy sentences.", encoding="utf-8")

    # load_single_document on empty file returns None
    assert load_single_document(kb_dir / "empty.md") is None
    assert load_single_document(kb_dir / "whitespace.md") is None

    # load_knowledge_base_documents gracefully skips empty documents
    docs = load_knowledge_base_documents(kb_dir)
    assert len(docs) == 1
    assert docs[0].doc_id == "valid"


# ============================================================================
# Fixed-Size Chunking Tests
# ============================================================================

def test_fixed_size_chunking_defaults():
    """Verify fixed-size character-based chunking with default 500 size and 100 overlap."""
    text = (
        "Cred Personal Loan eligibility requires Indian citizenship and net monthly income of INR 35,000. "
        * 10  # Approx 990 characters
    )
    chunks = chunk_fixed_size(text, parent_document_id="doc_loan", source="data/loan.md")

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert isinstance(c, Chunk)
        assert c.chunk_id == f"fixed_doc_loan_{i}"
        assert c.parent_document_id == "doc_loan"
        assert c.source == "data/loan.md"
        assert c.chunking_strategy == "fixed_size"
        assert c.chunk_index == i
        assert len(c.text) > 0, "Chunk text must be non-empty"
        assert len(c.text) <= 500, "Chunk must not exceed chunk_size"


def test_fixed_size_overlap_verification():
    """Verify exact character overlap between consecutive fixed-size chunks."""
    # Construct deterministic string of 800 characters
    sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * 25  # 900 chars
    chunk_size = 500
    overlap = 100
    step = chunk_size - overlap  # 400

    chunks = chunk_fixed_size(
        sample_text,
        parent_document_id="doc_test",
        source="test.md",
        chunk_size=chunk_size,
        overlap=overlap,
    )

    assert len(chunks) >= 2
    # The tail of chunk 0 (last 100 chars) must equal the head of chunk 1 (first 100 chars)
    chunk0_tail = chunks[0].text[-overlap:]
    chunk1_head = chunks[1].text[:overlap]
    assert chunk0_tail == chunk1_head, "Chunk overlap mismatch between consecutive chunks"


def test_fixed_size_invalid_parameters():
    """Verify that invalid overlap or chunk_size values raise ValueError to prevent infinite loops."""
    # overlap >= chunk_size
    with pytest.raises(ValueError, match="strictly less than"):
        chunk_fixed_size("Some text", "doc", "src", chunk_size=200, overlap=200)

    with pytest.raises(ValueError, match="strictly less than"):
        chunk_fixed_size("Some text", "doc", "src", chunk_size=200, overlap=250)

    # negative values
    with pytest.raises(ValueError, match="positive"):
        chunk_fixed_size("Some text", "doc", "src", chunk_size=-10, overlap=0)

    with pytest.raises(ValueError, match="non-negative"):
        chunk_fixed_size("Some text", "doc", "src", chunk_size=200, overlap=-20)


# ============================================================================
# Sentence-Based Chunking Tests
# ============================================================================

def test_sentence_based_chunking_defaults():
    """Verify sentence-based chunking groups 3 consecutive sentences by default."""
    sentences = [
        "First sentence regarding loan eligibility.",
        "Second sentence covering CIBIL score requirements.",
        "Third sentence outlining income verification.",
        "Fourth sentence detailing co-applicant guidelines.",
        "Fifth sentence stating underwriting escalation desks.",
    ]
    text = " ".join(sentences)

    chunks = chunk_sentence_based(
        text,
        parent_document_id="doc_policy",
        source="policy.md",
        sentences_per_chunk=3,
    )

    assert len(chunks) == 2
    # Chunk 0 contains first 3 sentences
    assert chunks[0].chunk_id == "sent_doc_policy_0"
    assert chunks[0].chunking_strategy == "sentence_based"
    assert chunks[0].chunk_index == 0
    assert "First sentence" in chunks[0].text
    assert "Third sentence" in chunks[0].text
    assert "Fourth sentence" not in chunks[0].text

    # Chunk 1 contains remaining 2 sentences
    assert chunks[1].chunk_id == "sent_doc_policy_1"
    assert chunks[1].chunking_strategy == "sentence_based"
    assert chunks[1].chunk_index == 1
    assert "Fourth sentence" in chunks[1].text
    assert "Fifth sentence" in chunks[1].text


def test_sentence_based_short_and_empty_documents():
    """Verify sentence chunking handles short documents (<3 sentences) and empty strings."""
    # 1 sentence doc
    c1 = chunk_sentence_based("Only one sentence here.", "doc1", "src1", sentences_per_chunk=3)
    assert len(c1) == 1
    assert c1[0].text == "Only one sentence here."

    # 2 sentence doc
    c2 = chunk_sentence_based("Sentence one. Sentence two.", "doc2", "src2", sentences_per_chunk=3)
    assert len(c2) == 1
    assert "Sentence one." in c2[0].text and "Sentence two." in c2[0].text

    # Empty document
    c_empty = chunk_sentence_based("", "doc_empty", "src_empty", sentences_per_chunk=3)
    assert c_empty == []


# ============================================================================
# Metadata Verification Tests
# ============================================================================

def test_chunk_metadata_completeness():
    """Verify that every chunk contains all 5 mandatory metadata attributes."""
    text = "Cred members require valid KYC documents. Video KYC must be completed within 72 hours."
    fixed = chunk_fixed_size(text, parent_document_id="doc_kyc", source="kyc.md", chunk_size=50, overlap=10)
    sent = chunk_sentence_based(text, parent_document_id="doc_kyc", source="kyc.md", sentences_per_chunk=1)

    for chunk_list, strat in [(fixed, "fixed_size"), (sent, "sentence_based")]:
        assert len(chunk_list) > 0
        for chk in chunk_list:
            assert chk.chunk_id is not None and len(chk.chunk_id) > 0
            assert chk.parent_document_id == "doc_kyc"
            assert chk.source == "kyc.md"
            assert chk.chunking_strategy == strat
            assert isinstance(chk.chunk_index, int)
            assert chk.text is not None and len(chk.text) > 0

