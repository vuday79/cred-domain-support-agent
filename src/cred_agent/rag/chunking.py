"""Chunking Strategies for Cred Domain Policy Documents.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag.chunking

This module implements:
1. Character-based fixed-size chunking with configurable overlap.
2. Sentence-based chunking grouping N consecutive sentences per chunk.
"""

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional
from src.cred_agent.rag.document_loader import Document


@dataclass
class Chunk:
    """Represents a text chunk produced from a parent policy document."""

    chunk_id: str
    parent_document_id: str
    source: str
    chunking_strategy: str  # 'fixed_size' or 'sentence_based'
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the character count of the chunk text."""
        return len(self.text)


# ============================================================================
# Strategy 1: Fixed-Size Character-Based Chunking with Overlap
# ============================================================================

def chunk_fixed_size(
    text: str,
    parent_document_id: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    """Split text into character-based fixed-size chunks with overlap.

    Guarantees:
        - Non-empty chunks only.
        - Strict validation of chunk_size > overlap >= 0 to prevent infinite loops.
        - Preserves parent document ID, source filename/path, and chunk index.

    Args:
        text: Text to chunk.
        parent_document_id: Identifier of the parent document.
        source: Source filename or filepath.
        chunk_size: Character length of each chunk window (default 500).
        overlap: Character overlap between consecutive chunks (default 100).

    Returns:
        List of Chunk objects with chunking_strategy='fixed_size'.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: List[Chunk] = []
    text_len = len(cleaned)
    step = chunk_size - overlap  # Guaranteed to be >= 1
    start = 0
    chunk_index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = cleaned[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"fixed_{parent_document_id}_{chunk_index}",
                    parent_document_id=parent_document_id,
                    source=source,
                    chunking_strategy="fixed_size",
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1

        if end == text_len:
            break

        start += step

    return chunks


# ============================================================================
# Strategy 2: Sentence-Based Chunking
# ============================================================================

def split_text_into_sentences(text: str) -> List[str]:
    """Split text into individual sentences handling punctuation, headers, and newlines.

    Args:
        text: Text to split.

    Returns:
        List of non-empty cleaned sentences.
    """
    if not text.strip():
        return []

    # Normalize line breaks and spaces
    normalized = re.sub(r"\s+", " ", text.strip())

    # Split on sentence boundary punctuation (. ! ?) followed by whitespace or string end
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Fallback if text didn't contain sentence terminators
    if not sentences and normalized:
        sentences = [normalized]

    return sentences


def chunk_sentence_based(
    text: str,
    parent_document_id: str,
    source: str,
    sentences_per_chunk: int = 3,
    stride: Optional[int] = None,
) -> List[Chunk]:
    """Group consecutive sentences into chunks, preserving complete linguistic propositions.

    Guarantees:
        - Handles short documents gracefully (e.g. fewer than 3 sentences).
        - Handles headings, blank lines, and whitespace without creating empty chunks.
        - Preserves parent document ID, source, and chunk index.

    Args:
        text: Text to chunk.
        parent_document_id: Identifier of the parent document.
        source: Source filename or filepath.
        sentences_per_chunk: Number of consecutive sentences per chunk (default 3).
        stride: Step size between chunk starts (defaults to sentences_per_chunk).

    Returns:
        List of Chunk objects with chunking_strategy='sentence_based'.

    Raises:
        ValueError: If sentences_per_chunk <= 0 or stride <= 0.
    """
    if sentences_per_chunk <= 0:
        raise ValueError(f"sentences_per_chunk must be positive, got {sentences_per_chunk}")

    effective_stride = stride if stride is not None else sentences_per_chunk
    if effective_stride <= 0:
        raise ValueError(f"stride must be positive, got {effective_stride}")

    sentences = split_text_into_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    chunk_index = 0

    for i in range(0, len(sentences), effective_stride):
        group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(group).strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"sent_{parent_document_id}_{chunk_index}",
                    parent_document_id=parent_document_id,
                    source=source,
                    chunking_strategy="sentence_based",
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1

    return chunks


# ============================================================================
# Document Batch Operations
# ============================================================================

def chunk_documents_fixed(
    documents: List[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    """Apply fixed-size chunking across a collection of Document objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_fixed_size(
            text=doc.text,
            parent_document_id=doc.doc_id,
            source=doc.source,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


def chunk_documents_sentence(
    documents: List[Document],
    sentences_per_chunk: int = 3,
    stride: Optional[int] = None,
) -> List[Chunk]:
    """Apply sentence-based chunking across a collection of Document objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        chunks = chunk_sentence_based(
            text=doc.text,
            parent_document_id=doc.doc_id,
            source=doc.source,
            sentences_per_chunk=sentences_per_chunk,
            stride=stride,
        )
        all_chunks.extend(chunks)
    return all_chunks


# Backward compatibility aliases
DocumentChunk = Chunk
chunk_all_documents_fixed = chunk_documents_fixed
chunk_all_documents_sentence = chunk_documents_sentence
chunk_text_fixed_size = chunk_fixed_size
chunk_text_sentence_based = chunk_sentence_based


if __name__ == "__main__":

    from src.cred_agent.rag.document_loader import load_knowledge_base_documents

    docs = load_knowledge_base_documents()
    fixed_chunks = chunk_documents_fixed(docs, chunk_size=500, overlap=100)
    sent_chunks = chunk_documents_sentence(docs, sentences_per_chunk=3)

    print(f"Loaded {len(docs)} documents.")
    print(f"Fixed-size chunks (size=500, overlap=100) : {len(fixed_chunks)} chunks")
    print(f"Sentence-based chunks (3 sentences/chunk)  : {len(sent_chunks)} chunks")

    sample_f = fixed_chunks[0]
    print(f"\nSample Fixed Chunk [{sample_f.chunk_id}]:")
    print(f"  Parent: {sample_f.parent_document_id} | Source: {sample_f.source} | Index: {sample_f.chunk_index}")
    print(f"  Length: {sample_f.char_count} chars\n  Text  : {sample_f.text[:120]}...")

    sample_s = sent_chunks[0]
    print(f"\nSample Sentence Chunk [{sample_s.chunk_id}]:")
    print(f"  Parent: {sample_s.parent_document_id} | Source: {sample_s.source} | Index: {sample_s.chunk_index}")
    print(f"  Length: {sample_s.char_count} chars\n  Text  : {sample_s.text[:120]}...")
