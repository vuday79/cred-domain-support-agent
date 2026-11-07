"""Local SentenceTransformers Embedding Module for Cred Domain Support Agent.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag.embeddings

This module provides local, offline vector embeddings using SentenceTransformers
('sentence-transformers/all-MiniLM-L6-v2'). It enforces singleton model caching
to load weights only once, returns 384-dimensional numerical vectors, and handles
empty or invalid inputs gracefully without paid APIs.
"""

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.cred_agent.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION
except ImportError:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

logger = logging.getLogger(__name__)

# Module-level singleton instance
_GLOBAL_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Retrieve or initialize the singleton SentenceTransformer instance.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Shared SentenceTransformer model instance loaded in memory.
    """
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        logger.info(f"Loading local SentenceTransformer model: '{model_name}'")
        _GLOBAL_MODEL = SentenceTransformer(model_name)
    return _GLOBAL_MODEL


class EmbeddingManager:
    """Manages generation of local vector embeddings with input validation and dimension guarantees."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize EmbeddingManager reusing the singleton model instance.

        Args:
            model_name: Identifier for the local SentenceTransformer model.
        """
        self.model_name = model_name
        self._model = get_embedding_model(model_name)
        self.dimension = EMBEDDING_DIMENSION

    def encode_text(self, text: Optional[str]) -> List[float]:
        """Generate a 384-dimensional embedding vector for a single text input.

        Gracefully handles empty, whitespace, or non-string inputs by returning a zero vector.

        Args:
            text: Single text string to embed.

        Returns:
            List of 384 floats representing the normalized embedding vector.
        """
        if text is None or not isinstance(text, str):
            logger.warning("Invalid input received for embedding; returning zero vector.")
            return [0.0] * self.dimension

        cleaned = text.strip()
        if not cleaned:
            logger.debug("Empty or whitespace string received; returning zero vector.")
            return [0.0] * self.dimension

        raw_emb = self._model.encode(
            [cleaned],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return [float(x) for x in raw_emb]

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate 384-dimensional embedding vectors for a batch of strings.

        Gracefully handles empty batches, empty texts, and invalid entries within the batch.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of 384-dimensional float vectors matching the order of input texts.
        """
        if not texts:
            return []

        results: List[List[float]] = [[0.0] * self.dimension] * len(texts)
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for idx, item in enumerate(texts):
            if item is not None and isinstance(item, str) and item.strip():
                valid_indices.append(idx)
                valid_texts.append(item.strip())
            else:
                results[idx] = [0.0] * self.dimension

        if valid_texts:
            embeddings = self._model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for orig_idx, emb in zip(valid_indices, embeddings):
                results[orig_idx] = [float(x) for x in emb]

        return results

    # Friendly convenience aliases
    def get_embedding(self, text: Optional[str]) -> List[float]:
        """Alias for encode_text."""
        return self.encode_text(text)

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Alias for encode_batch."""
        return self.encode_batch(texts, batch_size=batch_size)

    def compute_cosine_similarity(
        self,
        vec1: Union[List[float], np.ndarray],
        vec2: Union[List[float], np.ndarray],
    ) -> float:
        """Compute cosine similarity between two vectors in [-1.0, 1.0]."""
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Backward compatibility alias
PolicyEmbedder = EmbeddingManager


if __name__ == "__main__":
    manager = EmbeddingManager()
    print(f"Embedding Manager initialized with model: {manager.model_name}")
    print(f"Vector dimension: {manager.dimension}")

    # Single text test
    v = manager.encode_text("What is Cred loan eligibility?")
    print(f"Single text embedding vector length: {len(v)}")
    assert len(v) == 384, f"Expected 384 dimensions, got {len(v)}"

    # Empty text test
    v_empty = manager.encode_text("")
    print(f"Empty text embedding vector length: {len(v_empty)}, sum: {sum(v_empty)}")
    assert len(v_empty) == 384 and sum(v_empty) == 0.0

    # Batch test
    batch = ["First policy document.", "", "Second policy document.", None]
    batch_vecs = manager.encode_batch(batch)
    print(f"Batch embedding count: {len(batch_vecs)}, dimensions: {[len(x) for x in batch_vecs]}")
    assert len(batch_vecs) == 4
    assert all(len(x) == 384 for x in batch_vecs)
    print("Embedding module self-test passed successfully!")
