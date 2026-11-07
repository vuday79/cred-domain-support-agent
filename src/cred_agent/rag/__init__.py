"""RAG (Retrieval-Augmented Generation) Core Package for Cred Domain Support Agent.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag

This package provides document loading, dual chunking strategies, local SentenceTransformer
embeddings, dual ChromaDB collections, grounded generation, and comparative evaluation.
"""

from src.cred_agent.rag.chunking import (
    Chunk,
    DocumentChunk,
    chunk_all_documents_fixed,
    chunk_all_documents_sentence,
    chunk_documents_fixed,
    chunk_documents_sentence,
    chunk_fixed_size,
    chunk_sentence_based,
)
from src.cred_agent.rag.document_loader import (
    DOCUMENT_TOPIC_MAP,
    Document,
    PolicyDocument,
    load_knowledge_base_documents,
    load_single_document,
)
from src.cred_agent.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingManager,
    PolicyEmbedder,
    get_embedding_model,
)
from src.cred_agent.rag.evaluation import (
    DEMONSTRATION_QUERIES,
    EVALUATION_BENCHMARK,
    QueryEvaluationResult,
    evaluate_query_at_document_level,
    evaluate_similarity_threshold,
    print_grounded_generation_demo,
    print_evaluation_report,
    print_similarity_threshold_report,
    run_benchmark_evaluation,
)
from src.cred_agent.rag.retrieval import (
    CALIBRATED_FALLBACK_THRESHOLD,
    FALLBACK_RESPONSE,
    GroundedGenerator,
    PolicyRetriever,
    calibrate_empirical_threshold,
)
from src.cred_agent.rag.vector_store import (
    COLLECTION_FIXED,
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE,
    COLLECTION_SENTENCE_BASED,
    ChromaVectorStore,
    get_vector_store,
    index_fixed_size_chunks,
    index_sentence_based_chunks,
    query_fixed_size,
    query_sentence_based,
)

__all__ = [
    # Document loading
    "Document",
    "PolicyDocument",
    "load_knowledge_base_documents",
    "load_single_document",
    "DOCUMENT_TOPIC_MAP",
    # Chunking
    "Chunk",
    "DocumentChunk",
    "chunk_documents_fixed",
    "chunk_documents_sentence",
    "chunk_all_documents_fixed",
    "chunk_all_documents_sentence",
    "chunk_fixed_size",
    "chunk_sentence_based",
    # Embeddings
    "EmbeddingManager",
    "PolicyEmbedder",
    "get_embedding_model",
    "DEFAULT_EMBEDDING_MODEL",
    # Vector store
    "ChromaVectorStore",
    "get_vector_store",
    "COLLECTION_FIXED_SIZE",
    "COLLECTION_SENTENCE_BASED",
    "COLLECTION_FIXED",
    "COLLECTION_SENTENCE",
    "index_fixed_size_chunks",
    "index_sentence_based_chunks",
    "query_fixed_size",
    "query_sentence_based",
    # Retrieval & generation
    "PolicyRetriever",
    "GroundedGenerator",
    "calibrate_empirical_threshold",
    "CALIBRATED_FALLBACK_THRESHOLD",
    "FALLBACK_RESPONSE",
    # Evaluation
    "QueryEvaluationResult",
    "EVALUATION_BENCHMARK",
    "DEMONSTRATION_QUERIES",
    "evaluate_query_at_document_level",
    "evaluate_similarity_threshold",
    "print_grounded_generation_demo",
    "run_benchmark_evaluation",
    "print_evaluation_report",
    "print_similarity_threshold_report",
]
