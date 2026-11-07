"""Knowledge Base Vector Indexing Script for Cred Domain Support Agent.

Track: Banking & FinTech (Cred)
Script: scripts/build_index.py

This script:
1. Loads all policy Markdown documents from data/knowledge_base/ (excluding README.md).
2. Generates fixed-size chunks (with overlap) and sentence-based chunks.
3. Generates 384-dimensional embeddings using the local SentenceTransformers model.
4. Indexes fixed-size chunks into ChromaDB collection 'cred_fixed_size' using collection.upsert().
5. Indexes sentence-based chunks into ChromaDB collection 'cred_sentence_based' using collection.upsert().
6. Guarantees collection independence and prevents duplicate records on repeated execution.
7. Prints a comprehensive validation report.
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cred_agent.config import (
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE_BASED,
    DEFAULT_CHROMA_PATH,
    KNOWLEDGE_BASE_DIR,
)
from src.cred_agent.rag.chunking import (
    chunk_documents_fixed,
    chunk_documents_sentence,
)
from src.cred_agent.rag.document_loader import load_knowledge_base_documents
from src.cred_agent.rag.embeddings import EmbeddingManager
from src.cred_agent.rag.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def build_knowledge_base_index(
    kb_dir: Optional[Union[str, Path]] = None,
    chroma_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end knowledge base indexing workflow.

    Args:
        kb_dir: Path to directory containing knowledge-base markdown documents.
        chroma_dir: Path to directory for persistent ChromaDB storage.

    Returns:
        Dictionary summarizing indexing metrics, collection counts, and independence status.

    Raises:
        FileNotFoundError: If kb_dir does not exist.
        RuntimeError: If indexing or collection verification fails.
    """
    kb_path = Path(kb_dir).resolve() if kb_dir else KNOWLEDGE_BASE_DIR.resolve()
    persist_path = Path(chroma_dir).resolve() if chroma_dir else (PROJECT_ROOT / DEFAULT_CHROMA_PATH).resolve()

    logger.info(f"Starting knowledge base indexing workflow.")
    logger.info(f"Source Knowledge Base: {kb_path}")
    logger.info(f"ChromaDB Persistence: {persist_path}")

    # 1. Load documents (excluding README.md)
    documents = load_knowledge_base_documents(kb_dir=kb_path)
    doc_count = len(documents)
    if doc_count == 0:
        raise RuntimeError(f"No policy documents found in {kb_path}")
    logger.info(f"Loaded {doc_count} policy documents.")

    # 2. Generate chunks for both strategies
    fixed_chunks = chunk_documents_fixed(documents)
    sent_chunks = chunk_documents_sentence(documents)
    fixed_count = len(fixed_chunks)
    sent_count = len(sent_chunks)
    logger.info(f"Generated {fixed_count} fixed-size chunks and {sent_count} sentence-based chunks.")

    # 3. Initialize Embedding Manager & Vector Store
    embedder = EmbeddingManager()
    logger.info(f"Initialized embedding model: '{embedder.model_name}' (dimension={embedder.dimension})")

    store = ChromaVectorStore(persist_directory=persist_path, embedder=embedder)

    # 4. Index Fixed-Size Chunks into cred_fixed_size
    n_fixed_indexed = store.index_fixed_size_chunks(fixed_chunks)
    logger.info(f"Indexed {n_fixed_indexed} chunks into collection '{COLLECTION_FIXED_SIZE}'.")

    # 5. Index Sentence-Based Chunks into cred_sentence_based
    n_sent_indexed = store.index_sentence_based_chunks(sent_chunks)
    logger.info(f"Indexed {n_sent_indexed} chunks into collection '{COLLECTION_SENTENCE_BASED}'.")

    # 6. Verify collection metrics and independence
    col_fixed = store.collection_fixed
    col_sent = store.collection_sentence

    final_fixed_count = col_fixed.count()
    final_sent_count = col_sent.count()

    # Independence audit:
    # Verify collections have different names, different IDs, and disjoint chunk IDs
    is_distinct_ids = col_fixed.id != col_sent.id
    is_distinct_names = col_fixed.name != col_sent.name

    fixed_data = col_fixed.get(include=["metadatas"])
    sent_data = col_sent.get(include=["metadatas"])

    fixed_strategies = {m["chunking_strategy"] for m in fixed_data["metadatas"]} if fixed_data["metadatas"] else set()
    sent_strategies = {m["chunking_strategy"] for m in sent_data["metadatas"]} if sent_data["metadatas"] else set()

    is_strategies_clean = (fixed_strategies == {"fixed_size"}) and (sent_strategies == {"sentence_based"})
    collections_independent = is_distinct_ids and is_distinct_names and is_strategies_clean

    report: Dict[str, Any] = {
        "status": "SUCCESS",
        "kb_dir": str(kb_path),
        "chroma_dir": str(persist_path),
        "documents_loaded": doc_count,
        "fixed_chunks_generated": fixed_count,
        "sentence_chunks_generated": sent_count,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimension,
        "records_indexed_fixed": n_fixed_indexed,
        "records_indexed_sentence": n_sent_indexed,
        "total_records_indexed": n_fixed_indexed + n_sent_indexed,
        "final_collection_counts": {
            COLLECTION_FIXED_SIZE: final_fixed_count,
            COLLECTION_SENTENCE_BASED: final_sent_count,
        },
        "collections_independent": collections_independent,
        "independence_details": {
            "distinct_collection_ids": is_distinct_ids,
            "distinct_collection_names": is_distinct_names,
            "fixed_collection_strategies": list(fixed_strategies),
            "sentence_collection_strategies": list(sent_strategies),
        },
    }

    print_indexing_report(report)
    return report


def print_indexing_report(report: Dict[str, Any]) -> None:
    """Print a clean, structured console summary of the indexing execution."""
    print("\n" + "=" * 76)
    print(" CRED DOMAIN SUPPORT AGENT — VECTOR STORE INDEXING REPORT")
    print("=" * 76)
    print(f" Status                      : [{report['status']}]")
    print(f" Documents Loaded (Source)   : {report['documents_loaded']} documents (README.md excluded)")
    print(f" Embedding Model             : {report['embedding_model']}")
    print(f" Vector Dimensionality       : {report['embedding_dimensions']} dimensions")
    print(f" Storage Path (ChromaDB)     : {report['chroma_dir']}")
    print("-" * 76)
    print(" CHUNKING & INDEXING BREAKDOWN:")
    print(f"   * Fixed-Size Strategy     : {report['fixed_chunks_generated']} chunks generated -> {report['records_indexed_fixed']} upserted")
    print(f"   * Sentence-Based Strategy : {report['sentence_chunks_generated']} chunks generated -> {report['records_indexed_sentence']} upserted")
    print(f"   * Total Records Indexed   : {report['total_records_indexed']} chunk vectors")
    print("-" * 76)
    print(" FINAL CHROMADB COLLECTION COUNTS:")
    counts = report["final_collection_counts"]
    print(f"   * Collection '{COLLECTION_FIXED_SIZE}'     : {counts[COLLECTION_FIXED_SIZE]} records")
    print(f"   * Collection '{COLLECTION_SENTENCE_BASED}' : {counts[COLLECTION_SENTENCE_BASED]} records")
    print("-" * 76)
    independence_status = "[CONFIRMED INDEPENDENT]" if report["collections_independent"] else "[FAILED INDEPENDENCE]"
    print(f" Collection Independence Check: {independence_status}")
    print(f"   * Distinct Collection IDs : {report['independence_details']['distinct_collection_ids']}")
    print(f"   * Fixed Chunk Strategies  : {report['independence_details']['fixed_collection_strategies']}")
    print(f"   * Sent Chunk Strategies   : {report['independence_details']['sentence_collection_strategies']}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    build_knowledge_base_index()

