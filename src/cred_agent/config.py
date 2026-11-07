"""Configuration settings for Cred Domain Support Agent.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.config
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR: Path = DATA_DIR / "knowledge_base"

# Vector Store Persistent Storage
DEFAULT_CHROMA_PATH: str = "./chroma_db"

# ChromaDB Collection Names
COLLECTION_FIXED_SIZE: str = "cred_fixed_size"
COLLECTION_SENTENCE_BASED: str = "cred_sentence_based"

# Local Embedding Model (Free, local SentenceTransformers)
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# Chunking Parameters
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100
DEFAULT_SENTENCES_PER_CHUNK: int = 3

