"""CrewAI Retrieval Agent backed by Cred's existing fixed-size RAG index."""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.cred_agent.config import (
    COLLECTION_FIXED_SIZE,
    COLLECTION_SENTENCE_BASED,
    DEFAULT_CHROMA_PATH,
    PROJECT_ROOT,
)
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.services.runtime_budget import RuntimeBudget
from src.cred_agent.rag.retrieval import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    PolicyRetriever,
)
from src.cred_agent.rag.vector_store import ChromaVectorStore

RETRIEVAL_SIMILARITY_THRESHOLD = 0.286211
RETRIEVAL_TOP_K = 3
DEFAULT_PERSIST_DIRECTORY = (PROJECT_ROOT / DEFAULT_CHROMA_PATH).resolve()


class CredKnowledgeQuestion(BaseModel):
    """Input schema for a customer question to the Cred knowledge base."""

    question: str = Field(
        min_length=1,
        description="The customer's policy or knowledge-base question.",
    )


class MissingRetrievalCollectionsError(RuntimeError):
    """Raised when persisted Chroma storage lacks an expected RAG collection."""


def load_fixed_size_retriever(
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
) -> PolicyRetriever:
    """Open existing cosine collections without creating or reindexing storage."""
    directory = Path(persist_directory).resolve()
    sqlite_path = directory / "chroma.sqlite3"
    if not directory.is_dir() or not sqlite_path.is_file():
        raise FileNotFoundError(
            f"ChromaDB persistence directory or database is missing: {directory}"
        )

    client = chromadb.PersistentClient(
        path=str(directory),
        settings=Settings(anonymized_telemetry=False),
    )
    existing_names = {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    }
    expected_names = {COLLECTION_FIXED_SIZE, COLLECTION_SENTENCE_BASED}
    missing = sorted(expected_names - existing_names)
    if missing:
        raise MissingRetrievalCollectionsError(
            f"Missing required ChromaDB collection(s) in {directory}: "
            f"{', '.join(missing)}. No collection was created."
        )

    for collection_name in sorted(expected_names):
        ChromaVectorStore._require_cosine_metric(
            client.get_collection(collection_name)
        )

    # ChromaVectorStore's constructor uses get_or_create_collection for both
    # collections; check all expected names above first to prevent accidental
    # creation when opening an incomplete persisted index.
    vector_store = ChromaVectorStore(persist_directory=directory)
    return PolicyRetriever(vector_store=vector_store)


class CredKnowledgeRetrievalTool(BaseTool):
    """Search approved Cred policy chunks and return only grounded results."""

    name: str = "cred_knowledge_base_search"
    description: str = (
        "Search the approved Cred policy knowledge base for a customer's "
        "question. Return the grounded answer, retrieved context, cosine "
        "similarities, and source-document attribution. If evidence is below "
        "the configured threshold, return the insufficient-information fallback."
    )
    args_schema: type[BaseModel] = CredKnowledgeQuestion
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Optional[PolicyRetriever] = None
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY

    def _run(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        retriever = self.retriever or load_fixed_size_retriever(
            self.persist_directory
        )
        result = GroundedGenerator(
            retriever=retriever,
            threshold=RETRIEVAL_SIMILARITY_THRESHOLD,
        ).generate(
            query=question.strip(),
            collection_name=COLLECTION_FIXED_SIZE,
            top_k=RETRIEVAL_TOP_K,
        )
        payload = {
            "question": result["query"],
            "answer": result["answer"],
            "is_grounded": result["is_grounded"],
            "collection": result["collection_used"],
            "similarity_threshold": result["threshold"],
            "top_1_cosine_similarity": result["top_1_similarity"],
            "source_documents": result["source_documents"],
            "retrieved_source_documents": result["retrieved_source_documents"],
            "retrieved_chunks": [
                {
                    "chunk_id": hit["chunk_id"],
                    "parent_document_id": hit["parent_document_id"],
                    "source": hit["source"],
                    "text": hit["text"],
                    "cosine_similarity": hit["cosine_similarity"],
                }
                for hit in result["retrieved_chunks"]
            ],
            "reason": result.get("reason", ""),
        }
        if not result["is_grounded"]:
            payload["answer"] = FALLBACK_RESPONSE
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def create_retrieval_agent(
    retriever: Optional[PolicyRetriever] = None,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    runtime_budget: Optional[RuntimeBudget] = None,
) -> Agent:
    """Create the deterministic Retrieval Agent and its RAG tool."""
    tool = CredKnowledgeRetrievalTool(
        retriever=retriever,
        persist_directory=Path(persist_directory).resolve(),
    )
    return Agent(
        role="Cred Knowledge Base Retrieval Specialist",
        goal=(
            "Retrieve accurate, relevant and grounded information from the "
            "Cred knowledge base."
        ),
        backstory=(
            "A specialist responsible for answering customer queries using "
            "approved knowledge-base documents."
        ),
        tools=[tool],
        llm=CredMockLLM(runtime_budget=runtime_budget),
        allow_delegation=False,
        max_iter=2,
        cache=False,
        verbose=False,
    )
