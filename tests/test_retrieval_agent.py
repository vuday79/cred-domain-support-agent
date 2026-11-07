"""Tests for the CrewAI Retrieval Agent and its grounded RAG tool."""

import json
import socket
from typing import Any, Dict, List

import pytest
from crewai import Crew, Process, Task

from src.cred_agent.agents.retrieval_agent import (
    RETRIEVAL_SIMILARITY_THRESHOLD,
    CredKnowledgeRetrievalTool,
    create_retrieval_agent,
)
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.rag.retrieval import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    PolicyRetriever,
)
from src.cred_agent.rag.vector_store import COLLECTION_FIXED


class FixtureRetriever(PolicyRetriever):
    def __init__(self, similarity: float, text: str, parent: str, source: str):
        self.similarity = similarity
        self.text = text
        self.parent = parent
        self.source = source
        self.calls: List[Dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "collection_name": collection_name,
                "top_k": top_k,
            }
        )
        return [
            {
                "chunk_id": "fixed_emi_0",
                "text": self.text,
                "parent_document_id": self.parent,
                "doc_id": self.parent,
                "source": self.source,
                "cosine_similarity": self.similarity,
                "similarity_score": self.similarity,
                "distance": 1.0 - self.similarity,
                "metadata": {
                    "parent_document_id": self.parent,
                    "source": self.source,
                },
                "chunking_strategy": "fixed_size",
            }
        ][:top_k]


def _tool_output(
    similarity: float = 0.8,
    text: str = "EMI is calculated using the reducing balance method.",
) -> tuple[dict[str, Any], FixtureRetriever]:
    retriever = FixtureRetriever(
        similarity=similarity,
        text=text,
        parent="emi_calculation",
        source="emi_calculation.md",
    )
    output = CredKnowledgeRetrievalTool(retriever=retriever).run(
        question="How is EMI calculated?"
    )
    return json.loads(output), retriever


def test_agent_initializes_with_required_configuration():
    agent = create_retrieval_agent(
        retriever=FixtureRetriever(
            0.8,
            "EMI is calculated using the reducing balance method.",
            "emi_calculation",
            "emi_calculation.md",
        )
    )

    assert agent.role == "Cred Knowledge Base Retrieval Specialist"
    assert agent.goal == (
        "Retrieve accurate, relevant and grounded information from the Cred "
        "knowledge base."
    )
    assert agent.backstory == (
        "A specialist responsible for answering customer queries using "
        "approved knowledge-base documents."
    )
    agent_tools = agent.tools
    assert agent_tools is not None
    assert len(agent_tools) == 1
    assert isinstance(agent_tools[0], CredKnowledgeRetrievalTool)
    agent_llm = agent.llm
    assert isinstance(agent_llm, CredMockLLM)
    assert agent_llm.api_key is None
    assert agent.allow_delegation is False


def test_retrieval_tool_returns_emi_content_and_fixed_collection():
    result, retriever = _tool_output()

    assert result["answer"] == "EMI is calculated using the reducing balance method."
    assert result["is_grounded"] is True
    assert result["collection"] == COLLECTION_FIXED
    assert result["similarity_threshold"] == RETRIEVAL_SIMILARITY_THRESHOLD
    assert retriever.calls == [
        {
            "query": "How is EMI calculated?",
            "collection_name": COLLECTION_FIXED,
            "top_k": 3,
        }
    ]


def test_retrieval_tool_includes_source_and_parent_attribution():
    result, _ = _tool_output()

    assert result["source_documents"] == ["emi_calculation"]
    assert result["retrieved_chunks"][0]["source"] == "emi_calculation.md"
    assert result["retrieved_chunks"][0]["parent_document_id"] == "emi_calculation"
    assert result["retrieved_chunks"][0]["cosine_similarity"] == pytest.approx(0.8)
    assert result["retrieved_chunks"][0]["text"].startswith("EMI is calculated")


def test_out_of_scope_question_returns_insufficient_information_fallback():
    retriever = FixtureRetriever(
        0.10,
        "KYC requires PAN and address documents.",
        "kyc_requirements",
        "kyc_requirements.md",
    )
    result = json.loads(
        CredKnowledgeRetrievalTool(retriever=retriever).run(
            question="Who won the latest IPL season?"
        )
    )

    assert result["answer"] == FALLBACK_RESPONSE
    assert result["is_grounded"] is False
    assert result["source_documents"] == []
    assert result["retrieved_source_documents"] == ["kyc_requirements"]
    assert result["retrieved_chunks"][0]["source"] == "kyc_requirements.md"


def test_similarity_threshold_is_enforced_at_boundary():
    below, _ = _tool_output(similarity=RETRIEVAL_SIMILARITY_THRESHOLD - 0.000001)
    at_threshold, _ = _tool_output(similarity=RETRIEVAL_SIMILARITY_THRESHOLD)

    assert below["answer"] == FALLBACK_RESPONSE
    assert below["is_grounded"] is False
    assert at_threshold["is_grounded"] is True


def test_unrelated_above_threshold_chunk_cannot_ground_below_threshold_evidence():
    class RankedEvidenceRetriever(PolicyRetriever):
        def retrieve(self, query, collection_name, top_k=3):
            return [
                {
                    "chunk_id": "fixed_joint_accounts_1",
                    "text": "Joint account holders must provide unanimous consent.",
                    "parent_document_id": "joint_accounts",
                    "source": "joint_accounts.md",
                    "cosine_similarity": 0.2989023328,
                    "distance": 1.0 - 0.2989023328,
                    "chunking_strategy": "fixed_size",
                    "metadata": {"chunk_index": 1},
                },
                {
                    "chunk_id": "fixed_emi_calculation_0",
                    "text": (
                        "Borrowers may select tenures from 6 to 60 months."
                    ),
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                    "cosine_similarity": 0.2829895616,
                    "distance": 1.0 - 0.2829895616,
                    "chunking_strategy": "fixed_size",
                    "metadata": {"chunk_index": 0},
                },
                {
                    "chunk_id": "fixed_emi_calculation_1",
                    "text": (
                        "Payments are auto-debited on the 5th calendar day."
                    ),
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                    "cosine_similarity": 0.279114604,
                    "distance": 1.0 - 0.279114604,
                    "chunking_strategy": "fixed_size",
                    "metadata": {"chunk_index": 1},
                },
            ][:top_k]

    result = GroundedGenerator(
        retriever=RankedEvidenceRetriever(),
        threshold=RETRIEVAL_SIMILARITY_THRESHOLD,
    ).generate(
        query="What tenure options are available?",
        collection_name=COLLECTION_FIXED,
        top_k=3,
    )

    assert result["top_1_similarity"] > RETRIEVAL_SIMILARITY_THRESHOLD
    assert result["answer"] == FALLBACK_RESPONSE
    assert result["is_grounded"] is False
    assert result["source_documents"] == []


def test_only_above_threshold_supporting_chunks_are_used_for_grounded_answer():
    class RankedEvidenceRetriever(PolicyRetriever):
        def retrieve(self, query, collection_name, top_k=3):
            return [
                {
                    "chunk_id": "fixed_emi_calculation_0",
                    "text": (
                        "Borrowers may select tenures from 6 to 60 months."
                    ),
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                    "cosine_similarity": 0.31,
                    "distance": 0.69,
                    "chunking_strategy": "fixed_size",
                    "metadata": {"chunk_index": 0},
                },
                {
                    "chunk_id": "fixed_joint_accounts_1",
                    "text": "Joint account holders must provide unanimous consent.",
                    "parent_document_id": "joint_accounts",
                    "source": "joint_accounts.md",
                    "cosine_similarity": 0.29,
                    "distance": 0.71,
                    "chunking_strategy": "fixed_size",
                    "metadata": {"chunk_index": 1},
                },
            ][:top_k]

    result = GroundedGenerator(
        retriever=RankedEvidenceRetriever(),
        threshold=RETRIEVAL_SIMILARITY_THRESHOLD,
    ).generate(
        query="What tenure options are available?",
        collection_name=COLLECTION_FIXED,
        top_k=2,
    )

    assert result["is_grounded"] is True
    assert result["source_documents"] == ["emi_calculation"]
    assert result["answer"] == "Borrowers may select tenures from 6 to 60 months."


def test_agent_native_tool_call_does_not_invoke_external_llm_api(monkeypatch):
    original_connect = socket.socket.connect

    def reject_external_connection(
        sock: socket.socket,
        address: Any,
    ) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Retrieval Agent must not call an external LLM API")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", reject_external_connection)
    retriever = FixtureRetriever(
        0.8,
        "EMI is calculated using the reducing balance method.",
        "emi_calculation",
        "emi_calculation.md",
    )
    agent = create_retrieval_agent(retriever=retriever)
    task = Task(
        description=(
            "Use the Cred knowledge-base search tool for the customer's "
            "question. Customer question: {question}"
        ),
        expected_output="Return the grounded tool result.",
        agent=agent,
        tools=agent.tools or [],
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"question": "How is EMI calculated?"})

    assert "EMI is calculated using the reducing balance method." in str(result)
    assert retriever.calls[0]["query"] == "How is EMI calculated?"
    agent_llm = agent.llm
    assert isinstance(agent_llm, CredMockLLM)
    assert agent_llm.provider == "mock"
    assert agent_llm.api_key is None
