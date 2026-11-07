"""Tests for deterministic routing and the integrated three-agent Crew workflow."""

import socket
from typing import Any, Dict, List

import pytest
from crewai import Crew
from langchain_core.messages import AIMessage

from data.dataset import LOAN_APPLICATIONS
from src.cred_agent.crew import route_customer_query, run_customer_query
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE, PolicyRetriever
from src.cred_agent.services.session_memory import SessionConversationMemory
from src.cred_agent.schemas import (
    LoanApplicationStatus,
    SupportResponse,
    SupportResponseType,
)


class FixtureRetriever(PolicyRetriever):
    def __init__(self, similarity: float = 0.8):
        self.similarity = similarity
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
                "chunk_id": "fixed_emi_calculation_0",
                "text": (
                    "EMI is calculated using the reducing balance method based "
                    "on principal, tenure, and the annual interest rate."
                ),
                "parent_document_id": "emi_calculation",
                "doc_id": "emi_calculation",
                "source": "emi_calculation.md",
                "cosine_similarity": self.similarity,
                "similarity_score": self.similarity,
                "distance": 1.0 - self.similarity,
                "metadata": {
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                },
                "chunking_strategy": "fixed_size",
            }
        ][:top_k]


def test_integrated_query_router_routes_kb_lookup_combined_and_oos_queries():
    rag_route = route_customer_query("How is EMI calculated?")
    lookup_route = route_customer_query(
        "What is the status of loan application APP-1001?"
    )
    combined_route = route_customer_query(
        "What is the status of APP-1001 and how is EMI calculated?"
    )
    out_of_scope_route = route_customer_query("Who won the latest IPL season?")

    assert rag_route.retrieval_required and not rag_route.lookup_required
    assert lookup_route.lookup_required and not lookup_route.retrieval_required
    assert combined_route.retrieval_required and combined_route.lookup_required
    assert combined_route.record_id == "APP-1001"
    assert out_of_scope_route.retrieval_required
    assert not out_of_scope_route.lookup_required


def test_rag_query_invokes_retrieval_tool_and_composer_receives_findings():
    retriever = FixtureRetriever()
    result = run_customer_query("How is EMI calculated?", retriever=retriever)

    assert retriever.calls == [
        {
            "query": "How is EMI calculated?",
            "collection_name": "cred_fixed_size",
            "top_k": 3,
        }
    ]
    assert result.tool_invocations[0]["tool_name"] == "cred_knowledge_base_search"
    assert result.tool_invocations[0]["tool_args"] == {
        "question": "How is EMI calculated?"
    }
    assert result.specialist_findings[0]["tool_result"]["is_grounded"] is True
    assert isinstance(result.structured_response, SupportResponse)
    assert result.structured_response.response_type is SupportResponseType.KNOWLEDGE_BASE
    assert result.structured_response.is_grounded
    assert result.structured_response.source_documents == ["emi_calculation"]
    assert result.structured_response.retrieved_context
    assert "emi_calculation.md" in result.final_response
    assert "reducing balance method" in result.final_response
    assert result.kickoff_count == 2


def test_loan_query_invokes_lookup_tool_and_preserves_service_result():
    result = run_customer_query("What is the status of loan application APP-1001?")

    assert result.tool_invocations[0]["tool_name"] == "cred_application_record"
    assert result.tool_invocations[0]["tool_args"] == {"record_id": "APP-1001"}
    finding = result.specialist_findings[0]["tool_result"]
    assert finding["found"] is True
    assert finding["record_id"] == "APP-1001"
    assert finding["status"] == "Submitted"
    assert finding["loan_amount_inr"] == 750000
    assert finding["escalation_score"] == 0.23
    assert result.structured_response.response_type is SupportResponseType.LOAN_APPLICATION
    assert result.structured_response.loan_application is not None
    assert result.structured_response.loan_application.status is LoanApplicationStatus.SUBMITTED
    assert result.structured_response.loan_application.escalation_score == 0.23
    assert result.structured_response.loan_application.escalation_required is False
    assert "Status: Submitted" in result.final_response
    assert "INR 750,000" in result.final_response


def test_out_of_scope_query_returns_safe_fallback():
    result = run_customer_query(
        "Who won the latest IPL season?",
        retriever=FixtureRetriever(similarity=0.1),
    )

    assert result.specialist_findings[0]["tool_result"]["is_grounded"] is False
    assert result.final_response == FALLBACK_RESPONSE
    assert (
        result.structured_response.response_type
        is SupportResponseType.KNOWLEDGE_BASE_FALLBACK
    )
    assert not result.structured_response.is_grounded
    assert result.structured_response.source_documents == []


def test_unknown_application_id_is_preserved_as_not_found():
    result = run_customer_query("Tell me the status of APP-9999.")

    finding = result.specialist_findings[0]["tool_result"]
    assert finding["found"] is False
    assert finding["record_id"] == "APP-9999"
    assert finding["error"] == "not_found"
    assert (
        result.structured_response.response_type
        is SupportResponseType.APPLICATION_NOT_FOUND
    )
    assert result.structured_response.record_id == "APP-9999"
    assert "No loan application was found for record_id 'APP-9999'." in (
        result.final_response
    )
    assert "loan amount" not in result.final_response.lower()


def test_combined_query_runs_both_specialists_and_passes_both_findings():
    retriever = FixtureRetriever()
    result = run_customer_query(
        "What is the status of APP-1001 and how is EMI calculated?",
        retriever=retriever,
    )

    assert [item["tool_name"] for item in result.tool_invocations] == [
        "cred_knowledge_base_search",
        "cred_application_record",
    ]
    assert len(result.specialist_findings) == 2
    assert result.structured_response.response_type is SupportResponseType.COMBINED
    assert "emi_calculation.md" in result.final_response
    assert "APP-1001" in result.final_response
    assert "INR 750,000" in result.final_response
    assert result.kickoff_count == 2


def test_final_rag_response_does_not_add_unsupported_claims():
    result = run_customer_query(
        "How is EMI calculated?",
        retriever=FixtureRetriever(),
    )

    assert "reducing balance method" in result.final_response
    assert "emi_calculation.md" in result.final_response
    assert "approved" not in result.final_response.lower()
    assert "guarantee" not in result.final_response.lower()
    assert "processing fee" not in result.final_response.lower()


def test_required_escalation_is_not_reported_as_already_completed():
    flagged_id = next(
        application["record_id"]
        for application in LOAN_APPLICATIONS
        if application["flagged_for_fraud_review"]
    )
    result = run_customer_query(f"Check application {flagged_id}.")

    assert "Escalation is required." in result.final_response
    assert "does not confirm that escalation has been completed." in (
        result.final_response
    )


def test_crew_kickoff_is_invoked_for_specialist_and_composer(monkeypatch):
    kickoff_calls = 0
    original_kickoff = Crew.kickoff

    def track_kickoff(self: Crew, *args: Any, **kwargs: Any) -> Any:
        nonlocal kickoff_calls
        kickoff_calls += 1
        return original_kickoff(self, *args, **kwargs)

    monkeypatch.setattr(Crew, "kickoff", track_kickoff)
    result = run_customer_query(
        "How is EMI calculated?",
        retriever=FixtureRetriever(),
    )

    assert kickoff_calls == 2
    assert result.kickoff_count == kickoff_calls


def test_workflow_makes_no_external_network_call(monkeypatch):
    original_connect = socket.socket.connect

    def reject_external_connection(
        sock: socket.socket,
        address: Any,
    ) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Integrated workflow attempted external networking")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", reject_external_connection)
    result = run_customer_query(
        "How is EMI calculated?",
        retriever=FixtureRetriever(),
    )

    assert result.final_response
    assert result.tool_invocations


def test_mock_llm_is_deterministic_for_same_composition_input():
    llm = CredMockLLM()
    prompt = (
        "Specialist findings: "
        '{"is_grounded": true, "answer": "EMI uses reducing balance.", '
        '"source_documents": ["emi_calculation"]}'
    )

    assert llm.call(prompt, tools=[]) == llm.call(prompt, tools=[])


def test_session_memory_stores_the_validated_structured_response():
    results = []

    def workflow(query, history):
        result = run_customer_query(query, retriever=FixtureRetriever())
        results.append(result)
        return result.final_response

    memory = SessionConversationMemory(workflow=workflow)
    response = memory.chat("structured-session", "How is EMI calculated?")

    assert results[0].structured_response.response == response
    assert memory.get_messages("structured-session")[-1] == AIMessage(
        content=results[0].structured_response.response
    )


def test_route_rejects_empty_and_multiple_application_ids():
    with pytest.raises(ValueError, match="non-empty"):
        route_customer_query(" ")
    with pytest.raises(ValueError, match="no more than one"):
        route_customer_query("Compare APP-1001 and APP-1002.")
