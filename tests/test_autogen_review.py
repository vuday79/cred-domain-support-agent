"""Tests for the independent, deterministic AutoGen review stage."""

import socket
from typing import Any, Dict, List

from autogen_agentchat.messages import StructuredMessage

from src.cred_agent.crew import run_customer_query
from src.cred_agent.rag.retrieval import PolicyRetriever
from src.cred_agent.schemas import (
    LoanApplicationDetails,
    LoanApplicationStatus,
    RetrievedContext,
    SimilarityMeasurement,
    SupportResponse,
    SupportResponseType,
)
from src.cred_agent.services.autogen_review import (
    DeterministicAutoGenModelClient,
    ReviewVerdict,
    create_review_team,
    review_response,
)
from src.cred_agent.services.guardrails import check_response_groundedness


class FixtureRetriever(PolicyRetriever):
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": "fixed_emi_calculation_0",
                "text": (
                    "EMIs are calculated using the reducing balance method based "
                    "on principal, tenure, and the annual interest rate."
                ),
                "parent_document_id": "emi_calculation",
                "source": "emi_calculation.md",
                "cosine_similarity": 0.9,
                "distance": 0.1,
                "chunking_strategy": "fixed_size",
                "metadata": {
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                    "chunk_index": 0,
                },
            }
        ][:top_k]


def _grounded_knowledge_response() -> SupportResponse:
    answer = (
        "Personal loan applicants must be Indian residents aged 21 to 58 years."
    )
    context = RetrievedContext(
        chunk_id="eligibility-0",
        parent_document_id="loan_eligibility",
        source="loan_eligibility.md",
        text=answer,
        cosine_similarity=0.9,
    )
    return SupportResponse(
        response=f"{answer}\nSource: loan_eligibility (loan_eligibility.md).",
        response_type=SupportResponseType.KNOWLEDGE_BASE,
        is_grounded=True,
        source_documents=["loan_eligibility"],
        retrieved_context=[context],
        similarity_measurements=[
            SimilarityMeasurement(
                chunk_id=context.chunk_id,
                cosine_similarity=context.cosine_similarity,
                distance=0.1,
            )
        ],
    )


def _grounded_loan_response() -> SupportResponse:
    loan = LoanApplicationDetails(
        record_id="APP-1001",
        category="Personal Loan",
        status=LoanApplicationStatus.SUBMITTED,
        loan_amount_inr=750000,
        days_since_created=7,
        flagged_for_fraud_review=False,
        escalation_score=0.23,
        escalation_required=False,
        escalation_reason=(
            "No fraud review flag; only recency contributes 0.2300 "
            "(7 days old). Score 0.2300 does not meet the 0.65 escalation threshold."
        ),
    )
    response = (
        "Application APP-1001. Category: Personal Loan. Status: Submitted. "
        "Loan amount: INR 750,000. Application age: 7 days. "
        "Fraud review flagged: No. Escalation score: 0.2300. "
        "Escalation is not required. Reason: "
        "No fraud review flag; only recency contributes 0.2300 (7 days old). "
        "Score 0.2300 does not meet the 0.65 escalation threshold."
    )
    return SupportResponse(
        response=response,
        response_type=SupportResponseType.LOAN_APPLICATION,
        is_grounded=True,
        source_documents=[],
        retrieved_context=[],
        similarity_measurements=[],
        record_id=loan.record_id,
        loan_application=loan,
    )


def test_team_has_reviewer_then_editor_two_turns_and_structured_message():
    team, reviewer_client, editor_client = create_review_team()

    assert team._participant_names == ["Reviewer", "Editor"]
    assert team._max_turns == 2
    assert team._message_factory._message_types[
        "StructuredMessage[ReviewVerdict]"
    ] is StructuredMessage[ReviewVerdict]
    assert team._participants[1]._output_content_type is ReviewVerdict
    assert team._participants[0]._tools == []
    assert team._participants[1]._tools == []
    assert isinstance(reviewer_client, DeterministicAutoGenModelClient)
    assert isinstance(editor_client, DeterministicAutoGenModelClient)


def test_grounded_knowledge_draft_is_approved_byte_for_byte_with_two_turns():
    draft = _grounded_knowledge_response()

    result = review_response(
        question="What are the personal loan age requirements?",
        draft=draft,
    )

    assert result.verdict.approved is True
    assert result.response.response == draft.response
    assert result.turn_count == 2
    assert result.agent_order == ["Reviewer", "Editor"]
    assert result.structured_message_received
    assert isinstance(result.verdict, ReviewVerdict)
    assert check_response_groundedness(result.response)[0]


def test_grounded_loan_lookup_draft_is_approved_unchanged():
    draft = _grounded_loan_response()

    result = review_response(
        question="What is the status of application APP-1001?",
        draft=draft,
    )

    assert result.verdict.approved is True
    assert result.response.response == draft.response
    assert result.response.loan_application == draft.loan_application
    assert result.turn_count == 2


def test_unsupported_claim_is_removed_and_supporting_attribution_retained():
    draft = _grounded_knowledge_response()
    injected = (
        draft.response
        + " Every loan application is guaranteed approval."
    )

    result = review_response(
        question="What are the personal loan age requirements?",
        draft=draft,
        draft_answer=injected,
    )

    assert result.verdict.approved is False
    assert "guaranteed approval" not in result.response.response.lower()
    assert (
        "Personal loan applicants must be Indian residents aged 21 to 58 years."
        in result.response.response
    )
    assert result.response.source_documents == ["loan_eligibility"]
    assert "loan_eligibility.md" in result.response.response
    assert check_response_groundedness(result.response)[0]


def test_only_unsupported_claims_produce_the_existing_safe_fallback():
    draft = _grounded_knowledge_response()

    result = review_response(
        question="What are the personal loan age requirements?",
        draft=draft,
        draft_answer="Every loan application is guaranteed approval.",
    )

    assert result.verdict.approved is False
    assert (
        result.response.response
        == "I don't have sufficient information in the available knowledge base "
        "to answer this question."
    )
    assert result.response.response_type is SupportResponseType.KNOWLEDGE_BASE_FALLBACK
    assert result.response.source_documents == []


def test_autogen_stage_is_integrated_after_actual_crewai_workflow():
    result = run_customer_query(
        "How is EMI calculated?",
        retriever=FixtureRetriever(),
    )

    assert result.review_approved is True
    assert result.review_turn_count == 2
    assert result.structured_response.response_type is SupportResponseType.KNOWLEDGE_BASE
    assert result.final_response == result.structured_response.response
    assert result.groundedness_passed is True


def test_review_pipeline_uses_no_external_network_connections(monkeypatch):
    original_connect = socket.socket.connect

    def reject_external_connection(
        sock: socket.socket,
        address: Any,
    ) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if host in {"127.0.0.1", "::1", "localhost"}:
            original_connect(sock, address)
            return
        raise AssertionError("AutoGen review must not make external network calls")

    monkeypatch.setattr(socket.socket, "connect", reject_external_connection)
    draft = _grounded_knowledge_response()

    result = review_response(
        question="What are the personal loan age requirements?",
        draft=draft,
    )

    assert result.turn_count == 2
    assert result.verdict.approved
