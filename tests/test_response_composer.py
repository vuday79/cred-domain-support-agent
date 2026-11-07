"""Tests for deterministic customer-facing composition of specialist findings."""

import json
import socket
from typing import Any

from crewai import Crew, Process

from data.dataset import LOAN_APPLICATIONS
from src.cred_agent.agents.lookup_agent import CredLoanApplicationLookupTool
from src.cred_agent.agents.response_composer import (
    create_response_composer_agent,
    create_response_composer_task,
)
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE
from src.cred_agent.schemas import SupportResponse
from src.cred_agent.services import loan_application_lookup


def _compose(specialist_findings: str) -> str:
    agent = create_response_composer_agent()
    task = create_response_composer_task(agent)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    output = crew.kickoff(inputs={"specialist_findings": specialist_findings})
    return output.tasks_output[-1].pydantic.response


def _rag_findings(
    answer: str = (
        "EMIs are calculated using the reducing balance method based on "
        "principal, tenure, and the applicable annual interest rate."
    ),
    is_grounded: bool = True,
) -> str:
    return json.dumps(
        {
            "question": "How is EMI calculated?",
            "answer": answer,
            "is_grounded": is_grounded,
            "source_documents": ["emi_calculation"],
            "retrieved_chunks": [
                {
                    "chunk_id": "fixed_emi_calculation_0",
                    "parent_document_id": "emi_calculation",
                    "source": "emi_calculation.md",
                    "text": answer,
                    "cosine_similarity": 0.45,
                }
            ],
        }
    )


def test_agent_initializes_with_required_role_goal_backstory_and_mock_llm():
    agent = create_response_composer_agent()

    assert agent.role == "Cred Customer Support Response Composer"
    assert agent.goal == (
        "Convert specialist agent findings into accurate, concise, "
        "customer-friendly support responses."
    )
    assert agent.backstory == (
        "An experienced customer-support communication specialist who combines "
        "verified knowledge-base information and loan application lookup results "
        "into clear responses. The specialist never invents facts or makes "
        "unsupported promises."
    )
    assert agent.tools == []
    assert isinstance(agent.llm, CredMockLLM)
    assert agent.llm.api_key is None


def test_composer_task_uses_crewai_supported_response_model():
    agent = create_response_composer_agent()
    task = create_response_composer_task(agent)

    assert task.agent is agent
    assert task.output_pydantic is SupportResponse
    assert task.response_model is SupportResponse
    assert "{specialist_findings}" in task.description
    assert task.tools == []


def test_successful_rag_composition_preserves_answer_and_source():
    response = _compose(_rag_findings())

    assert (
        "EMIs are calculated using the reducing balance method based on "
        "principal, tenure, and the applicable annual interest rate."
    ) in response
    assert "emi_calculation" in response
    assert "emi_calculation.md" in response


def test_successful_lookup_composition_preserves_dataset_facts_and_escalation():
    finding = CredLoanApplicationLookupTool().run(record_id="APP-1001")
    record = next(item for item in LOAN_APPLICATIONS if item["record_id"] == "APP-1001")
    response = _compose(finding)

    assert "APP-1001" in response
    assert record["category"] in response
    assert record["status"] in response
    assert f"INR {record['loan_amount_inr']:,}" in response
    assert f"{record['days_since_created']} days" in response
    assert f"{loan_application_lookup.check_loan_application_status('APP-1001')['escalation_score']:.4f}" in response
    assert "Escalation is not required." in response
    assert "escalation has been completed" not in response.lower()


def test_required_escalation_is_not_reported_as_completed():
    flagged_record = next(
        record for record in LOAN_APPLICATIONS
        if record["flagged_for_fraud_review"]
    )
    finding = CredLoanApplicationLookupTool().run(
        record_id=flagged_record["record_id"]
    )
    response = _compose(finding)

    assert "Escalation is required." in response
    assert "does not confirm that escalation has been completed." in response


def test_unknown_application_is_communicated_without_invented_details():
    finding = CredLoanApplicationLookupTool().run(record_id="APP-9999")
    response = _compose(finding)

    assert "No loan application was found for record_id 'APP-9999'." in response
    assert "loan amount" not in response.lower()
    assert "status:" not in response.lower()
    assert "escalation score" not in response.lower()


def test_insufficient_knowledge_base_findings_preserve_fallback():
    response = _compose(
        _rag_findings(answer=FALLBACK_RESPONSE, is_grounded=False)
    )

    assert response == FALLBACK_RESPONSE


def test_composer_does_not_add_unsupported_policy_or_application_facts():
    response = _compose(_rag_findings())

    assert "reducing balance method" in response
    assert "emi_calculation.md" in response
    assert "approval" not in response.lower()
    assert "fee" not in response.lower()
    assert "business days" not in response.lower()
    assert "guarantee" not in response.lower()


def test_composer_output_is_deterministic_for_repeated_inputs():
    findings = _rag_findings()

    assert _compose(findings) == _compose(findings)


def test_composer_task_uses_mock_llm_without_external_network(monkeypatch):
    original_connect = socket.socket.connect

    def reject_external_connection(
        sock: socket.socket,
        address: Any,
    ) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Response Composer must not call an external API")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", reject_external_connection)
    findings = CredLoanApplicationLookupTool().run(record_id="APP-1001")

    response = _compose(findings)

    assert "APP-1001" in response
    assert isinstance(create_response_composer_agent().llm, CredMockLLM)
