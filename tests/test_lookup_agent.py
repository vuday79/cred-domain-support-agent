"""Tests for the CrewAI Loan Application Lookup Agent and authorized tool."""

import json
import socket
from typing import Any

import pytest
from crewai import Crew, Process, Task
from pydantic import ValidationError

from data.dataset import LOAN_APPLICATIONS, get_loan_application_by_id
from src.cred_agent.agents.lookup_agent import (
    CredLoanApplicationLookupTool,
    LoanApplicationRecordId,
    create_lookup_agent,
)
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.services import loan_application_lookup
from src.cred_agent.services.loan_application_lookup import (
    check_loan_application_status,
)


def test_lookup_agent_initializes_with_only_the_authorized_tool():
    agent = create_lookup_agent()

    assert agent.role == "Cred Loan Application Support Specialist"
    assert agent.goal == (
        "Retrieve accurate loan application details and escalation "
        "information using the authorized lookup tool."
    )
    assert agent.backstory == (
        "A specialist responsible for helping customers understand their loan "
        "application status. The agent must rely exclusively on the application "
        "lookup tool for application-specific information."
    )
    agent_tools = agent.tools
    assert agent_tools is not None
    assert len(agent_tools) == 1
    assert isinstance(agent_tools[0], CredLoanApplicationLookupTool)
    agent_llm = agent.llm
    assert isinstance(agent_llm, CredMockLLM)
    assert agent_llm.api_key is None
    assert agent.allow_delegation is False


def test_declared_tool_schema_accepts_application_ids_and_rejects_bad_formats():
    assert LoanApplicationRecordId.model_validate(
        {"record_id": "APP-1001"}
    ).record_id == "APP-1001"

    for invalid_id in ("", "1001", "APP-ABC", "APP-"):
        with pytest.raises(ValidationError):
            LoanApplicationRecordId.model_validate({"record_id": invalid_id})


def test_tool_returns_complete_existing_dataset_lookup_result():
    raw_application = get_loan_application_by_id("APP-1001")
    assert raw_application is not None
    expected = check_loan_application_status("APP-1001")

    result = json.loads(
        CredLoanApplicationLookupTool().run(record_id="APP-1001")
    )

    assert result == expected
    assert result["found"] is True
    assert result["record_id"] == raw_application["record_id"]
    assert result["category"] == raw_application["category"]
    assert result["status"] == raw_application["status"]
    assert result["loan_amount_inr"] == raw_application["loan_amount_inr"]
    assert result["days_since_created"] == raw_application["days_since_created"]
    assert result["flagged_for_fraud_review"] == raw_application[
        "flagged_for_fraud_review"
    ]
    assert result["escalation_score"] == expected["escalation_score"]
    assert result["escalation_required"] is expected["escalation_required"]
    assert result["escalation_reason"] == expected["escalation_reason"]


def test_unknown_application_id_returns_service_not_found_result():
    result = json.loads(
        CredLoanApplicationLookupTool().run(record_id="APP-9999")
    )

    assert result == check_loan_application_status("APP-9999")
    assert result["found"] is False
    assert result["error"] == "not_found"
    assert result["record_id"] == "APP-9999"
    assert "category" not in result
    assert "status" not in result
    assert "escalation_score" not in result


def test_tool_invokes_existing_lookup_service(monkeypatch: pytest.MonkeyPatch):
    expected = {
        "found": False,
        "record_id": "APP-1001",
        "error": "not_found",
        "message": "test result",
    }
    calls: list[str] = []

    def fake_lookup(record_id: str) -> dict[str, Any]:
        calls.append(record_id)
        return expected

    monkeypatch.setattr(
        loan_application_lookup,
        "check_loan_application_status",
        fake_lookup,
    )

    result = json.loads(
        CredLoanApplicationLookupTool().run(record_id="APP-1001")
    )

    assert calls == ["APP-1001"]
    assert result == expected


def test_mock_llm_requires_registered_lookup_tool_and_record_id_schema():
    agent = create_lookup_agent()
    llm = agent.llm
    assert isinstance(llm, CredMockLLM)
    record_schema_tool = {
        "type": "function",
        "function": {
            "name": "cred_application_record",
            "description": "Unrelated tool label.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    calls = llm.call(
        [{"role": "user", "content": "Please check application APP-1001."}],
        tools=[record_schema_tool],  # type: ignore[arg-type]
        from_agent=agent,
    )

    assert isinstance(calls, list)
    call = calls[0]
    assert isinstance(call, dict)
    function = call["function"]
    assert isinstance(function, dict)
    assert function["name"] == "cred_application_record"
    assert json.loads(function["arguments"]) == {"record_id": "APP-1001"}


def test_lookup_tool_is_invoked_in_crew_without_external_llm_api(monkeypatch):
    original_connect = socket.socket.connect

    def reject_external_connection(
        sock: socket.socket,
        address: Any,
    ) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Lookup Agent must not call an external LLM API")
        original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", reject_external_connection)
    agent = create_lookup_agent()
    task = Task(
        description=(
            "Use the authorized application lookup tool for the customer's "
            "application. Customer question: Please check application APP-1001."
        ),
        expected_output="Return the complete structured lookup result.",
        agent=agent,
        tools=agent.tools or [],
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    assert '"record_id": "APP-1001"' in str(result)
    assert '"escalation_score": 0.23' in str(result)
    assert '"escalation_required": false' in str(result)
    agent_tools = agent.tools
    assert agent_tools is not None
    assert isinstance(agent_tools[0], CredLoanApplicationLookupTool)


def test_lookup_tool_does_not_change_dataset():
    original = get_loan_application_by_id("APP-1001")

    CredLoanApplicationLookupTool().run(record_id="APP-1001")

    assert get_loan_application_by_id("APP-1001") == original
    assert len(LOAN_APPLICATIONS) >= 1
