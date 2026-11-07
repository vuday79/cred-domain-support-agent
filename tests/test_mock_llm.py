"""Tests for the deterministic, schema-driven CrewAI MOCK_LLM."""

import json
import socket
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.cred_agent.mock_llm import (
    MOCK_UNSUPPORTED_RESPONSE,
    CredMockLLM,
)
from src.cred_agent.services.tool_permissions import (
    LOOKUP_AGENT_ROLE,
    RETRIEVAL_AGENT_ROLE,
)

RETRIEVAL_AGENT = SimpleNamespace(role=RETRIEVAL_AGENT_ROLE)
LOOKUP_AGENT = SimpleNamespace(role=LOOKUP_AGENT_ROLE)


def _function_tool(
    name: str,
    properties: dict,
    required: list[str],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "A test-only tool",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def test_initializes_without_api_key_and_uses_local_defaults():
    llm = CredMockLLM()

    assert llm.model == "cred-mock-v1"
    assert llm.provider == "mock"
    assert llm.api_key is None
    assert not CredMockLLM.__abstractmethods__
    assert llm.supports_function_calling()


def test_telemetry_is_disabled_in_environment():
    import os

    assert os.environ["CREWAI_DISABLE_TELEMETRY"].lower() == "true"
    assert os.environ["CREWAI_DISABLE_TRACKING"].lower() == "true"
    assert os.environ["OTEL_SDK_DISABLED"].lower() == "true"


def test_initialization_and_calls_do_not_open_network_connections(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("MOCK_LLM must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    llm = CredMockLLM()
    rag_tool = _function_tool(
        "cred_knowledge_base_search",
        {"question": {"type": "string"}},
        ["question"],
    )

    result = llm.call(
        [{"role": "user", "content": "What is the loan policy?"}],
        tools=[rag_tool],
        from_agent=RETRIEVAL_AGENT,
    )

    assert result[0]["function"]["name"] == "cred_knowledge_base_search"


def test_tool_call_is_deterministic_and_uses_declared_query_schema():
    llm = CredMockLLM()
    rag_tool = _function_tool(
        "cred_knowledge_base_search",
        {"question": {"type": "string", "minLength": 1}},
        ["question"],
    )
    messages = [{"role": "user", "content": "How is EMI calculated?"}]

    first = llm.call(messages, tools=[rag_tool], from_agent=RETRIEVAL_AGENT)
    second = llm.call(messages, tools=[rag_tool], from_agent=RETRIEVAL_AGENT)

    assert first == second
    assert first[0]["function"]["name"] == "cred_knowledge_base_search"
    assert json.loads(first[0]["function"]["arguments"]) == {
        "question": "How is EMI calculated?"
    }
    assert first[0]["type"] == "function"


def test_tool_call_extracts_customer_question_from_crewai_task_prompt():
    llm = CredMockLLM()
    rag_tool = _function_tool(
        "cred_knowledge_base_search",
        {"question": {"type": "string", "minLength": 1}},
        ["question"],
    )
    task_prompt = (
        "Current Task: Search policy. Customer question: How is EMI calculated?\n\n"
        "This is the expected criteria for your final answer: cite sources."
    )

    result = llm.call(
        [{"role": "user", "content": task_prompt}],
        tools=[rag_tool],
        from_agent=RETRIEVAL_AGENT,
    )

    assert json.loads(result[0]["function"]["arguments"]) == {
        "question": "How is EMI calculated?"
    }


def test_query_schema_remains_eligible_when_prompt_also_contains_application_id():
    llm = CredMockLLM()
    rag_tool = _function_tool(
        "cred_knowledge_base_search",
        {"question": {"type": "string", "minLength": 1}},
        ["question"],
    )
    prompt = (
        "Current Task: Retrieve policy. Customer question: "
        "How is EMI calculated for application APP-1001?"
    )

    result = llm.call(
        [{"role": "user", "content": prompt}],
        tools=[rag_tool],
        from_agent=RETRIEVAL_AGENT,
    )

    assert json.loads(result[0]["function"]["arguments"]) == {
        "question": "How is EMI calculated for application APP-1001?"
    }


def test_lookup_tool_selection_uses_record_id_argument_not_tool_name():
    llm = CredMockLLM()
    lookup_tool = _function_tool(
        "cred_application_record",
        {
            "record_id": {
                "type": "string",
                "pattern": r"APP-",
            }
        },
        ["record_id"],
    )

    result = llm.call(
        [{"role": "user", "content": "Check application APP-1042"}],
        tools=[lookup_tool],
        from_agent=LOOKUP_AGENT,
    )

    assert result[0]["function"]["name"] == "cred_application_record"
    assert json.loads(result[0]["function"]["arguments"]) == {
        "record_id": "APP-1042"
    }


def test_tool_result_follow_up_uses_structured_tool_message():
    llm = CredMockLLM()
    result = llm.call(
        [
            {
                "role": "system",
                "content": "Observation: this prompt example is not a tool result.",
            },
            {"role": "user", "content": "How is EMI calculated?"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "rag_lookup",
                "content": "EMI uses principal, rate, and tenure.",
            },
        ],
        tools=[],
    )

    assert result == "Tool result: EMI uses principal, rate, and tenure."
    assert "this prompt example" not in result


def test_tool_result_response_is_deterministic():
    llm = CredMockLLM()
    messages = [
        {"role": "user", "content": "Check application APP-1042"},
        {"role": "tool", "content": '{"status": "Under Review"}'},
    ]

    assert llm.call(messages) == llm.call(messages)
    assert llm.call(messages) == 'Tool result: {"status": "Under Review"}'


def test_invalid_declared_argument_type_is_rejected():
    llm = CredMockLLM()
    invalid_tool = _function_tool(
        "cred_knowledge_base_search",
        {"question": {"type": "integer"}},
        ["question"],
    )

    with pytest.raises(ValueError, match="must be declared with type 'string'"):
        llm.call(
            [{"role": "user", "content": "What is the loan policy?"}],
            tools=[invalid_tool],
            from_agent=RETRIEVAL_AGENT,
        )


def test_generated_arguments_are_checked_against_schema_constraints():
    llm = CredMockLLM()
    restricted_tool = _function_tool(
        "cred_knowledge_base_search",
        {
            "question": {
                "type": "string",
                "maxLength": 5,
            }
        },
        ["question"],
    )

    with pytest.raises(ValueError, match="exceeds its declared maxLength"):
        llm.call(
            [{"role": "user", "content": "How is EMI calculated?"}],
            tools=[restricted_tool],
            from_agent=RETRIEVAL_AGENT,
        )


def test_unsupported_request_returns_explicit_deterministic_message():
    llm = CredMockLLM()
    assert llm.call("Tell me about cricket.", tools=[]) == MOCK_UNSUPPORTED_RESPONSE
    with pytest.raises(PermissionError, match="cannot invoke tool"):
        llm.call(
            [{"role": "user", "content": "Tell me about cricket."}],
            tools=[
                _function_tool(
                    "tool_with_other_arguments",
                    {"amount": {"type": "number"}},
                    ["amount"],
                )
            ],
            from_agent=RETRIEVAL_AGENT,
        )


def test_response_model_without_supported_mapping_returns_unsupported_response():
    class UnknownResponse(BaseModel):
        explanation: str

    result = CredMockLLM().call(
        [{"role": "user", "content": "Give me a response."}],
        response_model=UnknownResponse,
    )

    assert result == MOCK_UNSUPPORTED_RESPONSE
