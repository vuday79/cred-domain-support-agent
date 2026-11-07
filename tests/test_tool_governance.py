"""Least-autonomy registration and pre-dispatch authorization tests."""

from types import SimpleNamespace

import pytest
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    run_before_tool_call_hooks,
)

from src.cred_agent.agents.lookup_agent import CredLoanApplicationLookupTool
from src.cred_agent.agents.response_composer import (
    create_response_composer_agent,
)
from src.cred_agent.agents.retrieval_agent import (
    CredKnowledgeRetrievalTool,
    create_retrieval_agent,
)
from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.services import loan_application_lookup
from src.cred_agent.services.autogen_review import create_review_team
from src.cred_agent.services.tool_permissions import (
    COMPOSER_AGENT_ROLE,
    LOOKUP_AGENT_ROLE,
    LOOKUP_TOOL_NAME,
    RETRIEVAL_AGENT_ROLE,
    RETRIEVAL_TOOL_NAME,
    validate_agent_tool_permissions,
)


def _function_tool(name: str, argument_name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {argument_name: {"type": "string"}},
                "required": [argument_name],
                "additionalProperties": False,
            },
        },
    }


def test_agent_factories_register_only_their_approved_tools():
    retrieval = create_retrieval_agent()
    lookup = SimpleNamespace(
        role=LOOKUP_AGENT_ROLE,
        tools=[CredLoanApplicationLookupTool()],
    )
    composer = create_response_composer_agent()

    validate_agent_tool_permissions([retrieval, lookup, composer])

    assert [tool.name for tool in retrieval.tools] == [RETRIEVAL_TOOL_NAME]
    assert [tool.name for tool in lookup.tools] == [LOOKUP_TOOL_NAME]
    assert composer.tools == []
    team, _, _ = create_review_team()
    assert all(not agent._tools for agent in team._participants)


def test_registration_rejects_lookup_tool_wired_to_another_agent():
    miswired_retriever = create_retrieval_agent()
    miswired_retriever.tools = [CredLoanApplicationLookupTool()]

    with pytest.raises(PermissionError, match="Tool registration"):
        validate_agent_tool_permissions([miswired_retriever])


@pytest.mark.parametrize(
    "agent_role",
    [
        RETRIEVAL_AGENT_ROLE,
        COMPOSER_AGENT_ROLE,
        "Reviewer",
        "Editor",
    ],
)
def test_actual_crewai_dispatch_hook_blocks_unauthorized_lookup_without_service_call(
    monkeypatch,
    agent_role,
):
    service_calls = []

    def lookup_service(record_id):
        service_calls.append(record_id)
        return {"found": True, "record_id": record_id}

    monkeypatch.setattr(
        loan_application_lookup,
        "check_loan_application_status",
        lookup_service,
    )
    context = ToolCallHookContext(
        tool_name=LOOKUP_TOOL_NAME,
        tool_input={"record_id": "APP-1001"},
        tool=CredLoanApplicationLookupTool(),
        agent=SimpleNamespace(role=agent_role),
    )

    assert run_before_tool_call_hooks(context) is True
    assert service_calls == []


@pytest.mark.parametrize(
    "tool_input",
    [
        {"record_id": "APP-1001", "query": "use another schema"},
        {"record_id": "not-an-application-id"},
        {"query": "APP-1001"},
    ],
)
def test_lookup_dispatch_rejects_alternate_or_invalid_argument_payloads(tool_input):
    context = ToolCallHookContext(
        tool_name=LOOKUP_TOOL_NAME,
        tool_input=tool_input,
        tool=CredLoanApplicationLookupTool(),
        agent=SimpleNamespace(role=LOOKUP_AGENT_ROLE),
    )

    assert run_before_tool_call_hooks(context) is True


def test_lookup_agent_is_allowed_and_mock_llm_cannot_choose_other_agents_tool():
    lookup_context = ToolCallHookContext(
        tool_name=LOOKUP_TOOL_NAME,
        tool_input={"record_id": "APP-1001"},
        tool=CredLoanApplicationLookupTool(),
        agent=SimpleNamespace(role=LOOKUP_AGENT_ROLE),
    )
    assert run_before_tool_call_hooks(lookup_context) is False

    with pytest.raises(PermissionError, match="cannot invoke tool"):
        CredMockLLM().call(
            [{"role": "user", "content": "Check application APP-1001"}],
            tools=[_function_tool(LOOKUP_TOOL_NAME, "record_id")],
            from_agent=SimpleNamespace(role=RETRIEVAL_AGENT_ROLE),
        )

    with pytest.raises(PermissionError, match="cannot invoke tool"):
        CredMockLLM().call(
            [{"role": "user", "content": "Check application APP-1001"}],
            tools=[_function_tool("lookup_alias", "record_id")],
            from_agent=SimpleNamespace(role=LOOKUP_AGENT_ROLE),
        )

    with pytest.raises(PermissionError, match="unauthorized argument schema"):
        CredMockLLM().call(
            [{"role": "user", "content": "Check application APP-1001"}],
            tools=[_function_tool(LOOKUP_TOOL_NAME, "query")],
            from_agent=SimpleNamespace(role=LOOKUP_AGENT_ROLE),
        )


def test_authorized_lookup_tool_executes_only_after_lookup_agent_permission(
    monkeypatch,
):
    service_calls = []

    def lookup_service(record_id):
        service_calls.append(record_id)
        return {"found": True, "record_id": record_id}

    monkeypatch.setattr(
        loan_application_lookup,
        "check_loan_application_status",
        lookup_service,
    )
    tool = CredLoanApplicationLookupTool()
    context = ToolCallHookContext(
        tool_name=LOOKUP_TOOL_NAME,
        tool_input={"record_id": "APP-1001"},
        tool=tool,
        agent=SimpleNamespace(role=LOOKUP_AGENT_ROLE),
    )

    assert run_before_tool_call_hooks(context) is False
    result = tool.run(record_id="APP-1001")

    assert "APP-1001" in result
    assert service_calls == ["APP-1001"]


def test_retrieval_agent_cannot_register_a_tool_with_a_lookup_alias():
    agent = SimpleNamespace(
        role=RETRIEVAL_AGENT_ROLE,
        tools=[
            CredKnowledgeRetrievalTool(),
            CredLoanApplicationLookupTool(),
        ],
    )

    with pytest.raises(PermissionError):
        validate_agent_tool_permissions([agent])
