"""Least-autonomy policy for CrewAI tool registration and dispatch."""

import re
from typing import Any, Iterable

from crewai.hooks import register_before_tool_call_hook
from crewai.hooks.tool_hooks import ToolCallHookContext

LOOKUP_AGENT_ROLE = "Cred Loan Application Support Specialist"
RETRIEVAL_AGENT_ROLE = "Cred Knowledge Base Retrieval Specialist"
COMPOSER_AGENT_ROLE = "Cred Customer Support Response Composer"

LOOKUP_TOOL_NAME = "cred_application_record"
RETRIEVAL_TOOL_NAME = "cred_knowledge_base_search"

_AGENT_TOOL_ALLOWLIST = {
    LOOKUP_AGENT_ROLE: frozenset({LOOKUP_TOOL_NAME}),
    RETRIEVAL_AGENT_ROLE: frozenset({RETRIEVAL_TOOL_NAME}),
    COMPOSER_AGENT_ROLE: frozenset(),
}


def is_tool_authorized(agent_role: str, tool_name: str) -> bool:
    """Return whether the exact agent role may execute the named tool."""
    return tool_name in _AGENT_TOOL_ALLOWLIST.get(agent_role, frozenset())


def validate_agent_tool_permissions(agents: Iterable[Any]) -> None:
    """Fail closed if a constructed Crew violates the declared tool policy."""
    for agent in agents:
        role = getattr(agent, "role", "")
        tools = getattr(agent, "tools", None) or []
        tool_names = {getattr(tool, "name", "") for tool in tools}
        permitted = _AGENT_TOOL_ALLOWLIST.get(role)
        if permitted is None or tool_names != permitted:
            raise PermissionError(
                f"Tool registration is not authorized for agent role {role!r}."
            )
        if role == LOOKUP_AGENT_ROLE and tools:
            from src.cred_agent.agents.lookup_agent import (
                CredLoanApplicationLookupTool,
            )

            if any(type(tool) is not CredLoanApplicationLookupTool for tool in tools):
                raise PermissionError(
                    "The lookup role may register only the approved lookup tool."
                )
        if role == RETRIEVAL_AGENT_ROLE and tools:
            from src.cred_agent.agents.retrieval_agent import (
                CredKnowledgeRetrievalTool,
            )

            if any(type(tool) is not CredKnowledgeRetrievalTool for tool in tools):
                raise PermissionError(
                    "The retrieval role may register only the approved retrieval tool."
                )


def enforce_tool_dispatch(context: ToolCallHookContext) -> bool | None:
    """CrewAI pre-dispatch hook denying lookup calls from every other agent."""
    if context.tool_name != LOOKUP_TOOL_NAME:
        return None
    agent_role = getattr(context.agent, "role", "")
    tool_input = context.tool_input
    valid_lookup_input = (
        isinstance(tool_input, dict)
        and set(tool_input) == {"record_id"}
        and isinstance(tool_input.get("record_id"), str)
        and re.fullmatch(r"APP-\d+", tool_input["record_id"]) is not None
    )
    if not is_tool_authorized(agent_role, context.tool_name) or not valid_lookup_input:
        return False
    return None


register_before_tool_call_hook(enforce_tool_dispatch)
