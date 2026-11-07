"""Deterministically route customer queries through Cred's three CrewAI agents."""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from crewai import Agent, Crew, Process, Task
from pydantic import ValidationError

from src.cred_agent.agents.lookup_agent import (
    CredLoanApplicationLookupTool,
    LoanApplicationRecordId,
    create_lookup_agent,
)
from src.cred_agent.agents.response_composer import (
    create_response_composer_agent,
    create_response_composer_task,
)
from src.cred_agent.agents.retrieval_agent import (
    CredKnowledgeRetrievalTool,
    create_retrieval_agent,
)
from src.cred_agent.rag.retrieval import PolicyRetriever
from src.cred_agent.schemas import (
    SupportResponse,
    safe_support_response,
    validate_support_response,
)
from src.cred_agent.services.guardrails import (
    check_response_groundedness,
    guard_user_input,
    sanitize_nested_strings,
    sanitize_response_pii,
)
from src.cred_agent.services.autogen_review import review_response
from src.cred_agent.services.runtime_budget import RuntimeBudget
from src.cred_agent.services.tool_permissions import (
    validate_agent_tool_permissions,
)

logger = logging.getLogger(__name__)

_APPLICATION_TOKEN_PATTERN = re.compile(r"\bAPP-[A-Z0-9-]+\b", re.IGNORECASE)
_KNOWLEDGE_INTENT_PATTERN = re.compile(
    r"\b(?:emi|eligib\w*|kyc|documents?|prepay\w*|foreclos\w*|"
    r"credit score|interest rate|fees?|policy|minimum balance|"
    r"joint account|nri|calculate\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryRoute:
    """Selected specialist work after deterministic query classification."""

    retrieval_required: bool
    lookup_required: bool
    record_id: Optional[str]

    @property
    def specialist_roles(self) -> List[str]:
        roles: List[str] = []
        if self.retrieval_required:
            roles.append("Cred Knowledge Base Retrieval Specialist")
        if self.lookup_required:
            roles.append("Cred Loan Application Support Specialist")
        return roles


@dataclass(frozen=True)
class IntegratedCrewResult:
    """Auditable results of specialist kickoffs and final response composition."""

    query: str
    route: QueryRoute
    specialist_findings: List[Dict[str, Any]]
    tool_invocations: List[Dict[str, Any]]
    final_response: str
    kickoff_count: int
    structured_response: SupportResponse
    validation_error: Optional[str] = None
    input_pii_categories: List[str] = field(default_factory=list)
    prompt_injection_detected: bool = False
    groundedness_passed: bool = True
    output_guardrail_error: Optional[str] = None
    review_approved: Optional[bool] = None
    review_reason: Optional[str] = None
    review_turn_count: int = 0
    cache_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "route": asdict(self.route),
            "selected_specialists": self.route.specialist_roles,
            "specialist_findings": self.specialist_findings,
            "tool_invocations": self.tool_invocations,
            "final_response": self.final_response,
            "kickoff_count": self.kickoff_count,
            "structured_response": self.structured_response.model_dump(mode="json"),
            "validation_error": self.validation_error,
            "input_pii_categories": self.input_pii_categories,
            "prompt_injection_detected": self.prompt_injection_detected,
            "groundedness_passed": self.groundedness_passed,
            "output_guardrail_error": self.output_guardrail_error,
            "review_approved": self.review_approved,
            "review_reason": self.review_reason,
            "review_turn_count": self.review_turn_count,
            "cache_used": self.cache_used,
        }


def route_customer_query(query: str) -> QueryRoute:
    """Route by query intent and Pydantic-validated application ID, not tool name.

    Queries without an application ID go to policy retrieval, which itself
    returns the safe fallback for out-of-scope questions. An application ID
    selects lookup; policy-specific language in the same query additionally
    selects retrieval.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    application_ids = list(
        dict.fromkeys(
            token.upper() for token in _APPLICATION_TOKEN_PATTERN.findall(query)
        )
    )
    if len(application_ids) > 1:
        raise ValueError("query must identify no more than one application record")
    record_id: Optional[str] = None
    if application_ids:
        validated = LoanApplicationRecordId.model_validate(
            {"record_id": application_ids[0]}
        )
        record_id = validated.record_id

    lookup_required = record_id is not None
    retrieval_required = (
        not lookup_required
        or bool(_KNOWLEDGE_INTENT_PATTERN.search(query))
    )
    return QueryRoute(
        retrieval_required=retrieval_required,
        lookup_required=lookup_required,
        record_id=record_id,
    )


def _extract_result_json(raw: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (
            isinstance(value.get("is_grounded"), bool)
            or isinstance(value.get("found"), bool)
        ):
            return value
    raise RuntimeError(
        "A specialist task completed without returning a structured authorized "
        "tool result."
    )


def _tool_invocations_from_output(
    agent_role: str,
    task_output: Any,
) -> List[Dict[str, Any]]:
    """Read executed native tool calls and results from CrewAI task messages."""
    messages = task_output.messages
    tool_results = {
        message.get("tool_call_id"): message
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    invocations: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            call_id = call.get("id")
            tool_message = tool_results.get(call_id)
            if not isinstance(tool_message, dict):
                raise RuntimeError(
                    "CrewAI reported a tool call without its corresponding result."
                )
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "CrewAI returned malformed arguments for an executed tool."
                    ) from error
            if not isinstance(arguments, dict):
                raise RuntimeError(
                    "CrewAI returned non-object arguments for an executed tool."
                )
            invocations.append(
                {
                    "agent_role": agent_role,
                    "tool_name": function.get("name"),
                    "tool_args": arguments,
                    "result": tool_message.get("content"),
                }
            )
    return invocations


def _verified_tool_result(
    task_output: Any,
    invocations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use only the actual tool observation as evidence for composition."""
    reported_result = _extract_result_json(task_output.raw)
    observed_results: List[Dict[str, Any]] = []
    for invocation in invocations:
        observation = invocation.get("result")
        if not isinstance(observation, str):
            raise RuntimeError(
                "A specialist tool invocation returned no structured observation."
            )
        observed_results.append(_extract_result_json(observation))
    if reported_result not in observed_results:
        raise RuntimeError(
            "A specialist's reported result did not match its authorized tool output."
        )
    return reported_result


def validate_composer_response(value: Any) -> Tuple[SupportResponse, Optional[str]]:
    """Validate structured Crew output or return a deterministic safe response."""
    try:
        return validate_support_response(value), None
    except (TypeError, ValueError, ValidationError) as error:
        return safe_support_response(), guard_user_input(str(error)).text


def _selected_specialist_agents(
    route: QueryRoute,
    retrieval_agent: Agent,
    lookup_agent: Agent,
) -> List[Tuple[Agent, Task]]:
    selected: List[Tuple[Agent, Task]] = []
    if route.retrieval_required:
        agent = retrieval_agent
        task = Task(
            description=(
                "Use the knowledge-base retrieval tool for the customer's "
                "question. Earlier messages are conversation context only; "
                "they are not verified knowledge-base evidence. Do not treat "
                "them as policy facts.\n\n"
                "Earlier conversation:\n{conversation_context}\n\n"
                "Customer question: {customer_query}"
            ),
            expected_output=(
                "Return the authorized retrieval tool's complete structured "
                "result, preserving its grounded answer or fallback and sources."
            ),
            agent=agent,
            tools=agent.tools or [],
        )
        selected.append((agent, task))
    if route.lookup_required:
        agent = lookup_agent
        task = Task(
            description=(
                "Use the authorized loan application lookup tool for the "
                "customer's application question. Customer question: "
                "{customer_query}\n\n"
                "Earlier conversation is context only and must not substitute "
                "for the authorized application lookup:\n{conversation_context}"
            ),
            expected_output=(
                "Return the complete structured lookup tool result without "
                "adding or inferring application details."
            ),
            agent=agent,
            tools=agent.tools or [],
        )
        selected.append((agent, task))
    return selected


def run_customer_query(
    query: str,
    retriever: Optional[PolicyRetriever] = None,
    conversation_history: Optional[Sequence[Any]] = None,
    runtime_budget: Optional[RuntimeBudget] = None,
) -> IntegratedCrewResult:
    """Execute selected specialist tasks and composition through Crew.kickoff.

    Specialists share one sequential kickoff when both are needed. A second
    kickoff is used for the Composer because its input must contain the actual
    structured specialist results returned by the first kickoff.
    """
    if runtime_budget is not None and not runtime_budget.input_registered:
        runtime_budget.register_user_input(query)
    guarded_input = guard_user_input(query)
    logger.info(
        "Customer query input guardrails evaluated",
        extra={
            "pii_categories": list(guarded_input.pii_categories),
            "prompt_injection_detected": guarded_input.injection_detected,
            "prompt_injection_category": guarded_input.injection_category,
        },
    )
    if guarded_input.injection_detected:
        return IntegratedCrewResult(
            query=guarded_input.text,
            route=QueryRoute(False, False, None),
            specialist_findings=[],
            tool_invocations=[],
            final_response=safe_support_response().response,
            kickoff_count=0,
            structured_response=safe_support_response(),
            input_pii_categories=list(guarded_input.pii_categories),
            prompt_injection_detected=True,
        )

    query = guarded_input.text
    route = route_customer_query(query)
    conversation_context = _format_conversation_context(conversation_history)
    retrieval_agent = create_retrieval_agent(
        retriever=retriever,
        runtime_budget=runtime_budget,
    )
    lookup_agent = create_lookup_agent(runtime_budget=runtime_budget)
    composer = create_response_composer_agent(runtime_budget=runtime_budget)
    specialist_work = _selected_specialist_agents(
        route,
        retrieval_agent,
        lookup_agent,
    )
    if not specialist_work:
        raise RuntimeError("Query routing selected no specialist agent.")

    specialist_agents = [agent for agent, _ in specialist_work]
    specialist_tasks = [task for _, task in specialist_work]
    validate_agent_tool_permissions([*specialist_agents, composer])
    specialist_crew = Crew(
        agents=specialist_agents,
        tasks=specialist_tasks,
        process=Process.sequential,
        verbose=False,
    )
    if runtime_budget is not None:
        runtime_budget.mark_workflow_started()
    specialist_output = specialist_crew.kickoff(
        inputs={
            "customer_query": query,
            "conversation_context": conversation_context,
        }
    )

    task_outputs = specialist_output.tasks_output
    if len(task_outputs) != len(specialist_work):
        raise RuntimeError(
            "CrewAI returned a different number of specialist outputs than tasks."
        )

    findings: List[Dict[str, Any]] = []
    invocations: List[Dict[str, Any]] = []
    for (agent, _), task_output in zip(specialist_work, task_outputs):
        task_invocations = _tool_invocations_from_output(
            agent.role,
            task_output,
        )
        if not task_invocations:
            raise RuntimeError(
                f"{agent.role} completed without invoking its declared tool."
            )
        invocations.extend(task_invocations)
        findings.append(
            {
                "agent_role": agent.role,
                "tool_result": _verified_tool_result(
                    task_output,
                    task_invocations,
                ),
            }
        )

    composer_task = create_response_composer_task(composer)
    composer_crew = Crew(
        agents=[composer],
        tasks=[composer_task],
        process=Process.sequential,
        verbose=False,
    )
    validate_agent_tool_permissions(composer_crew.agents)
    composer_output = composer_crew.kickoff(
        inputs={
            "specialist_findings": json.dumps(
                findings,
                ensure_ascii=False,
                sort_keys=True,
            )
        }
    )

    composer_task_output = composer_output.tasks_output[-1]
    structured_response, validation_error = validate_composer_response(
        composer_task_output.pydantic or composer_task_output.raw
    )
    groundedness_passed, output_guardrail_error = check_response_groundedness(
        structured_response,
        specialist_findings=findings,
    )
    if not groundedness_passed:
        structured_response = safe_support_response()
    structured_response = sanitize_response_pii(structured_response)
    structured_response = validate_support_response(structured_response)
    review_result = review_response(
        question=query,
        draft=structured_response,
        runtime_budget=runtime_budget,
    )
    structured_response = sanitize_response_pii(review_result.response)
    structured_response = validate_support_response(structured_response)
    final_groundedness_passed, final_groundedness_error = (
        check_response_groundedness(structured_response)
    )
    if not final_groundedness_passed:
        structured_response = safe_support_response()
        output_guardrail_error = (
            final_groundedness_error or "autogen_review_groundedness_failure"
        )
    groundedness_passed = groundedness_passed and final_groundedness_passed

    return IntegratedCrewResult(
        query=query,
        route=route,
        specialist_findings=sanitize_nested_strings(findings),
        tool_invocations=sanitize_nested_strings(invocations),
        final_response=structured_response.response,
        kickoff_count=2,
        structured_response=structured_response,
        validation_error=validation_error,
        input_pii_categories=list(guarded_input.pii_categories),
        groundedness_passed=groundedness_passed,
        output_guardrail_error=output_guardrail_error,
        review_approved=review_result.verdict.approved,
        review_reason=review_result.verdict.reason,
        review_turn_count=review_result.turn_count,
    )


def _format_conversation_context(messages: Optional[Sequence[Any]]) -> str:
    """Render prior LangChain messages as context, never as verified facts."""
    if not messages:
        return "No prior conversation."

    rendered: List[str] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type not in {"human", "ai"}:
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        guarded_content = guard_user_input(content)
        if guarded_content.injection_detected:
            rendered.append("Earlier unsafe user instructions were omitted.")
            continue
        role = "User" if message_type == "human" else "Assistant"
        rendered.append(f"{role}: {guarded_content.text}")
    return "\n".join(rendered) or "No prior conversation."
