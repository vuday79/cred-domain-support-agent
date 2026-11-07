"""Two-agent deterministic AutoGen review of CrewAI customer responses."""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.cred_agent.agents.retrieval_agent import RETRIEVAL_SIMILARITY_THRESHOLD
from src.cred_agent.rag.retrieval import FALLBACK_RESPONSE
from src.cred_agent.schemas import (
    RetrievedContext,
    SimilarityMeasurement,
    SupportResponse,
    SupportResponseType,
    safe_support_response,
    validate_support_response,
)
from src.cred_agent.services.guardrails import check_response_groundedness
from src.cred_agent.services.runtime_budget import (
    RuntimeBudget,
    RuntimeBudgetExceeded,
)

_REVIEW_INPUT_MARKER = "AUTO_GEN_REVIEW_INPUT_JSON:"
_SOURCE_LINE_PATTERN = re.compile(r"(?im)^\s*source\s*:\s*.*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_REVIEWER_SYSTEM_MESSAGE = (
    "Review the original customer question and CrewAI draft against only "
    "the supplied retrieved text and validated lookup observation. Check "
    "each factual statement, contradictions, invented financial terms, "
    "and any unsupported claim that escalation was completed. Do not use "
    "tools, external information, or other documents. Report supported or "
    "unsupported claims for the Editor."
)
_EDITOR_SYSTEM_MESSAGE = (
    "Return a structured ReviewVerdict. Approve fully supported drafts "
    "without changing a character. Otherwise remove unsupported claims "
    "using only the supplied evidence, preserving supported facts and "
    "source attribution, or return the safe fallback if nothing is "
    "supported. Do not use tools or external sources."
)


class ReviewVerdict(BaseModel):
    """Structured verdict emitted by AutoGen's Final Editor."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    final_answer: str
    reason: str


class AutoGenReviewResult(BaseModel):
    """Validated review result plus bounded group-chat execution evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: SupportResponse
    verdict: ReviewVerdict
    turn_count: int
    agent_order: List[str]
    structured_message_received: bool


class _ReviewRequest(BaseModel):
    question: str
    draft_answer: str
    evidence_response: SupportResponse


def _parse_review_request(messages: Sequence[Any]) -> _ReviewRequest:
    for message in messages:
        if not isinstance(message, UserMessage) or not isinstance(message.content, str):
            continue
        marker_index = message.content.find(_REVIEW_INPUT_MARKER)
        if marker_index < 0:
            continue
        payload = message.content[marker_index + len(_REVIEW_INPUT_MARKER) :].strip()
        return _ReviewRequest.model_validate_json(payload)
    raise ValueError("AutoGen review task did not contain its evidence payload.")


def _candidate_response(
    evidence_response: SupportResponse,
    answer: str,
) -> SupportResponse:
    return evidence_response.model_copy(update={"response": answer})


def _strip_known_attribution(
    answer: str,
    evidence_response: SupportResponse,
) -> str:
    """Remove only known citation labels; never discard text appended to a citation."""
    known_sources: Dict[str, str] = {}
    for context in evidence_response.retrieved_context:
        filename = context.source.replace("\\", "/").rsplit("/", 1)[-1]
        known_sources[context.parent_document_id] = filename

    cleaned_lines: List[str] = []
    for line in answer.splitlines():
        source_match = re.match(r"^\s*source\s*:\s*(.*)$", line, re.IGNORECASE)
        if source_match is None:
            cleaned_lines.append(line)
            continue
        remainder = source_match.group(1)
        for source_id, filename in sorted(
            known_sources.items(),
            key=lambda item: len(item[0]) + len(item[1]),
            reverse=True,
        ):
            labels = [source_id]
            if filename:
                labels.insert(0, f"{source_id} ({filename})")
            for label in labels:
                remainder = re.sub(
                    re.escape(label),
                    "",
                    remainder,
                    flags=re.IGNORECASE,
                )
        remainder = re.sub(r"^[\s;,.]+", "", remainder)
        remainder = re.sub(r"^[\s;,.]+", "", remainder)
        if remainder.strip():
            cleaned_lines.append(remainder.strip())
    return "\n".join(cleaned_lines).strip()


def _is_supported(
    evidence_response: SupportResponse,
    answer: str,
) -> bool:
    candidate = _candidate_response(
        evidence_response,
        _strip_known_attribution(answer, evidence_response),
    )
    grounded, _ = check_response_groundedness(candidate)
    return grounded


def _supporting_source_ids(
    evidence_response: SupportResponse,
    sentence: str,
) -> List[str]:
    if not evidence_response.retrieved_context:
        return []
    supported: List[str] = []
    document_ids = list(
        dict.fromkeys(
            context.parent_document_id
            for context in evidence_response.retrieved_context
        )
    )
    for document_id in document_ids:
        contexts = [
            context
            for context in evidence_response.retrieved_context
            if context.parent_document_id == document_id
            and context.cosine_similarity >= RETRIEVAL_SIMILARITY_THRESHOLD
        ]
        if not contexts:
            continue
        chunk_ids = {context.chunk_id for context in contexts}
        measurements = [
            measurement
            for measurement in evidence_response.similarity_measurements
            if measurement.chunk_id in chunk_ids
        ]
        candidate = evidence_response.model_copy(
            update={
                "response": sentence,
                "source_documents": [document_id],
                "retrieved_context": contexts,
                "similarity_measurements": measurements,
            }
        )
        grounded, _ = check_response_groundedness(candidate)
        if grounded:
            supported.append(document_id)
    return supported


def _sentences(
    answer: str,
    evidence_response: SupportResponse,
) -> List[str]:
    without_sources = _strip_known_attribution(answer, evidence_response)
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(without_sources)
        if sentence.strip()
    ]


def _attributed_answer(
    sentences: List[str],
    source_ids: List[str],
    evidence_response: SupportResponse,
) -> str:
    answer = " ".join(sentences).strip()
    if not source_ids:
        return answer
    attribution = []
    for source_id in source_ids:
        matching = next(
            (
                context.source
                for context in evidence_response.retrieved_context
                if context.parent_document_id == source_id
            ),
            "",
        )
        filename = matching.replace("\\", "/").rsplit("/", 1)[-1]
        attribution.append(
            f"{source_id} ({filename})" if filename else source_id
        )
    return f"{answer}\nSource: {'; '.join(attribution)}."


def _revised_response(
    evidence_response: SupportResponse,
    draft_answer: str,
) -> tuple[SupportResponse, str]:
    supported_sentences: List[str] = []
    used_source_ids: List[str] = []
    for sentence in _sentences(draft_answer, evidence_response):
        if not _is_supported(evidence_response, sentence):
            continue
        supported_sentences.append(sentence)
        for source_id in _supporting_source_ids(evidence_response, sentence):
            if source_id not in used_source_ids:
                used_source_ids.append(source_id)

    if not supported_sentences:
        if evidence_response.response_type is SupportResponseType.KNOWLEDGE_BASE:
            fallback = SupportResponse(
                response=FALLBACK_RESPONSE,
                response_type=SupportResponseType.KNOWLEDGE_BASE_FALLBACK,
                is_grounded=False,
                source_documents=[],
                retrieved_context=[],
                similarity_measurements=[],
            )
        else:
            fallback = safe_support_response()
        return fallback, fallback.response

    if (
        evidence_response.response_type is SupportResponseType.KNOWLEDGE_BASE
        and not used_source_ids
    ):
        fallback = SupportResponse(
            response=FALLBACK_RESPONSE,
            response_type=SupportResponseType.KNOWLEDGE_BASE_FALLBACK,
            is_grounded=False,
            source_documents=[],
            retrieved_context=[],
            similarity_measurements=[],
        )
        return fallback, fallback.response

    final_answer = _attributed_answer(
        supported_sentences,
        used_source_ids,
        evidence_response,
    )
    used_chunks = [
        context
        for context in evidence_response.retrieved_context
        if context.parent_document_id in used_source_ids
    ]
    used_chunk_ids = {context.chunk_id for context in used_chunks}
    used_measurements = [
        item
        for item in evidence_response.similarity_measurements
        if item.chunk_id in used_chunk_ids
    ]

    response_type = evidence_response.response_type
    loan_application = evidence_response.loan_application
    record_id = evidence_response.record_id
    if response_type is SupportResponseType.COMBINED and not used_source_ids:
        if loan_application is not None:
            response_type = SupportResponseType.LOAN_APPLICATION
        else:
            response_type = SupportResponseType.APPLICATION_NOT_FOUND
    revised = SupportResponse(
        response=final_answer,
        response_type=response_type,
        is_grounded=True,
        source_documents=used_source_ids,
        retrieved_context=used_chunks,
        similarity_measurements=used_measurements,
        record_id=record_id,
        loan_application=loan_application,
    )
    grounded, _ = check_response_groundedness(revised)
    if not grounded:
        fallback = safe_support_response()
        return fallback, fallback.response
    return validate_support_response(revised), final_answer


def _assess_draft(request: _ReviewRequest) -> ReviewVerdict:
    if _is_supported(request.evidence_response, request.draft_answer):
        return ReviewVerdict(
            approved=True,
            final_answer=request.draft_answer,
            reason="Every factual claim in the draft is supported by supplied evidence.",
        )

    revised, final_answer = _revised_response(
        request.evidence_response,
        request.draft_answer,
    )
    if revised.response_type in {
        SupportResponseType.KNOWLEDGE_BASE_FALLBACK,
        SupportResponseType.SAFE_ERROR,
    }:
        reason = (
            "The draft contained unsupported claims and no factual sentence could "
            "be retained from the supplied evidence; returned the safe fallback."
        )
    else:
        reason = (
            "Unsupported or contradictory claims were removed; the remaining "
            "answer is limited to supplied evidence."
        )
    return ReviewVerdict(
        approved=False,
        final_answer=final_answer,
        reason=reason,
    )


class DeterministicAutoGenModelClient(ChatCompletionClient):
    """Offline AutoGen-compatible model client for Reviewer and Editor agents."""

    mode: str
    model: str = "cred-autogen-mock-v1"
    _call_count: int = PrivateAttr(default=0)
    _runtime_budget: Optional[RuntimeBudget] = PrivateAttr(default=None)

    def __init__(
        self,
        mode: str,
        runtime_budget: Optional[RuntimeBudget] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self._call_count = 0
        self._runtime_budget = runtime_budget

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
            "multiple_system_messages": True,
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
        }

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = ()) -> int:
        del tools
        return sum(
            len(message.content)
            for message in messages
            if isinstance(message, (SystemMessage, UserMessage, AssistantMessage))
            and isinstance(message.content, str)
        )

    def remaining_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = ()) -> int:
        del messages, tools
        return 1_000_000

    async def close(self) -> None:
        return None

    async def create(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        del tools, tool_choice, extra_create_args, cancellation_token
        self._call_count += 1
        request = _parse_review_request(messages)
        verdict = _assess_draft(request)
        budget_already_exceeded = (
            self._runtime_budget is not None
            and self._runtime_budget.outcome != "within_budget"
        )
        if self.mode == "reviewer" and budget_already_exceeded:
            content = "Reviewer halted because the request runtime budget was exceeded."
        elif self.mode == "reviewer":
            content = (
                "Reviewer assessment: "
                + ("SUPPORTED" if verdict.approved else "UNSUPPORTED")
                + ". "
                + verdict.reason
            )
        elif self.mode == "editor":
            if json_output is not ReviewVerdict:
                raise ValueError("Editor must be configured for ReviewVerdict output.")
            if budget_already_exceeded:
                content = ReviewVerdict(
                    approved=False,
                    final_answer=FALLBACK_RESPONSE,
                    reason="The request exceeded its runtime budget; returned the safe fallback.",
                ).model_dump_json()
            else:
                content = verdict.model_dump_json()
        else:
            raise ValueError(f"Unknown deterministic AutoGen role {self.mode!r}.")
        if self._runtime_budget is not None and not budget_already_exceeded:
            prompt = json.dumps(
                [
                    {
                        "type": type(message).__name__,
                        "content": getattr(message, "content", ""),
                    }
                    for message in messages
                ],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            try:
                self._runtime_budget.charge_model_call(prompt, content)
            except RuntimeBudgetExceeded:
                if self.mode == "reviewer":
                    content = (
                        "Reviewer halted because the request runtime budget was exceeded."
                    )
                else:
                    content = ReviewVerdict(
                        approved=False,
                        final_answer=FALLBACK_RESPONSE,
                        reason=(
                            "The request exceeded its runtime budget; returned "
                            "the safe fallback."
                        ),
                    ).model_dump_json()
        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result


def create_review_team(
    runtime_budget: Optional[RuntimeBudget] = None,
) -> tuple[
    RoundRobinGroupChat,
    DeterministicAutoGenModelClient,
    DeterministicAutoGenModelClient,
]:
    """Create the required tool-less, two-agent, exactly-two-turn AutoGen team."""
    reviewer_client = DeterministicAutoGenModelClient(
        mode="reviewer",
        runtime_budget=runtime_budget,
    )
    editor_client = DeterministicAutoGenModelClient(
        mode="editor",
        runtime_budget=runtime_budget,
    )
    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=reviewer_client,
        system_message=_REVIEWER_SYSTEM_MESSAGE,
        tools=[],
    )
    editor = AssistantAgent(
        name="Editor",
        model_client=editor_client,
        system_message=_EDITOR_SYSTEM_MESSAGE,
        tools=[],
        output_content_type=ReviewVerdict,
    )
    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
        name="CredResponseReview",
        description="Independent deterministic review of CrewAI support drafts.",
    )
    return team, reviewer_client, editor_client


async def _run_team(
    team: RoundRobinGroupChat,
    task: str,
) -> Any:
    return await team.run(task=task)


def _run_async_safely(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def review_response(
    question: str,
    draft: SupportResponse,
    draft_answer: Optional[str] = None,
    runtime_budget: Optional[RuntimeBudget] = None,
) -> AutoGenReviewResult:
    """Run Reviewer then Editor through AutoGen and validate the final verdict."""
    request = _ReviewRequest(
        question=question,
        draft_answer=draft_answer if draft_answer is not None else draft.response,
        evidence_response=draft,
    )
    task = _REVIEW_INPUT_MARKER + request.model_dump_json()
    team, reviewer_client, editor_client = create_review_team(runtime_budget)
    if runtime_budget is not None:
        expected_verdict = _assess_draft(request)
        reviewer_output = (
            "Reviewer assessment: "
            + ("SUPPORTED" if expected_verdict.approved else "UNSUPPORTED")
            + ". "
            + expected_verdict.reason
        )
        reviewer_prompt = json.dumps(
            [
                {"type": "SystemMessage", "content": _REVIEWER_SYSTEM_MESSAGE},
                {"type": "UserMessage", "content": task},
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        editor_prompt = json.dumps(
            [
                {"type": "SystemMessage", "content": _EDITOR_SYSTEM_MESSAGE},
                {"type": "UserMessage", "content": task},
                {"type": "AssistantMessage", "content": reviewer_output},
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        def reserve_call_tokens(prompt: str, completion: str) -> int:
            estimated = RuntimeBudget.estimate_tokens(prompt) + (
                RuntimeBudget.estimate_tokens(completion)
            )
            return estimated + (estimated // 3) + 128

        runtime_budget.reserve_model_calls(
            [
                reserve_call_tokens(reviewer_prompt, reviewer_output),
                reserve_call_tokens(editor_prompt, expected_verdict.model_dump_json()),
            ]
        )
    result = _run_async_safely(_run_team(team, task))
    if runtime_budget is not None:
        runtime_budget.raise_if_exceeded()

    messages = result.messages
    agent_messages = messages[1:] if messages and messages[0].source == "user" else messages
    if len(agent_messages) != 2:
        raise RuntimeError(
            "AutoGen review expected exactly 2 agent turns, "
            f"received {len(agent_messages)}."
        )
    if [message.source for message in agent_messages] != ["Reviewer", "Editor"]:
        raise RuntimeError("AutoGen review agents did not execute in the required order.")
    final_message = agent_messages[-1]
    if not isinstance(final_message, StructuredMessage) or not isinstance(
        final_message.content, ReviewVerdict
    ):
        raise RuntimeError("AutoGen Editor did not return a structured verdict.")
    verdict = ReviewVerdict.model_validate(final_message.content)

    if verdict.approved:
        if verdict.final_answer != request.draft_answer:
            raise RuntimeError("Approved AutoGen verdict changed the draft answer.")
        final_response = _candidate_response(draft, verdict.final_answer)
    else:
        final_response, expected_answer = _revised_response(
            draft,
            request.draft_answer,
        )
        if verdict.final_answer != expected_answer:
            raise RuntimeError("AutoGen Editor verdict did not match evidence-based editing.")

    grounded, _ = check_response_groundedness(final_response)
    if not grounded:
        final_response = (
            SupportResponse(
                response=FALLBACK_RESPONSE,
                response_type=SupportResponseType.KNOWLEDGE_BASE_FALLBACK,
                is_grounded=False,
                source_documents=[],
                retrieved_context=[],
                similarity_measurements=[],
            )
            if draft.response_type is SupportResponseType.KNOWLEDGE_BASE
            else safe_support_response()
        )
        verdict = ReviewVerdict(
            approved=False,
            final_answer=final_response.response,
            reason="The edited response failed the existing groundedness guard; "
            "returned the safe fallback.",
        )
    final_response = validate_support_response(final_response)

    return AutoGenReviewResult(
        response=final_response,
        verdict=verdict,
        turn_count=len(agent_messages),
        agent_order=[message.source for message in agent_messages],
        structured_message_received=True,
    )
