"""Tests for LangChain-backed session-isolated conversation memory."""

from types import SimpleNamespace
from typing import List, Sequence, Tuple

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from src.cred_agent import crew
from src.cred_agent.services.session_memory import SessionConversationMemory


class RecordingWorkflow:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, List[BaseMessage]]] = []

    def __call__(
        self,
        message: str,
        history: Sequence[BaseMessage],
    ) -> str:
        self.calls.append((message, list(history)))
        return f"Deterministic response to: {message}"


def test_first_message_starts_with_empty_history_and_is_appended():
    workflow = RecordingWorkflow()
    memory = SessionConversationMemory(workflow=workflow)

    response = memory.chat("session-a", "What is the EMI calculation method?")

    assert response == (
        "Deterministic response to: What is the EMI calculation method?"
    )
    assert workflow.calls[0][1] == []
    assert memory.get_messages("session-a") == [
        HumanMessage(content="What is the EMI calculation method?"),
        AIMessage(content=response),
    ]


def test_follow_up_receives_previous_user_and_assistant_context():
    workflow = RecordingWorkflow()
    memory = SessionConversationMemory(workflow=workflow)
    memory.chat("session-a", "What is the EMI calculation method?")

    memory.chat("session-a", "What tenure options are available?")

    assert workflow.calls[1][0] == "What tenure options are available?"
    assert workflow.calls[1][1] == [
        HumanMessage(content="What is the EMI calculation method?"),
        AIMessage(
            content=(
                "Deterministic response to: What is the EMI calculation method?"
            )
        ),
    ]


def test_a_new_session_has_no_previous_context():
    workflow = RecordingWorkflow()
    memory = SessionConversationMemory(workflow=workflow)
    memory.chat("session-a", "First session question")

    assert memory.get_messages("session-b") == []
    memory.chat("session-b", "Independent question")

    assert workflow.calls[1][1] == []


def test_sessions_remain_isolated_after_multiple_messages():
    workflow = RecordingWorkflow()
    memory = SessionConversationMemory(workflow=workflow)
    memory.chat("session-a", "A1")
    memory.chat("session-a", "A2")
    memory.chat("session-b", "B1")

    assert [message.content for message in memory.get_messages("session-a")] == [
        "A1",
        "Deterministic response to: A1",
        "A2",
        "Deterministic response to: A2",
    ]
    assert [message.content for message in memory.get_messages("session-b")] == [
        "B1",
        "Deterministic response to: B1",
    ]
    assert workflow.calls[2][1] == []


def test_history_persists_for_the_lifetime_of_the_service_instance():
    workflow = RecordingWorkflow()
    memory = SessionConversationMemory(workflow=workflow)
    first_history = memory.get_session_history("session-a")
    assert isinstance(first_history, InMemoryChatMessageHistory)

    memory.chat("session-a", "Persist this turn")

    assert memory.get_session_history("session-a") is first_history
    assert len(first_history.messages) == 2
    assert len(memory.get_messages("session-a")) == 2


@pytest.mark.parametrize("session_id", ["", "  ", None])
def test_empty_or_invalid_session_id_is_rejected(session_id):
    memory = SessionConversationMemory(workflow=RecordingWorkflow())

    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        memory.chat(session_id, "A question")


def test_empty_message_is_rejected_without_changing_history():
    memory = SessionConversationMemory(workflow=RecordingWorkflow())

    with pytest.raises(ValueError, match="message must be a non-empty string"):
        memory.chat("session-a", "  ")

    assert memory.get_messages("session-a") == []


def test_default_workflow_receives_history_and_delegates_to_integrated_crew(
    monkeypatch,
):
    calls = []

    def run_customer_query(message, conversation_history=None):
        calls.append((message, list(conversation_history or [])))
        return SimpleNamespace(final_response=f"Crew response to: {message}")

    monkeypatch.setattr(crew, "run_customer_query", run_customer_query)
    memory = SessionConversationMemory()
    memory.chat("session-a", "What is EMI?")

    response = memory.chat("session-a", "And what tenure options?")

    assert response == "Crew response to: And what tenure options?"
    assert calls[0] == ("What is EMI?", [])
    assert calls[1][0] == "And what tenure options?"
    assert calls[1][1] == [
        HumanMessage(content="What is EMI?"),
        AIMessage(content="Crew response to: What is EMI?"),
    ]
