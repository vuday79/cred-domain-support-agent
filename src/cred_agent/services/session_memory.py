"""In-process, session-isolated conversation memory for Cred support."""

import logging
from typing import Any, Callable, Dict, List, Sequence

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.cred_agent.schemas import safe_support_response
from src.cred_agent.services.guardrails import guard_user_input
from src.cred_agent.services.runtime_budget import RuntimeBudget

ConversationWorkflow = Callable[..., str]
logger = logging.getLogger(__name__)


def _run_customer_support(
    message: str,
    history: Sequence[BaseMessage],
    runtime_budget: RuntimeBudget | None = None,
) -> str:
    from src.cred_agent import crew

    if runtime_budget is None:
        return crew.run_customer_query(
            message,
            conversation_history=history,
        ).final_response
    return crew.run_customer_query(
        message,
        conversation_history=history,
        runtime_budget=runtime_budget,
    ).final_response


class SessionConversationMemory:
    """Run the support workflow with LangChain-managed, per-session history.

    Histories live for the lifetime of this service instance. Each session ID
    owns an ``InMemoryChatMessageHistory`` and is isolated by
    ``RunnableWithMessageHistory``'s ``session_id`` configuration.
    """

    def __init__(
        self,
        workflow: ConversationWorkflow = _run_customer_support,
    ) -> None:
        self._workflow = workflow
        self._histories: Dict[str, InMemoryChatMessageHistory] = {}
        self._runnable = RunnableWithMessageHistory(
            RunnableLambda(self._process_message),
            self.get_session_history,
            input_messages_key="message",
            history_messages_key="history",
            output_messages_key="response",
        )

    def get_session_history(
        self,
        session_id: str,
    ) -> InMemoryChatMessageHistory:
        """Return or create the LangChain message history for one session."""
        normalized_id = self._validate_session_id(session_id)
        if normalized_id not in self._histories:
            self._histories[normalized_id] = InMemoryChatMessageHistory()
        return self._histories[normalized_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Return a snapshot of messages currently stored for a session."""
        return list(self.get_session_history(session_id).messages)

    def chat(
        self,
        session_id: str,
        message: str,
        runtime_budget: RuntimeBudget | None = None,
    ) -> str:
        """Run one user turn and append its user and assistant messages."""
        normalized_id = self._validate_session_id(session_id)
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if runtime_budget is not None and not runtime_budget.input_registered:
            runtime_budget.register_user_input(message)

        guarded_input = guard_user_input(message)
        logger.info(
            "Session input guardrails evaluated",
            extra={
                "pii_categories": list(guarded_input.pii_categories),
                "prompt_injection_detected": guarded_input.injection_detected,
                "prompt_injection_category": guarded_input.injection_category,
            },
        )
        if guarded_input.injection_detected:
            history = self.get_session_history(normalized_id)
            history.add_messages(
                [
                    HumanMessage(content="[Rejected unsafe request]"),
                    AIMessage(content=safe_support_response().response),
                ]
            )
            return safe_support_response().response

        result: Any = self._runnable.invoke(
            {
                "message": guarded_input.text,
                "runtime_budget": runtime_budget,
            },
            config={"configurable": {"session_id": normalized_id}},
        )
        response = result.get("response") if isinstance(result, dict) else None
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                "The customer-support workflow returned no assistant response."
            )
        return response

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        return session_id.strip()

    def _process_message(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        message = inputs.get("message")
        history = inputs.get("history")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if not isinstance(history, list) or not all(
            isinstance(item, BaseMessage) for item in history
        ):
            raise TypeError("LangChain did not provide a valid message history")

        runtime_budget = inputs.get("runtime_budget")
        if runtime_budget is None:
            response = self._workflow(message, history)
        else:
            response = self._workflow(message, history, runtime_budget)
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                "The customer-support workflow returned no assistant response."
            )
        return {"response": guard_user_input(response).text}
