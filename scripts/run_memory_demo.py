"""Demonstrate session-isolated conversation history in the support workflow."""

from datetime import datetime
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The retrieval model is local; keep the demonstration offline.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_core.messages import BaseMessage

from src.cred_agent.services.session_memory import SessionConversationMemory


def _format_history(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(empty)"
    return "\n".join(
        f"- {'User' if message.type == 'human' else 'Assistant'}: "
        f"{message.content}"
        for message in messages
    )


def run_memory_demo() -> str:
    """Run the two-session example and save/print its readable transcript."""
    memory = SessionConversationMemory()
    session_a_id = "session-a"
    session_b_id = "session-b"
    session_a_history = memory.get_session_history(session_a_id)
    transcript = [
        "# Session Memory Demo Transcript",
        "",
        f"**Transcript generated:** {_timestamp()}",
        "",
        "Each turn shows the LangChain history supplied to the workflow before "
        "the request and the stored history after the response. Timestamps use "
        "the machine's local timezone.",
        "",
    ]

    turns = (
        ("SESSION A", session_a_id, "What is the EMI calculation method?"),
        ("SESSION A", session_a_id, "What tenure options are available?"),
        ("SESSION B", session_b_id, "What tenure options are available?"),
    )
    session_b_isolated = False
    session_a_before_b: list[BaseMessage] = []
    for interaction_number, (label, session_id, user_message) in enumerate(
        turns,
        start=1,
    ):
        started_at = _timestamp()
        history_before = memory.get_messages(session_id)
        if label == "SESSION B":
            session_b_history = memory.get_session_history(session_id)
            session_a_before_b = memory.get_messages(session_a_id)
            session_b_isolated = (
                session_b_history is not session_a_history
                and not history_before
                and bool(session_a_before_b)
            )
            if not session_b_isolated:
                raise RuntimeError(
                    "SESSION B did not start with a fresh independent history."
                )

        response = memory.chat(session_id, user_message)
        history_after = memory.get_messages(session_id)
        completed_at = _timestamp()

        transcript.extend(
            [
                f"## {label}: interaction {interaction_number}",
                "",
                f"**Session ID:** `{session_id}`",
                "",
                f"**Started:** {started_at}",
                "",
                f"**Completed:** {completed_at}",
                "",
                "**History before interaction (context supplied to workflow):**",
                _format_history(history_before),
                "",
                f"**User:** {user_message}",
                "",
                f"**Assistant:** {response}",
                "",
                "**History after interaction:**",
                _format_history(history_after),
                "",
                "---",
                "",
            ]
        )

    if memory.get_messages(session_a_id) != session_a_before_b:
        raise RuntimeError("SESSION B changed SESSION A's stored conversation.")
    if not session_b_isolated:
        raise RuntimeError("SESSION B isolation could not be verified.")
    transcript.extend(
        [
            (
                "**Isolation evidence:** SESSION B uses a separate history "
                "object and had empty history before its first interaction. "
                "It could not access SESSION A's messages; SESSION A's history "
                "was unchanged after SESSION B completed."
            ),
            "",
            "**Retrieval note:** Adjacent fixed-size chunks are rejoined using "
            "their exact overlap before extractive sentence matching. Retrieved "
            "chunk records remain unchanged, while the composed answers below "
            "contain the complete tenure sentence once.",
        ]
    )
    rendered = "\n".join(transcript) + "\n"
    report_path = PROJECT_ROOT / "evaluation" / "session_memory_transcript.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Transcript saved to: {report_path}")
    return rendered


def _timestamp() -> str:
    """Return a local, timezone-aware timestamp for the transcript."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    run_memory_demo()
