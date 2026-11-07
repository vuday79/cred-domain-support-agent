"""Tests for the persistent, readable session-memory demo transcript."""

import importlib.util
from datetime import datetime
from pathlib import Path

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = PROJECT_ROOT / "scripts" / "run_memory_demo.py"


class DemoMemory:
    """Small deterministic memory double for testing transcript rendering."""

    def __init__(self) -> None:
        self.histories = {}

    def get_session_history(self, session_id):
        if session_id not in self.histories:
            self.histories[session_id] = InMemoryChatMessageHistory()
        return self.histories[session_id]

    def get_messages(self, session_id):
        return list(self.get_session_history(session_id).messages)

    def chat(self, session_id, message):
        response = f"Demo response: {message}"
        history = self.get_session_history(session_id)
        history.add_messages(
            [
                HumanMessage(content=message),
                AIMessage(content=response),
            ]
        )
        return response


def test_memory_demo_saves_timestamped_sessions_and_isolation_evidence(
    monkeypatch,
    tmp_path,
    capsys,
):
    spec = importlib.util.spec_from_file_location("memory_demo", DEMO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SessionConversationMemory", DemoMemory)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    rendered = module.run_memory_demo()

    transcript_path = tmp_path / "evaluation" / "session_memory_transcript.md"
    assert transcript_path.read_text(encoding="utf-8") == rendered
    assert rendered.count("**Session ID:** `session-a`") == 2
    assert rendered.count("**Session ID:** `session-b`") == 1
    assert rendered.count("**Started:** ") == 3
    assert rendered.count("**Completed:** ") == 3
    for line in rendered.splitlines():
        if line.startswith("**Started:** ") or line.startswith("**Completed:** "):
            datetime.fromisoformat(line.split(":** ", 1)[1])

    assert "**User:** What is the EMI calculation method?" in rendered
    assert "**User:** What tenure options are available?" in rendered
    assert "**Assistant:** Demo response: What is the EMI calculation method?" in (
        rendered
    )
    assert "**Assistant:** Demo response: What tenure options are available?" in (
        rendered
    )
    assert "(empty)" in rendered
    assert (
        "**Isolation evidence:** SESSION B uses a separate history object and "
        "had empty history before its first interaction."
    ) in rendered
    assert "It could not access SESSION A's messages" in rendered
    assert "Transcript saved to:" in capsys.readouterr().out


def test_memory_demo_uses_two_turns_in_session_a_and_a_fresh_session_b(
    monkeypatch,
    tmp_path,
):
    spec = importlib.util.spec_from_file_location("memory_demo", DEMO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SessionConversationMemory", DemoMemory)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    rendered = module.run_memory_demo()

    session_a_second_turn = rendered.index("## SESSION A: interaction 2")
    session_b_turn = rendered.index("## SESSION B: interaction 3")
    assert rendered.index("## SESSION A: interaction 1") < session_a_second_turn
    assert session_a_second_turn < session_b_turn
    assert (
        "**History before interaction (context supplied to workflow):**\n"
        "- User: What is the EMI calculation method?"
    ) in rendered
    session_b_section = rendered[session_b_turn:]
    assert (
        "**History before interaction (context supplied to workflow):**\n"
        "(empty)"
    ) in session_b_section
