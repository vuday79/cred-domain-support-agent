"""Run the deterministic Cred Retrieval Agent against the persisted index."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crewai import Crew, Process, Task

from src.cred_agent.agents.retrieval_agent import (
    RETRIEVAL_SIMILARITY_THRESHOLD,
    create_retrieval_agent,
    load_fixed_size_retriever,
)
from src.cred_agent.config import DEFAULT_CHROMA_PATH

DEMO_QUESTIONS = (
    "How is EMI calculated?",
    "Who won the latest IPL season?",
)


def run_retrieval_agent_demo() -> None:
    """Demonstrate one grounded policy answer and one out-of-scope fallback."""
    persist_directory = (PROJECT_ROOT / DEFAULT_CHROMA_PATH).resolve()
    retriever = load_fixed_size_retriever(persist_directory)

    print("Cred Retrieval Agent | deterministic MOCK_LLM")
    print(f"Collection: cred_fixed_size | cosine threshold: {RETRIEVAL_SIMILARITY_THRESHOLD}")
    for question in DEMO_QUESTIONS:
        agent = create_retrieval_agent(
            retriever=retriever,
            persist_directory=persist_directory,
        )
        task = Task(
            description=(
                "Use the Cred knowledge-base search tool for the customer's "
                "question. Customer question: {question}"
            ),
            expected_output=(
                "Return the retrieval tool's grounded answer with its source "
                "attribution, similarity scores, and retrieved context. Preserve "
                "the insufficient-information fallback when returned."
            ),
            agent=agent,
            tools=agent.tools,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        output = crew.kickoff(inputs={"question": question})
        print(f"\nQuestion: {question}")
        print(str(output))


if __name__ == "__main__":
    run_retrieval_agent_demo()
