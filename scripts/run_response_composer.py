"""Demonstrate deterministic customer response composition from specialist findings."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The RAG demo uses the existing local SentenceTransformer. Refuse model
# downloads or Hub checks so this standalone workflow remains offline.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from crewai import Crew, Process

from src.cred_agent.agents.lookup_agent import CredLoanApplicationLookupTool
from src.cred_agent.agents.response_composer import (
    create_response_composer_agent,
    create_response_composer_task,
)
from src.cred_agent.agents.retrieval_agent import CredKnowledgeRetrievalTool


def _compose(specialist_findings: str) -> str:
    agent = create_response_composer_agent()
    task = create_response_composer_task(agent)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff(inputs={"specialist_findings": specialist_findings}))


def run_response_composer_demo() -> None:
    """Compose one grounded policy response and one application response."""
    rag_findings = CredKnowledgeRetrievalTool().run(
        question="How is EMI calculated?"
    )
    print("RAG response:")
    print(_compose(rag_findings))

    lookup_findings = CredLoanApplicationLookupTool().run(record_id="APP-1001")
    print("\nLoan application response:")
    print(_compose(lookup_findings))


if __name__ == "__main__":
    run_response_composer_demo()
