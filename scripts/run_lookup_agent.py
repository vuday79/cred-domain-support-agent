"""Demonstrate the deterministic Cred Loan Application Lookup Agent."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crewai import Crew, Process, Task

from src.cred_agent.agents.lookup_agent import create_lookup_agent

DEMO_RECORD_IDS = ("APP-1001", "APP-9999")


def run_lookup_agent_demo() -> None:
    """Run valid and not-found IDs through the CrewAI native tool workflow."""
    for record_id in DEMO_RECORD_IDS:
        agent = create_lookup_agent()
        task = Task(
            description=(
                "Use the authorized application lookup tool for the customer's "
                f"application. Customer question: Please check application {record_id}."
            ),
            expected_output=(
                "Return the complete structured lookup tool result, including "
                "the not-found response when the ID is unknown."
            ),
            agent=agent,
            tools=agent.tools or [],
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        print(f"\nApplication requested: {record_id}")
        print(str(result))


if __name__ == "__main__":
    run_lookup_agent_demo()
