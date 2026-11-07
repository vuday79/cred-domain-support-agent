"""Demonstrate deterministic query routing through Cred's three-agent workflow."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The prebuilt embedding model must be cached; avoid any Hugging Face Hub access.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.cred_agent.crew import run_customer_query

DEMO_QUERIES = (
    "How is EMI calculated?",
    "What is the status of loan application APP-1001?",
    "Who won the latest IPL season?",
    "Tell me the status of APP-9999.",
)


def run_crew_demo() -> None:
    """Print routing, actual tool calls, findings, and composed responses."""
    for query in DEMO_QUERIES:
        result = run_customer_query(query)
        print(f"\nIncoming query: {query}")
        print("Selected specialist agent(s):")
        for role in result.route.specialist_roles:
            print(f"  - {role}")
        print("Actual tool invocation(s):")
        for invocation in result.tool_invocations:
            print(
                f"  - {invocation['agent_role']}: {invocation['tool_name']} "
                f"arguments={invocation['tool_args']}"
            )
        print("Specialist findings:")
        for finding in result.specialist_findings:
            print(f"  - {finding['agent_role']}: {finding['tool_result']}")
        print("Final composed response:")
        print(result.final_response)


if __name__ == "__main__":
    run_crew_demo()
