"""CrewAI Response Composer for verified specialist findings."""

from crewai import Agent, Task

from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.schemas import SupportResponse
from src.cred_agent.services.runtime_budget import RuntimeBudget


def create_response_composer_agent(
    runtime_budget: RuntimeBudget | None = None,
) -> Agent:
    """Create the deterministic customer-support Response Composer."""
    return Agent(
        role="Cred Customer Support Response Composer",
        goal=(
            "Convert specialist agent findings into accurate, concise, "
            "customer-friendly support responses."
        ),
        backstory=(
            "An experienced customer-support communication specialist who "
            "combines verified knowledge-base information and loan application "
            "lookup results into clear responses. The specialist never invents "
            "facts or makes unsupported promises."
        ),
        tools=[],
        llm=CredMockLLM(runtime_budget=runtime_budget),
        allow_delegation=False,
        max_iter=1,
        cache=False,
        verbose=False,
    )


def create_response_composer_task(agent: Agent) -> Task:
    """Create a composition task that accepts serialized specialist findings."""
    return Task(
        description=(
            "Write a concise, professional customer-facing response based only "
            "on the verified specialist findings below. Preserve factual details "
            "and any source attribution. Clearly state when information is "
            "unavailable. Preserve application status, amount, and escalation "
            "details. Do not claim a required escalation has been completed.\n\n"
            "Specialist findings:\n{specialist_findings}"
        ),
        expected_output=(
            "A validated SupportResponse containing a concise customer response, "
            "grounding status, attribution and retrieved context, plus complete "
            "application and escalation fields when the lookup agent supplied them."
        ),
        agent=agent,
        output_pydantic=SupportResponse,
        response_model=SupportResponse,
    )
