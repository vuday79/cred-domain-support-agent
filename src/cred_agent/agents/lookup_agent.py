"""CrewAI Lookup Agent backed by the existing loan application service."""

import json

from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.cred_agent.mock_llm import CredMockLLM
from src.cred_agent.services import loan_application_lookup
from src.cred_agent.services.runtime_budget import RuntimeBudget


class LoanApplicationRecordId(BaseModel):
    """Validated input for looking up an existing loan application."""

    record_id: str = Field(
        min_length=1,
        pattern=r"^APP-\d+$",
        description="Application ID in the format APP-1001.",
    )


class CredLoanApplicationLookupTool(BaseTool):
    """Return the existing lookup service's complete application result."""

    name: str = "cred_application_record"
    description: str = (
        "Look up an existing Cred loan application by its record_id, such as "
        "APP-1001. Return the complete application status and escalation result. "
        "For an unknown ID, return the service's explicit not-found result; "
        "never infer or invent application information."
    )
    args_schema: type[BaseModel] = LoanApplicationRecordId
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, record_id: str) -> str:
        """Call the authoritative lookup service and serialize its full result."""
        result = loan_application_lookup.check_loan_application_status(record_id)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)


def create_lookup_agent(
    runtime_budget: RuntimeBudget | None = None,
) -> Agent:
    """Create the deterministic agent with only the authorized lookup tool."""
    return Agent(
        role="Cred Loan Application Support Specialist",
        goal=(
            "Retrieve accurate loan application details and escalation "
            "information using the authorized lookup tool."
        ),
        backstory=(
            "A specialist responsible for helping customers understand their "
            "loan application status. The agent must rely exclusively on the "
            "application lookup tool for application-specific information."
        ),
        tools=[CredLoanApplicationLookupTool()],
        llm=CredMockLLM(runtime_budget=runtime_budget),
        allow_delegation=False,
        max_iter=2,
        cache=False,
        verbose=False,
    )
