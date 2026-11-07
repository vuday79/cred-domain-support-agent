# Cred Loan Application Lookup Agent

`src/cred_agent/agents/lookup_agent.py` defines a CrewAI Lookup Agent and its
single authorized tool, `CredLoanApplicationLookupTool`. The agent uses the
project's deterministic `CredMockLLM` and makes no external LLM API calls.

## Tool and input schema

The tool accepts `record_id` as a string validated by a Pydantic schema. The
expected form is `APP-` followed by digits (for example, `APP-1001`). The tool
calls the existing `check_loan_application_status(record_id)` service and
returns its complete result as JSON, including `found`, application data, the
escalation score and decision, and the score reason. The tool does not
reimplement scoring. Unknown IDs return the service's explicit `not_found`
result and do not acquire fabricated application fields.

MOCK_LLM routes calls by the declared `record_id` argument schema. CrewAI
executes the returned native tool call, and MOCK_LLM forwards the structured
tool result on the follow-up call. The agent is assigned only the authorized
loan application tool and disallows delegation.

## Examples

Run the standalone demonstration from the project root:

```powershell
python scripts\run_lookup_agent.py
```

The script checks `APP-1001` and `APP-9999`. Current generated data returns
`APP-1001` as a Personal Loan with status `Submitted`, amount INR 750,000,
7 days since creation, score `0.23`, and `escalation_required=false`. `APP-9999`
returns `found=false`, `error=not_found`, and no application or score details.
