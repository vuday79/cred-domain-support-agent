# Integrated Three-Agent Crew

`src/cred_agent/crew.py` routes a customer request through the existing
Retrieval Agent, Lookup Agent, and Response Composer. All three are constructed
for each request; only the specialists selected by the deterministic router
receive specialist tasks, and the Composer always handles the gathered results.

## Routing

- A query with no application ID goes to the Retrieval Agent. Unsupported
  questions therefore reach the existing RAG threshold/fallback logic.
- An ID matching `APP-` followed by letters, digits, or hyphens is validated
  against the Lookup Agent's Pydantic `record_id` schema. A query with only an
  application status intent goes to the Lookup Agent.
- If an application ID and explicit policy intent (for example EMI, KYC,
  eligibility, prepayment, or credit score) are both present, both specialists
  are selected.
- Multiple application IDs are rejected as ambiguous.

Tool selection is not based on tool names. Each specialist is assigned its
existing tool and its declared argument schema. The deterministic MOCK_LLM
uses the task prompt and declared schema to generate CrewAI's native structured
tool call, and CrewAI executes that call.

## Crew execution sequence

The specialist tasks run together in a sequential Crew kickoff. A separate
Composer Crew kickoff follows because the specialist findings are only known
after the specialist kickoff completes. The actual structured tool results are
passed as JSON to the existing Composer task; actual tool name, arguments, and
result are exposed in `IntegratedCrewResult.tool_invocations` for auditing.
Both phases use `Crew.kickoff()`.

The Retrieval Agent retains the existing `cred_fixed_size` collection and
0.286211 cosine threshold. The Lookup Agent delegates to
`check_loan_application_status` and preserves its escalation score and
decision. The Composer only formats recognized specialist-result fields;
unknown application IDs and unsupported knowledge questions remain explicit,
and escalation-required is not described as completed.

The final Composer task uses CrewAI 1.15.37's `Task.response_model` and
`Task.output_pydantic` with `SupportResponse`. CrewAI's installed `Crew` and
`Crew.kickoff()` do not expose a `response_format` parameter. MOCK_LLM returns a
local deterministic Pydantic response built from verified specialist findings.
The orchestration validates the final model again and replaces malformed output
with a schema-valid safe error response. See
`evaluation/structured_response.md`.

## Keyless deterministic execution

Every agent is configured with `CredMockLLM`, which requires no API key and
makes no external LLM API calls. The demo puts Hugging Face/Transformers in
offline mode for the locally cached SentenceTransformer retrieval model.

Run the four requested examples from the project root:

```powershell
python scripts\run_crew_demo.py
```

Run the focused integration tests with:

```powershell
python -m pytest tests\test_integrated_crew.py -q
```
