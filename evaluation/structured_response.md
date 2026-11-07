# Structured Crew Responses

The final Crew response is represented by
`src/cred_agent/schemas.py::SupportResponse`. It includes the customer-facing
response, response classification, grounding flag, source document IDs,
retrieved context and similarity measurements, and complete loan application
and escalation fields when supplied by the Lookup Agent. Application statuses
are constrained to the values used by the generated dataset. Pydantic
cross-field validation prevents inconsistent response types, missing source
attribution for grounded knowledge answers, and invalid escalation values.

## CrewAI 1.15.37 API

The installed `Crew` model has no `response_format` field, and
`Crew.kickoff()` accepts `inputs`, `input_files`, and `from_checkpoint` only.
The installed `Task` does support `response_model`, described by CrewAI as the
Pydantic model for native provider structured outputs. The Composer task sets
`response_model=SupportResponse` and `output_pydantic=SupportResponse`;
CrewAI 1.15.37 passes this response model to the agent executor and exports the
TaskOutput's Pydantic model. The deterministic MOCK_LLM recognizes that model
request and constructs the same schema directly from the structured specialist
findings without contacting an external provider.

`run_customer_query` validates the final composer task output before returning
it. On a Pydantic validation failure it returns a separately validated,
deterministic `safe_error` response, records the validation error for
diagnostics, and does not expose the malformed output as a customer response.
`IntegratedCrewResult.final_response` remains the customer-facing text for
existing callers; `structured_response` carries the complete schema, and
`to_dict()` includes its JSON-compatible fields.

The LangChain memory service continues to store the final validated response
text as its assistant message. Specialist routing, retrieval attribution,
loan lookup facts and escalation scoring are unchanged.

## Reproduce the evidence

From the project root:

```powershell
python scripts\run_structured_response_demo.py
```

The script writes a successful grounded RAG response and a deliberately
invalid response with Pydantic's validation failure and safe replacement to
`evaluation/structured_response_transcript.md`. It runs the embedding model
offline from its local cache and uses the existing deterministic `CredMockLLM`.

Focused tests:

```powershell
python -m pytest tests\test_structured_response.py tests\test_integrated_crew.py tests\test_response_composer.py -q
```
