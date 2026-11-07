# Cred Customer Support Response Composer

`src/cred_agent/agents/response_composer.py` defines the **Cred Customer
Support Response Composer** and a reusable CrewAI Task factory. The agent uses
the deterministic `CredMockLLM`, has no tools of its own, and does not make
external LLM API calls.

## Input and output

The task accepts `specialist_findings` as JSON text from the existing Retrieval
Agent's RAG tool or Lookup Agent's application tool. MOCK_LLM recognizes the
explicit `Specialist findings:` section, reads only the structured result
following it, and deterministically formats the fields it receives.

For grounded RAG findings, the response preserves the answer and source
document attribution. For application results, it preserves the ID, status,
category, loan amount, application age, fraud-review flag, escalation score,
decision, and service-provided reason when present. Unknown IDs and insufficient
knowledge-base results remain explicit; missing facts are not filled in.

If a lookup says escalation is required, the response says it is required and
explicitly notes that the result does not confirm completion. It does not claim
that escalation or any other operational action has already occurred.

The response task uses the Task 9 Pydantic schema:
`SupportResponse` is set as both the CrewAI `response_model` and
`output_pydantic`. The integrated Crew validates the final output again before
exposing its customer-facing text. See `evaluation/structured_response.md` for
the schema, validation behavior, and executable evidence.

## Run the standalone demonstration

From the project root:

```powershell
python scripts\run_response_composer.py
```

The script composes one existing RAG result for “How is EMI calculated?” and
one lookup result for `APP-1001`. It runs the existing embedding model in
Hugging Face/Transformers offline mode; the model must already be cached
locally, otherwise startup fails rather than downloading it. Run focused tests
with:

```powershell
python -m pytest tests\test_response_composer.py -q
```
