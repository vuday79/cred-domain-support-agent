# Input and Output Guardrails

`src/cred_agent/services/guardrails.py` provides deterministic, offline
guardrails for the customer-support workflow.

## Input policy

`guard_user_input` masks PAN-shaped values, 12-digit Aadhaar-shaped values,
and 9–18 digit bank-account values when explicitly labeled as an account
number. The label requirement for bank accounts avoids masking unrelated
amounts, tenures, and ordinary numeric identifiers. Fixed-format PII is
replaced with category placeholders while the surrounding question remains
available to the agents.

The `run_customer_query` entry point masks before routing or Crew construction.
`SessionConversationMemory.chat` masks before invoking
`RunnableWithMessageHistory`, so the stored HumanMessage is already redacted.
Prior conversation is also sanitized before it is added to a task prompt.
The application logs only PII categories and injection decisions, never the
input text.

Prompt-injection patterns cover instruction overrides, requests to disclose
internal prompts, safety bypasses, data exfiltration, and user-content
priority overrides. The deterministic policy is **reject before Crew kickoff**.
Direct workflow calls return the existing schema-valid safe error response and
show zero kickoffs. Session calls append a generic rejected-request marker and
the safe response; the attempted instruction itself is not retained in history.

## Output policy

After Pydantic validation, the response guard checks customer-facing claims
against retrieved chunk text and validated application fields. Citations are
removed from the claim text and never count as evidence. Knowledge responses
must preserve source attribution that maps to retrieved parent documents.
Numeric claims must match evidence values. Unsupported claims are replaced by
the schema-valid `safe_error` response. The established knowledge-base
fallback, application not-found result, loan status, and escalation fields
remain intact when supported.

At the workflow boundary the final text must also match the deterministic
composition of the actual specialist tool findings. This prevents an added
claim from passing solely because some of its words happen to occur in
retrieved material.

The claim check is intentionally conservative and deterministic: it uses
sentence-level content-word coverage, exact numeric support, required evidence
types, and attribution consistency. It is not a probabilistic semantic
entailment model; ambiguous paraphrases may be safely rejected rather than
accepted without a reliable keyless judge.

Final structured content and auditable workflow findings are redacted again
before exposure, then the structured response is validated once more.

## Evidence and tests

Run the demonstration from the project root:

```powershell
python scripts\run_guardrails_demo.py
```

The script writes `evaluation/guardrails_transcript.md`. It uses synthetic PII
only, withholds raw values from the transcript, verifies their absence from
history/results/log records, demonstrates pre-kickoff injection rejection,
and compares a real grounded answer with a deliberately altered unsupported
claim.

Focused tests:

```powershell
python -m pytest tests\test_guardrails.py tests\test_integrated_crew.py tests\test_session_memory.py tests\test_structured_response.py -q
```
