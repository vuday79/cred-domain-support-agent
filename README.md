# Cred Domain Support Agent

Cred (Banking & FinTech) capstone project. The local support API wraps the
existing deterministic CrewAI workflow, LangChain session memory, input/output
guardrails, structured Pydantic response, and persistent Chroma collections.

## Run locally

From the project root on Windows:

```powershell
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m uvicorn src.cred_agent.api:app --host 127.0.0.1 --port 8000
```

The local SentenceTransformers model must already be cached. MOCK_LLM requires
no API keys and makes no external LLM calls. The API sets
`CREWAI_DISABLE_TELEMETRY=true` before importing the agent workflow.
Structured API JSONL logs are written outside the repository by default:
`%LOCALAPPDATA%\CredDomainSupportAgent\logs\api_requests.jsonl` on Windows.
Set `CRED_API_LOG_PATH` to override the file location. When running locally,
disable Uvicorn access logs if session IDs in WebSocket URL paths must also be
excluded from server-level access logs:

```powershell
$env:CRED_API_LOG_PATH = "$env:LOCALAPPDATA\CredDomainSupportAgent\logs\api_requests.jsonl"
python -m uvicorn src.cred_agent.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

## Endpoints

### `POST /ask`

Request:

```json
{
  "query": "How is EMI calculated?",
  "session_id": "customer-session-1"
}
```

Returns an API envelope with a trace ID, response text, validated
`structured_response`, selected specialist roles, guardrail decisions, and the
current session history size. Requests with the same session ID share
in-process `InMemoryChatMessageHistory`; a restart clears that history.
PAN/Aadhaar/account values are masked before the query is recorded or sent to
the workflow. Injection attempts return a safe refusal without Crew kickoff.
The public structured response includes source document IDs, filename-only
attribution, response type, groundedness, and verified loan fields. Retrieved
chunk text, similarity diagnostics, and absolute local paths are not returned.

### `POST /add-document`

Request:

```json
{
  "document_id": "repayment_notice",
  "title": "Repayment notice",
  "content": "Contact support before the repayment due date.",
  "metadata": {
    "owner": "lending-operations",
    "revision": 1
  }
}
```

Documents are validated, PII-masked, passed through the existing document
cleaning and both established chunkers, embedded locally, and upserted into
`cred_fixed_size` and `cred_sentence_based` through the existing
`ChromaVectorStore` interface. Duplicate parent document IDs are rejected;
reserved Chroma metadata keys cannot be overridden.

### `WS /ws/{session_id}`

Send JSON messages such as `{"message":"What are KYC requirements?"}`. The
server returns the same structured envelope as `/ask`, supports multiple
turns, returns a generic error for malformed messages, and handles
`WebSocketDisconnect` without exposing an exception.

## Task 16: In-memory response cache

The API owns a bounded, thread-safe, process-local LRU cache. Query keys use
trimmed, case-folded text with repeated whitespace collapsed, plus a hash of
answer-affecting retrieval profile and cache generation. Successfully indexed
documents clear the cache to prevent stale policy answers. Configure capacity
with `CRED_RESPONSE_CACHE_MAX_SIZE` (default `256`).

Cache lookup happens only after request validation, input guardrails, and
request-budget registration. Hits return the validated public answer, preserve
its source attribution and response type, skip CrewAI and AutoGen, and still
flow through session-memory bookkeeping. Only history-free, PII-free
knowledge-base responses that passed Pydantic validation, AutoGen approval,
and final groundedness checks can be stored. Loan lookups, combined responses,
fallbacks, injection rejections, PII-bearing inputs or outputs, budget
rejections, validation failures, and session-history-dependent answers are
excluded. Cache hits incur input estimation only; skipped agent calls are not
charged to that request's synthetic budget.

The cache is not persistent or distributed: a process restart clears entries,
and independent server workers do not share them. Cache hit, miss, and LRU
eviction counters are available through the app's cache stats for local
diagnostics.

Generate the reproducible cache demonstration and run its focused tests:

```powershell
python scripts\run_task16_cache_demo.py
python -m pytest tests\test_response_cache.py tests\test_api_logging.py -q
```

The transcript is written to
`evaluation/response_cache_transcript.md`; API cache behavior is documented in
`evaluation/api.md`.

## Demonstration and tests

```powershell
python scripts\run_api_demo.py
python scripts\run_logging_demo.py
python -m pytest tests\test_api.py -q
python -m pytest tests\test_api_logging.py -q
python -m pytest -q
```

The demo writes request/response evidence to
`evaluation/api_transcript.md`. API architecture and the error-handling
boundaries are documented in `evaluation/api.md`. The JSONL demo writes
sanitized request-log examples to `evaluation/logging_transcript.md`.

## Task 13 evaluation

Run the deterministic evaluation from the project root:

```powershell
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts\run_task13_evaluation.py
```

The script loads the existing persistent `cred_fixed_size` and
`cred_sentence_based` Chroma collections and executes every case in
`evaluation/evaluation_dataset.json` through
`src.cred_agent.crew.run_customer_query`. A local cached SentenceTransformers
model and the deterministic `CredMockLLM` are required; the evaluator makes no
external LLM or network calls. It does not silently omit failed workflow runs.
The rubric reports accuracy, grounding, completeness, and safety on a 0-5
scale, with per-score reasons and a predeclared pass rule: each case must
average at least 4.0/5, and every dimension must score at least 3/5.

Generated reports:

- `evaluation/evaluation_report.json` - machine-readable case results and
  mathematically computed aggregates.
- `evaluation/evaluation_report.md` - methodology, per-case score details,
  aggregate results, and any failures.

## Task 14: AutoGen Reviewer and Editor

After CrewAI creates and validates its response, a tool-less AutoGen
`RoundRobinGroupChat` runs the deterministic local `Reviewer` and `Editor` for
exactly two turns. They receive the original guarded question, draft, retrieved
evidence, and validated lookup observation; neither agent can call retrieval,
lookup, external tools, or an external model. The Editor returns a validated
Pydantic `ReviewVerdict` through AutoGen's registered structured message type.
Grounded drafts remain unchanged; unsupported claims are removed using supplied
evidence only, and the final response passes the existing groundedness check.

Generate the reproducible demonstration from the project root:

```powershell
python scripts\run_autogen_review_demo.py
python -m pytest tests\test_autogen_review.py -q
```

The generated [AutoGen review transcript](evaluation/autogen_review_transcript.md)
shows grounded knowledge-base and loan lookup approval, removal of an injected
unsupported claim, safe fallback when no draft claim is supported, structured
verdict validation, and the two-turn Reviewer/Editor order.

## Task 15: Governance and runtime budget

The application enforces tool permissions through agent-specific tool
registration, MOCK_LLM call-selection checks, and a CrewAI pre-dispatch hook.
Only the Lookup Agent may run `cred_application_record`, and its dispatched
payload must contain a valid `record_id`; the Retrieval Agent can use only the
knowledge-base tool, while the Composer, Reviewer, and Editor have no lookup
capability.

Each `/ask` and WebSocket message receives a fresh request-scoped budget.
Defaults can be adjusted with `CRED_REQUEST_MAX_INPUT_TOKENS`,
`CRED_REQUEST_MAX_TOTAL_TOKENS`, `CRED_REQUEST_MAX_ESTIMATED_COST`, and
`CRED_SYNTHETIC_COST_PER_1000_TOKENS`. The deterministic estimator uses one
estimated token per four UTF-8 bytes rounded up. Estimated costs are
illustrative synthetic values, **not API charges**. Input is rejected before
agent execution or session-history mutation when it exceeds the input cap.
CrewAI model calls are charged before their tool dispatch; the bounded two-turn
AutoGen stage is pre-reserved and reconciled with its actual local calls. Budget
rejections use a schema-valid safe response, are logged by outcome only, and do
not append an incomplete turn to session memory.

The project classifies itself as **HIGH risk** because it handles financial
policy and loan-status information, escalation/fraud indicators, and possible
PII. Implemented controls and residual limitations are detailed in
[governance-and-risk.md](docs/governance-and-risk.md).

Run Task 15 evidence and focused regressions from the project root:

```powershell
python scripts\run_task15_governance_demo.py
python -m pytest tests\test_tool_governance.py tests\test_runtime_budget.py -q
```

The reproducible transcript is generated at
`evaluation/task15_governance_transcript.md`.
