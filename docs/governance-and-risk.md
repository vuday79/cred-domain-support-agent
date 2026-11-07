# Governance and Risk Classification

## Risk classification: HIGH

This is a financial-services customer-support system. It answers questions
about loan policies and amounts, retrieves loan application statuses, reports
fraud-review flags, and presents escalation recommendations. Inputs may contain
PAN, Aadhaar, and bank-account identifiers. Fabricated policy terms or incorrect
application details could mislead customers; prompt injection or unauthorized
tool execution could expose or misuse application information. These factors
justify a **HIGH** risk classification for this capstone. This classification
is a project risk assessment, not a production approval.

## Four-layer controls in this implementation

| Layer | Implemented controls |
|---|---|
| Application | Explicit agent/tool allowlists; retrieval, lookup, and composer tools are registered only for their intended roles. A CrewAI pre-dispatch hook checks the executing agent and exact lookup argument shape before the lookup service can run. Reviewer and Editor have no tools. |
| Data | The loan dataset and policy knowledge base are synthetic. Existing input guardrails detect and mask PII before workflow processing; privacy-safe JSONL logs record categories and outcomes, not prompts, responses, chunks, or raw identifiers. |
| Model | CrewAI and AutoGen use deterministic, local MOCK_LLM adapters. Output is schema-validated, constrained to retrieved or verified observations, checked by the existing groundedness guard, then reviewed by the two-turn AutoGen Reviewer/Editor stage. No external model APIs or keys are used. |
| Runtime | Each incoming request gets a separate token/cost budget. User input is checked before agent execution or session-history mutation; CrewAI calls charge before tool dispatch; the AutoGen stage reserves its two bounded turns before starting and validates actual calls against those reservations. Budget outcomes use generic, schema-valid safe responses and are recorded without request content. |

## Runtime budget method

The estimator assigns one token per four UTF-8 bytes, rounded up (and at least
one for non-empty text). It is deliberately deterministic, not a tokenizer and
not a provider usage measurement. The illustrative rate defaults to
`0.002` synthetic currency units per 1,000 estimated tokens. It is not a real
API charge, cost saving, or bill.

Default configurable limits:

- `CRED_REQUEST_MAX_INPUT_TOKENS=512`
- `CRED_REQUEST_MAX_TOTAL_TOKENS=16384`
- `CRED_REQUEST_MAX_ESTIMATED_COST=0.05`
- `CRED_SYNTHETIC_COST_PER_1000_TOKENS=0.002`

Changing these environment variables affects each new request; each request
owns its own budget object. The initial query is checked independently against
the input cap. Total token and synthetic-cost limits aggregate estimated
CrewAI and AutoGen prompt/completion usage, including tool descriptions and
retrieved evidence carried in model inputs. Known deterministic AutoGen turns
are conservatively pre-reserved before AutoGen execution, with measured calls
charged against those reservations.

## Residual risks and limitations

- Token counts are byte-based approximations; they do not represent provider
  tokenizer output, latency, CPU/GPU resource consumption, or API billing.
- The HIGH classification does not establish regulatory compliance, security
  certification, or suitability for production use.
- PII masking and injection detection are pattern/heuristic controls and may
  miss novel formats or attacks. Synthetic data does not represent live-data
  operational controls.
- The in-memory session history and request budget are process-local. They are
  not distributed, persistent, or coordinated across multiple service workers.
- A request rejected by a cumulative budget after CrewAI work may already have
  performed a bounded retrieval or authorized lookup. Its final response is
  safely rejected and the incomplete turn is not appended to session memory;
  the preceding authorized read is not rolled back.
- The dispatcher policy depends on the pinned/installed CrewAI hook lifecycle.
  Any framework upgrade should rerun the tool-dispatch regression tests and
  verify the before-tool hook remains on the execution path.
- No monitoring, human escalation service, access-control system, or production
  incident-response process is provided by this capstone.

## Reproducible evidence

Run `python scripts\run_task15_governance_demo.py` from the project root. The
generated `evaluation/task15_governance_transcript.md` records normal knowledge,
lookup and combined requests, an oversized input, a cumulative budget rejection,
authorized/unauthorized tool dispatch checks, and this risk/mitigation summary.
