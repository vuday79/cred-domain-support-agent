# FastAPI and WebSocket Layer

`src/cred_agent/api.py` exposes:

- `POST /ask`: a validated `query` and `session_id` are passed through the
  existing `SessionConversationMemory`, which calls the integrated
  `run_customer_query` workflow. Its input guardrails, CrewAI tools, Pydantic
  `SupportResponse`, source attribution, and output groundedness checks remain
  active. The public structured response retains response type, groundedness,
  source document IDs with filename-only attribution, and verified loan
  application fields. Retrieved chunk text, similarity diagnostics, and local
  filesystem paths stay internal.
- `POST /add-document`: validates document ID, title, content, and scalar
  metadata; masks PII and rejects injection-like content; then uses the
  existing document cleaning, fixed and sentence chunkers, local embeddings,
  and `ChromaVectorStore` upserts. Only the established
  `cred_fixed_size` and `cred_sentence_based` collections are written. When the
  API uses an injected vector store, its retrieval workflow reads the same
  collections so newly added documents are immediately queryable.
- `WS /ws/{session_id}`: accepts repeated `{"message": "..."}` JSON frames,
  uses the same session workflow, returns the same structured response as
  `/ask`, handles malformed frames without echoing their contents, and exits
  cleanly on `WebSocketDisconnect`.

The fixed-size retrieval threshold remains `0.286211`. A chunk can contribute
to an extractive answer only if that chunk itself meets the threshold; a high
similarity score from an unrelated top-ranked chunk cannot authorize weaker
supporting chunks. Thus the follow-up tenure question shown in the WebSocket
transcript safely falls back: its joint-account chunk scores `0.2989`, while
its EMI evidence scores `0.2830` and `0.2791`.

The application returns generic validation and processing errors; response
validation details and exception text are not returned. API request events use
the shared JSONL schema implemented in
`src/cred_agent/services/api_jsonl_logging.py`. Each HTTP request and WebSocket
message gets a unique trace ID, UTC ISO 8601 start/completion timestamps, a
monotonic duration, a safe outcome/error category, response classification,
guardrail results, and the actual `cache_used` value.
Session IDs are represented only by a truncated HMAC-SHA256 digest using a
process-local random key, so logged pseudonyms are not reversible by offline
guessing and are not stable across process restarts.

Logs never contain request prompts, assistant response text, retrieved chunks,
raw PII, or exception messages/stack traces. Existing `guard_user_input`
processing supplies PII categories. The default file is
`%LOCALAPPDATA%\\CredDomainSupportAgent\\logs\\api_requests.jsonl` on Windows;
set `CRED_API_LOG_PATH` to select another path. The file logger is app-scoped
and closed with the FastAPI lifespan. For Uvicorn deployments, disable or
sanitize server access logs to avoid recording the raw WebSocket URL path.
CrewAI telemetry is forced off before the CrewAI workflow is imported.

## In-memory response cache

Each FastAPI app owns a thread-safe `InMemoryResponseCache` with a configurable
capacity (`CRED_RESPONSE_CACHE_MAX_SIZE`, default 256) and least-recently-used
eviction. Keys are SHA-256 hashes of normalized query text, a retrieval-profile
identifier, and an app cache generation. Normalization trims leading/trailing
whitespace, case-folds, and collapses repeated whitespace. A successful
`/add-document` indexing operation increments the generation and clears
entries so cached answers cannot outlive knowledge-base updates.

The cache is checked after request schema validation, input guardrails, and
budget input accounting, but before CrewAI. It is consulted only for a
history-free, PII-free request that does not require loan lookup. A hit returns
the schema-validated stored `SupportResponse`, keeps its citations/response
type, bypasses CrewAI and AutoGen, and is still recorded by session memory.
The request budget still accounts for the user input but adds no model-call
charges for skipped work.

Insertion requires a grounded `knowledge_base` response with source
documents, successful Pydantic validation, a successful two-turn AutoGen
approval, final groundedness validation, and no detected PAN, Aadhaar, or bank
account identifiers in response or retrieved evidence. The policy excludes
loan application and combined responses, out-of-scope fallback, validation or
budget errors, groundedness failures, PII-bearing inputs/outputs, injection
rejections, and answers evaluated with non-empty session history. Thus this
cache does not serve answers that depend on prior turns. `cache_used` in API
JSONL records is true only for actual response-cache hits. The cache's hit,
miss, and eviction statistics are process-local.

The cache is deliberately ephemeral and per-process: it is neither persistent
nor shared across workers or server instances. Run
`python scripts\run_task16_cache_demo.py` to generate
`evaluation/response_cache_transcript.md`, which demonstrates hits, skips,
timings, counters, exclusions, and LRU eviction.

The synchronous CrewAI kickoff is run from FastAPI's synchronous REST handlers
and Starlette's worker thread for WebSocket messages, avoiding CrewAI's
`kickoff()`-inside-a-running-event-loop restriction while keeping the existing
workflow unchanged.

Run locally and create the redacted API transcript using the commands in
`README.md`. `evaluation/api_transcript.md` includes REST requests and
responses, document ingestion, multiple WebSocket turns, and a client
disconnect. `scripts/run_logging_demo.py` writes sanitized JSONL evidence to
`evaluation/logging_transcript.md`. The API memory remains in-process and does
not persist across server restarts. Added documents are indexed in both
existing collections and are persistent in the configured Chroma storage.
