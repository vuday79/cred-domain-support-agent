# Structured JSONL Logging Demonstration

**Generated:** 2026-11-08T01:27:07+05:30

All requests ran locally with deterministic MOCK_LLM and offline embeddings. The transcript contains sanitized log records only: prompts, assistant response text, document chunks, raw session IDs, and synthetic PII values are deliberately omitted.

## Successful knowledge-base request

```json
{
  "cache_used": false,
  "duration_ms": 203.932,
  "endpoint": "/ask",
  "error_category": null,
  "event_type": "http_request",
  "http_status": 200,
  "method": "POST",
  "outcome": "success",
  "output_groundedness_passed": true,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:06.859+00:00",
  "response_type": "knowledge_base",
  "session_id_hash": "hmac-sha256:469e80b7e84ca5e519ba",
  "timestamp": "2026-11-07T19:57:07.062+00:00",
  "trace_id": "07536b02-2928-4a69-8476-722ce1837806"
}
```

## Loan application lookup

```json
{
  "cache_used": false,
  "duration_ms": 126.088,
  "endpoint": "/ask",
  "error_category": null,
  "event_type": "http_request",
  "http_status": 200,
  "method": "POST",
  "outcome": "success",
  "output_groundedness_passed": true,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.067+00:00",
  "response_type": "loan_application",
  "session_id_hash": "hmac-sha256:5edc38f3649a29e4a1b3",
  "timestamp": "2026-11-07T19:57:07.193+00:00",
  "trace_id": "bac919a9-2c5e-4937-80a5-0142ce0c528e"
}
```

## Prompt injection rejection

```json
{
  "cache_used": false,
  "duration_ms": 2.37,
  "endpoint": "/ask",
  "error_category": "prompt_injection",
  "event_type": "http_request",
  "http_status": 200,
  "method": "POST",
  "outcome": "rejected",
  "output_groundedness_passed": false,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": true,
  "request_started_at": "2026-11-07T19:57:07.196+00:00",
  "response_type": "safe_error",
  "session_id_hash": "hmac-sha256:c9c9f0093e036fd25fc6",
  "timestamp": "2026-11-07T19:57:07.199+00:00",
  "trace_id": "baf66cad-278f-48fc-bada-7767e93b7d27"
}
```

## Synthetic PII-bearing request

```json
{
  "cache_used": false,
  "duration_ms": 142.686,
  "endpoint": "/ask",
  "error_category": null,
  "event_type": "http_request",
  "http_status": 200,
  "method": "POST",
  "outcome": "success",
  "output_groundedness_passed": true,
  "pii_categories": [
    "aadhaar",
    "bank_account",
    "pan"
  ],
  "pii_detected": true,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.201+00:00",
  "response_type": "loan_application",
  "session_id_hash": "hmac-sha256:1f9bcb9d887b99060932",
  "timestamp": "2026-11-07T19:57:07.344+00:00",
  "trace_id": "c1411973-5f09-415e-a897-f99e2bd1bd0a"
}
```

## Out-of-scope request

```json
{
  "cache_used": false,
  "duration_ms": 124.538,
  "endpoint": "/ask",
  "error_category": null,
  "event_type": "http_request",
  "http_status": 200,
  "method": "POST",
  "outcome": "success",
  "output_groundedness_passed": true,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.348+00:00",
  "response_type": "knowledge_base_fallback",
  "session_id_hash": "hmac-sha256:a77b8bf2362da512c19e",
  "timestamp": "2026-11-07T19:57:07.472+00:00",
  "trace_id": "7b624238-ceda-4b61-bdd4-a5e3eedd95ed"
}
```

## Safe validation failure

```json
{
  "cache_used": false,
  "duration_ms": 0.835,
  "endpoint": "/ask",
  "error_category": "validation_error",
  "event_type": "http_request",
  "http_status": 422,
  "method": "POST",
  "outcome": "validation_error",
  "output_groundedness_passed": null,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.475+00:00",
  "response_type": null,
  "session_id_hash": null,
  "timestamp": "2026-11-07T19:57:07.476+00:00",
  "trace_id": "3636adec-d9e4-4ac0-b1f2-46e4b133d656"
}
```

## WebSocket message

```json
{
  "cache_used": false,
  "duration_ms": 147.735,
  "endpoint": "/ws/{session_id}",
  "error_category": null,
  "event_type": "websocket_message",
  "http_status": null,
  "method": "WEBSOCKET",
  "outcome": "success",
  "output_groundedness_passed": true,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.480+00:00",
  "response_type": "knowledge_base",
  "session_id_hash": "hmac-sha256:0430076cf564baecdac7",
  "timestamp": "2026-11-07T19:57:07.628+00:00",
  "trace_id": "d630877d-d548-4a7c-9d00-6e2f947bad59"
}
```

## WebSocket disconnect lifecycle

The client closed the WebSocket after receiving its reply. The disconnect is separately traceable and does not include message data.

```json
{
  "cache_used": false,
  "duration_ms": 0.571,
  "endpoint": "/ws/{session_id}",
  "error_category": null,
  "event_type": "websocket_disconnect",
  "http_status": null,
  "method": "WEBSOCKET",
  "outcome": "disconnected",
  "output_groundedness_passed": null,
  "pii_categories": [],
  "pii_detected": false,
  "prompt_injection_detected": false,
  "request_started_at": "2026-11-07T19:57:07.628+00:00",
  "response_type": null,
  "session_id_hash": "hmac-sha256:0430076cf564baecdac7",
  "timestamp": "2026-11-07T19:57:07.629+00:00",
  "trace_id": "1fd1b8ce-fdfa-4234-8dfe-9b1cf72eb36a"
}
```

## Configuration and privacy

- Set `CRED_API_LOG_PATH` to an absolute or user-relative JSONL file path. By default, Windows uses `%LOCALAPPDATA%\CredDomainSupportAgent\logs\api_requests.jsonl`; other platforms use the user state directory.
- Session identifiers use a truncated, process-keyed HMAC-SHA256 pseudonym (not stable across process restarts).
- Durations use `time.perf_counter()`; timestamps are UTC ISO 8601.
- PII values, prompts, responses, chunks, exception messages, and environment values are not fields in the log schema.
- No response cache is configured; `cache_used` is always `false`.
