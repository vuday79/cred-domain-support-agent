# Deterministic CrewAI MOCK_LLM

`src/cred_agent/mock_llm.py` provides `CredMockLLM`, a local implementation of
CrewAI's `BaseLLM` for demonstrations and tests. It uses no API key, external
LLM, or network client. Its output is deterministic for the same messages and
tool schemas.

## Tool calls and grounded results

The implementation targets the installed CrewAI 1.15.37 native tool-call
protocol. It selects a tool from its declared argument schema (`record_id`,
`query`, or `question`), validates the generated arguments against the schema,
and returns a CrewAI-native function-call object. CrewAI executes that tool and
sends the result back in a structured message with `role="tool"`. The next model
call returns that tool result as text; it does not search the system prompt for
an `"Observation:"` marker. Unrecognized or ambiguous requests receive the
fixed `MOCK_UNSUPPORTED_RESPONSE` text.
For CrewAI task prompts that label the user's input as `Customer question:`,
the mock passes that explicit question to the tool instead of including
expected-output instructions in the tool arguments.

This component does not itself execute a tool or compose policy answers. For a
grounded RAG response, the retrieval/generation tool must return its answer and
attribution in its tool result. The mock model only forwards that returned
content.

Passing a `response_model` is currently unsupported and returns the fixed
unsupported response rather than fabricating fields. This should be extended
only when the project's concrete structured-output schemas are implemented.

Telemetry is disabled before CrewAI imports by setting
`CREWAI_DISABLE_TELEMETRY=true`, `CREWAI_DISABLE_TRACKING=true`, and
`OTEL_SDK_DISABLED=true`.

## Run the tests

From the project root:

```powershell
python -m pytest tests\test_mock_llm.py -q
```
