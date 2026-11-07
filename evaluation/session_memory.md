# Session-Based Conversation Memory

The support application uses LangChain Core's `InMemoryChatMessageHistory` and
`RunnableWithMessageHistory` in
`src/cred_agent/services/session_memory.py`. `SessionConversationMemory` keeps
one LangChain history object per normalized `session_id`. Each successful
`chat(session_id, message)` call appends a HumanMessage and the resulting
AIMessage to that session; a new ID starts with an empty history. Histories
remain available for the lifetime of the service instance and are not written
to disk or shared with another process.

The wrapper invokes the existing three-agent `run_customer_query` workflow.
Earlier turns are passed as conversation context to specialist tasks so
follow-up questions can be interpreted in context. That history is explicitly
not treated as verified policy or application information: RAG and the
authorized application lookup remain the sources of facts. The Retrieval
Agent, Lookup Agent, Response Composer, deterministic `CredMockLLM`, and
`Crew.kickoff()` phases remain in use.

The grounded RAG answer generator reconstructs consecutive overlapping
fixed-size chunks from the same parent document before sentence matching. This
avoids composing truncated fragments from chunk boundaries while preserving
the original retrieved chunk records and source attribution.

## Run

From the project root:

```powershell
python scripts\run_memory_demo.py
```

The script prints the transcript and saves it to
`evaluation/session_memory_transcript.md`. It shows history before and after
each interaction in SESSION A and SESSION B, including explicit session IDs,
local timezone-aware timestamps, and evidence that SESSION B begins empty and
does not alter SESSION A's history.

Focused tests:

```powershell
python -m pytest tests\test_session_memory.py -q
```

## Compatibility note

The verified environment has `langchain-core==1.7.6`; the required classes
currently emit LangChain deprecation warnings (the history class and
`RunnableWithMessageHistory`). They are intentionally not filtered or
suppressed because this assignment requires these interfaces. `langchain-core`
was already listed in `requirements.txt`; no dependency versions were changed.
