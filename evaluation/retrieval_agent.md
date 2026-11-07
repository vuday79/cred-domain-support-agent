# Cred Retrieval Agent

The Retrieval Agent is implemented in
`src/cred_agent/agents/retrieval_agent.py`. It uses CrewAI's installed
`Agent`/`BaseTool` interfaces and the deterministic `CredMockLLM`; no external
LLM API or credential is needed.

## Retrieval tool behavior

`CredKnowledgeRetrievalTool` accepts one customer question and delegates
retrieval and extractive generation to the existing `PolicyRetriever` and
`GroundedGenerator`. It queries only `cred_fixed_size`, returns the grounded
answer together with chunk text, chunk ID, parent document ID, filename, and
cosine similarity, and uses the calibrated top-1 threshold **0.286211**. When
the threshold is not met or retrieved context cannot support an extractive
answer, the tool returns the existing safe fallback:

> I don't have sufficient information in the available knowledge base to answer this question.

The persisted collection loader checks that the database and both expected
collections already exist and use cosine distance before constructing the
existing vector store. It does not create, rename, or reindex collections.

The agent is configured with the role **Cred Knowledge Base Retrieval
Specialist**, its approved-knowledge-base goal and backstory, one retrieval
tool, no delegation, and a two-iteration limit for deterministic execution.
CrewAI executes the native structured tool call; MOCK_LLM forwards the
structured tool result rather than creating policy content.

## Run a demonstration

From the project root:

```powershell
python scripts\run_retrieval_agent.py
```

The script runs the question “How is EMI calculated?” and an out-of-scope IPL
question through the CrewAI agent using the existing project-root `chroma_db`.
If the index is missing or incomplete, the loader reports that and stops
without silently creating a collection.

Run the focused tests with:

```powershell
python -m pytest tests\test_retrieval_agent.py -q
```
