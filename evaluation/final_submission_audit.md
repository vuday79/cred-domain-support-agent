# Final Capstone Submission Audit

**Audit date:** 2026-11-08  
**Scope:** Submission readiness and evidence inventory only; no feature development.

## Summary

**The implementation and evidence for Tasks 1–16 are present, but do not publish
from the current Git workspace yet.** The active Git root is the parent
`repos` directory, not this project. The project currently has 0 tracked files
and 207 untracked project entries before this audit report was added (208
afterward); the broader worktree had 318 untracked entries at that point,
spanning other projects. There is no project `.gitignore`, so runtime
Chroma databases, Python bytecode, and pytest cache files are also unignored.
The RAG and structured-response transcripts contain local absolute paths with
the workstation username.

## Task coverage and evidence

| Task | Evidence present |
|---|---|
| 1. Deterministic loan dataset | [`data/dataset.py`](../data/dataset.py), dataset tests, [`evaluation/knowledge_base_validation.md`](knowledge_base_validation.md). Runtime check: 50 records; seed 42, fraud probability 0.20, one attempt; category counts 17/5/10/7/11; status counts 11/16/9/4/10; 14% flagged; observed amount range INR 100,000–4,740,000. |
| 2. Knowledge base | 12 policy documents under [`data/knowledge_base/`](../data/knowledge_base), with topic inventory and simulated-policy disclaimer in its README. |
| 3. Chunking and vector index | [`scripts/build_index.py`](../scripts/build_index.py), RAG chunking/vector-store modules, and focused indexing tests; two persisted Chroma collections are present locally. |
| 4. Grounded retrieval/generation | [`src/cred_agent/rag/retrieval.py`](../src/cred_agent/rag/retrieval.py), [`scripts/run_rag_demo.py`](../scripts/run_rag_demo.py), [`evaluation/rag_evaluation_report.md`](rag_evaluation_report.md). |
| 5. Chunking comparison | Per-query document-level Precision@3/Recall@3 arithmetic and recommendation in [`evaluation/rag_evaluation_report.md`](rag_evaluation_report.md). |
| 6. Loan lookup/escalation | [`src/cred_agent/services/loan_application_lookup.py`](../src/cred_agent/services/loan_application_lookup.py), [`evaluation/loan_application_lookup.md`](loan_application_lookup.md), lookup tests. |
| 7. CrewAI multi-agent workflow | [`src/cred_agent/crew.py`](../src/cred_agent/crew.py), agent modules under [`src/cred_agent/agents/`](../src/cred_agent/agents), and [`evaluation/integrated_crew.md`](integrated_crew.md). |
| 8. Session memory | [`src/cred_agent/services/session_memory.py`](../src/cred_agent/services/session_memory.py), [`evaluation/session_memory_transcript.md`](session_memory_transcript.md), session-memory tests. |
| 9. Structured response | [`src/cred_agent/schemas.py`](../src/cred_agent/schemas.py), [`evaluation/structured_response_transcript.md`](structured_response_transcript.md), schema tests. |
| 10. Guardrails | [`src/cred_agent/services/guardrails.py`](../src/cred_agent/services/guardrails.py), [`evaluation/guardrails_transcript.md`](guardrails_transcript.md), guardrail tests. |
| 11. FastAPI/REST/WebSocket | [`src/cred_agent/api.py`](../src/cred_agent/api.py), [`evaluation/api_transcript.md`](api_transcript.md), API tests. |
| 12. Structured privacy-safe logging | [`src/cred_agent/services/api_jsonl_logging.py`](../src/cred_agent/services/api_jsonl_logging.py), [`evaluation/logging_transcript.md`](logging_transcript.md), logging tests. |
| 13. Evaluation framework | [`src/cred_agent/evaluation_judge.py`](../src/cred_agent/evaluation_judge.py), [`evaluation/evaluation_dataset.json`](evaluation_dataset.json), JSON/Markdown reports and evaluation tests. Latest observed report: 19/19 passed. |
| 14. AutoGen Reviewer/Editor | [`src/cred_agent/services/autogen_review.py`](../src/cred_agent/services/autogen_review.py), [`evaluation/autogen_review_transcript.md`](autogen_review_transcript.md), AutoGen tests. Demo command was run successfully during this audit. |
| 15. Governance and runtime budget | [`docs/governance-and-risk.md`](../docs/governance-and-risk.md), [`evaluation/task15_governance_transcript.md`](task15_governance_transcript.md), governance/budget tests. Demo command completed successfully in the preceding validation. |
| 16. In-memory response cache | [`src/cred_agent/services/response_cache.py`](../src/cred_agent/services/response_cache.py), [`evaluation/response_cache_transcript.md`](response_cache_transcript.md), cache tests. Demo command completed successfully in the preceding validation. |

This is an inventory of implementation and evidence files, not a second
line-by-line code review.

## README and reproducibility check

| Requested README item | Result |
|---|---|
| Project overview and architecture | Present at a high level. |
| Installation/environment setup | **Incomplete:** no Python-version guidance or dependency-install command (`pip install -r requirements.txt`). |
| Offline MOCK_LLM | Present; local SentenceTransformers weights must already be cached. |
| Dataset seed, categories/statuses, weights, amount range, generation details | **Missing:** the runtime design is reproducible in code, but README does not state it. The generator uses uniform category/status choices, seed 42, 0.20 fraud probability, and deterministic retry; see observed statistics above. |
| Build local vector database | **Missing:** `scripts/build_index.py` exists, but README does not give its command. |
| API and WebSocket run instructions | Present. |
| Evaluation command and report locations/results | Command and report paths are present; **the 19/19 result and score averages are not summarized in README**. |
| Governance, AutoGen, and cache demo commands | Present. |
| Synthetic/simulated-policy limitation | **Not explicit in README.** It is stated in the knowledge-base README and governance evidence. |
| Process-local cache/session memory | Present. |
| Deterministic MOCK_LLM limitation | Deterministic/offline mode is stated; it should be identified explicitly as a non-production LLM limitation. |
| Illustrative budget/token estimation | Present and clearly not represented as real API billing. |
| HIGH-risk classification/governance limitation | HIGH risk is stated and linked to detailed governance documentation; the non-certification/production limitation is clearer in that document than in README. |

`requirements.txt` exists but uses unpinned package names and there is no lock
file. The relevant demo/evaluation commands ran in the configured environment;
the model weights were locally available. A fresh machine still needs those
weights and compatible dependency versions.

## Sensitive-data and generated-file scan

- No common API-key/private-key markers were found in source, reports, or
  configuration text during the scan.
- PAN/Aadhaar-shaped values found are explicitly synthetic test/evaluation
  fixtures, as required for guardrail demonstrations. No `.env` or JSONL
  runtime log file was found in the project inventory.
- **Local path disclosure:** [`evaluation/rag_evaluation_report.md`](rag_evaluation_report.md)
  records the absolute Chroma path; [`evaluation/structured_response_transcript.md`](structured_response_transcript.md)
  records absolute source paths. Both include the workstation username and
  should be made repository-relative before public publication.
- **Unignored runtime artifacts:** `chroma_db/`, `data/chroma_db/`,
  `.pytest_cache/`, and Python `__pycache__`/`.pyc` files are present.
  Inspect whether any persisted index is intentionally part of the deliverable;
  do not stage these by accident.

## Git readiness and blockers

1. **Critical — wrong Git boundary:** `git rev-parse --show-toplevel` resolves
   to the parent `repos` directory. The active branch is `main`, it has no
   commits, and `origin` points to `vuday79/repos`. The project has no tracked
   files. Publishing from this root risks including unrelated sibling
   workspaces and duplicate project copies. Use a clean, dedicated repository
   root for this capstone and confirm its remote before staging.
2. **Critical — missing ignore policy:** there is no project `.gitignore`;
   runtime Chroma data, caches, and bytecode appear as untracked files. Add and
   review ignore rules before staging, while preserving only intentionally
   required text evidence and source.
3. **Submission blocker — README omissions:** document the dataset design
   choices and exact local-index build/setup commands required by the
   assignment, add the latest evaluation result, and make synthetic/simulated
   and non-production limitations explicit.
4. **Privacy cleanup:** replace the absolute local paths in the two reports
   identified above with relative paths before public release.

## Checks actually run/observed

- `python -m pytest -q`: **232 passed** in the Task 16 implementation
  validation immediately before this audit.
- `python scripts\run_task13_evaluation.py`: **19 passed, 0 failed** in that
  validation; reports exist under `evaluation/`.
- `python scripts\run_task15_governance_demo.py` and
  `python scripts\run_task16_cache_demo.py`: completed successfully in that
  validation.
- `python scripts\run_autogen_review_demo.py`: completed successfully during
  this audit with offline environment variables set and local model weights
  available.
- Dataset import/statistics check completed during this audit.
- Git root/status, README, artifact inventory, API-key marker, and formatted
  PII-fixture scans were inspected. The RAG report's absolute-path finding is
  confirmed.

## Commands before publishing

From PowerShell, first move the project into/use a **dedicated clean Git
repository** (do not use `git add .` from the current parent `repos` root).
After correcting the README, adding reviewed ignore rules, and sanitizing the
two paths above, run these from that project root:

```powershell
python -m pip install -r requirements.txt
$env:CREWAI_DISABLE_TELEMETRY = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest -q
python scripts\build_index.py
python scripts\run_task13_evaluation.py
python scripts\run_autogen_review_demo.py
python scripts\run_task15_governance_demo.py
python scripts\run_task16_cache_demo.py
git status --short --untracked-files=all
git diff --check
```

Index construction is listed as a fresh-clone reproducibility check; run it
only after confirming the destination is the intended local index, since it
writes persistent Chroma data. Review the final staged file list explicitly,
confirm the dedicated GitHub remote, then commit and push. Do not push the
current parent workspace.

## Recommendation

**Do not publish from the current workspace yet.** Resolve the Git boundary,
ignore/runtime-file selection, README omissions, and path disclosures first.
Once those submission-hygiene items are corrected and the commands above pass
from a dedicated project repository, stop development and submit; no further
feature work is indicated.
