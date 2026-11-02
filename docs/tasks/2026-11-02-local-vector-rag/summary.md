# Local vector RAG implementation evidence

Status: local implementation complete. The full roadmap is not complete: reviewed held-out quality acceptance, generated agent answers, and AWS migration remain open.

Completed T-1 through T-5 in the scoped local plan. The original blueprint remains unchanged. This directory was documentation-only and is not a Git repository; all implementation files are new and no existing code was replaced.

## Implemented

- Validated cleaned JSONL and configuration, deterministic tokenizer-aware chunks, CPU embeddings, persistent Qdrant and atomic backend-specific completion manifests.
- Fresh full-snapshot builds, explicit version selection/rollback, deletion/shortening behavior, failed-write isolation and count/contract checks.
- Mandatory tenant/group filtering before reranking, CrossEncoder pair limits, stage timings and selected evidence.
- Paired recall/MRR evaluation with deduplicated document IDs and separate undefined metrics for unanswerable examples.
- Question-only LangChain tool with captured trusted identity, citations and an exact complete-response token budget.
- Local OpenSearch dense and BM25 search, filtered rank fusion, bulk failure handling, shared manifest semantics and pinned development Compose configuration.
- Resolved Python 3.11 dependencies, immutable model revisions, README, architecture, structural code map, fixtures, reports and opt-in integration tests.

## Acceptance evidence

| Criterion | Result | Evidence |
|---|---|---|
| AC-1: input validation before writes | PASS | Duplicate identity and missing permissions tests; Pydantic rejects unknown/missing fields. |
| AC-2: persistence, isolation and lifecycle | PASS | Real embedded Qdrant tests cover retries, reopening, failed writes, damaged counts, contract mismatch, missing manifests, deletion, shortening and rollback. |
| AC-3: metrics, trusted tool and budget | PASS for implemented tool | Tests verify document deduplication, undefined empty-label metrics, paired aggregation, captured identity, schema and full serialized evidence budget. Generated answers are unverified. |
| AC-4: OpenSearch contracts | PASS locally | Request tests plus real OpenSearch 3.3.2 tests verify identical-vector isolation, dense/hybrid filters, persistence, idempotent writes and fresh-version protection. |
| Real CPU path | PASS | Pinned models ingested four records; integration checks verified refund ranking, real tokenizer bounds for Unicode/code/tables/unspaced text, ACL boundaries and reopen equivalence. |
| Held-out quality and latency gates | NOT ESTABLISHED | Supplied toy set has only two answerable and one unanswerable question. Human-reviewed corpus and latency budget are still needed. |
| Agent-generated answers | UNRUN | No generation provider was selected; only the retrieval tool was implemented and tested. |
| S3, Bedrock, Glue and AWS OpenSearch | DEFERRED | Original plan describes these as later migration work; no AWS adapters/deployment or cloud calls were performed. |

## Checks executed

Working directory: project root.

| Check | Result |
|---|---|
| `RAG_MODEL_TESTS=1 RAG_OPENSEARCH_URL=http://127.0.0.1:19200 .venv/bin/python -m pytest -q` | 13 passed, including real models and real local engine. |
| `.venv/bin/python -m pytest tests/test_evaluate.py -q` after adding evaluator aggregation coverage | 1 passed. Total 14 distinct passing tests across these runs. |
| `.venv/bin/python -m compileall -q rag_local.py validation.py opensearch_store.py evaluate.py agent_tool.py metrics.py` | Passed. |
| `uv pip check --python .venv/bin/python` | All 81 installed packages compatible. |
| `docker compose config --quiet` | Passed. |
| Real Qdrant CLI ingestion/query and `evaluate.py` | Passed; four documents/chunks, refund passage ranked first, zero observed access violations. |
| Real OpenSearch CLI ingestion and hybrid `evaluate.py` | Passed; four documents/chunks, zero observed access violations. |
| Final source/artifact inspection | Completed; source compiled, code map generated from AST and reports inspected. |

Reports: `reports/toy-evaluation.json` and `reports/opensearch-hybrid-evaluation.json`. Both measured candidate recall, final recall and MRR of 1.0 on the two answerable toy questions. Reranking did not improve these already-perfect toy rankings. Unanswerable queries still return nearest evidence: no calibrated abstention is claimed. Timings are local warm measurements, not performance acceptance evidence. The newer hybrid report includes model/configuration and startup timing metadata.

The isolated validation container was stopped and removed after testing. Its test indexes were disposable. The downloaded image remains cached. No unrelated containers or browser sessions were changed. Qdrant artifacts remain under ignored `.state`; `knowledge_pinned_v1` is the completed reusable local collection. No publishing, Git operations, remote uploads or deployment occurred.

## Deviations and next steps

Validation was separated into `validation.py`, and OpenSearch/evaluation/tool responsibilities were extracted rather than mechanically preserving the one-file blueprint or introducing the later full `app/` refactor. Model construction stays in object initialization, with imports at module scope to follow Python conventions. Manifests include backend identity to prevent local/OpenSearch collisions. Complete tool output is budgeted, including status and citation metadata. Dependencies and model revisions were resolved and actually exercised on macOS ARM/Python 3.11.

Next: obtain a human-reviewed 30–50-question tuning/held-out corpus, agree on a latency budget and evaluate the quality gates. Select a generator and test citation/abstention/prompt-injection behavior if answers are needed. Then implement the separately described S3/Bedrock/Glue migration, replay controls, deployment-specific authentication and promotion. Equal dimensions do not make models compatible; re-embed and reevaluate when moving to Titan. No cloud implementation or production readiness is claimed.
