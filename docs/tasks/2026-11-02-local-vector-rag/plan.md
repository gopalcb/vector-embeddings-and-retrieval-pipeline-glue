# Local vector RAG implementation

Implement the repository blueprint in ../../../local-vector-rag-implementation-plan.md.

Scope: local milestones 1–7. AWS migration remains a future stage; no cloud provisioning, publishing or paid generation calls.

- T-1: Materialize baseline and fixtures, centralize validation; AC-1: invalid records fail before writes.
- T-2: Versioned ingestion and filtered retrieval; AC-2: persistence, isolation, failure, deletion and contract tests pass.
- T-3: Evaluator and trusted-identity agent tool; AC-3: metrics deduplicate documents and exact tool output respects token budget.
- T-4: Selectable local OpenSearch adapter and hybrid search; AC-4: mapping/query/filter and failure contracts pass focused tests. Live engine verification requires a running local instance.
- T-5: Resolve dependencies, run tests and model-backed CLI/evaluation if downloads work; document evidence and limitations.

Use the blueprint directly, retaining fresh collection builds and explicit serving names. Keep model initialization outside query timings. Toy evaluation cannot establish held-out quality. No LLM is needed. The project has no existing Git root or source to preserve.
