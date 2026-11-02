# Architecture

This small CPU pipeline turns complete cleaned JSONL snapshots into authorized evidence. It does not extract binaries or generate answers.

`rag_local.py` owns token-aware chunking, embeddings, persistent Qdrant, ingestion, completion contracts, reranking and CLI orchestration. `validation.py` owns the document/configuration contracts and input checks. `opensearch_store.py` implements the equivalent versioned storage contract, Lucene filtered k-NN, BM25 and rank fusion. `metrics.py` and `evaluate.py` compute document-level recall/MRR and compare retrieval with reranking. `agent_tool.py` binds trusted identity to a question-only tool and budgets the complete response with a caller-supplied generator tokenizer. `tests_support.py` contains deterministic inference substitutes used only by tests.

Ingestion validates the complete snapshot before writes, splits using the embedding tokenizer, creates stable tenant/document/position/text UUIDs, writes normalized vectors to a fresh collection, checks its count and atomically publishes a local manifest. Failure leaves an incomplete build without a serving marker. No automatic collection promotion or in-place deletion occurs. A fresh complete snapshot implements deletion, shortening and access changes while preserving the old version for explicit rollback.

Retrieval loads models once, verifies the manifest against embedding/chunk configuration, verifies the stored count, embeds the question, performs mandatory tenant/group filtering, optionally fuses lexical/dense ranks, checks paired reranker token limits and selects up to five passages. Citations and the evidence budget are applied by the agent adapter. Source text remains untrusted.

`Config` defaults to pinned MiniLM embedding and CrossEncoder revisions, 200-token chunks with 30-token overlap, 20 candidates and 5 results. `--state-dir` controls Qdrant and backend-specific manifest paths. Qdrant embedded mode requires single-process ownership. `compose.yaml` provides a loopback-only development OpenSearch engine; its storage is distinct from Qdrant. Dependencies are resolved for the tested Python 3.11/macOS ARM environment.

Tests separate model-free persistence/lifecycle checks, real CPU tokenizer/inference checks, mocked OpenSearch request contracts and opt-in real engine access/persistence checks. Reports record toy relevance and stage timings. Held-out corpus quality, agent generation and AWS adapters/deployment are outside the validated local baseline; preserve the embedding, ACL and completion contracts when adding them.
