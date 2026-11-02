"""Opt-in CPU integration tests requiring the pinned downloaded model weights."""

import os
from pathlib import Path
import pytest
from rag_local import (
    Config,
    LocalEmbedder,
    Retriever,
    build_chunks,
    ingest,
    read_documents,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_MODEL_TESTS") != "1",
    reason="Set RAG_MODEL_TESTS=1 to run pinned CPU models",
)


# Exercise pinned CPU models on difficult chunk formats and the policy fixtures, checking
# real token limits, relevance ordering, authorization, question limits and persistence
# across reopening.
def test_real_tokenizers_and_retrieval(tmp_path):
    """Verify real splitting, paired limits, ranking, persistence and ACL boundaries."""
    config = Config(collection="real_v1", state_dir=tmp_path)
    embedder = LocalEmbedder(config)
    documents = read_documents(Path("data/cleaned/documents.jsonl"))

    for text in [
        "你好🌍" * 400,
        "identifier_" * 500,
        "```python\nx = 1\n```\n" * 200,
        "| ID | Value |\n| a | b |\n" * 200,
    ]:
        documents.append(
            documents[0].model_copy(
                update={"doc_id": str(len(documents)), "text": text}
            )
        )
    chunks = build_chunks(documents, config, embedder)
    assert chunks == build_chunks(documents, config, embedder)
    assert all(
        embedder.token_count(hit["text"]) <= embedder.max_tokens for hit in chunks
    )
    ingest(Path("data/cleaned/documents.jsonl"), config)
    retriever = Retriever(config)

    try:

        for hit in chunks:
            tokens = retriever.reranker.tokenizer.encode(
                "What is the policy?",
                hit["text"],
                add_special_tokens=True,
                truncation=False,
            )
            assert len(tokens) <= retriever.reranker.max_seq_length
        result = retriever.retrieve("What is the refund deadline?", "demo", ["support"])
        assert result["selected"][0]["doc_id"] == "refund-policy"
        assert all(
            hit["tenant_id"] == "demo" and "support" in hit["allowed_groups"]
            for hit in result["candidates"] + result["selected"]
        )
        assert retriever.retrieve("refund", "demo", ["unknown"])["selected"] == []

        with pytest.raises(ValueError, match="64 model tokens"):
            retriever.retrieve("refund " * 100, "demo", ["support"])
    finally:
        retriever.close()
    reopened = Retriever(config)

    try:
        assert (
            reopened.retrieve("What is the refund deadline?", "demo", ["support"])[
                "selected"
            ]
            == result["selected"]
        )
    finally:
        reopened.close()
