"""Exercise complete snapshot lifecycle with deterministic model substitutes."""

import json
from dataclasses import replace
import pytest
import rag_local as rag
from tests_support import FakeEmbedder, FakeReranker


# Replace only embedding and reranking inference with deterministic substitutes; keep the
# production splitter and persistent Qdrant behavior active for lifecycle tests.
@pytest.fixture
def models(monkeypatch):
    """Replace inference only; use real splitting and persistent Qdrant."""
    monkeypatch.setattr(rag, "LocalEmbedder", FakeEmbedder)
    monkeypatch.setattr(rag, "CrossEncoder", FakeReranker)


# Write the supplied texts as one complete synthetic JSONL snapshot with stable numbered
# document IDs and support-group permissions, returning its path for ingestion.
def source(tmp_path, texts):
    """Write a complete synthetic snapshot."""
    path = tmp_path / "input.jsonl"
    records = [
        {
            "doc_id": str(i),
            "tenant_id": "demo",
            "title": "Policy",
            "text": text,
            "allowed_groups": ["support"],
            "source_uri": f"kb://{i}",
            "updated_at": "2026-11-02",
        }
        for i, text in enumerate(texts)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))

    return path


# Build successive full snapshots with shortened and deleted content, verify the old version
# still serves unchanged results, and reject incompatible configurations or missing
# completion markers.
def test_snapshot_deletion_rollback_and_contract(tmp_path, models):
    """A new full snapshot removes old text without changing the old version."""
    config = rag.Config(
        collection="v1", state_dir=tmp_path, chunk_tokens=30, overlap_tokens=3
    )
    path = source(tmp_path, ["Refund within thirty days.", "Deleted policy."])
    rag.ingest(path, config)
    first = rag.Retriever(config)
    before = first.retrieve("refund", "demo", ["support"])
    assert len(before["selected"]) == 2
    assert first.retrieve("refund", "demo", ["hr"])["selected"] == []
    first.close()
    path = source(tmp_path, ["Refund within ten days."])
    next_config = replace(config, collection="v2")
    rag.ingest(path, next_config)
    second = rag.Retriever(next_config)
    assert [
        hit["text"]
        for hit in second.retrieve("refund", "demo", ["support"])["selected"]
    ] == ["Refund within ten days."]
    second.close()
    reopened = rag.Retriever(config)
    assert (
        reopened.retrieve("refund", "demo", ["support"])["selected"]
        == before["selected"]
    )
    reopened.close()

    with pytest.raises(ValueError, match="contract mismatch"):
        rag.Retriever(replace(config, chunk_tokens=31))

    with pytest.raises(FileNotFoundError):
        rag.Retriever(replace(config, collection="missing"))


# Interrupt a fresh build during upsert and require both an absent manifest and refusal to
# serve the incomplete collection.
def test_failure_never_publishes_completion(tmp_path, models, monkeypatch):
    """A write failure leaves an unservable build."""
    config = rag.Config(collection="failed", state_dir=tmp_path)
    path = source(tmp_path, ["Refund policy."])

    # Raise a simulated write interruption at the store boundary so the test can verify that
    # failed ingestion never publishes a completion manifest.
    def fail(self, chunks, vectors):
        """Simulate an interrupted write."""
        raise RuntimeError("interrupted")

    monkeypatch.setattr(rag.LocalStore, "upsert", fail)

    with pytest.raises(RuntimeError, match="interrupted"):
        rag.ingest(path, config)
    assert not rag.manifest_path(config).exists()

    with pytest.raises(FileNotFoundError):
        rag.Retriever(config)


# Split Unicode, code and table fixtures with deterministic token lengths, checking
# repeatable content-derived IDs, uniqueness and embedding-capacity bounds.
def test_chunk_ids_and_bounds(tmp_path, models):
    """Unicode, code and unspaced content split deterministically within limits."""
    config = rag.Config(
        collection="chunks", state_dir=tmp_path, chunk_tokens=30, overlap_tokens=3
    )
    path = source(
        tmp_path,
        [
            "你好🌍" * 100,
            "```python\nx = 1\n```\n" * 40,
            "| ID | Value |\n| a | b |\n" * 30,
        ],
    )
    documents = rag.read_documents(path)
    embedder = FakeEmbedder(config)
    chunks = rag.build_chunks(documents, config, embedder)
    assert chunks == rag.build_chunks(documents, config, embedder)
    assert all(embedder.token_count(c["text"]) <= embedder.max_tokens for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


# Delete stored points after a successful build and verify that the retriever refuses a
# collection whose count no longer matches its completion manifest.
def test_incomplete_point_count_is_not_servable(tmp_path, models):
    """A modified completed collection fails its manifest count check."""
    config = rag.Config(collection="damaged", state_dir=tmp_path)
    rag.ingest(source(tmp_path, ["Refund policy."]), config)
    store = rag.LocalStore(config)

    try:
        store.client.delete(
            collection_name=config.collection,
            points_selector=rag.models.FilterSelector(filter=rag.models.Filter()),
            wait=True,
        )
    finally:
        store.close()

    with pytest.raises(ValueError, match="no longer matches"):
        rag.Retriever(config)
