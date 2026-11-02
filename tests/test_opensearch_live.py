"""Opt-in integration contracts against an isolated local OpenSearch instance."""

import os
import uuid
import pytest
from opensearch_store import OpenSearchStore
from validation import Config

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAG_OPENSEARCH_URL"),
    reason="Set RAG_OPENSEARCH_URL to an isolated local engine",
)


# Use an isolated real OpenSearch index to verify deterministic upserts, fresh-version
# protection and tenant/group isolation for identical vectors in dense and hybrid retrieval;
# delete only the test index afterward.
def test_live_engine_access_persistence_and_hybrid(tmp_path):
    """Identical vectors cannot bypass filters in dense or hybrid search."""
    config = Config(
        collection=f"test_{uuid.uuid4().hex}",
        state_dir=tmp_path,
        backend="opensearch",
        opensearch_url=os.environ.get("RAG_OPENSEARCH_URL"),
    )
    store = OpenSearchStore(config)
    chunks = []

    for tenant, group, text in [
        ("demo", "support", "Refunds within thirty days."),
        ("demo", "hr", "Payroll within seven days."),
        ("other", "support", "Refunds within ninety days."),
    ]:
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": str(len(chunks)),
                "tenant_id": tenant,
                "allowed_groups": [group],
                "text": text,
                "title": "Policy",
                "source_uri": "kb://test",
            }
        )

    try:
        store.create_empty(3)
        store.upsert(chunks, [[1.0, 0.0, 0.0]] * 3)
        store.upsert(chunks, [[1.0, 0.0, 0.0]] * 3)
        assert store.count() == 3

        with pytest.raises(ValueError, match="fresh version"):
            store.create_empty(3)
    finally:
        store.close()
    reopened = OpenSearchStore(config)

    try:

        for hits in [
            reopened.search([1.0, 0.0, 0.0], "demo", ["support"], 20),
            reopened.hybrid_search("Refunds", [1.0, 0.0, 0.0], "demo", ["support"], 20),
        ]:
            assert [hit["doc_id"] for hit in hits] == ["0"]
        assert reopened.search([1.0, 0.0, 0.0], "demo", ["unknown"], 20) == []
    finally:
        reopened.client.indices.delete(index=config.collection)
        reopened.close()
