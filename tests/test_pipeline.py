"""Verify snapshot validation and persistent tenant/group filtering."""

import json

import pytest

from rag_local import Config, LocalStore, read_documents


# Create a valid cleaned-document fixture with selectable document, tenant and access
# groups. Allow explicit empty groups so validation tests can construct denied inputs.
def record(doc_id="refunds", tenant="demo", groups=None):
    """Build a source record, preserving explicitly empty groups for rejection tests."""

    return {
        "doc_id": doc_id,
        "tenant_id": tenant,
        "title": "Example",
        "text": "Refunds are permitted within thirty days.",
        "source_uri": f"kb://{tenant}/{doc_id}",
        "updated_at": "2026-11-01T12:00:00Z",
        "allowed_groups": ["support"] if groups is None else groups,
    }


# Write two records with the same tenant/document identity and require input validation to
# reject the snapshot before any indexing work starts.
def test_duplicate_document_identity_fails_before_indexing(tmp_path):
    """Reject repeated document identities during complete-snapshot validation."""
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(record()) for _ in range(2)))

    with pytest.raises(ValueError, match="Duplicate"):
        read_documents(path)


# Provide a document with an empty permission list and verify that schema validation rejects
# it rather than indexing unrestricted content.
def test_missing_permissions_are_rejected(tmp_path):
    """Reject source records without explicit access groups."""
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(record(groups=[])))

    with pytest.raises(ValueError, match="Invalid record"):
        read_documents(path)


# Write identical vectors across groups and tenants, retry the same IDs and reopen the
# store. Verify persistence, duplicate-free counts, fail-closed access and protection
# against overwriting a collection.
def test_filtering_and_persistence(tmp_path):
    """Preserve idempotent writes and access filtering after reopening the store."""
    config = Config(collection="test_v1", state_dir=tmp_path)
    store = LocalStore(config)

    try:
        store.create_empty(3)
        chunks = [
            {**record("support"), "chunk_id": "00000000-0000-0000-0000-000000000001"},
            {
                **record("hr", groups=["hr"]),
                "chunk_id": "00000000-0000-0000-0000-000000000002",
            },
            {
                **record("other", tenant="other"),
                "chunk_id": "00000000-0000-0000-0000-000000000003",
            },
        ]
        # Identical vectors ensure access filtering, not semantic separation,
        # is what prevents unauthorized chunks from being returned.
        store.upsert(chunks, [[1.0, 0.0, 0.0]] * 3)
        store.upsert(chunks, [[1.0, 0.0, 0.0]] * 3)
        assert store.count() == 3  # Retry of the same IDs adds no duplicates.
    finally:
        store.close()
    reopened = LocalStore(config)

    try:
        hits = reopened.search([1.0, 0.0, 0.0], "demo", ["support"], 20)
        assert [hit["doc_id"] for hit in hits] == ["support"]
        assert reopened.search([1.0, 0.0, 0.0], "demo", [], 20) == []

        with pytest.raises(ValueError, match="fresh version"):
            reopened.create_empty(3)
    finally:
        reopened.close()
