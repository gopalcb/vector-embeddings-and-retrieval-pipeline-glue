"""Verify tool boundaries, evaluation metrics and OpenSearch request contracts."""

import json
from unittest.mock import MagicMock
import pytest
from agent_tool import make_retrieval_tool
from metrics import recall_and_mrr
from opensearch_store import OpenSearchStore, fuse_ranks
from validation import Config
from tests_support import FakeTokenizer


# Check that repeated chunks do not inflate document recall or rank and that empty relevance
# labels produce undefined metrics.
def test_metrics_deduplicate_and_exclude_unanswerable():
    """Repeated chunks must not inflate document recall or rank."""
    hits = [{"doc_id": doc_id} for doc_id in ["a", "a", "b"]]
    assert recall_and_mrr(hits, ["b"]) == {"recall": 1, "mrr": 0.5}
    assert recall_and_mrr(hits, []) == {"recall": None, "mrr": None}


# Check the question-only schema, immutable captured group membership and exact serialized-
# response budget, including rejection of a budget too small for an empty response.
def test_tool_identity_and_exact_budget():
    """The model chooses only a question; full serialized results fit the budget."""
    retriever = MagicMock()
    hit = {
        key: "example" for key in ["doc_id", "chunk_id", "title", "source_uri", "text"]
    }
    retriever.retrieve.return_value = {"selected": [hit, {**hit, "text": "x" * 1000}]}
    groups = ["support"]
    tool = make_retrieval_tool(retriever, "demo", groups, FakeTokenizer(), 300)
    groups.append("hr")
    assert set(tool.args) == {"question"}
    output = tool.invoke({"question": "refund"})
    retriever.retrieve.assert_called_once_with("refund", "demo", ["support"])
    assert len(json.dumps(output, ensure_ascii=False)) <= 300
    assert len(output["evidence"]) == 1

    with pytest.raises(ValueError, match="empty tool response"):
        make_retrieval_tool(retriever, "demo", ["support"], FakeTokenizer(), 1)


# Inspect mocked OpenSearch mappings and both retrieval branches for mandatory access
# predicates. Confirm bulk failures propagate and absent groups prevent a search request.
def test_opensearch_filter_mapping_and_failure(monkeypatch):
    """Both search branches carry ACL filters and bulk failures propagate."""
    client = MagicMock()
    monkeypatch.setattr("opensearch_store.OpenSearch", lambda **kwargs: client)
    store = OpenSearchStore(Config(collection="v1", backend="opensearch"))
    client.indices.exists.return_value = False
    store.create_empty(384)
    mapping = client.indices.create.call_args.kwargs["body"]
    assert mapping["mappings"]["properties"]["embedding"]["dimension"] == 384
    client.search.return_value = {"hits": {"hits": []}}
    store.hybrid_search("refund", [1.0] * 384, "demo", ["support"], 20)
    dense, lexical = [call.kwargs["body"] for call in client.search.call_args_list]
    expected = [
        {"term": {"tenant_id": "demo"}},
        {"terms": {"allowed_groups": ["support"]}},
    ]
    assert dense["query"]["knn"]["embedding"]["filter"]["bool"]["filter"] == expected
    assert lexical["query"]["bool"]["filter"] == expected
    monkeypatch.setattr(
        "opensearch_store.helpers.bulk",
        MagicMock(side_effect=RuntimeError("bulk failed")),
    )

    with pytest.raises(RuntimeError, match="bulk failed"):
        store.upsert([], [])
    client.search.reset_mock()
    assert store.search([1.0], "demo", [], 20) == []
    client.search.assert_not_called()
    store.close()


# Verify that a chunk appearing in both search branches receives both rank contributions and
# rises above a chunk present in only one branch.
def test_rank_fusion():
    """A shared result gains rank credit from both branches."""
    rankings = [[{"chunk_id": "a"}, {"chunk_id": "b"}], [{"chunk_id": "b"}]]
    assert [hit["chunk_id"] for hit in fuse_ranks(rankings, 2)] == ["b", "a"]
