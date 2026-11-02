"""Check paired evaluation aggregation without any model downloads."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from evaluate import evaluate


# Evaluate paired synthetic questions in hybrid and reranked modes, checking that
# unanswerable labels stay out of relevance averages and both modes retain the same query
# set.
def test_evaluation_excludes_unanswerable_and_compares_modes(tmp_path):
    """Empty labels stay undefined and both modes use the same trusted queries."""
    path = tmp_path / "questions.jsonl"
    queries = [
        {
            "question": "refund?",
            "tenant_id": "demo",
            "groups": ["support"],
            "relevant_doc_ids": ["refund"],
        },
        {
            "question": "lunar?",
            "tenant_id": "demo",
            "groups": ["support"],
            "relevant_doc_ids": [],
        },
    ]
    path.write_text("\n".join(json.dumps(query) for query in queries))
    hit = {"doc_id": "refund", "tenant_id": "demo", "allowed_groups": ["support"]}
    retriever = MagicMock()
    retriever.config = SimpleNamespace(hybrid=True)
    retriever.retrieve.return_value = {
        "candidates": [hit, hit],
        "selected": [hit],
        "timings_ms": {"embedding": 1, "search": 2, "rerank_and_select": 3},
    }
    output = evaluate(retriever, path)
    assert set(output["reports"]) == {"hybrid_only", "reranked"}

    for report in output["reports"].values():
        assert (
            report["candidate_recall"] == report["final_recall"] == report["mrr"] == 1
        )
        assert report["unanswerable_questions"] == report["answerable_questions"] == 1
        assert report["authorization_violations"] == 0
    assert retriever.retrieve.call_count == 5  # One warmup plus paired query sets.
