"""Document-level retrieval metrics with explicit undefined recall."""


# Collapse multiple chunks from the same document into its earliest appearance, preserving
# retrieval order for document-level relevance metrics.
def unique_doc_ids(hits: list[dict]) -> list[str]:
    """Deduplicate documents while preserving their earliest rank."""

    return list(dict.fromkeys(hit["doc_id"] for hit in hits))


# Compute relevant-document coverage and the reciprocal rank of the first relevant document
# after deduplication. Return undefined metrics when there are no relevance labels.
def recall_and_mrr(hits: list[dict], relevant: list[str]) -> dict:
    """Measure recall and reciprocal rank; empty labels have undefined metrics."""
    ordered = unique_doc_ids(hits)
    relevant = set(relevant)

    if not relevant:

        return {"recall": None, "mrr": None}
    recall = len(set(ordered) & relevant) / len(relevant)
    mrr = next(
        (1.0 / rank for rank, doc_id in enumerate(ordered, 1) if doc_id in relevant),
        0.0,
    )

    return {"recall": recall, "mrr": mrr}
