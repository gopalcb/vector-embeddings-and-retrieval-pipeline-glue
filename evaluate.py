"""Compare dense retrieval and reranking on labelled questions; no generator required."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from metrics import recall_and_mrr
from rag_local import Config, Retriever


# Aggregate recall and reciprocal rank over answerable questions only, count authorization
# violations and report warm stage p50/p95 latency. Keep unanswerable examples visible
# without assigning them artificial recall.
def summarize(samples: list[dict]) -> dict:
    """Report document metrics, authorization violations and warm stage latency."""
    answerable = [sample for sample in samples if sample["final"]["recall"] is not None]
    metrics = {}

    for field, stage, metric in [
        ("candidate_recall", "candidate", "recall"),
        ("final_recall", "final", "recall"),
        ("mrr", "final", "mrr"),
    ]:
        metrics[field] = (
            statistics.mean(s[stage][metric] for s in answerable)
            if answerable
            else None
        )
    timings = {}

    for stage in ["embedding", "search", "rerank_and_select"]:
        values = sorted(sample["timings_ms"][stage] for sample in samples)
        timings[stage] = (
            {
                "p50": statistics.median(values),
                "p95": values[math.ceil(len(values) * 0.95) - 1],
            }
            if values
            else None
        )

    return {
        **metrics,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(samples) - len(answerable),
        "authorization_violations": sum(s["authorization_violations"] for s in samples),
        "warm_timings_ms": timings,
        "queries": samples,
    }


# Warm the retriever once and run the same labelled questions with and without reranking.
# Save per-question evidence metrics and access checks, then aggregate each mode for
# comparison.
def evaluate(retriever, path: Path) -> dict:
    """Evaluate both rankings, keeping unanswerable examples separate from recall."""
    questions = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    if not questions:
        raise ValueError("Evaluation set must not be empty")
    first = questions[0]
    # Exclude lazy model/backend startup from the per-question warm timings below.
    retriever.retrieve(first["question"], first["tenant_id"], first["groups"])
    reports = {}

    for rerank in [False, True]:
        samples = []

        for query in questions:
            result = retriever.retrieve(
                query["question"], query["tenant_id"], query["groups"], rerank=rerank
            )
            violations = sum(
                hit["tenant_id"] != query["tenant_id"]
                or not set(hit["allowed_groups"]) & set(query["groups"])
                for hit in result["candidates"] + result["selected"]
            )
            samples.append(
                {
                    "question": query["question"],
                    "candidate": recall_and_mrr(
                        result["candidates"], query["relevant_doc_ids"]
                    ),
                    "final": recall_and_mrr(
                        result["selected"], query["relevant_doc_ids"]
                    ),
                    "selected_doc_ids": [hit["doc_id"] for hit in result["selected"]],
                    "authorization_violations": violations,
                    "timings_ms": result["timings_ms"],
                }
            )
        mode = (
            "reranked"
            if rerank
            else ("hybrid_only" if retriever.config.hybrid else "vector_only")
        )
        reports[mode] = summarize(samples)

    return {
        "note": "Toy fixture is a smoke check, not held-out quality evidence. No abstention threshold is calibrated.",
        "reports": reports,
    }


# Load the selected completed collection, measure retriever startup separately from warm
# inference, run paired evaluation and save a JSON report with configuration metadata.
# Always close the retriever.
def main() -> None:
    """Evaluate an explicitly selected completed collection and save its report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--state-dir", type=Path, default=Path(".state"))
    parser.add_argument("--input", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["qdrant", "opensearch"], default="qdrant")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--opensearch-url", default="http://localhost:9200")
    parser.add_argument("--embedding-revision", default=Config.embedding_revision)
    parser.add_argument("--reranker-revision", default=Config.reranker_revision)
    args = parser.parse_args()
    started = time.perf_counter()
    retriever = Retriever(
        Config(
            collection=args.collection,
            state_dir=args.state_dir,
            backend=args.backend,
            hybrid=args.hybrid,
            opensearch_url=args.opensearch_url,
            embedding_revision=args.embedding_revision,
            reranker_revision=args.reranker_revision,
        )
    )
    startup_ms = round((time.perf_counter() - started) * 1000, 2)

    try:
        output = evaluate(retriever, args.input)
        output["startup_ms"] = startup_ms
        output["configuration"] = {
            key: getattr(retriever.config, key)
            for key in [
                "collection",
                "backend",
                "embedding_model",
                "embedding_revision",
                "reranker_model",
                "reranker_revision",
                "retrieval_k",
                "final_n",
                "hybrid",
            ]
        }
    finally:
        retriever.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
