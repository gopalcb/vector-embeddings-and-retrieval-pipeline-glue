"""Ingest complete snapshots and retrieve authorized, reranked local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from validation import CleanDocument, Config, read_documents, validate_question
from opensearch_store import OpenSearchStore


class LocalEmbedder:
    """Own the CPU embedding model and its tokenizer-based input limits."""

    # Load the configured embedding model on CPU and retain its tokenizer, vector dimension,
    # token limit and batch size for subsequent document and query encoding.
    def __init__(self, config: Config) -> None:
        """Load the pinned CPU embedding model once per process."""
        self.model = SentenceTransformer(
            config.embedding_model,
            revision=config.embedding_revision,
            device="cpu",
        )
        self.tokenizer = self.model.tokenizer
        self.dimension = int(self.model.get_embedding_dimension())
        self.max_tokens = int(self.model.max_seq_length)
        self.batch_size = config.embedding_batch

    # Measure the complete embedding input, including special tokens, so callers can check
    # model limits before content is silently truncated.
    def token_count(self, text: str) -> int:
        """Count tokens including special tokens."""

        return len(self.tokenizer.encode(text, add_special_tokens=True))

    # Convert a batch of texts into normalized embedding vectors. Reject overlong inputs and
    # return ordinary Python lists suitable for either storage adapter.
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Normalize embeddings and reject oversized inputs."""

        if not texts:

            return []

        # Fail rather than silently losing document content to model truncation.
        if any(self.token_count(text) > self.max_tokens for text in texts):
            raise ValueError("Embedding input exceeds the model token limit")
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return vectors.tolist()


# Split each validated document with the embedding tokenizer and preserve its access
# metadata and source references. Derive repeatable chunk IDs from tenant, document,
# position and text so repeated input has stable identities.
def build_chunks(
    documents: list[CleanDocument], config: Config, embedder: LocalEmbedder
) -> list[dict[str, Any]]:
    """Split documents with their embedding tokenizer and retain stable provenance."""

    # Reserve the two special tokens added by the pinned embedding tokenizer.
    if config.chunk_tokens + 2 > embedder.max_tokens:
        raise ValueError("Chunk budget exceeds the embedding model capacity")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedder.tokenizer,
        chunk_size=config.chunk_tokens,
        chunk_overlap=config.overlap_tokens,
    )
    chunks = []

    for document in documents:

        for position, text in enumerate(splitter.split_text(document.text)):

            if not text.strip():
                continue

            if embedder.token_count(text) > embedder.max_tokens:
                raise ValueError("Splitter produced an oversized chunk")
            # IDs are deterministic within this ingestion contract. Including
            # tenant prevents identical source IDs from colliding across tenants.
            identity = json.dumps(
                [document.tenant_id, document.doc_id, position, text],
                ensure_ascii=False,
            )
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
            payload = document.model_dump(exclude={"text"})
            chunks.append(
                {
                    **payload,
                    "chunk_id": chunk_id,
                    "position": position,
                    "text": text,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )

    if not chunks:
        raise ValueError("No indexable chunks were produced")

    return chunks


class LocalStore:
    """Own one versioned collection and the embedded Qdrant directory lock."""

    # Create the local state directory if needed and open persistent Qdrant storage for the
    # selected collection. The client owns the embedded directory lock until close is
    # called.
    def __init__(self, config: Config) -> None:
        """Open one persistent embedded store; one process owns its directory."""
        config.state_dir.mkdir(parents=True, exist_ok=True)
        self.collection = config.collection
        self.client = QdrantClient(path=str(config.state_dir / "qdrant"))

    # Create a new cosine-vector collection with the embedding dimension. Refuse an existing
    # name so ingestion cannot overwrite a completed version or reuse a partial build.
    def create_empty(self, dimension: int) -> None:
        # Never erase an existing collection as a side effect of ingestion.
        """Create a fresh collection without overwriting an older version."""

        if self.client.collection_exists(self.collection):
            raise ValueError("Collection exists; choose a fresh version name")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    # Pair every chunk with exactly one vector, write points under deterministic chunk IDs
    # and wait for completion. A repeated write replaces the same points rather than adding
    # duplicates.
    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """Write deterministic points and wait for completion."""
        points = [
            models.PointStruct(id=chunk["chunk_id"], vector=vector, payload=chunk)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    # Read the exact stored point count for ingestion completion checks and for verifying
    # that a collection still matches its published manifest.
    def count(self) -> int:
        """Count persisted points exactly."""

        return self.client.count(
            collection_name=self.collection,
            exact=True,
        ).count

    # Retrieve nearest neighbors only inside the requested tenant and at least one trusted
    # access group. Missing identity returns no hits; results include readable payloads and
    # vector scores.
    def search(
        self, vector: list[float], tenant_id: str, groups: list[str], k: int
    ) -> list[dict]:
        """Find candidates within mandatory tenant and group restrictions."""

        if not tenant_id or not groups:

            return []  # Fail closed for missing identity information.
        access_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="allowed_groups",
                    match=models.MatchAny(any=groups),
                ),
            ]
        )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=access_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        ).points

        return [
            {**(hit.payload or {}), "vector_score": float(hit.score)} for hit in hits
        ]

    # Release the embedded client and its directory lock so another CLI command or retriever
    # can safely open the same local storage.
    def close(self) -> None:
        """Release the store lock and client resources."""
        self.client.close()


# Describe the embedding space and chunk transformation that produced an index. Readers
# compare this contract to reject incompatible models or chunk settings while permitting
# query-time ranking changes.
def index_contract(config: Config, dimension: int) -> dict:
    """Identity of the vector space and chunk transformation, not query knobs."""

    return {
        "schema_version": 1,
        "backend": config.backend,
        "embedding_model": config.embedding_model,
        "embedding_revision": config.embedding_revision,
        "dimension": dimension,
        "normalized": True,
        "distance": "cosine",
        "splitter": "recursive_huggingface_tokenizer",
        "chunk_tokens": config.chunk_tokens,
        "overlap_tokens": config.overlap_tokens,
    }


# Choose a completion-marker path scoped by backend and collection, keeping Qdrant and
# OpenSearch builds with the same collection name separate.
def manifest_path(config: Config) -> Path:
    """Locate the backend-specific completion marker."""

    return config.state_dir / "manifests" / f"{config.backend}-{config.collection}.json"


# Validate a complete source snapshot, generate chunks and embeddings, then write a fresh
# collection. Publish its manifest atomically only after the persisted chunk count matches;
# always close the storage client.
def ingest(path: Path, config: Config) -> dict:
    """Publish completion only after a full snapshot is written and counted."""
    # Complete validation and chunking before creating any persistent collection.
    documents = read_documents(path)
    embedder = LocalEmbedder(config)
    chunks = build_chunks(documents, config, embedder)
    store = (
        OpenSearchStore(config)
        if config.backend == "opensearch"
        else LocalStore(config)
    )

    try:
        store.create_empty(embedder.dimension)

        for start in range(0, len(chunks), config.embedding_batch):
            batch = chunks[start : start + config.embedding_batch]
            vectors = embedder.encode([chunk["text"] for chunk in batch])
            store.upsert(batch, vectors)

        if store.count() != len(chunks):
            raise RuntimeError("Indexed point count does not match chunk count")
        manifest = {
            "contract": index_contract(config, embedder.dimension),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        target = manifest_path(config)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Only publish completion after all points have been written and counted.
        temporary.replace(target)

        return manifest
    finally:
        store.close()


class Retriever:
    """Serve a verified snapshot with reusable embedding and reranking models."""

    # Load the query and reranking models, require a matching completion manifest and check
    # the stored chunk count. Close an opened store if its completion check fails.
    def __init__(self, config: Config) -> None:
        """Open only a completed collection matching the requested embedding contract."""
        self.config = config
        self.embedder = LocalEmbedder(config)
        completed = json.loads(manifest_path(config).read_text(encoding="utf-8"))

        if completed["contract"] != index_contract(config, self.embedder.dimension):
            raise ValueError("Index contract mismatch; rebuild a new collection")
        self.reranker = CrossEncoder(
            config.reranker_model,
            revision=config.reranker_revision,
            device="cpu",
            max_length=512,
        )
        self.store = (
            OpenSearchStore(config)
            if config.backend == "opensearch"
            else LocalStore(config)
        )

        try:

            if self.store.count() != completed["chunk_count"]:
                raise ValueError("Collection no longer matches its build manifest")
        except Exception:
            self.store.close()
            raise

    # Validate and embed the question, retrieve authorized dense or hybrid candidates,
    # optionally score question-passage pairs with the reranker, and return selected
    # evidence with stage timings.
    def retrieve(
        self, question: str, tenant_id: str, groups: list[str], rerank: bool = True
    ) -> dict:
        """Retrieve authorized evidence and optionally rerank it with paired token checks."""
        validate_question(question, self.embedder.token_count(question))
        started = time.perf_counter()
        vector = self.embedder.encode([question])[0]
        embedded = time.perf_counter()

        if self.config.hybrid:
            candidates = self.store.hybrid_search(
                question, vector, tenant_id, groups, self.config.retrieval_k
            )
        else:
            candidates = self.store.search(
                vector, tenant_id, groups, self.config.retrieval_k
            )
        searched = time.perf_counter()
        ordered = candidates

        if candidates and rerank:
            # Score only the already authorized candidates; reranking cannot widen access.
            pairs = [(question, item["text"]) for item in candidates]

            # Check the actual paired tokenizer input; different models tokenize
            # differently, so the embedding chunk limit alone is insufficient.
            for query, passage in pairs:
                ids = self.reranker.tokenizer.encode(
                    query,
                    passage,
                    add_special_tokens=True,
                    truncation=False,
                )

                if len(ids) > self.reranker.max_seq_length:
                    raise ValueError("Question–chunk pair exceeds reranker capacity")
            scores = self.reranker.predict(pairs, batch_size=16)
            ordered = sorted(
                [
                    {**item, "rerank_score": float(score)}
                    for item, score in zip(candidates, scores, strict=True)
                ],
                key=lambda item: item["rerank_score"],
                reverse=True,
            )
        selected = ordered[: self.config.final_n]
        finished = time.perf_counter()

        # Full-text chunks remain in the tool result for transparent inspection.
        # The agent adapter applies its own generator-specific context budget.
        return {
            "question": question,
            "status": "evidence_candidates" if selected else "no_authorized_hits",
            "candidates": candidates,
            "selected": selected,
            "timings_ms": {
                "embedding": round((embedded - started) * 1000, 2),
                "search": round((searched - embedded) * 1000, 2),
                "rerank_and_select": round((finished - searched) * 1000, 2),
            },
        }

    # Close the owned storage adapter at session shutdown or after a CLI query, releasing
    # local locks or network connections.
    def close(self) -> None:
        """Release the store lock and client resources."""
        self.store.close()


# Translate CLI arguments into validated pipeline configuration, run ingestion or querying,
# print the JSON result and guarantee retriever cleanup after queries.
def main() -> None:
    """Parse CLI input and run ingestion or retrieval with resource cleanup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "query"])
    parser.add_argument("--collection", required=True)
    parser.add_argument("--state-dir", type=Path, default=Path(".state"))
    parser.add_argument("--backend", choices=["qdrant", "opensearch"], default="qdrant")
    parser.add_argument("--opensearch-url", default="http://localhost:9200")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--embedding-revision", default=Config.embedding_revision)
    parser.add_argument("--reranker-revision", default=Config.reranker_revision)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--tenant")
    parser.add_argument("--groups", nargs="+", default=[])
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    config = Config(
        collection=args.collection,
        state_dir=args.state_dir,
        backend=args.backend,
        opensearch_url=args.opensearch_url,
        hybrid=args.hybrid,
        embedding_revision=args.embedding_revision,
        reranker_revision=args.reranker_revision,
    )

    if args.command == "ingest":

        if args.input is None:
            parser.error("ingest requires --input")
        output = ingest(args.input, config)
    else:

        if not args.question or not args.tenant or not args.groups:
            parser.error("query requires --question, --tenant and --groups")
        retriever = Retriever(config)

        try:
            output = retriever.retrieve(
                args.question,
                args.tenant,
                args.groups,
                rerank=not args.no_rerank,
            )
        finally:
            retriever.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
