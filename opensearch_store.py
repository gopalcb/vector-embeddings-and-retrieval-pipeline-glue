"""Local OpenSearch storage with filtered dense and reciprocal-rank hybrid search."""

from urllib.parse import urlparse
from opensearchpy import OpenSearch, helpers


class OpenSearchStore:
    """Own a local client and one immutable versioned index."""

    # Open the client for one versioned index, rejecting non-loopback hosts and embedded
    # credentials because this adapter is intended for unauthenticated local development.
    def __init__(self, config) -> None:
        """Restrict this unauthenticated adapter to loopback development."""
        url = urlparse(config.opensearch_url)

        if (
            url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
        ):
            raise ValueError(
                "Local OpenSearch requires a loopback URL without credentials"
            )
        self.collection = config.collection
        self.client = OpenSearch(hosts=[config.opensearch_url])

    # Create a fresh index with explicit vector and metadata mappings, Lucene HNSW cosine
    # search and strict payload fields. Refuse an existing index to preserve previous builds.
    def create_empty(self, dimension: int) -> None:
        """Create a fresh index; never erase an existing build."""

        if self.client.indices.exists(index=self.collection):
            raise ValueError("Index exists; choose a fresh version name")
        properties = {
            name: {"type": "keyword"}
            for name in [
                "doc_id",
                "tenant_id",
                "allowed_groups",
                "source_uri",
                "chunk_id",
                "updated_at",
                "content_sha256",
            ]
        }
        properties.update(
            {
                "text": {"type": "text"},
                "title": {"type": "text"},
                "position": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
            }
        )
        self.client.indices.create(
            index=self.collection,
            body={
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
                "mappings": {"dynamic": "strict", "properties": properties},
            },
        )

    # Bulk-index chunks under deterministic IDs and fail on item or transport errors.
    # Refresh successful writes so subsequent count and retrieval checks see the indexed points.
    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """Fail on item errors so incomplete writes cannot publish a manifest."""
        actions = [
            {
                "_op_type": "index",
                "_index": self.collection,
                "_id": chunk["chunk_id"],
                "_source": {**chunk, "embedding": vector},
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        helpers.bulk(
            self.client,
            actions,
            chunk_size=100,
            raise_on_error=True,
            raise_on_exception=True,
            request_timeout=60,
        )
        self.client.indices.refresh(index=self.collection)

    # Return the indexed document count used to verify snapshot completion and detect
    # disagreement with a serving manifest.
    def count(self) -> int:
        """Return the number of searchable chunks."""

        return self.client.count(index=self.collection)["count"]

    # Issue a vector query with tenant and group predicates inside k-NN filtering, then
    # return payloads and scores without embedding arrays. Missing identity fails closed.
    def search(
        self, vector: list[float], tenant_id: str, groups: list[str], k: int
    ) -> list[dict]:
        """Apply access restrictions inside approximate nearest neighbor retrieval."""

        if not tenant_id or not groups:

            return []
        response = self.client.search(
            index=self.collection,
            body={
                "size": k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": vector,
                            "k": k,
                            "filter": {
                                "bool": {"filter": access_conditions(tenant_id, groups)}
                            },
                        }
                    }
                },
            },
        )

        return [
            {**hit["_source"], "vector_score": float(hit["_score"])}
            for hit in response["hits"]["hits"]
        ]

    # Run dense retrieval and a separately access-filtered BM25 text query for the same
    # principal. Fuse chunk ranks into a bounded candidate list before downstream reranking.
    def hybrid_search(
        self,
        question: str,
        vector: list[float],
        tenant_id: str,
        groups: list[str],
        k: int,
    ) -> list[dict]:
        """Fuse independently permission-filtered lexical and dense ranks."""

        if not tenant_id or not groups:

            return []
        dense = self.search(vector, tenant_id, groups, k)
        response = self.client.search(
            index=self.collection,
            body={
                "size": k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "filter": access_conditions(tenant_id, groups),
                        "must": [{"match": {"text": question}}],
                    }
                },
            },
        )
        lexical = [
            {**hit["_source"], "bm25_score": float(hit["_score"])}
            for hit in response["hits"]["hits"]
        ]

        return fuse_ranks([dense, lexical], k)

    # Release the OpenSearch transport connections when ingestion or a retrieval session finishes.
    def close(self) -> None:
        """Release the transport."""
        self.client.close()


# Build the shared mandatory tenant and group predicates so lexical and vector retrieval
# enforce the same authorization boundary.
def access_conditions(tenant_id: str, groups: list[str]) -> list[dict]:
    """Use identical mandatory access predicates for both query branches."""

    return [{"term": {"tenant_id": tenant_id}}, {"terms": {"allowed_groups": groups}}]


# Merge candidate lists by chunk ID using reciprocal rank fusion with offset 60. Count each
# chunk once per ranking and select the highest fused scores without averaging incompatible raw scores.
def fuse_ranks(rankings: list[list[dict]], k: int) -> list[dict]:
    """Combine ranks without comparing incompatible raw score scales."""
    fused = {}

    for ranking in rankings:
        seen = set()

        for rank, hit in enumerate(ranking, 1):
            identity = hit["chunk_id"]

            if identity in seen:
                continue
            seen.add(identity)
            # A chunk may earn credit once in each branch, never twice in one branch.
            entry = fused.setdefault(identity, {**hit, "fusion_score": 0.0})
            entry["fusion_score"] += 1 / (60 + rank)

    return sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)[
        :k
    ]
