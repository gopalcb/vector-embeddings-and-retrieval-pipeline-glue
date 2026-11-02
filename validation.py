"""Validated input and index configuration for local retrieval."""

from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CleanDocument(BaseModel):
    """Require explicit provenance and access groups for every source document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    allowed_groups: list[str] = Field(min_length=1)

    # Reject required document fields containing only whitespace, while preserving the
    # original accepted text and identifiers for provenance.
    @field_validator("doc_id", "tenant_id", "title", "text", "source_uri", "updated_at")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Reject whitespace-only source values."""

        if not value.strip():
            raise ValueError("Field must contain non-whitespace text")

        return value

    # Require each supplied access-group name to contain text and return a sorted,
    # deduplicated list so permission metadata is consistent across ingestion runs.
    @field_validator("allowed_groups")
    @classmethod
    def validate_groups(cls, groups: list[str]) -> list[str]:
        """Reject blank permissions and deduplicate group names."""

        if any(not group.strip() for group in groups):
            raise ValueError("Access group names must not be blank")

        return sorted(set(groups))


@dataclass(frozen=True)
class Config:
    """Keep model, chunking, storage and retrieval settings in one validated value."""

    collection: str
    state_dir: Path = Path(".state")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    # Set immutable Hugging Face commit hashes after initial exploration.
    embedding_revision: str | None = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    reranker_revision: str | None = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    chunk_tokens: int = 200
    overlap_tokens: int = 30
    embedding_batch: int = 32
    retrieval_k: int = 20
    final_n: int = 5
    backend: str = "qdrant"
    opensearch_url: str = "http://localhost:9200"
    hybrid: bool = False

    # Reject unsupported storage choices, invalid collection names and inconsistent chunking
    # or retrieval limits before pipeline resources are opened.
    def __post_init__(self) -> None:
        """Validate storage selection and numeric pipeline limits."""

        if self.backend not in {"qdrant", "opensearch"}:
            raise ValueError("Unknown storage backend")

        if self.hybrid and self.backend != "opensearch":
            raise ValueError("Hybrid search requires OpenSearch")

        if not self.collection or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for char in self.collection
        ):
            raise ValueError("Use a lowercase collection name with digits, _ or -")

        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Overlap must be smaller than chunk size")

        if not 1 <= self.final_n <= self.retrieval_k:
            raise ValueError("Require 1 <= final_n <= retrieval_k")

        if self.embedding_batch < 1:
            raise ValueError("Embedding batch size must be positive")


# Parse and validate the full JSONL snapshot before indexing begins. Reject malformed
# records, duplicate tenant/document identities and empty input, adding line numbers to
# record-validation errors.
def read_documents(path: Path) -> list[CleanDocument]:
    """Validate the complete small input before modifying the vector store."""
    documents = []
    seen = set()

    with path.open(encoding="utf-8") as handle:

        for line_number, line in enumerate(handle, start=1):

            if not line.strip():
                continue

            try:
                document = CleanDocument.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid record on line {line_number}") from exc
            identity = (document.tenant_id, document.doc_id)

            if identity in seen:
                raise ValueError(f"Duplicate document identity: {identity}")
            seen.add(identity)
            documents.append(document)

    if not documents:
        raise ValueError("No cleaned documents found")

    return documents


# Reject blank questions and questions exceeding the baseline token allowance before
# embedding, leaving capacity for paired reranker inputs.
def validate_question(question: str, token_count: int) -> None:
    """Reject blank or oversized questions before running inference."""

    if not question.strip():
        raise ValueError("Question must not be blank")

    if token_count > 64:
        raise ValueError("Local baseline accepts questions up to 64 model tokens")
