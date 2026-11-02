"""Expose evidence retrieval with application-owned identity and exact output budgeting."""

import json
from langchain_core.tools import tool


# Bind a trusted tenant and a snapshot of its groups into a question-only LangChain tool.
# Validate the evidence budget with the generator tokenizer before returning the closure.
def make_retrieval_tool(
    retriever,
    tenant_id: str,
    groups: list[str],
    generator_tokenizer,
    context_budget: int = 2500,
):
    """Bind one trusted principal; only the question is visible in the tool schema."""

    if (
        not tenant_id.strip()
        or not groups
        or any(not group.strip() for group in groups)
    ):
        raise ValueError("Trusted tenant and groups are required")

    if context_budget < 1:
        raise ValueError("Context budget must be positive")
    # Snapshot membership so later changes to the caller's list cannot widen access.
    trusted_groups = tuple(groups)
    empty = {"status": "no_context", "evidence": []}

    if (
        len(generator_tokenizer.encode(json.dumps(empty, ensure_ascii=False)))
        > context_budget
    ):
        raise ValueError("Budget cannot fit the empty tool response")

    # Retrieve using the captured principal, attach citation labels and source metadata, and
    # include only passages whose complete serialized response fits the generator-token
    # budget.
    @tool
    def search_knowledge(question: str) -> dict:
        """Search authorized knowledge and return untrusted evidence with citations."""
        result = retriever.retrieve(question, tenant_id, list(trusted_groups))
        output = {"status": "no_context", "evidence": []}

        for hit in result["selected"]:
            candidate = {
                key: hit[key]
                for key in ["doc_id", "chunk_id", "title", "source_uri", "text"]
            }
            candidate["citation"] = f"S{len(output['evidence']) + 1}"
            proposed = {
                "status": "evidence_candidates",
                "evidence": output["evidence"] + [candidate],
            }

            # Count metadata and citations too. A skipped large hit does not prevent
            # a later, shorter passage from fitting in the remaining budget.
            if (
                len(
                    generator_tokenizer.encode(json.dumps(proposed, ensure_ascii=False))
                )
                <= context_budget
            ):
                output = proposed

        return output

    return search_knowledge
