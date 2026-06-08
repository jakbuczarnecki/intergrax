# © Artur Czarnecki. All rights reserved.

"""Explore delegation runner — isolated child context + synthesis handoff (MEM-DEPTH-4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from intergrax.contracts.delegation import DelegationSpec, ExploreDelegationProfile
from intergrax.runtime.architecture.hybrid_retrieval import (
    ChannelRetrievalHit,
    HybridRetrievalRequest,
    RetrievalChannel,
    execute_hybrid_retrieval,
)


@dataclass(frozen=True, slots=True)
class ExploreFinding:
    source: str
    summary: str
    score: float


@dataclass(frozen=True, slots=True)
class ExploreDelegationResult:
    findings: List[ExploreFinding] = field(default_factory=list)
    synthesis_text: str = ""
    memory_namespace: str = ""


class ExploreDelegationRunner:
    """
    Runs parallel retrieval in an isolated namespace and returns synthesis-only payload.
    """

    def __init__(self, profile: ExploreDelegationProfile | None = None) -> None:
        self._profile = profile or ExploreDelegationProfile()

    def run(
        self,
        spec: DelegationSpec,
        *,
        task_id: str,
        node_id: str,
        vector_hits: Sequence[ChannelRetrievalHit] | None = None,
        keyword_hits: Sequence[ChannelRetrievalHit] | None = None,
        graph_hits: Sequence[ChannelRetrievalHit] | None = None,
    ) -> ExploreDelegationResult:
        profile = spec.explore or self._profile
        namespace = spec.resolved_memory_namespace(task_id=task_id, node_id=node_id)

        merged_ids: list[str] = []
        if profile.enable_hybrid_retrieval:
            hybrid = execute_hybrid_retrieval(
                HybridRetrievalRequest(
                    query_id=f"{task_id}:{node_id}",
                    vector_hits=list(vector_hits or ()),
                    keyword_hits=list(keyword_hits or ()),
                    graph_hits=list(graph_hits or ()),
                    top_k=profile.parallel_search_budget,
                )
            )
            merged_ids = list(hybrid.merged_document_ids)

        findings: List[ExploreFinding] = []
        for index, document_id in enumerate(merged_ids[: profile.parallel_search_budget]):
            findings.append(
                ExploreFinding(
                    source=document_id,
                    summary=f"Retrieved evidence #{index + 1} from {document_id}",
                    score=float(profile.parallel_search_budget - index),
                )
            )

        synthesis_lines = [
            f"- [{finding.source}] {finding.summary}" for finding in findings
        ]
        synthesis = "Explore synthesis:\n" + "\n".join(synthesis_lines) if synthesis_lines else spec.objective

        if profile.synthesis_only_return and len(synthesis) > profile.max_child_context_tokens * 4:
            synthesis = synthesis[: profile.max_child_context_tokens * 4]

        return ExploreDelegationResult(
            findings=findings,
            synthesis_text=synthesis,
            memory_namespace=namespace,
        )
