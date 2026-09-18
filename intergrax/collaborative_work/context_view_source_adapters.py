# © Artur Czarnecki. All rights reserved.

"""MP-5F-B5 replaceable ContextView source adapters (reference-only translation)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.collaborative_work.context_view_async_reference_read import (
    ContextViewAsyncReferenceReadRunner,
)
from intergrax.collaborative_work.context_view_source_mapping import (
    ContextViewSourceAdapterConfigurationError,
    candidate_scope_for_collaborative_work_ref,
    candidate_scope_for_ucl_ref,
    candidate_scope_from_knowledge_evaluated,
    candidate_scope_from_memory_evaluated,
    collaborative_work_read_request_from_context_view,
    collaborative_work_ref_within_request_scope,
    knowledge_evaluated_scope_within_request,
    knowledge_read_request_from_context_view,
    knowledge_ref_within_evaluated_scope,
    map_collaborative_work_artifact_ref,
    map_collaborative_work_item_ref,
    map_collaborative_work_version_ref,
    map_domain_read_outcome_to_context_view,
    map_knowledge_chunk_to_context_view_ref,
    map_memory_record_to_context_view_ref,
    map_ucl_ref_to_context_view_ref,
    memory_evaluated_scope_within_request,
    memory_read_request_from_context_view,
    memory_ref_within_evaluated_scope,
    request_identity_from_source_request,
    suggested_visibility_from_request,
    ucl_read_request_from_context_view,
    ucl_ref_within_request_scope,
)
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceReadPort,
)
from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.context_view_source_ports import (
    ContextViewCollaborativeWorkSourceCandidate,
    ContextViewCollaborativeWorkSourceCandidatesResult,
    ContextViewCollaborativeWorkSourceRequest,
    ContextViewKnowledgeSourceCandidate,
    ContextViewKnowledgeSourceCandidatesResult,
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceCandidate,
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    ContextViewUclSourceCandidate,
    ContextViewUclSourceCandidatesResult,
    ContextViewUclSourceRequest,
)
from intergrax.knowledge.contracts.knowledge_reference_read import KnowledgeReferenceReadPort
from intergrax.memory.contracts.memory_reference_read import MemoryReferenceReadPort
from intergrax.ucl.contracts.ucl_reference_read import UclReferenceReadPort

__all__ = [
    "DefaultCollaborativeWorkContextSource",
    "DefaultKnowledgeContextSource",
    "DefaultMemoryContextSource",
    "DefaultUclContextSource",
]


@dataclass
class DefaultMemoryContextSource:
    """MP-5D Memory port — translates ``MemoryReferenceReadPort`` results only."""

    reader: MemoryReferenceReadPort
    async_runner: ContextViewAsyncReferenceReadRunner

    def __post_init__(self) -> None:
        if self.reader is None:
            raise ContextViewSourceAdapterConfigurationError("memory reader is required")
        if self.async_runner is None:
            raise ContextViewSourceAdapterConfigurationError(
                "async_runner is required for Memory reference read"
            )

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        identity = request_identity_from_source_request(request)
        read_request = memory_read_request_from_context_view(request, identity=identity)
        try:
            result = self.async_runner.run(
                self.reader.read_references(identity, read_request),
            )
        except Exception:
            return ContextViewMemorySourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
            )
        outcome = map_domain_read_outcome_to_context_view(result.outcome)
        if outcome is not ContextViewSourceOutcome.OK:
            return ContextViewMemorySourceCandidatesResult(outcome=outcome)
        evaluated_scope = result.evaluated_scope
        if evaluated_scope is None:
            return ContextViewMemorySourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
            )
        if not memory_evaluated_scope_within_request(
            request=request,
            evaluated_scope=evaluated_scope,
        ):
            return ContextViewMemorySourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
            )
        visibility = suggested_visibility_from_request(request)
        candidate_scope = candidate_scope_from_memory_evaluated(evaluated_scope)
        candidates: list[ContextViewMemorySourceCandidate] = []
        for ref in result.references:
            if not memory_ref_within_evaluated_scope(ref=ref, evaluated_scope=evaluated_scope):
                return ContextViewMemorySourceCandidatesResult(
                    outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
                )
            candidates.append(
                ContextViewMemorySourceCandidate(
                    source_ref=map_memory_record_to_context_view_ref(ref),
                    candidate_scope=candidate_scope,
                    suggested_visibility=visibility,
                )
            )
        return ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=tuple(candidates),
        )


@dataclass
class DefaultKnowledgeContextSource:
    """MP-5D Knowledge port — translates ``KnowledgeReferenceReadPort`` results only."""

    reader: KnowledgeReferenceReadPort

    def __post_init__(self) -> None:
        if self.reader is None:
            raise ContextViewSourceAdapterConfigurationError("knowledge reader is required")

    def list_candidates(
        self,
        request: ContextViewKnowledgeSourceRequest,
    ) -> ContextViewKnowledgeSourceCandidatesResult:
        identity = request_identity_from_source_request(request)
        read_request = knowledge_read_request_from_context_view(request)
        try:
            result = self.reader.read_references(identity, read_request)
        except Exception:
            return ContextViewKnowledgeSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
            )
        outcome = map_domain_read_outcome_to_context_view(result.outcome)
        if outcome is not ContextViewSourceOutcome.OK:
            return ContextViewKnowledgeSourceCandidatesResult(outcome=outcome)
        evaluated_scope = result.evaluated_scope
        if evaluated_scope is None:
            return ContextViewKnowledgeSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
            )
        if not knowledge_evaluated_scope_within_request(
            request=request,
            evaluated_scope=evaluated_scope,
        ):
            return ContextViewKnowledgeSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
            )
        visibility = suggested_visibility_from_request(request)
        candidate_scope = candidate_scope_from_knowledge_evaluated(
            evaluated_scope,
            request_scope=request.scope,
        )
        candidates: list[ContextViewKnowledgeSourceCandidate] = []
        for ref in result.references:
            if not knowledge_ref_within_evaluated_scope(
                ref=ref,
                evaluated_scope=evaluated_scope,
            ):
                return ContextViewKnowledgeSourceCandidatesResult(
                    outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
                )
            candidates.append(
                ContextViewKnowledgeSourceCandidate(
                    source_ref=map_knowledge_chunk_to_context_view_ref(ref),
                    candidate_scope=candidate_scope,
                    suggested_visibility=visibility,
                )
            )
        return ContextViewKnowledgeSourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=tuple(candidates),
        )


@dataclass
class DefaultUclContextSource:
    """MP-5D UCL port — translates ``UclReferenceReadPort`` results only."""

    reader: UclReferenceReadPort
    async_runner: ContextViewAsyncReferenceReadRunner

    def __post_init__(self) -> None:
        if self.reader is None:
            raise ContextViewSourceAdapterConfigurationError("ucl reader is required")
        if self.async_runner is None:
            raise ContextViewSourceAdapterConfigurationError(
                "async_runner is required for UCL reference read"
            )

    def list_candidates(
        self,
        request: ContextViewUclSourceRequest,
    ) -> ContextViewUclSourceCandidatesResult:
        read_request = ucl_read_request_from_context_view(request)
        if read_request is None:
            return ContextViewUclSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.INVALID_REQUEST,
            )
        identity = request_identity_from_source_request(request)
        try:
            result = self.async_runner.run(
                self.reader.read_references(identity, read_request),
            )
        except Exception:
            return ContextViewUclSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
            )
        outcome = map_domain_read_outcome_to_context_view(result.outcome)
        if outcome is not ContextViewSourceOutcome.OK:
            return ContextViewUclSourceCandidatesResult(outcome=outcome)
        visibility = suggested_visibility_from_request(request)
        candidates: list[ContextViewUclSourceCandidate] = []
        for ref in result.references:
            if not ucl_ref_within_request_scope(ref=ref, request=request):
                return ContextViewUclSourceCandidatesResult(
                    outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
                )
            candidates.append(
                ContextViewUclSourceCandidate(
                    source_ref=map_ucl_ref_to_context_view_ref(ref),
                    candidate_scope=candidate_scope_for_ucl_ref(
                        ref=ref,
                        request_scope=request.scope,
                    ),
                    suggested_visibility=visibility,
                )
            )
        return ContextViewUclSourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=tuple(candidates),
        )


@dataclass
class DefaultCollaborativeWorkContextSource:
    """MP-5D Collaborative Work port — translates CW reference read only."""

    reader: CollaborativeWorkReferenceReadPort

    def __post_init__(self) -> None:
        if self.reader is None:
            raise ContextViewSourceAdapterConfigurationError(
                "collaborative work reader is required"
            )

    def list_candidates(
        self,
        request: ContextViewCollaborativeWorkSourceRequest,
    ) -> ContextViewCollaborativeWorkSourceCandidatesResult:
        identity = request_identity_from_source_request(request)
        read_request = collaborative_work_read_request_from_context_view(request)
        try:
            result = self.reader.read_references(identity, read_request)
        except Exception:
            return ContextViewCollaborativeWorkSourceCandidatesResult(
                outcome=ContextViewSourceOutcome.SOURCE_UNAVAILABLE,
            )
        outcome = map_domain_read_outcome_to_context_view(result.outcome)
        if outcome is not ContextViewSourceOutcome.OK:
            return ContextViewCollaborativeWorkSourceCandidatesResult(outcome=outcome)
        visibility = suggested_visibility_from_request(request)
        candidates: list[ContextViewCollaborativeWorkSourceCandidate] = []
        for ref in result.references:
            if not collaborative_work_ref_within_request_scope(ref=ref, request=request):
                return ContextViewCollaborativeWorkSourceCandidatesResult(
                    outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
                )
            if isinstance(ref, CollaborativeWorkItemCanonicalRef):
                source_ref = map_collaborative_work_item_ref(ref)
            elif isinstance(ref, CollaborativeWorkArtifactCanonicalRef):
                source_ref = map_collaborative_work_artifact_ref(ref)
            elif isinstance(ref, WorkArtifactVersionRef):
                source_ref = map_collaborative_work_version_ref(ref)
            else:
                return ContextViewCollaborativeWorkSourceCandidatesResult(
                    outcome=ContextViewSourceOutcome.SCOPE_REJECTED,
                )
            candidates.append(
                ContextViewCollaborativeWorkSourceCandidate(
                    source_ref=source_ref,
                    candidate_scope=candidate_scope_for_collaborative_work_ref(
                        ref=ref,
                        request_scope=request.scope,
                    ),
                    suggested_visibility=visibility,
                )
            )
        return ContextViewCollaborativeWorkSourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=tuple(candidates),
        )
