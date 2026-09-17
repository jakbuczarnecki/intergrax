# © Artur Czarnecki. All rights reserved.

"""MP-5E default principal ContextView composer (Collaborative Work).

Flow: approved ContextViewPolicyDecision → injected MP-5D source ports → validated ContextView.
No policy re-evaluation, retrieval, hydration, or concrete domain adapters.
"""

from __future__ import annotations

import hashlib
import json

from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewEntry,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewUclSourceRef,
)
from intergrax.contracts.context_view_composition import (
    ContextViewCandidateOrderingStrategy,
    ContextViewCompositionCandidateIsolationError,
    ContextViewCompositionInvariantError,
    ContextViewCompositionPolicyDeniedError,
    ContextViewCompositionRequest,
    ContextViewCompositionRequestAlignmentError,
    ContextViewCompositionSourceFailureError,
    ContextViewCompositionValidatedCandidate,
    ContextViewEntryIdentityStrategy,
    ContextViewIdentityStrategy,
    context_view_source_ref_identity_key,
    validate_composition_request_alignment,
    validate_context_view_matches_composition_request,
)
from intergrax.contracts.context_view_composition import DefaultContextViewComposerConfig
from intergrax.contracts.context_view_source_ports import (
    CollaborativeWorkContextSourcePort,
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
    KnowledgeContextSourcePort,
    MemoryContextSourcePort,
    UclContextSourcePort,
    validate_collaborative_work_source_candidate_isolation,
    validate_knowledge_source_candidate_isolation,
    validate_memory_source_candidate_isolation,
    validate_ucl_source_candidate_isolation,
)
from intergrax.contracts.context_view_visibility_policy import ContextViewPolicyOutcome


class DefaultContextViewCategoryOrderingStrategy:
    """Platform default: policy eligible category order, then locator identity."""

    def order(
        self,
        *,
        eligible_category_order: tuple[ContextViewCategory, ...],
        candidates: tuple[ContextViewCompositionValidatedCandidate, ...],
    ) -> tuple[ContextViewCompositionValidatedCandidate, ...]:
        rank = {category: index for index, category in enumerate(eligible_category_order)}

        def sort_key(candidate: ContextViewCompositionValidatedCandidate) -> tuple[object, ...]:
            return (
                rank.get(candidate.category, len(rank)),
                context_view_source_ref_identity_key(candidate.source_ref),
            )

        return tuple(sorted(candidates, key=sort_key))


class Sha256ContextViewEntryIdentityStrategy:
    """Deterministic entry_id from category, scope, and typed source locator."""

    def entry_id_for_candidate(
        self,
        *,
        candidate: ContextViewCompositionValidatedCandidate,
        composition_request: ContextViewCompositionRequest,
    ) -> str:
        payload = {
            "category": candidate.category.value,
            "scope": candidate.entry_scope.model_dump(mode="json"),
            "source_ref": candidate.source_ref.model_dump(mode="json"),
            "acting_principal_id": composition_request.request.acting_principal_id,
            "policy_id": composition_request.policy_decision.policy_id,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()
        return f"context_view_entry:{digest}"


class Sha256ContextViewIdentityStrategy:
    """Deterministic view_id from policy scope, principal, and ordered entry ids."""

    def view_id_for_composition(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        entry_ids: tuple[str, ...],
    ) -> str:
        decision = composition_request.policy_decision
        payload = {
            "acting_principal_id": composition_request.request.acting_principal_id,
            "effective_scope": decision.effective_scope.model_dump(mode="json"),
            "policy_id": decision.policy_id,
            "entry_ids": list(entry_ids),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()
        return f"context_view:{digest}"


class DefaultContextViewComposer:
    """Platform default composer — injected MP-5D ports and replaceable strategies only."""

    def __init__(
        self,
        *,
        memory_source: MemoryContextSourcePort | None = None,
        knowledge_source: KnowledgeContextSourcePort | None = None,
        ucl_source: UclContextSourcePort | None = None,
        collaborative_work_source: CollaborativeWorkContextSourcePort | None = None,
        config: DefaultContextViewComposerConfig | None = None,
        ordering_strategy: ContextViewCandidateOrderingStrategy | None = None,
        entry_identity_strategy: ContextViewEntryIdentityStrategy | None = None,
        view_identity_strategy: ContextViewIdentityStrategy | None = None,
    ) -> None:
        self._memory_source = memory_source
        self._knowledge_source = knowledge_source
        self._ucl_source = ucl_source
        self._collaborative_work_source = collaborative_work_source
        self._config = config or DefaultContextViewComposerConfig()
        self._ordering_strategy = ordering_strategy or DefaultContextViewCategoryOrderingStrategy()
        self._entry_identity_strategy = (
            entry_identity_strategy or Sha256ContextViewEntryIdentityStrategy()
        )
        self._view_identity_strategy = view_identity_strategy or Sha256ContextViewIdentityStrategy()

    def compose(self, composition_request: ContextViewCompositionRequest) -> ContextView:
        decision = composition_request.policy_decision
        if decision.outcome is ContextViewPolicyOutcome.DENY:
            raise ContextViewCompositionPolicyDeniedError(
                "composition blocked when policy outcome is DENY",
            )
        validate_composition_request_alignment(composition_request=composition_request)

        validated = self._collect_validated_candidates(composition_request)
        ordered = self._ordering_strategy.order(
            eligible_category_order=decision.eligible_categories,
            candidates=validated,
        )
        entries = self._materialize_entries(
            composition_request=composition_request,
            ordered_candidates=ordered,
        )
        view_id = self._view_identity_strategy.view_id_for_composition(
            composition_request=composition_request,
            entry_ids=tuple(entry.entry_id for entry in entries),
        )
        view = ContextView(
            view_id=view_id,
            scope=decision.effective_scope,
            acting_principal_id=composition_request.request.acting_principal_id,
            entries=entries,
        )
        validate_context_view_matches_composition_request(
            view=view,
            composition_request=composition_request,
        )
        return view

    def _collect_validated_candidates(
        self,
        composition_request: ContextViewCompositionRequest,
    ) -> tuple[ContextViewCompositionValidatedCandidate, ...]:
        decision = composition_request.policy_decision
        request = composition_request.request
        seen: set[tuple[str, ...]] = set()
        collected: list[ContextViewCompositionValidatedCandidate] = []

        for category in decision.eligible_categories:
            port = self._port_for_category(category)
            if port is None:
                raise ContextViewCompositionSourceFailureError(
                    f"missing source port for eligible category {category.value}",
                )
            source_request = self._build_source_request(
                category=category,
                composition_request=composition_request,
            )
            result = self._invoke_port(category=category, port=port, source_request=source_request)
            self._raise_on_source_failure(category=category, outcome=result.outcome)
            for candidate in result.candidates:
                self._validate_candidate_isolation(
                    category=category,
                    source_request=source_request,
                    candidate=candidate,
                )
                validated = self._to_validated_candidate(category=category, candidate=candidate)
                if validated.visibility not in decision.eligible_visibility_classes:
                    raise ContextViewCompositionInvariantError(
                        "candidate visibility is not in policy eligible_visibility_classes",
                    )
                identity = context_view_source_ref_identity_key(validated.source_ref)
                if identity in seen:
                    continue
                seen.add(identity)
                collected.append(validated)

        return tuple(collected)

    def _port_for_category(
        self,
        category: ContextViewCategory,
    ) -> (
        MemoryContextSourcePort
        | KnowledgeContextSourcePort
        | UclContextSourcePort
        | CollaborativeWorkContextSourcePort
        | None
    ):
        if category is ContextViewCategory.MEMORY:
            return self._memory_source
        if category is ContextViewCategory.KNOWLEDGE:
            return self._knowledge_source
        if category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
            return self._ucl_source
        if category is ContextViewCategory.COLLABORATIVE_WORK:
            return self._collaborative_work_source
        raise ContextViewCompositionInvariantError(f"unsupported category {category.value}")

    def _build_source_request(
        self,
        *,
        category: ContextViewCategory,
        composition_request: ContextViewCompositionRequest,
    ) -> (
        ContextViewMemorySourceRequest
        | ContextViewKnowledgeSourceRequest
        | ContextViewUclSourceRequest
        | ContextViewCollaborativeWorkSourceRequest
    ):
        decision = composition_request.policy_decision
        request = composition_request.request
        base = {
            "scope": decision.effective_scope,
            "acting_principal_id": request.acting_principal_id,
            "eligible_visibility_classes": decision.eligible_visibility_classes,
        }
        if category is ContextViewCategory.MEMORY:
            return ContextViewMemorySourceRequest(**base)
        if category is ContextViewCategory.KNOWLEDGE:
            return ContextViewKnowledgeSourceRequest(**base)
        if category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
            return ContextViewUclSourceRequest(**base)
        if category is ContextViewCategory.COLLABORATIVE_WORK:
            return ContextViewCollaborativeWorkSourceRequest(**base)
        raise ContextViewCompositionInvariantError(f"unsupported category {category.value}")

    def _invoke_port(
        self,
        *,
        category: ContextViewCategory,
        port: (
            MemoryContextSourcePort
            | KnowledgeContextSourcePort
            | UclContextSourcePort
            | CollaborativeWorkContextSourcePort
        ),
        source_request: (
            ContextViewMemorySourceRequest
            | ContextViewKnowledgeSourceRequest
            | ContextViewUclSourceRequest
            | ContextViewCollaborativeWorkSourceRequest
        ),
    ) -> (
        ContextViewMemorySourceCandidatesResult
        | ContextViewKnowledgeSourceCandidatesResult
        | ContextViewUclSourceCandidatesResult
        | ContextViewCollaborativeWorkSourceCandidatesResult
    ):
        if category is ContextViewCategory.MEMORY:
            memory_port: MemoryContextSourcePort = port
            memory_request: ContextViewMemorySourceRequest = source_request
            return memory_port.list_candidates(memory_request)
        if category is ContextViewCategory.KNOWLEDGE:
            knowledge_port: KnowledgeContextSourcePort = port
            knowledge_request: ContextViewKnowledgeSourceRequest = source_request
            return knowledge_port.list_candidates(knowledge_request)
        if category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
            ucl_port: UclContextSourcePort = port
            ucl_request: ContextViewUclSourceRequest = source_request
            return ucl_port.list_candidates(ucl_request)
        if category is ContextViewCategory.COLLABORATIVE_WORK:
            collaborative_port: CollaborativeWorkContextSourcePort = port
            collaborative_request: ContextViewCollaborativeWorkSourceRequest = source_request
            return collaborative_port.list_candidates(collaborative_request)
        raise ContextViewCompositionInvariantError(f"unsupported category {category.value}")

    @staticmethod
    def _raise_on_source_failure(
        *,
        category: ContextViewCategory,
        outcome: ContextViewSourceOutcome,
    ) -> None:
        if outcome is ContextViewSourceOutcome.OK:
            return
        raise ContextViewCompositionSourceFailureError(
            f"source port for {category.value} returned {outcome.value}",
        )

    def _validate_candidate_isolation(
        self,
        *,
        category: ContextViewCategory,
        source_request: object,
        candidate: object,
    ) -> None:
        try:
            if category is ContextViewCategory.MEMORY:
                validate_memory_source_candidate_isolation(
                    request=source_request,  # type: ignore[arg-type]
                    candidate=candidate,  # type: ignore[arg-type]
                )
            elif category is ContextViewCategory.KNOWLEDGE:
                validate_knowledge_source_candidate_isolation(
                    request=source_request,  # type: ignore[arg-type]
                    candidate=candidate,  # type: ignore[arg-type]
                )
            elif category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
                validate_ucl_source_candidate_isolation(
                    request=source_request,  # type: ignore[arg-type]
                    candidate=candidate,  # type: ignore[arg-type]
                )
            elif category is ContextViewCategory.COLLABORATIVE_WORK:
                validate_collaborative_work_source_candidate_isolation(
                    request=source_request,  # type: ignore[arg-type]
                    candidate=candidate,  # type: ignore[arg-type]
                )
        except ValueError as exc:
            raise ContextViewCompositionCandidateIsolationError(str(exc)) from exc

    @staticmethod
    def _to_validated_candidate(
        *,
        category: ContextViewCategory,
        candidate: (
            ContextViewMemorySourceCandidate
            | ContextViewKnowledgeSourceCandidate
            | ContextViewUclSourceCandidate
            | ContextViewCollaborativeWorkSourceCandidate
        ),
    ) -> ContextViewCompositionValidatedCandidate:
        source_ref: (
            ContextViewMemorySourceRef
            | ContextViewKnowledgeSourceRef
            | ContextViewUclSourceRef
            | ContextViewCollaborativeWorkSourceRef
        )
        if category is ContextViewCategory.MEMORY:
            memory_candidate = candidate
            if not isinstance(memory_candidate, ContextViewMemorySourceCandidate):
                raise ContextViewCompositionInvariantError("memory candidate type mismatch")
            source_ref = memory_candidate.source_ref
        elif category is ContextViewCategory.KNOWLEDGE:
            knowledge_candidate = candidate
            if not isinstance(knowledge_candidate, ContextViewKnowledgeSourceCandidate):
                raise ContextViewCompositionInvariantError("knowledge candidate type mismatch")
            source_ref = knowledge_candidate.source_ref
        elif category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
            ucl_candidate = candidate
            if not isinstance(ucl_candidate, ContextViewUclSourceCandidate):
                raise ContextViewCompositionInvariantError("ucl candidate type mismatch")
            source_ref = ucl_candidate.source_ref
        else:
            collaborative_candidate = candidate
            if not isinstance(collaborative_candidate, ContextViewCollaborativeWorkSourceCandidate):
                raise ContextViewCompositionInvariantError("collaborative candidate type mismatch")
            source_ref = collaborative_candidate.source_ref

        return ContextViewCompositionValidatedCandidate(
            category=category,
            source_ref=source_ref,
            entry_scope=candidate.candidate_scope,
            visibility=candidate.suggested_visibility,
        )

    def _materialize_entries(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        ordered_candidates: tuple[ContextViewCompositionValidatedCandidate, ...],
    ) -> tuple[ContextViewEntry, ...]:
        entries: list[ContextViewEntry] = []
        for candidate in ordered_candidates:
            entry_id = self._entry_identity_strategy.entry_id_for_candidate(
                candidate=candidate,
                composition_request=composition_request,
            )
            entries.append(
                ContextViewEntry(
                    entry_id=entry_id,
                    source_ref=candidate.source_ref,
                    visibility=candidate.visibility,
                    entry_scope=candidate.entry_scope,
                ),
            )
        return tuple(entries)
