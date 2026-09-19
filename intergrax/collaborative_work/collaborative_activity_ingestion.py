# © Artur Czarnecki. All rights reserved.

"""MP-6C Collaborative Activity publication / ingestion boundary.

Policy-governed ingress: trusted publisher context → ingestion policy → append intent
→ append store. No source-domain semantics, no direct publication → store path.
"""

from __future__ import annotations

from intergrax.contracts.collaborative_activity import (
    CollaborativeActivity,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityAppendStore,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityPublication,
    CollaborativeActivityPublicationPort,
    CollaborativeActivityTypeId,
    CollaborativeActivityWritePort,
)
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityAdmissionRejected,
    CollaborativeActivityIngestionAppendError,
    CollaborativeActivityIngestionDecision,
    CollaborativeActivityIngestionDenialReason,
    CollaborativeActivityIngestionOutcome,
    CollaborativeActivityIngestionPolicy,
    CollaborativeActivityIngestionPolicyError,
    CollaborativeActivityIngestionRequest,
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
    DefaultCollaborativeActivityIngestionPolicyConfig,
    fail_closed_collaborative_activity_ingestion_decision,
)

_RESERVED_NAMESPACES = frozenset({"intergrax", "platform"})

_DURABILITY_STRICTNESS: dict[CollaborativeActivityDurabilityClass, int] = {
    CollaborativeActivityDurabilityClass.INFORMATIONAL: 1,
    CollaborativeActivityDurabilityClass.COLLABORATIVE: 2,
    CollaborativeActivityDurabilityClass.AUDIT_CRITICAL: 3,
}

_PLATFORM_MINIMUM_DURABILITY: dict[str, CollaborativeActivityDurabilityClass] = {
    CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION.qualified_id: (
        CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    ),
    CollaborativeActivityBuiltinType.AUTHORITY_RELEVANT_ACTION.qualified_id: (
        CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    ),
    CollaborativeActivityBuiltinType.DELEGATION_USED.qualified_id: (
        CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    ),
}


def _resolve_effective_durability(
    *,
    activity_type: CollaborativeActivityTypeId,
    requested: CollaborativeActivityDurabilityClass,
) -> CollaborativeActivityDurabilityClass:
    minimum = _PLATFORM_MINIMUM_DURABILITY.get(
        activity_type.qualified_id,
        CollaborativeActivityDurabilityClass.COLLABORATIVE,
    )
    requested_rank = _DURABILITY_STRICTNESS[requested]
    minimum_rank = _DURABILITY_STRICTNESS[minimum]
    if minimum_rank >= requested_rank:
        return minimum
    return requested


class DefaultCollaborativeActivityIngestionPolicy:
    """Platform default fail-closed ingestion policy (deterministic, no I/O)."""

    def __init__(
        self,
        config: DefaultCollaborativeActivityIngestionPolicyConfig | None = None,
    ) -> None:
        self._config = config or DefaultCollaborativeActivityIngestionPolicyConfig()

    @property
    def policy_id(self) -> str:
        return self._config.policy_id

    def evaluate(
        self,
        request: CollaborativeActivityIngestionRequest,
    ) -> CollaborativeActivityIngestionDecision:
        publication = request.publication
        publisher = request.publisher_context
        policy_id = self.policy_id

        if publisher.tenant_id != publication.scope.tenant_id:
            return fail_closed_collaborative_activity_ingestion_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityIngestionDenialReason.TENANT_MISMATCH,
            )

        if publisher.allowed_workspace_ids:
            if publication.scope.workspace_id not in publisher.allowed_workspace_ids:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.WORKSPACE_RESTRICTED,
                )

        source_ns = publication.idempotency_key.source.namespace
        type_ns = publication.activity_type.namespace

        if publisher.kind is CollaborativeActivityPublisherKind.PLUGIN:
            owned = (publisher.owned_namespace or "").strip().lower()
            if source_ns in _RESERVED_NAMESPACES or type_ns in _RESERVED_NAMESPACES:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.RESERVED_NAMESPACE_SPOOF,
                )
            if source_ns != owned:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.SOURCE_NAMESPACE_UNAUTHORIZED,
                )
            if type_ns != owned:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.TYPE_NAMESPACE_UNAUTHORIZED,
                )
        else:
            if source_ns not in _RESERVED_NAMESPACES:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.SOURCE_NAMESPACE_UNAUTHORIZED,
                )
            if type_ns not in _RESERVED_NAMESPACES:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityIngestionDenialReason.TYPE_NAMESPACE_UNAUTHORIZED,
                )

        if source_ns != type_ns and not (
            source_ns in _RESERVED_NAMESPACES and type_ns in _RESERVED_NAMESPACES
        ):
            return fail_closed_collaborative_activity_ingestion_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityIngestionDenialReason.SOURCE_TYPE_NAMESPACE_MISMATCH,
            )

        if publication.activity_type == CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION:
            if publisher.kind is not CollaborativeActivityPublisherKind.PLATFORM:
                return fail_closed_collaborative_activity_ingestion_decision(
                    policy_id=policy_id,
                    denial_reason=(
                        CollaborativeActivityIngestionDenialReason.CORRECTION_PUBLISHER_UNAUTHORIZED
                    ),
                )

        effective = _resolve_effective_durability(
            activity_type=publication.activity_type,
            requested=publication.requested_durability_class,
        )

        return CollaborativeActivityIngestionDecision(
            outcome=CollaborativeActivityIngestionOutcome.ALLOW,
            policy_id=policy_id,
            effective_durability_class=effective,
        )


class CollaborativeActivityIngestionService(
    CollaborativeActivityPublicationPort,
    CollaborativeActivityWritePort,
):
    """Canonical MP-6C ingestion orchestrator — instance-bound to trusted publisher context."""

    def __init__(
        self,
        *,
        publisher_context: CollaborativeActivityPublisherContext,
        ingestion_policy: CollaborativeActivityIngestionPolicy,
        append_store: CollaborativeActivityAppendStore,
    ) -> None:
        self._publisher_context = publisher_context
        self._ingestion_policy = ingestion_policy
        self._append_store = append_store

    def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        return self._ingest(publication)

    def append(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        return self._ingest(publication)

    def _ingest(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        request = CollaborativeActivityIngestionRequest(
            publication=publication,
            publisher_context=self._publisher_context,
        )
        try:
            decision = self._ingestion_policy.evaluate(request)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise CollaborativeActivityIngestionPolicyError(
                f"{self._ingestion_policy.policy_id}: policy evaluation failed",
            ) from exc

        if decision.outcome is not CollaborativeActivityIngestionOutcome.ALLOW:
            reason = decision.denial_reason or CollaborativeActivityIngestionDenialReason.POLICY_AMBIGUITY
            raise CollaborativeActivityAdmissionRejected(
                denial_reason=reason,
                policy_id=decision.policy_id,
            )

        intent = CollaborativeActivityAppendIntent(
            publication=publication,
            effective_durability_class=decision.effective_durability_class,
        )
        try:
            return self._append_store.append_idempotent(intent)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise CollaborativeActivityIngestionAppendError(
                f"{self._ingestion_policy.policy_id}: append store failed",
            ) from exc


__all__ = [
    "CollaborativeActivityIngestionService",
    "DefaultCollaborativeActivityIngestionPolicy",
]
