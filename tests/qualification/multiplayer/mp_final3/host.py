# © Artur Czarnecki. All rights reserved.

"""Composition-only fixture for MP-FINAL-3 capability-wide backend E2E.

Qualification composition root: seeds authority, wires public Collaborative Work
services, ContextView (CW reference source only), and Activity ingestion/read.
Does not own business semantics and is not a production orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from intergrax.collaborative_work.artifact_service import (
    CollaborativeWorkArtifactService,
    TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
    TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_ingestion_service,
    build_collaborative_activity_read_service,
)
from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
    MappingCollaborativeActivityPublisherAuthoritySource,
)
from intergrax.collaborative_work.collaborative_activity_read import CollaborativeActivityReadService
from intergrax.collaborative_work.collaborative_activity_source_adapters import (
    CollaborativeDecisionBindingServiceWithActivityPublication,
    CollaborativeWorkArtifactServiceWithActivityPublication,
    CollaborativeWorkServiceWithActivityPublication,
    ContextViewComposerWithActivityPublication,
)
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    FixedCollaborativeActivityActorPrincipalKindResolver,
)
from intergrax.collaborative_work.collaborative_activity_source_wiring import (
    wire_collaborative_decision_binding_service_with_activity_publication,
    wire_collaborative_work_artifact_service_with_activity_publication,
    wire_collaborative_work_service_with_activity_publication,
    wire_context_view_composer_with_activity_publication,
)
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkReferenceReadOutcome,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadResult,
)
from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultCollaborativeWorkContextSource,
)
from intergrax.collaborative_work.context_view_visibility import (
    ContextViewVisibilityEvaluator,
    DefaultContextViewVisibilityPolicy,
)
from intergrax.collaborative_work.decision_binding_service import (
    CollaborativeDecisionBindingService,
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
)
from intergrax.collaborative_work.default_collaborative_work_reference_reader import (
    CollaborativeWorkReferenceReadCapabilityBinding,
    DefaultCollaborativeWorkReferenceReader,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithArtifacts,
    collaborative_activity_append_store_from_sqlite_bundle,
    collaborative_activity_read_store_from_sqlite_bundle,
    open_sqlite_collaborative_work_repositories,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import CreateWorkspaceMembershipCommand
from intergrax.collaborative_work.repository_backed_reference_catalog import (
    RepositoryBackedCollaborativeWorkReferenceCatalog,
)
from intergrax.collaborative_work.service import (
    CollaborativeWorkService,
    TRUSTED_OPERATION_ASSIGNMENT_CREATE,
    TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
    TRUSTED_OPERATION_WORK_ITEM_CREATE,
    TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity import CollaborativeActivityReadPort
from intergrax.contracts.collaborative_activity_publisher_authority import (
    verified_collaborative_activity_publisher_identity_from_request_identity,
)
from intergrax.contracts.collaborative_activity_read import (
    COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
)
from intergrax.contracts.collaborative_work import (
    MembershipResolutionMode,
    MembershipStatus,
    PrincipalKind,
    WorkspaceMembershipRole,
)
from intergrax.contracts.context_view_composition import (
    ContextViewComposer,
    DefaultContextViewComposerConfig,
)
from intergrax.contracts.context_view_visibility_policy import CONTEXT_VIEW_READ_AUTHORITY_SCOPE
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from tests.qualification.mp6.mp6g_harness import seed_workspace_principal_authority
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    platform_publisher_registration,
)

TENANT_A: Final = "mpf3-tenant-a"
TENANT_B: Final = "mpf3-tenant-b"
WS_A: Final = "mpf3-workspace-a"
WS_B: Final = "mpf3-workspace-b"

AUTHORIZED_PRINCIPAL: Final = "mpf3-authorized"
UNAUTHORIZED_PRINCIPAL: Final = "mpf3-member-no-grant"
READER_PRINCIPAL: Final = "mpf3-reader"
FOREIGN_PRINCIPAL: Final = "mpf3-tenant-b-actor"

PUBLISHER_PRINCIPAL: Final = "mpf3-platform-activity-publisher"
_WORK_AUTHORITY: Final = "collaborative_work.manage"

FIXED_CLOCK = datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
FIXED_UTC_NOW = datetime(2026, 9, 20, 10, 5, tzinfo=UTC)


def fixed_clock() -> datetime:
    return FIXED_CLOCK


def fixed_utc_now() -> datetime:
    return FIXED_UTC_NOW


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal shared-work mutations",
            policy_rule_id="mpf3.unexpected_runtime",
        )


class _WorkspaceRoutedCollaborativeWorkReader:
    """Composition helper: route CW reference reads by workspace capability binding."""

    def __init__(
        self,
        readers: dict[str, DefaultCollaborativeWorkReferenceReader],
    ) -> None:
        self._readers = readers

    def read_references(
        self,
        identity: RequestIdentity,
        request: CollaborativeWorkReferenceReadRequest,
    ) -> CollaborativeWorkReferenceReadResult:
        reader = self._readers.get(request.scope.workspace_id)
        if reader is None:
            return CollaborativeWorkReferenceReadResult(
                outcome=CollaborativeWorkReferenceReadOutcome.UNAVAILABLE,
                reason="no_reader_for_workspace",
            )
        return reader.read_references(identity, request)


@dataclass(slots=True)
class MpFinal3Host:
    """Public composition surface for capability-wide E2E (test layer only)."""

    bundle: CollaborativeWorkRepositoriesWithArtifacts
    work_service: CollaborativeWorkServiceWithActivityPublication
    artifact_service: CollaborativeWorkArtifactServiceWithActivityPublication
    decision_binding_service: CollaborativeDecisionBindingServiceWithActivityPublication
    context_view_composer: ContextViewComposerWithActivityPublication
    raw_context_view_composer: ContextViewComposer
    context_view_evaluator: ContextViewVisibilityEvaluator
    activity_read_service: CollaborativeActivityReadService
    activity_read_port: CollaborativeActivityReadPort
    authority_resolver: CollaborativeWorkAuthorityResolver
    clock: Callable[[], datetime]
    utc_now: Callable[[], datetime]

    def close(self) -> None:
        self.bundle.close()


def _build_enforcement_gate(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    clock: Callable[[], datetime],
) -> CollaborativeWorkEnforcementGate:
    return CollaborativeWorkEnforcementGate(
        profile_repository=bundle.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=bundle.membership,
            delegation_repository=bundle.delegation,
            principal_authority_repository=bundle.principal_authority,
            clock=clock,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(bundle.policy),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )


def _seed_host_authority(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    clock: Callable[[], datetime],
) -> None:
    work_ops = (
        TRUSTED_OPERATION_WORK_ITEM_CREATE,
        TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
        TRUSTED_OPERATION_ASSIGNMENT_CREATE,
        TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
        TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
        TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
        TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=AUTHORIZED_PRINCIPAL,
        membership_id="mpf3-mem-authorized-a",
        grant_id="mpf3-grant-authorized-a",
        authority_scopes=(_WORK_AUTHORITY, CONTEXT_VIEW_READ_AUTHORITY_SCOPE),
        operation_ids=work_ops,
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=READER_PRINCIPAL,
        membership_id="mpf3-mem-reader-a",
        grant_id="mpf3-grant-reader-a",
        authority_scopes=(
            COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
            CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
        ),
        operation_ids=(),
        clock=clock,
    )
    # Membership without authority grant — identity/membership ≠ mutation authority.
    bundle.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            membership_id="mpf3-mem-unauthorized-a",
            principal_id=UNAUTHORIZED_PRINCIPAL,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_B,
        workspace_id=WS_B,
        principal_id=FOREIGN_PRINCIPAL,
        membership_id="mpf3-mem-foreign-b",
        grant_id="mpf3-grant-foreign-b",
        authority_scopes=(
            _WORK_AUTHORITY,
            COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
            CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
        ),
        operation_ids=(TRUSTED_OPERATION_WORK_ITEM_CREATE,),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_B,
        workspace_id=WS_B,
        principal_id=READER_PRINCIPAL,
        membership_id="mpf3-mem-reader-b",
        grant_id="mpf3-grant-reader-b",
        authority_scopes=(
            COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
            CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
        ),
        operation_ids=(),
        clock=clock,
    )


def build_mp_final3_host(db_path: str | Path) -> MpFinal3Host:
    """Compose SQLite-backed Multiplayer surfaces for deterministic E2E qualification."""
    clock = fixed_clock
    utc_now = fixed_utc_now
    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    _seed_host_authority(bundle, clock=clock)

    append_store = collaborative_activity_append_store_from_sqlite_bundle(bundle, utc_now=utc_now)
    read_port = collaborative_activity_read_store_from_sqlite_bundle(bundle)

    publisher_registrations = (
        platform_publisher_registration(TENANT_A, PUBLISHER_PRINCIPAL),
        platform_publisher_registration(TENANT_B, PUBLISHER_PRINCIPAL),
    )
    authority_source = MappingCollaborativeActivityPublisherAuthoritySource(
        registrations=publisher_registrations,
    )
    resolver = DefaultCollaborativeActivityPublisherContextResolver(authority_source)
    verified_a = verified_collaborative_activity_publisher_identity_from_request_identity(
        RequestIdentity(
            tenant_id=TENANT_A,
            auth_subject=PUBLISHER_PRINCIPAL,
            principal_type=PrincipalType.SERVICE,
        ),
    )
    ingestion_a = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_a,
        publisher_context_resolver=resolver,
        append_store=append_store,
    )

    gate = _build_enforcement_gate(bundle, clock=clock)
    raw_work = CollaborativeWorkService(
        work_item_repository=bundle.work_item,
        assignment_repository=bundle.assignment,
        enforcement_gate=gate,
        clock=clock,
    )
    raw_artifact = CollaborativeWorkArtifactService(
        work_item_repository=bundle.work_item,
        work_artifact_repository=bundle.artifact,
        artifact_publication_repository=bundle.publication,
        enforcement_gate=gate,
        clock=clock,
    )
    raw_binding = CollaborativeDecisionBindingService(
        work_item_repository=bundle.work_item,
        work_artifact_version_repository=bundle.version,
        binding_repository=bundle.decision_binding,
        enforcement_gate=gate,
        clock=clock,
    )

    catalog = RepositoryBackedCollaborativeWorkReferenceCatalog(
        work_item_repository=bundle.work_item,
        work_artifact_repository=bundle.artifact,
        work_artifact_version_repository=bundle.version,
    )
    cw_readers = {
        WS_A: DefaultCollaborativeWorkReferenceReader(
            catalog=catalog,
            capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
                tenant_id=TENANT_A,
                workspace_id=WS_A,
            ),
        ),
        WS_B: DefaultCollaborativeWorkReferenceReader(
            catalog=catalog,
            capability_binding=CollaborativeWorkReferenceReadCapabilityBinding(
                tenant_id=TENANT_B,
                workspace_id=WS_B,
            ),
        ),
    }
    cw_reader = _WorkspaceRoutedCollaborativeWorkReader(cw_readers)
    raw_composer: ContextViewComposer = DefaultContextViewComposer(
        collaborative_work_source=DefaultCollaborativeWorkContextSource(reader=cw_reader),
        config=DefaultContextViewComposerConfig(
            knowledge_reference_read_query_text="mpf3-unused-knowledge-query",
        ),
    )

    principal_kind_resolver = FixedCollaborativeActivityActorPrincipalKindResolver(
        principal_kind=PrincipalKind.HUMAN,
    )
    work_service = wire_collaborative_work_service_with_activity_publication(
        inner=raw_work,
        publication_port=ingestion_a,
        principal_kind_resolver=principal_kind_resolver,
    )
    artifact_service = wire_collaborative_work_artifact_service_with_activity_publication(
        inner=raw_artifact,
        publication_port=ingestion_a,
        principal_kind_resolver=principal_kind_resolver,
    )
    decision_binding_service = wire_collaborative_decision_binding_service_with_activity_publication(
        inner=raw_binding,
        publication_port=ingestion_a,
        principal_kind_resolver=principal_kind_resolver,
    )
    context_view_composer = wire_context_view_composer_with_activity_publication(
        inner=raw_composer,
        publication_port=ingestion_a,
        principal_kind_resolver=principal_kind_resolver,
    )

    authority_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=bundle.membership,
        delegation_repository=bundle.delegation,
        principal_authority_repository=bundle.principal_authority,
        clock=clock,
    )
    context_view_evaluator = ContextViewVisibilityEvaluator(
        authority_resolver=authority_resolver,
        visibility_policy=DefaultContextViewVisibilityPolicy(),
        delegation_repository=bundle.delegation,
    )
    activity_read_service = build_collaborative_activity_read_service(
        authority_resolver=authority_resolver,
        read_port=read_port,
    )

    return MpFinal3Host(
        bundle=bundle,
        work_service=work_service,
        artifact_service=artifact_service,
        decision_binding_service=decision_binding_service,
        context_view_composer=context_view_composer,
        raw_context_view_composer=raw_composer,
        context_view_evaluator=context_view_evaluator,
        activity_read_service=activity_read_service,
        activity_read_port=read_port,
        authority_resolver=authority_resolver,
        clock=clock,
        utc_now=utc_now,
    )


__all__ = [
    "AUTHORIZED_PRINCIPAL",
    "FOREIGN_PRINCIPAL",
    "MembershipResolutionMode",
    "MpFinal3Host",
    "PUBLISHER_PRINCIPAL",
    "READER_PRINCIPAL",
    "TENANT_A",
    "TENANT_B",
    "UNAUTHORIZED_PRINCIPAL",
    "WS_A",
    "WS_B",
    "build_mp_final3_host",
    "fixed_clock",
    "fixed_utc_now",
]
