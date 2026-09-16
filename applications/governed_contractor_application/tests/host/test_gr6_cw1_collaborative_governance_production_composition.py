# © Artur Czarnecki. All rights reserved.

"""GR-6-CW1 — contract-based Collaborative Governance production composition."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
    wire_governed_contractor_production_external_work_settings,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_fixture_authority_clock,
    gr6_seeded_collaborative_work_repositories,
)
from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
    ExternalWorkAdapter,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from governed_contractor_application.host.collaborative_work_boundary import (
    build_external_work_authorization_boundary,
)
from governed_contractor_application.host.collaborative_work_local_fixture import (
    build_in_memory_collaborative_work_repositories,
    seed_external_work_collaborative_governance_state,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.repository import (
    CollaborativeWorkRepositoryCapabilities,
    CreateWorkspaceMembershipCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipRepository,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.collaborative_work import (
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_execution_authorization,
    decision_governance_policy_context,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.execution.decision_governed_side_effect import (
    DecisionGovernedSideEffectInputs,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BOUNDARY_MODULE = (
    Path(__file__).resolve().parents[2] / "host" / "collaborative_work_boundary.py"
)
_DIGEST = "sha256:" + ("ef" * 32)
_T0 = datetime(2026, 9, 16, 10, 0, 0, tzinfo=timezone.utc)
_TENANT = "gr6cw1-tenant"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "gr6cw1-user"
_PROVIDER = "gec3_deterministic_fake"
_INITIAL_RECORD_REVISION = 0


@dataclass(frozen=True, slots=True)
class _Payload:
    note: str


class _NoopCollaborativeWorkStore:
    def close(self) -> None:
        return None


class _PluginMembershipRepository:
    """Minimal non-InMemory WorkspaceMembershipRepository for pluginability proof."""

    plugin_marker = "gr6cw1-custom-membership"

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], WorkspaceMembership] = {}
        self.create_calls = 0

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id=self.plugin_marker,
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership:
        self.create_calls += 1
        record = WorkspaceMembership(
            membership_id=command.membership_id,
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            principal_id=command.principal_id,
            role=command.role,
            status=command.status,
            revision=_INITIAL_RECORD_REVISION,
        )
        key = (command.tenant_id, command.workspace_id, command.membership_id)
        self._records[key] = record
        return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        return self._records.get((tenant_id, workspace_id, membership_id))

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        for record in self._records.values():
            if (
                record.tenant_id == tenant_id
                and record.workspace_id == workspace_id
                and record.principal_id == principal_id
            ):
                return record
        return None

    def update(self, command: UpdateWorkspaceMembershipCommand) -> WorkspaceMembership:
        raise NotImplementedError("plugin membership repo update not required for GR-6-CW1")


def _policy_bundle():
    return build_immutable_runtime_policy_bundle(
        bundle_id="gr6cw1-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="gr6cw1.CREATE",
                description="allow create",
                effect="allow",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
            PolicyBundleRule(
                rule_id="gr6cw1.ACCEPT",
                description="allow accept",
                effect="allow",
                match_action=ACTION_ACCEPT_QUOTE,
            ),
        ),
        issued_at=_T0,
    )


def _runtime_policy() -> DeterministicMeaningfulSideEffectPolicy:
    return DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        policy_bundle=_policy_bundle(),
    )


def _create_meta(task_id: str, run_id: str) -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "gr6cw1 scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gr6cw1-create",
        META_CORRELATION_ID: "corr-gr6cw1",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("10.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
        "external_work.workspace_ref": _WORKSPACE,
    }


def _in_memory_stores():
    return (
        InMemoryGovernedExecutionStore(),
        InMemoryProofReceiptStore(),
        InMemoryPolicyBundleArtifactStore(),
        InMemoryContinuationStateStore(),
    )


def test_production_boundary_module_does_not_import_in_memory_repositories() -> None:
    tree = ast.parse(_BOUNDARY_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "in_memory_repository" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "in_memory_repository" not in alias.name


def test_injected_repository_state_allows_governance_create() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    boundary = build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=repositories,
        authority_clock=gr6_fixture_authority_clock,
        task_scope=StaticActiveTaskScope(task_id),
    )
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, authorization_boundary=boundary)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        created = adapter.create_and_map(
            adapter.build_create_request(
                task_id=str(task_id),
                run_id=str(run_id),
                metadata=_create_meta(str(task_id), str(run_id)),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
        )
    assert created.used is True
    assert fake.create_calls == 1


def test_empty_injected_repositories_denies_side_effect() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    empty_repositories = build_in_memory_collaborative_work_repositories()
    boundary = build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=empty_repositories,
        authority_clock=gr6_fixture_authority_clock,
        task_scope=StaticActiveTaskScope(task_id),
    )
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, authorization_boundary=boundary)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = adapter.create_and_map(
            adapter.build_create_request(
                task_id=str(task_id),
                run_id=str(run_id),
                metadata=_create_meta(str(task_id), str(run_id)),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
        )
    assert denied.used is False
    assert denied.policy_decision is not None
    assert denied.policy_decision.action is PolicyAction.DENY
    assert fake.create_calls == 0


def test_injected_policy_repository_controls_allow_and_deny() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    allow_repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    allow_boundary = build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=allow_repositories,
        authority_clock=gr6_fixture_authority_clock,
        task_scope=StaticActiveTaskScope(task_id),
    )
    fake = DeterministicExternalWorkFake()
    allow_adapter = ExternalWorkAdapter(fake, authorization_boundary=allow_boundary)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        allowed = allow_adapter.create_and_map(
            allow_adapter.build_create_request(
                task_id=str(task_id),
                run_id=str(run_id),
                metadata=_create_meta(str(task_id), str(run_id)),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
        )
    assert allowed.used is True

    deny_repositories = build_in_memory_collaborative_work_repositories()
    seed_external_work_collaborative_governance_state(
        deny_repositories,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        seed_workspace_policy=False,
    )
    deny_boundary = build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=deny_repositories,
        authority_clock=gr6_fixture_authority_clock,
        task_scope=StaticActiveTaskScope(task_id),
    )
    deny_adapter = ExternalWorkAdapter(fake, authorization_boundary=deny_boundary)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = deny_adapter.create_and_map(
            deny_adapter.build_create_request(
                task_id=str(task_id),
                run_id=str(run_id),
                metadata=_create_meta(str(task_id), str(run_id)),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
        )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert fake.create_calls == 1


def test_custom_membership_repository_is_pluggable() -> None:
    plugin_membership = _PluginMembershipRepository()
    repositories = CollaborativeWorkRepositories(
        membership=plugin_membership,
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=_NoopCollaborativeWorkStore(),
    )
    seed_external_work_collaborative_governance_state(
        repositories,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    assert plugin_membership.create_calls >= 1
    assert isinstance(repositories.membership, WorkspaceMembershipRepository)
    assert getattr(repositories.membership, "plugin_marker") == "gr6cw1-custom-membership"


def _accepted_decision(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    subject: str = _DIGEST,
) -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    return AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="external_work", subject=subject),
            tenant_id=_TENANT,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr6cw1.accept"),
            content=_Payload(note="accept"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(version)),
    )


def _decision_inputs(
    decision: AuthoritativeAcceptedDecision[_Payload],
    *,
    action_kind: str = ACTION_ACCEPT_QUOTE,
    subject: str = _DIGEST,
) -> DecisionGovernedSideEffectInputs[_Payload]:
    action = decision_execution_action(kind=action_kind, subject=subject)
    policy = decision_governance_policy_context(policy_provenance_digest="gr6cw1-policy")
    governance = DecisionGovernanceDecision(
        disposition=DecisionGovernanceDisposition.ALLOW,
        decision_ref=authoritative_decision_ref(decision),
        action=action,
        policy_context=policy,
        tenant_id=decision.identity.tenant_id,
    )
    authorization = decision_execution_authorization(governance_decision=governance)
    return DecisionGovernedSideEffectInputs(
        decision=decision,
        authorization=authorization,
        action=action,
        policy_context=policy,
    )


def test_gr6_decision_flow_regression_on_injected_composition() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    cw_repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    execution_store, receipt_store, bundle_store, continuation_store = _in_memory_stores()
    runtime = build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),
        collaborative_work_repositories=cw_repositories,
        execution_store=execution_store,
        receipt_store=receipt_store,
        bundle_store=bundle_store,
        continuation_store=continuation_store,
        clock=lambda: _T0,
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    ):
        created = runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            execution_id="exec-gr6cw1-create",
        )
    assert created.adapter_result is not None
    assert fake.create_calls == 1
    acceptance = QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-gr6cw1",
            "quote_id": created.adapter_result.quote.quote_id,
            "quote_version": 1,
            "scope_digest": _DIGEST,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id=_PRINCIPAL,
                tenant_id=_TENANT,
            ),
            "accepted_at": _T0 + timedelta(minutes=1),
        }
    )
    accept_calls_before = fake.accept_calls
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = runtime.orchestrator.accept(
            execution_id="exec-gr6cw1-accept-deny",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key="idem-gr6cw1-accept",
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
        )
    assert denied.adapter_result is not None
    assert denied.adapter_result.used is False
    assert fake.accept_calls == accept_calls_before

    decision = _accepted_decision(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        accepted = runtime.orchestrator.accept(
            execution_id="exec-gr6cw1-accept-ok",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key="idem-gr6cw1-accept",
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            decision_governance=_decision_inputs(decision),
        )
    assert accepted.adapter_result is not None
    assert accepted.adapter_result.used is True
    assert fake.accept_calls == 1

    wrong_decision = _accepted_decision(
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        lineage_denied = runtime.orchestrator.accept(
            execution_id="exec-gr6cw1-wrong-resource",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key="idem-gr6cw1-accept-wrong",
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            decision_governance=_decision_inputs(
                wrong_decision,
                action_kind=ACTION_CREATE_EXTERNAL_WORK,
            ),
        )
    assert lineage_denied.adapter_result is not None
    assert lineage_denied.adapter_result.used is False
    assert fake.accept_calls == 1


def test_production_settings_wire_uses_injected_collaborative_work_repositories() -> None:
    fake = DeterministicExternalWorkFake()
    cw_repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    wired = wire_governed_contractor_production_external_work_settings(
        replace(
            GovernedContractorBackendSettings.from_env(),
            external_work_integration=fake,
            runtime_policy_bundle=_policy_bundle(),
            collaborative_work_repositories=cw_repositories,
        ),
        task_scope=StaticActiveTaskScope(mint_task_id()),
    )
    assert wired.collaborative_work_repositories is cw_repositories
    assert wired.meaningful_side_effect_authorization_boundary is not None
