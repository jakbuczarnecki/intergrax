# © Artur Czarnecki. All rights reserved.

"""GR-6-WIRE — production Decision → Governance → External Work accept composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
    wire_governed_contractor_production_external_work_settings,
)
from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
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
from intergrax.runtime.execution.decision_governed_side_effect import (
    DecisionGovernedSideEffectInputs,
)
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("ef" * 32)
_T0 = datetime(2026, 9, 16, 10, 0, 0, tzinfo=timezone.utc)
_TENANT = "gr6wire-tenant"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "gr6wire-user"
_PROVIDER = "gec3_deterministic_fake"
_CREATE_IDEMP = "idem-gr6wire-create"
_ACCEPT_IDEMP = "idem-gr6wire-accept"


@dataclass(frozen=True, slots=True)
class _Payload:
    note: str


def _create_meta(task_id: str, run_id: str) -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "gr6wire scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _CREATE_IDEMP,
        META_CORRELATION_ID: "corr-gr6wire",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("10.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
        "external_work.workspace_ref": _WORKSPACE,
    }


def _acceptance(quote_id: str) -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-gr6wire",
            "quote_id": quote_id,
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


def _accepted_decision(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    scope_digest: str = _DIGEST,
) -> AuthoritativeAcceptedDecision[_Payload]:
    version = initial_decision_version()
    return AuthoritativeAcceptedDecision(
        identity=DecisionIdentity(
            decision_id=mint_decision_id(),
            version=version,
            scope=DecisionScope(namespace="external_work", subject=scope_digest),
            tenant_id=_TENANT,
            execution=DecisionExecutionLineage(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        ),
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("gr6wire.accept"),
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
    policy = decision_governance_policy_context(policy_provenance_digest="gr6wire-policy")
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


def _production_runtime(
    fake: DeterministicExternalWorkFake,
    task_id: TaskId,
    *,
    decision_requirement_policy: object | None = None,
):
    policy = decision_requirement_policy
    return build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        decision_requirement_policy=policy,  # type: ignore[arg-type]
        attestor=build_deterministic_test_attestor(clock=lambda: _T0),
        clock=lambda: _T0,
    )


def _create_with_runtime(runtime, task_id: TaskId, run_id: RunId):
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    ):
        return runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            execution_id="exec-gr6wire-create",
        )


def test_production_accept_denies_without_decision_material() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(fake, task_id)
    created = _create_with_runtime(runtime, task_id, run_id)
    assert created.adapter_result is not None
    assert fake.create_calls == 1

    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    accept_calls_before = fake.accept_calls
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = runtime.orchestrator.accept(
            execution_id="exec-gr6wire-accept",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key=_ACCEPT_IDEMP,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
        )
    assert denied.adapter_result is not None
    assert denied.adapter_result.used is False
    assert denied.adapter_result.policy_decision is not None
    assert denied.adapter_result.policy_decision.action is PolicyAction.DENY
    assert fake.accept_calls == accept_calls_before
    assert denied.state is GovernedExternalWorkHostState.ACCEPT_POLICY_DENIED


def test_production_accept_allows_with_authoritative_decision() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(fake, task_id)
    created = _create_with_runtime(runtime, task_id, run_id)
    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    decision = _accepted_decision(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    inputs = _decision_inputs(decision)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        accepted = runtime.orchestrator.accept(
            execution_id="exec-gr6wire-accept-ok",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key=_ACCEPT_IDEMP,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            decision_governance=inputs,
        )
    assert accepted.adapter_result is not None
    assert accepted.adapter_result.used is True
    assert fake.accept_calls == 1


def test_production_accept_denies_wrong_decision_action() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(fake, task_id)
    created = _create_with_runtime(runtime, task_id, run_id)
    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    decision = _accepted_decision(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    wrong_inputs = _decision_inputs(
        decision,
        action_kind=ACTION_CREATE_EXTERNAL_WORK,
        subject=_DIGEST,
    )
    accept_calls_before = fake.accept_calls
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = runtime.orchestrator.accept(
            execution_id="exec-gr6wire-wrong-action",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key=_ACCEPT_IDEMP,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            decision_governance=wrong_inputs,
        )
    assert denied.adapter_result is not None
    assert denied.adapter_result.used is False
    assert fake.accept_calls == accept_calls_before


def test_production_accept_denies_execution_lineage_mismatch() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(fake, task_id)
    created = _create_with_runtime(runtime, task_id, run_id)
    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    other_run = mint_run_id()
    decision = _accepted_decision(
        task_id=task_id,
        run_id=other_run,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    inputs = _decision_inputs(decision)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        denied = runtime.orchestrator.accept(
            execution_id="exec-gr6wire-mismatch",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key=_ACCEPT_IDEMP,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            decision_governance=inputs,
        )
    assert denied.adapter_result is not None
    assert denied.adapter_result.used is False
    assert fake.accept_calls == 0


def test_production_create_not_required_without_decision() -> None:
    task_id, run_id, _, _ = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(fake, task_id)
    created = _create_with_runtime(runtime, task_id, run_id)
    assert created.adapter_result is not None
    assert created.adapter_result.used is True
    assert fake.create_calls == 1


def test_production_composition_accepts_injected_decision_requirement_policy() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _production_runtime(
        fake,
        task_id,
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
    )
    created = _create_with_runtime(runtime, task_id, run_id)
    acceptance = _acceptance(created.adapter_result.quote.quote_id)  # type: ignore[union-attr]
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        accepted = runtime.orchestrator.accept(
            execution_id="exec-gr6wire-permissive",
            create_result=created.adapter_result,
            acceptance=acceptance,
            idempotency_key="idem-gr6wire-permissive",
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
        )
    assert accepted.adapter_result is not None
    assert accepted.adapter_result.used is True
    assert fake.accept_calls == 1


def test_production_settings_wire_authorization_boundary_idempotent() -> None:
    fake = DeterministicExternalWorkFake()
    base = GovernedContractorBackendSettings.from_env()
    wired = wire_governed_contractor_production_external_work_settings(
        base,
        integration=fake,
        task_scope=StaticActiveTaskScope(mint_task_id()),
    )
    assert wired.meaningful_side_effect_authorization_boundary is not None
    assert wired.external_work_integration is fake
    assert wired.decision_requirement_policy is not None
    rewired = wire_governed_contractor_production_external_work_settings(wired)
    assert rewired is wired
