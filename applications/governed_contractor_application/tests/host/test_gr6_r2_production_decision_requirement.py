# © Artur Czarnecki. All rights reserved.

"""GR-6-R2 — production External Work host wires DecisionRequirementPolicy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from applications.governed_contractor_application.host.collaborative_work_boundary import (
    build_external_work_authorization_boundary,
    default_external_work_decision_requirement_policy,
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
    default_external_work_decision_requirement_policy as host_default_policy,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.decision_requirement_policy import DecisionRequirement
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
    classify_decision_requirement,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("ef" * 32)
_T0 = datetime(2026, 9, 16, 10, 0, 0, tzinfo=timezone.utc)
_TENANT = "gr6r2-tenant"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "gr6r2-user"
_CREATE_IDEMP = "idem-gr6r2-create"
_ACCEPT_IDEMP = "idem-gr6r2-accept"


def _policy_bundle():
    return build_immutable_runtime_policy_bundle(
        bundle_id="gr6r2-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="gr6r2.CREATE_EXTERNAL_WORK",
                description="allow create",
                effect="allow",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
            PolicyBundleRule(
                rule_id="gr6r2.ACCEPT_QUOTE",
                description="allow accept",
                effect="allow",
                match_action=ACTION_ACCEPT_QUOTE,
            ),
        ),
        issued_at=_T0,
    )


def _runtime_policy() -> DeterministicMeaningfulSideEffectPolicy:
    bundle = _policy_bundle()
    return DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        policy_bundle=bundle,
    )


def _production_boundary(task_id):
    repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    return build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=repositories,
        authority_clock=gr6_fixture_authority_clock,
        task_scope=StaticActiveTaskScope(task_id),
    )


def _create_meta(task_id: str, run_id: str) -> dict[str, object]:
    return {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "gr6r2 scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _CREATE_IDEMP,
        META_CORRELATION_ID: "corr-gr6r2",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("10.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
        "external_work.workspace_ref": _WORKSPACE,
    }


def test_default_policy_classifies_accept_quote_as_required() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    policy = default_external_work_decision_requirement_policy()
    from intergrax.contracts.decision_requirement_policy import DecisionRequirementContext
    from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest

    side_effect = MeaningfulSideEffectRequest(
        action=ACTION_ACCEPT_QUOTE,
        kinds=(MeaningfulSideEffectKind.COMMITMENT, MeaningfulSideEffectKind.MUTATION),
        side_effect_scope_id=_ACCEPT_IDEMP,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        tenant_id=_TENANT,
        principal_id=_PRINCIPAL,
    )
    context = DecisionRequirementContext.from_meaningful_side_effect(
        operation_id=ACTION_ACCEPT_QUOTE,
        resource_scope=_DIGEST,
        side_effect=side_effect,
    )
    assert classify_decision_requirement(policy, context) is DecisionRequirement.REQUIRED

    create_context = context.model_copy(
        update={
            "operation_id": ACTION_CREATE_EXTERNAL_WORK,
            "action": ACTION_CREATE_EXTERNAL_WORK,
        },
    )
    assert (
        classify_decision_requirement(policy, create_context)
        is DecisionRequirement.NOT_REQUIRED
    )


def test_production_boundary_denies_accept_without_decision_material() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, authorization_boundary=_production_boundary(task_id))
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

        acceptance = QuoteAcceptanceEvidence.model_validate(
            {
                "acceptance_id": "acc-gr6r2",
                "quote_id": created.quote.quote_id,  # type: ignore[union-attr]
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
        denied = adapter.forward_quote_acceptance(
            created.snapshot.correlation,  # type: ignore[union-attr]
            acceptance,
            idempotency_key=_ACCEPT_IDEMP,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert denied.used is False
    assert denied.policy_decision is not None
    assert denied.policy_decision.action is PolicyAction.DENY
    assert fake.accept_calls == accept_calls_before


def test_production_boundary_create_remains_not_required() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, authorization_boundary=_production_boundary(task_id))
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


def test_production_composition_accepts_injected_permissive_policy() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    boundary = build_external_work_authorization_boundary(
        _runtime_policy(),
        collaborative_work_repositories=repositories,
        authority_clock=gr6_fixture_authority_clock,
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        task_scope=StaticActiveTaskScope(task_id),
    )
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
        acceptance = QuoteAcceptanceEvidence.model_validate(
            {
                "acceptance_id": "acc-gr6r2-permissive",
                "quote_id": created.quote.quote_id,  # type: ignore[union-attr]
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
        accepted = adapter.forward_quote_acceptance(
            created.snapshot.correlation,  # type: ignore[union-attr]
            acceptance,
            idempotency_key="idem-gr6r2-permissive-accept",
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert accepted.used is True
    assert fake.accept_calls == 1


def test_host_exports_same_default_policy_factory() -> None:
    assert host_default_policy().evaluate is not None
