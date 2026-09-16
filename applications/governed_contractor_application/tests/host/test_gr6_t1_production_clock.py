# © Artur Czarnecki. All rights reserved.

"""GR-6-T1 — production runtime clock must not use fixed timestamps."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from decimal import Decimal

from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
    InMemoryProviderInvocationStore,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 16, 10, 0, 0, tzinfo=timezone.utc)
_TENANT = "gr6t1-tenant"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "gr6t1-user"
_PROVIDER = "gec3_deterministic_fake"
_CREATE_IDEMP = "idem-gr6t1-create"
_DIGEST = "sha256:" + ("ef" * 32)

_HISTORICAL_FIXED_PRODUCTION_CLOCK = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

_PRODUCTION_COMPOSITION_PATH = (
    Path(__file__).resolve().parents[2] / "host" / "production_external_work_composition.py"
)


def _in_memory_stores() -> tuple[
    InMemoryGovernedExecutionStore,
    InMemoryProofReceiptStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryContinuationStateStore,
    InMemoryProviderInvocationStore,
]:
    return (
        InMemoryGovernedExecutionStore(),
        InMemoryProofReceiptStore(),
        InMemoryPolicyBundleArtifactStore(),
        InMemoryContinuationStateStore(),
        InMemoryProviderInvocationStore(),
    )


def _policy_bundle() -> object:
    return build_immutable_runtime_policy_bundle(
        bundle_id="gr6t1-production-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="gr6t1.CREATE_EXTERNAL_WORK",
                description="allow create",
                effect="allow",
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
        ),
        issued_at=_T0,
    )


def _create_meta(task_id: str, run_id: str) -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "gr6t1 scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _CREATE_IDEMP,
        META_CORRELATION_ID: f"corr-{task_id}-{run_id}",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("10.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
        "external_work.workspace_ref": _WORKSPACE,
    }


def _build_runtime(
    fake: DeterministicExternalWorkFake,
    task_id: object,
    *,
    clock: object | None = None,
):
    execution_store, receipt_store, bundle_store, continuation_store, invocation_store = (
        _in_memory_stores()
    )
    cw_repositories = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    return build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),  # type: ignore[arg-type]
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),  # type: ignore[arg-type]
        collaborative_work_repositories=cw_repositories,
        execution_store=execution_store,
        receipt_store=receipt_store,
        bundle_store=bundle_store,
        continuation_store=continuation_store,
        provider_invocation_store=invocation_store,
        clock=clock,  # type: ignore[arg-type]
    )


def test_production_composition_has_no_fixed_runtime_clock_constant() -> None:
    source = _PRODUCTION_COMPOSITION_PATH.read_text(encoding="utf-8")
    assert "_PRODUCTION_POLICY_ISSUED_AT" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "datetime":
            continue
        if len(node.args) < 6:
            continue
        year = node.args[0]
        month = node.args[1]
        day = node.args[2]
        hour = node.args[3]
        if (
            isinstance(year, ast.Constant)
            and year.value == 2026
            and isinstance(month, ast.Constant)
            and month.value == 9
            and isinstance(day, ast.Constant)
            and day.value == 16
            and isinstance(hour, ast.Constant)
            and hour.value == 12
        ):
            pytest.fail(
                "production_external_work_composition must not embed fixed runtime clock datetime",
            )


def test_default_production_runtime_uses_dynamic_utc_not_fixed_fallback() -> None:
    task_id, run_id, _, _ = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _build_runtime(fake, task_id, clock=None)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    ):
        step = runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            execution_id="exec-gr6t1-dynamic-clock",
        )
    assert step.adapter_result is not None
    decision = step.adapter_result.policy_decision
    assert decision is not None
    evaluated_raw = decision.audit_payload.get("evaluated_at")
    assert evaluated_raw is not None
    evaluated_at = datetime.fromisoformat(str(evaluated_raw))
    assert evaluated_at != _HISTORICAL_FIXED_PRODUCTION_CLOCK
    assert evaluated_at.tzinfo is not None

    ger = runtime.orchestrator.get_result("exec-gr6t1-dynamic-clock")
    assert ger is not None
    assert ger.execution_started_at != _HISTORICAL_FIXED_PRODUCTION_CLOCK
    assert ger.execution_completed_at != _HISTORICAL_FIXED_PRODUCTION_CLOCK


def test_injected_clock_is_shared_between_policy_evaluator_and_orchestrator() -> None:
    task_id, run_id, _, _ = default_gr3_identity_bundle()
    fake = DeterministicExternalWorkFake()
    runtime = _build_runtime(fake, task_id, clock=lambda: _T0)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    ):
        step = runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id)),
            execution_id="exec-gr6t1-shared-clock",
        )
    assert step.adapter_result is not None
    decision = step.adapter_result.policy_decision
    assert decision is not None
    evaluated_at = datetime.fromisoformat(str(decision.audit_payload["evaluated_at"]))
    assert evaluated_at == _T0
    assert runtime.policy_evaluator.last_evaluation is not None
    assert runtime.policy_evaluator.last_evaluation.evaluated_at == _T0

    ger = runtime.orchestrator.get_result("exec-gr6t1-shared-clock")
    assert ger is not None
    assert ger.execution_started_at == _T0
    assert ger.execution_completed_at == _T0
    assert ger.provider_outcome.completed_at == _T0
