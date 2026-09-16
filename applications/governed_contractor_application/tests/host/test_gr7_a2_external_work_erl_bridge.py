# © Artur Czarnecki. All rights reserved.

"""GR-7-A2 — External Work observation → ERL admission bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
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
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
    InMemoryProviderInvocationStore,
)
from intergrax.contracts.enterprise_reliability.admission_boundary import (
    ExternalEffectAdmissionCaseError,
    ExternalEffectAdmissionPhase,
    ExternalEffectAdmissionRequest,
    ExternalEffectAdmissionResult,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reliability_boundary import (
    ExternalEffectReliabilityInteraction,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.enterprise_reliability.admission_boundary import (
    admit_external_effect_into_enterprise_reliability,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 16, 10, 0, 0, tzinfo=timezone.utc)
_TENANT = "gr7a2-tenant"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "gr7a2-user"
_PROVIDER = "gec3_deterministic_fake"
_DIGEST = "sha256:" + ("ef" * 32)
_CREATE_IDEMP = "idem-gr7a2-create"
_UNCERTAIN_IDEMP = "idem-gr7a2-uncertain"
_FAIL_IDEMP = "idem-gr7a2-fail"
_ERL_RUNTIME = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "admission_boundary.py"
)


@dataclass
class RecordingAdmissionPort:
    requests: list[ExternalEffectAdmissionRequest] = field(default_factory=list)
    fail: bool = False

    def admit_external_effect(
        self,
        request: ExternalEffectAdmissionRequest,
    ) -> ExternalEffectAdmissionResult:
        self.requests.append(request)
        if self.fail:
            raise ExternalEffectAdmissionCaseError("injected admission failure")
        return admit_external_effect_into_enterprise_reliability(request)


def _stores() -> tuple[
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


def _policy_bundle(*, deny_create: bool = False) -> object:
    effect = "deny" if deny_create else "allow"
    return build_immutable_runtime_policy_bundle(
        bundle_id="gr7a2-policy",
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="gr7a2.CREATE_EXTERNAL_WORK",
                description="create side effect",
                effect=effect,
                match_action=ACTION_CREATE_EXTERNAL_WORK,
            ),
        ),
        issued_at=_T0,
    )


def _create_meta(task_id: str, run_id: str, *, idempotency_key: str = _CREATE_IDEMP) -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "gr7a2 scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: idempotency_key,
        META_CORRELATION_ID: f"corr-{task_id}",
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
    deny_create: bool = False,
    admission_port: RecordingAdmissionPort | None = None,
    capabilities: ExternalWorkProviderCapabilities | None = None,
):
    execution_store, receipt_store, bundle_store, continuation_store, invocation_store = (
        _stores()
    )
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    port = admission_port or RecordingAdmissionPort()
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(admission_port=port)
    runtime = build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),  # type: ignore[arg-type]
        capabilities=capabilities
        or quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(deny_create=deny_create),  # type: ignore[arg-type]
        collaborative_work_repositories=cw,
        execution_store=execution_store,
        receipt_store=receipt_store,
        bundle_store=bundle_store,
        continuation_store=continuation_store,
        provider_invocation_store=invocation_store,
        reliability_bridge=bridge,
    )
    return runtime, port


def _create_step(
    runtime,
    fake: DeterministicExternalWorkFake,
    task_id: object,
    run_id: object,
    attempt_id: object,
    execution_id: object,
    *,
    idempotency_key: str = _CREATE_IDEMP,
):
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        step = runtime.orchestrator.create(
            task_id=str(task_id),
            run_id=str(run_id),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            metadata=_create_meta(str(task_id), str(run_id), idempotency_key=idempotency_key),
            execution_id="exec-gr7a2",
        )
    return step, fake


def test_success_admits_external_effect_success() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id)
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 1
    assert step.external_effect_outcome is ExternalEffectOutcome.SUCCESS
    assert step.reliability_admission is not None
    assert step.reliability_admission.admission_error is None
    assert len(port.requests) == 1
    assert port.requests[0].effect_outcome is ExternalEffectOutcome.SUCCESS
    assert step.reliability_admission is not None
    assert step.reliability_admission.admission is not None
    assert (
        step.reliability_admission.admission.projection.interaction
        is ExternalEffectReliabilityInteraction.NO_FAILURE_CLASSIFICATION
    )
    assert step.governed_result is not None
    assert (
        step.governed_result.provider_outcome.status is ProviderInvocationStatus.SUCCEEDED
    )


def test_definitive_failure_admits_failure_without_ger() -> None:
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={_FAIL_IDEMP: ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE},
    )
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id)
    step, fake = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_FAIL_IDEMP,
    )
    assert fake.create_calls == 1
    assert step.state is GovernedExternalWorkHostState.EXECUTION_FAILED
    assert step.governed_result is None
    assert step.external_effect_outcome is ExternalEffectOutcome.FAILURE
    assert port.requests[0].effect_outcome is ExternalEffectOutcome.FAILURE


def test_unknown_admits_uncertainty_case() -> None:
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={
            _UNCERTAIN_IDEMP: ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
        },
    )
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id)
    step, fake = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_UNCERTAIN_IDEMP,
    )
    assert fake.create_calls == 1
    assert step.external_effect_outcome is ExternalEffectOutcome.UNKNOWN
    admission = step.reliability_admission
    assert admission is not None
    assert admission.admission is not None
    assert admission.admission.phase is ExternalEffectAdmissionPhase.HANDED_OFF
    assert admission.admission.case_record is not None
    assert port.requests[0].effect_outcome is ExternalEffectOutcome.UNKNOWN


def test_discover_failure_zero_provider_calls_no_admission() -> None:
    fake = DeterministicExternalWorkFake(
        discover_error_code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE,
    )
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id)
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 0
    assert step.external_effect_outcome is None
    assert port.requests == []
    assert step.adapter_result is not None
    assert not step.adapter_result.provider_mutation_dispatched


def test_admission_contract_reflects_provider_capabilities() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER).model_copy(
        update={"supports_idempotency": False, "supports_status_polling": False},
    )
    runtime, port = _build_runtime(fake, task_id, capabilities=caps)
    _create_step(runtime, fake, task_id, run_id, attempt_id, execution_id)
    contract = port.requests[0].contract
    assert contract.safety.idempotency is ExternalEffectCapabilitySupport.NOT_SUPPORTED
    assert contract.safety.reconciliation is ExternalEffectCapabilitySupport.NOT_SUPPORTED


def test_governance_deny_zero_provider_calls_no_admission() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(
        fake,
        task_id,
        deny_create=True,
    )
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 0
    assert step.state is GovernedExternalWorkHostState.CREATE_POLICY_DENIED
    assert step.external_effect_outcome is None
    assert port.requests == []


def test_erl_admission_failure_does_not_retry_provider() -> None:
    fake = DeterministicExternalWorkFake()
    port = RecordingAdmissionPort(fail=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(
        fake,
        task_id,
        admission_port=port,
    )
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 1
    assert step.external_effect_outcome is ExternalEffectOutcome.SUCCESS
    assert step.reliability_admission is not None
    assert step.reliability_admission.admission_error is not None
    assert step.governed_result is not None


def test_admission_identity_binding() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, port = _build_runtime(fake, task_id)
    _create_step(runtime, fake, task_id, run_id, attempt_id, execution_id)
    req = port.requests[0]
    assert req.correlation_id == f"corr-{task_id}"
    assert req.external_effect_ref == f"ext:external_work:{_CREATE_IDEMP}"
    assert req.source_context.source_ref == "exec-gr7a2"


def test_erl_runtime_does_not_import_external_work_adapter() -> None:
    source = _ERL_RUNTIME.read_text(encoding="utf-8")
    forbidden = (
        "external_contractor_adapter",
        "ExternalWorkAdapter",
        "ExternalWorkErrorCode",
    )
    for token in forbidden:
        assert token not in source


def test_platform_erl_admission_still_side_effect_free() -> None:
    source = _ERL_RUNTIME.read_text(encoding="utf-8")
    for token in (
        "plan_external_effect_reconciliation",
        "execute_external_effect_reconciliation_probe",
    ):
        assert token not in source
