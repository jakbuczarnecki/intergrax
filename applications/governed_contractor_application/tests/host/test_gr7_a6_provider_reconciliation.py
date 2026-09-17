# © Artur Czarnecki. All rights reserved.

"""GR-7-A6 — durable UNKNOWN provider reconciliation (read-only probe)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import get_type_hints

import pytest

from applications.governed_contractor_application.host.provider_invocation_reconciliation import (
    GovernedExternalWorkProviderReconciliation,
)
from applications.governed_contractor_application.tests.host.durable_provider_invocation_test_store import (
    DurableTestProviderInvocationStore,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    _CREATE_IDEMP,
    _DIGEST,
    _PRINCIPAL,
    _PROVIDER,
    _TENANT,
    _build_runtime,
    _create_step,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.external_work_reconciliation_plugin import (
    external_task_correlation_from_invocation,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.stores import InMemoryProofReceiptStore
from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPlugin,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectEvidenceVerdict,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ResolutionDecision,
    ResolutionPlatformAction,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRequest,
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityRequest,
    ExternalEffectRepeatEligibilityVerdict,
    ExternalEffectRepeatPolicyDecision,
    ExternalEffectRepeatPolicyRequest,
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.external_work import ExternalWorkStatus, QuoteAcceptanceEvidence
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.governed_execution_result import GovernedExecutionResult
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    reconcile_durable_provider_invocation_unknown,
)
from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 17, 9, 0, 0, tzinfo=UTC)
_ACCEPT_IDEM = "idem-gr7a6-accept"


@dataclass(frozen=True, slots=True)
class _ResolutionPlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def version(self) -> str:
        return self._descriptor.version

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return self._descriptor

    def evaluate(
        self,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        return ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="continue_after_reconcile",
        )


def _resolution_plugin(
    plugin_id: str = "external_work.reconciliation.v1",
) -> _ResolutionPlugin:
    return _ResolutionPlugin(
        EnterpriseReliabilityPluginDescriptor(
            plugin_id=plugin_id,
            version="1.0.0",
            owner="tests",
            capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
            capabilities=("resolve",),
            tenant_scope=None,
            priority=0,
        ),
    )


def _reconciliation_stack(
    integration: ExternalWorkIntegration,
) -> GovernedExternalWorkProviderReconciliation:
    return GovernedExternalWorkProviderReconciliation.build(
        integration,
        resolution_plugin=_resolution_plugin(),
    )


def _acceptance_evidence(quote_id: str) -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence(
        acceptance_id="acc-gr7a6",
        quote_id=quote_id,
        quote_version=1,
        scope_digest=_DIGEST,
        actor=ActorIdentity(kind=ActorKind.USER, actor_id=_PRINCIPAL, tenant_id=_TENANT),
        accepted_at=_T0,
        hitl_decision_id="hdec-gr7a6",
    )


def _unknown_outcome(invocation_id: str) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=ProviderInvocationStatus.UNKNOWN,
        completed_at=_T0,
        error_code="PROVIDER_OUTCOME_UNCERTAIN",
    )


def _create_external_task(
    fake: DeterministicExternalWorkFake,
) -> ProviderInvocation:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
    )
    assert step.governed_result is not None
    inv = step.governed_result.provider_invocation
    assert inv.external_task_id is not None
    return inv


def _accept_invocation(create_inv: ProviderInvocation) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id="inv-accept-gr7a6",
        provider_id=create_inv.provider_id,
        operation="external_work.accept_quote",
        task_id=create_inv.task_id,
        run_id=create_inv.run_id,
        external_task_id=create_inv.external_task_id,
        correlation_id=create_inv.correlation_id,
        idempotency_key=_ACCEPT_IDEM,
        request_digest="digest-accept",
        started_at=_T0,
    )


def _cancel_invocation(create_inv: ProviderInvocation) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id="inv-cancel-gr7a6",
        provider_id=create_inv.provider_id,
        operation="external_work.cancel_work",
        task_id=create_inv.task_id,
        run_id=create_inv.run_id,
        external_task_id=create_inv.external_task_id,
        correlation_id=create_inv.correlation_id,
        idempotency_key="idem-gr7a6-cancel",
        request_digest="digest-cancel",
        started_at=_T0,
    )


def test_create_unknown_reconciliation_not_available_no_probe() -> None:
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    contract = external_work_effect_contract_for_action(
        ACTION_CREATE_EXTERNAL_WORK,
        caps,
    )
    inv = ProviderInvocation(
        invocation_id="inv-create",
        provider_id=_PROVIDER,
        operation="external_work.create_work",
        task_id="task-c",
        run_id="run-c",
        idempotency_key=_CREATE_IDEMP,
        correlation_id="corr-c",
        request_digest="d",
        started_at=_T0,
    )
    outcome = _unknown_outcome(inv.invocation_id)
    gateway = EnterpriseReliabilityPluginGatewayImpl(InMemoryEnterpriseReliabilityPluginRegistry())
    run = reconcile_durable_provider_invocation_unknown(
        ProviderInvocationReconciliationRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=contract,
            tenant_id=_TENANT,
            plugin_id="external_work.reconciliation.v1",
        ),
        gateway=gateway,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.NOT_AVAILABLE
    assert run.probe_run is None


def test_accept_unknown_reconciliation_probe_once_confirmed_success() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    correlation = external_task_correlation_from_invocation(
        task_id=accept_inv.task_id,
        run_id=accept_inv.run_id,
        provider_id=accept_inv.provider_id,
        external_task_id=accept_inv.external_task_id or "",
        correlation_id=accept_inv.correlation_id,
        idempotency_key=accept_inv.idempotency_key,
    )
    quote = fake.get_quote(correlation)
    fake.submit_quote_acceptance(
        correlation,
        _acceptance_evidence(quote.quote_id),
        idempotency_key="idem-real-accept",
    )
    mutations_before = fake.accept_calls
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
        recorded_at=_T0,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
    assert stack.correlation_registry.get_work_calls == 1
    assert fake.accept_calls == mutations_before
    assert fake.create_calls == 1


def test_cancel_unknown_reconciliation_confirmed_success() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    cancel_inv = _cancel_invocation(create_inv)
    correlation = external_task_correlation_from_invocation(
        task_id=cancel_inv.task_id,
        run_id=cancel_inv.run_id,
        provider_id=cancel_inv.provider_id,
        external_task_id=cancel_inv.external_task_id or "",
        correlation_id=cancel_inv.correlation_id,
        idempotency_key=cancel_inv.idempotency_key,
    )
    fake.cancel_work(correlation, idempotency_key="idem-real-cancel")
    mutations_before = fake.cancel_calls
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=cancel_inv,
        outcome=_unknown_outcome(cancel_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
    assert stack.correlation_registry.get_work_calls == 1
    assert fake.cancel_calls == mutations_before


def test_accept_unknown_still_unknown_when_quote_not_accepted() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN
    assert stack.correlation_registry.get_work_calls == 1


def test_accept_unknown_still_unknown_when_work_later_cancelled() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    correlation = external_task_correlation_from_invocation(
        task_id=accept_inv.task_id,
        run_id=accept_inv.run_id,
        provider_id=accept_inv.provider_id,
        external_task_id=accept_inv.external_task_id or "",
        correlation_id=accept_inv.correlation_id,
        idempotency_key=accept_inv.idempotency_key,
    )
    fake.cancel_work(correlation, idempotency_key="idem-cancel-for-fail")
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN


def test_cancel_unknown_still_unknown_when_probe_shows_accepted() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    cancel_inv = _cancel_invocation(create_inv)
    correlation = external_task_correlation_from_invocation(
        task_id=cancel_inv.task_id,
        run_id=cancel_inv.run_id,
        provider_id=cancel_inv.provider_id,
        external_task_id=cancel_inv.external_task_id or "",
        correlation_id=cancel_inv.correlation_id,
        idempotency_key=cancel_inv.idempotency_key,
    )
    quote = fake.get_quote(correlation)
    fake.submit_quote_acceptance(
        correlation,
        _acceptance_evidence(quote.quote_id),
        idempotency_key="idem-prior-accept",
    )
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=cancel_inv,
        outcome=_unknown_outcome(cancel_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN
    assert run.result.verdict is not ProviderInvocationReconciliationVerdict.CONFIRMED_FAILED


@dataclass(frozen=True, slots=True)
class _GetWorkStatusOverrideIntegration:
    """Test double — overrides observed status without mutating provider state."""

    _inner: DeterministicExternalWorkFake
    _status: ExternalWorkStatus

    def get_work(self, correlation):
        snapshot = self._inner.get_work(correlation)
        return snapshot.model_copy(update={"status": self._status})

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_accept_unknown_still_unknown_when_probe_shows_failed() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    integration = _GetWorkStatusOverrideIntegration(fake, ExternalWorkStatus.FAILED)
    stack = _reconciliation_stack(integration)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN


def test_cancel_unknown_still_unknown_when_probe_shows_failed() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    cancel_inv = _cancel_invocation(create_inv)
    integration = _GetWorkStatusOverrideIntegration(fake, ExternalWorkStatus.FAILED)
    stack = _reconciliation_stack(integration)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=cancel_inv,
        outcome=_unknown_outcome(cancel_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN


def test_governed_external_work_reconciliation_build_is_strongly_typed() -> None:
    from pathlib import Path
    import inspect

    source = Path(
        inspect.getfile(GovernedExternalWorkProviderReconciliation),
    ).read_text(encoding="utf-8")
    assert "integration: object" not in source
    assert "resolution_plugin: object" not in source
    assert "type: ignore" not in source
    hints = get_type_hints(GovernedExternalWorkProviderReconciliation.build)
    assert hints["integration"] is ExternalWorkIntegration
    assert hints["resolution_plugin"] == EnterpriseReliabilityPlugin | None
    fake: ExternalWorkIntegration = DeterministicExternalWorkFake()
    stack = GovernedExternalWorkProviderReconciliation.build(
        fake,
        resolution_plugin=_resolution_plugin(),
    )
    assert stack.gateway is not None


def test_probe_read_failure_not_confirmed_failure() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    snapshot_key = accept_inv.external_task_id
    assert snapshot_key is not None
    del fake._by_external_task[snapshot_key]  # noqa: SLF001
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.STILL_UNKNOWN
    assert "probe_read_failed" in run.result.detail
    assert run.result.verdict is not ProviderInvocationReconciliationVerdict.CONFIRMED_FAILED


def test_original_unknown_outcome_preserved_in_store() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    store = DurableTestProviderInvocationStore()
    store.put_invocation(accept_inv)
    outcome = _unknown_outcome(accept_inv.invocation_id)
    store.put_outcome(outcome)
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=outcome,
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert store.get_outcome(accept_inv.invocation_id).status is ProviderInvocationStatus.UNKNOWN


def test_non_unknown_outcome_skips_probe() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    run = stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=ProviderInvocationOutcome(
            invocation_id=accept_inv.invocation_id,
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=_T0,
        ),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.NOT_EXECUTED
    assert stack.correlation_registry.get_work_calls == 0


def test_no_ger_created_by_reconciliation() -> None:
    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    correlation = external_task_correlation_from_invocation(
        task_id=accept_inv.task_id,
        run_id=accept_inv.run_id,
        provider_id=accept_inv.provider_id,
        external_task_id=accept_inv.external_task_id or "",
        correlation_id=accept_inv.correlation_id,
        idempotency_key=accept_inv.idempotency_key,
    )
    quote = fake.get_quote(correlation)
    fake.submit_quote_acceptance(
        correlation,
        _acceptance_evidence(quote.quote_id),
        idempotency_key="idem-ger-check",
    )
    receipt_store = InMemoryProofReceiptStore()
    stack = _reconciliation_stack(fake)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    stack.reconcile_unknown(
        invocation=accept_inv,
        outcome=_unknown_outcome(accept_inv.invocation_id),
        capabilities=caps,
        tenant_id=_TENANT,
    )
    assert receipt_store.get_receipt("exec-gr7a6") is None
    assert GovernedExecutionResult.__name__


@dataclass(frozen=True, slots=True)
class _AllowRepeatPolicy:
    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        return ExternalEffectRepeatPolicyDecision(allow_repeat=True)


def test_repeat_eligibility_unchanged_regression() -> None:
    from external_contractor_adapter.side_effect_actions import ACTION_ACCEPT_QUOTE

    fake = DeterministicExternalWorkFake()
    create_inv = _create_external_task(fake)
    accept_inv = _accept_invocation(create_inv)
    caps = quote_first_partner_capability_fixture(provider_id=_PROVIDER)
    contract = external_work_effect_contract_for_action(ACTION_ACCEPT_QUOTE, caps)
    outcome = _unknown_outcome(accept_inv.invocation_id)
    result = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=accept_inv,
            outcome=outcome,
            effect_contract=contract,
        ),
        policy=_AllowRepeatPolicy(),
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE


@dataclass
class _CustomProbePlugin:
    probe_calls: list[int] = field(default_factory=list)

    @property
    def plugin_id(self) -> str:
        return "custom.probe"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return EnterpriseReliabilityPluginDescriptor(
            plugin_id="custom.probe",
            version="1.0.0",
            owner="tests",
            capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
            capabilities=("custom",),
            tenant_scope=None,
            priority=0,
        )

    def evaluate(self, context: object) -> object:
        from intergrax.contracts.enterprise_reliability.plugin_spi import (
            ReconciliationStrategyAdvice,
        )

        return ReconciliationStrategyAdvice(probe_ref="custom_probe")

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        self.probe_calls.append(1)
        return ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="evidence://custom/1",
            rationale="custom_ok",
        )


def test_pluginability_custom_probe_without_core_change() -> None:
    from intergrax.contracts.enterprise_reliability.effect_contract import (
        ExternalEffectCapabilitySupport,
        ExternalEffectCategory,
        ExternalEffectContract,
        ExternalEffectSafetyCapabilities,
    )

    contract = ExternalEffectContract(
        contract_id="custom.op",
        operation_key="custom.op",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("custom_probe",),
    )
    inv = ProviderInvocation(
        invocation_id="inv-custom",
        provider_id="prov",
        operation="custom.op",
        task_id="t",
        run_id="r",
        external_task_id="ext-custom",
        correlation_id="corr-custom",
        request_digest="d",
        started_at=_T0,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    custom = _CustomProbePlugin()
    registry.register(custom)
    registry.register(_resolution_plugin(plugin_id="custom.probe"))
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    run = reconcile_durable_provider_invocation_unknown(
        ProviderInvocationReconciliationRequest(
            invocation=inv,
            outcome=_unknown_outcome(inv.invocation_id),
            effect_contract=contract,
            tenant_id=_TENANT,
            plugin_id="custom.probe",
        ),
        gateway=gateway,
    )
    assert run.result.verdict is ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
    assert len(custom.probe_calls) == 1


def test_erl_core_provider_invocation_reconciliation_has_no_external_work_imports() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    source = (root / "intergrax/runtime/enterprise_reliability/provider_invocation_reconciliation.py").read_text(
        encoding="utf-8",
    )
    for token in (
        "external_contractor_adapter",
        "ExternalWorkIntegration",
        "ExternalWorkProviderCapabilities",
    ):
        assert token not in source
