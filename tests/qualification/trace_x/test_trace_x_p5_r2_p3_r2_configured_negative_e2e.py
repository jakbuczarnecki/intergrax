# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R2 mandatory configured-path negative E2E matrix (15 cases)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from unittest.mock import MagicMock

import pytest

from intergrax.autonomous_work.configured_capability_execution_subject_builder import (
    build_configured_capability_execution_subject,
)
from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
    WorkerConfiguredCapabilityFulfillmentService,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
)
from intergrax.contracts.autonomous_work.worker_configured_capability_fulfillment import (
    WorkerConfiguredCapabilityFulfillmentFailureReason,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    MarketplaceToolExecutionProvenanceKind,
    UcaMarketplaceToolExecutionProvenance,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoptionError,
    ExecutionIntegrationConfigurationAdoptionFailureReason,
    validate_configured_adoption_match,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
    IntegrationMaterializationKind,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    MARKETPLACE_TOOL_EXECUTION_HANDLER_ID,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResolver,
)
from tests.qualification.trace_x._trace_x_p5_r2_p3_r2_configured_negative_support import (
    ConfiguredNegativeExpectation,
    _ATTEMPT_ID,
    _RUN_ID,
    _TENANT,
    _TASK_ID,
    acquisition_decision,
    adoption,
    bound_dispatch,
    configured_intent,
    configured_target,
    opportunity,
    opportunity_tenant_lookup_error,
    realization_error_correlation,
    realization_success,
    configured_subject,
    recovery_outcome,
    uca_target_on_configured_path,
)
from intergrax.contracts.execution_identity import ExecutionId

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def _assert_expectation(exp: ConfiguredNegativeExpectation) -> None:
    assert exp.case_id
    assert exp.detection_layer
    assert exp.disposition_or_reason


@dataclass
class _CountingCatalogInvoker:
    calls: int = 0

    @property
    def caller_agent_id(self) -> str:
        return "agent-test"

    def invoke(self, request):
        self.calls += 1
        return MagicMock(success=False)


@dataclass
class _CountingStageRepo:
    calls: int = 0

    def get(self, *, tenant_id: str, handoff_id: str):
        self.calls += 1
        return None


@dataclass
class _CountingActivation:
    calls: int = 0
    outcome: QualifiedMarketplaceToolActivationOutcome = (
        QualifiedMarketplaceToolActivationOutcome.RESOLUTION_FAILURE
    )
    reason_detail: str = "release_unresolved"

    def ensure_exact_active(self, **_kwargs):
        self.calls += 1
        return MagicMock(
            outcome=self.outcome,
            reason_detail=self.reason_detail,
            registry_tool_id=None,
        )

    def ensure_exact_active_for_identity(self, **_kwargs):
        self.calls += 1
        return MagicMock(
            outcome=self.outcome,
            reason_detail=self.reason_detail,
            registry_tool_id=None,
        )


@dataclass
class _CountingMaterial:
    calls: int = 0

    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        self.calls += 1
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.INVALID,
            reason_detail="governance_deny",
        )


@dataclass
class _RecordingIntentRepo:
    recorded: list[object] = field(default_factory=list)

    def record(self, intent):
        self.recorded.append(intent)
        return MagicMock(outcome=MagicMock())

    def get(self, *, execution_request_id: str):
        if not self.recorded:
            return None
        return self.recorded[0]


def _activation_resolver(counter: _CountingActivation) -> MagicMock:
    resolver = MagicMock(spec=QualifiedMarketplaceToolActivationResolver)
    resolver.ensure_exact_active.side_effect = counter.ensure_exact_active
    resolver.ensure_exact_active_for_identity.side_effect = (
        counter.ensure_exact_active_for_identity
    )
    return resolver


def _handler(
    *,
    intent_repo: _RecordingIntentRepo | None = None,
    activation: _CountingActivation | None = None,
    material: _CountingMaterial | None = None,
    stage_repo: _CountingStageRepo | None = None,
    catalog_invoker: _CountingCatalogInvoker | None = None,
    package_resolver: MagicMock | None = None,
) -> MarketplaceToolQualifiedCapabilityExecutionHandler:
    activation_counter = activation or _CountingActivation()
    return MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repo or _RecordingIntentRepo(),  # pyright: ignore[reportArgumentType]
        stage_repository=stage_repo or _CountingStageRepo(),  # pyright: ignore[reportArgumentType]
        activation_resolver=_activation_resolver(activation_counter),
        material_provider=material or _CountingMaterial(),  # pyright: ignore[reportArgumentType]
        invocation_resolver=MagicMock(),
        catalog_tool_invoker=catalog_invoker or _CountingCatalogInvoker(),  # pyright: ignore[reportArgumentType]
        configured_invocation_projection=MagicMock(),
        package_resolver=package_resolver or MagicMock(),
    )


def test_txp5r2p3r2_neg01_tenant_mismatch_at_subject_builder() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="01",
        detection_layer="ConfiguredCapabilityExecutionSubject builder",
        disposition_or_reason="subject=None → FAIL_CLOSED",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    subject = build_configured_capability_execution_subject(
        tenant_id="other-tenant",
        worker_need_id="need-1",
        recovery_decision_id="recovery-1",
        decision=acquisition_decision(),
        adoption=adoption(),
        selected_operations=("database.query",),
    )
    assert subject is None
    _assert_expectation(exp)


def test_txp5r2p3r2_neg02_missing_capability_identity_key() -> None:
    from intergrax.tools.marketplace_configured_tool_execution_intent_preparation import (
        MarketplaceConfiguredToolExecutionIntentPreparation,
        MarketplaceConfiguredToolExecutionIntentPreparationOutcome,
    )

    exp = ConfiguredNegativeExpectation(
        case_id="02",
        detection_layer="intent preparation (_validate_configured_capability_identity)",
        disposition_or_reason="INTEGRITY_FAILURE",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    skill_identity = CapabilityIdentityKey(
        kind=CapabilityKind.SKILL,
        source_id="skills.local",
        source_kind=CapabilitySourceKind.LOCAL,
        logical_id="skills.data.query",
    )
    subject = configured_subject(capability_identity=skill_identity)
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=MagicMock())
    result = preparation.prepare(
        MagicMock(
            subject=subject,
            execution_request_id="exec-req-1",
            binding_operation_id="bind-1",
            tenant_id=_TENANT,
            task_id=_TASK_ID,
            worker_need_id="need-1",
        ),
    )
    assert (
        result.outcome
        is MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE
    )
    _assert_expectation(exp)


def test_txp5r2p3r2_neg03_opportunity_correlation_tenant_mismatch() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="03",
        detection_layer="WorkerConfiguredCapabilityFulfillmentService opportunity read",
        disposition_or_reason="OPPORTUNITY_TENANT_MISMATCH",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    read = MagicMock()
    read.read_exact.side_effect = opportunity_tenant_lookup_error()
    service = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=MagicMock(),
        principal_binding_resolver=MagicMock(),
    )
    request = MagicMock()
    request.tenant_id = _TENANT
    request.worker_instance_id = MagicMock()
    request.acquisition_request.need.recovery_decision_id = "recovery-1"
    result = service.fulfill_configure_existing(
        request,
        recovery_outcome(correlation_id="corr-mismatch"),
        acquisition_decision(),
    )
    assert result.adoption is None
    assert (
        result.failure_reason
        is WorkerConfiguredCapabilityFulfillmentFailureReason.OPPORTUNITY_TENANT_MISMATCH
    )
    read.read_exact.assert_called_once()
    _assert_expectation(exp)


def test_txp5r2p3r2_neg04_realization_failure() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="04",
        detection_layer="INT-CONFIG realization port",
        disposition_or_reason="REALIZATION_DENIED",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    read = MagicMock()
    read.read_exact.return_value = opportunity()
    realize = MagicMock()
    realize.realize.side_effect = realization_error_correlation()
    principal = MagicMock()
    principal.tenant_id = _TENANT
    principal.principal_id = "p-1"
    resolver = MagicMock()
    resolver.resolve.return_value = principal
    service = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=realize,
        principal_binding_resolver=resolver,
    )
    request = MagicMock()
    request.tenant_id = _TENANT
    request.run_id = None
    request.task_id = _TASK_ID
    request.worker_instance_id = MagicMock()
    request.acquisition_request.need.recovery_decision_id = "recovery-1"
    result = service.fulfill_configure_existing(
        request,
        recovery_outcome(),
        acquisition_decision(),
    )
    assert result.adoption is None
    assert (
        result.failure_reason
        is WorkerConfiguredCapabilityFulfillmentFailureReason.REALIZATION_DENIED
    )
    _assert_expectation(exp)


def test_txp5r2p3r2_neg05_adoption_binding_mismatch() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="05",
        detection_layer="binding continuity validation post-realization",
        disposition_or_reason="BINDING_IDENTITY_MISMATCH",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    read = MagicMock()
    read.read_exact.return_value = opportunity()
    realize = MagicMock()
    bad_binding = replace(
        realization_success().configured_binding,
        provider_id="postgres",
    )
    realize.realize.return_value = replace(
        realization_success(),
        configured_binding=bad_binding,
    )
    principal = MagicMock()
    principal.tenant_id = _TENANT
    principal.principal_id = "p-1"
    resolver = MagicMock()
    resolver.resolve.return_value = principal
    service = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=realize,
        principal_binding_resolver=resolver,
    )
    request = MagicMock()
    request.tenant_id = _TENANT
    request.run_id = None
    request.task_id = _TASK_ID
    request.worker_instance_id = MagicMock()
    request.acquisition_request.need.recovery_decision_id = "recovery-1"
    result = service.fulfill_configure_existing(
        request,
        recovery_outcome(),
        acquisition_decision(),
    )
    assert result.failure_reason is (
        WorkerConfiguredCapabilityFulfillmentFailureReason.BINDING_IDENTITY_MISMATCH
    )
    _assert_expectation(exp)


def test_txp5r2p3r2_neg06_invalid_configured_subject_empty_operations() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="06",
        detection_layer="subject builder",
        disposition_or_reason="subject=None",
        execution_exists=False,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    subject = build_configured_capability_execution_subject(
        tenant_id=_TENANT,
        worker_need_id="need-1",
        recovery_decision_id="recovery-1",
        decision=acquisition_decision(),
        adoption=adoption(),
        selected_operations=(),
    )
    assert subject is None
    _assert_expectation(exp)


def test_txp5r2p3r2_neg07_fake_uca_binding_provider_on_configured_intent() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="07",
        detection_layer="MarketplaceToolQualifiedCapabilityExecutionHandler",
        disposition_or_reason="binding_provider_provenance_mismatch",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent())
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, catalog_invoker=catalog)
    target = configured_target()
    target = target.model_copy(
        update={"binding_provider_id": "uca.fake.provider"},
    )
    result = handler.dispatch_once(
        bound_dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "c" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert result.reason_detail == "binding_provider_provenance_mismatch"
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg08_wrong_execution_handler_id() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="08",
        detection_layer="QualifiedCapabilityExecutionBindingHandlerRegistry",
        disposition_or_reason="execution_handler_unavailable",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    canonical_handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=MagicMock(),
        stage_repository=MagicMock(),
        activation_resolver=MagicMock(),
        material_provider=MagicMock(),
        invocation_resolver=MagicMock(),
        catalog_tool_invoker=MagicMock(),
    )
    registry = QualifiedCapabilityExecutionBindingHandlerRegistry((canonical_handler,))
    wrong_target = configured_target(execution_handler_id="handler.other")
    assert registry.resolve(wrong_target.execution_handler_id) is None
    _assert_expectation(exp)


def test_txp5r2p3r2_neg09_missing_handler_registry_entry() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="09",
        detection_layer="QualifiedCapabilityExecutionBindingHandlerRegistry",
        disposition_or_reason="handler=None",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    registry = QualifiedCapabilityExecutionBindingHandlerRegistry(())
    assert registry.resolve(MARKETPLACE_TOOL_EXECUTION_HANDLER_ID) is None
    _assert_expectation(exp)


def test_txp5r2p3r2_neg10_missing_intent() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="10",
        detection_layer="handler intent repository",
        disposition_or_reason="intent_not_found",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, catalog_invoker=catalog)
    result = handler.dispatch_once(
        bound_dispatch(configured_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "d" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "intent_not_found"
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg11_wrong_provenance_kind_uca_on_configured_target() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="11",
        detection_layer="handler provenance branch",
        disposition_or_reason="binding_provider_provenance_mismatch",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    intent = configured_intent(provenance_kind=MarketplaceToolExecutionProvenanceKind.UCA)
    assert isinstance(intent.provenance, UcaMarketplaceToolExecutionProvenance)
    repo.recorded.append(intent)
    stage = _CountingStageRepo()
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, stage_repo=stage, catalog_invoker=catalog)
    result = handler.dispatch_once(
        bound_dispatch(configured_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "e" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "binding_provider_provenance_mismatch"
    assert stage.calls == 0
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg12_tool_release_unresolved() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="12",
        detection_layer="QualifiedMarketplaceToolActivationResolver",
        disposition_or_reason="release_unresolved",
        execution_exists=True,
        activation_performed=True,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    binding_id = "bind-configured-1"
    repo.recorded.append(configured_intent(binding_operation_id=binding_id))
    activation = _CountingActivation(
        outcome=QualifiedMarketplaceToolActivationOutcome.RESOLUTION_FAILURE,
        reason_detail="release_unresolved",
    )
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, activation=activation, catalog_invoker=catalog)
    result = handler.dispatch_once(
        bound_dispatch(configured_target(binding_operation_id=binding_id)),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "f" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "release_unresolved"
    assert activation.calls == 1
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg13_activation_failure() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="13",
        detection_layer="activation resolver",
        disposition_or_reason="activation_failed",
        execution_exists=True,
        activation_performed=True,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent())
    activation = _CountingActivation(
        outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATION_FAILURE,
        reason_detail="activation_failed",
    )
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, activation=activation, catalog_invoker=catalog)
    result = handler.dispatch_once(
        bound_dispatch(configured_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "0" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "activation_failed"
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg14_governance_material_deny() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="14",
        detection_layer="QualifiedToolInvocationMaterialProvider",
        disposition_or_reason="governance_deny",
        execution_exists=True,
        activation_performed=True,
        provider_materialized=False,
        provider_business_call_count=0,
    )
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent())
    activation = _CountingActivation(
        outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
        reason_detail="activated",
    )

    def _activated(**_kwargs):
        activation.calls += 1
        return MagicMock(
            outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
            registry_tool_id="tool-1",
        )

    activation.ensure_exact_active_for_identity = _activated  # type: ignore[method-assign]
    material = _CountingMaterial()
    catalog = _CountingCatalogInvoker()
    handler = _handler(
        intent_repo=repo,
        activation=activation,
        material=material,
        catalog_invoker=catalog,
    )
    result = handler.dispatch_once(
        bound_dispatch(configured_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "1" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "governance_deny"
    assert material.calls == 1
    assert catalog.calls == 0
    _assert_expectation(exp)


def test_txp5r2p3r2_neg15_configured_effective_provider_mismatch_before_io() -> None:
    exp = ConfiguredNegativeExpectation(
        case_id="15",
        detection_layer="validate_configured_adoption_match (ExecutionBoundIntegrationResolution)",
        disposition_or_reason="CONFIGURED_ADOPTION_PROVIDER_MISMATCH",
        execution_exists=True,
        activation_performed=False,
        provider_materialized=True,
        provider_business_call_count=0,
    )
    effective = EffectiveIntegrationIdentity(
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="postgres",
        materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
    )
    with pytest.raises(ExecutionIntegrationConfigurationAdoptionError) as exc:
        validate_configured_adoption_match(
            adoption=adoption(),
            effective=effective,
            expected_tenant_id=_TENANT,
        )
    assert (
        exc.value.reason
        is ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_PROVIDER_MISMATCH
    )
    _assert_expectation(exp)


def test_txp5r2p3r2_neg_fail_closed_no_uca_stage_fallback_on_configured_target() -> None:
    """Configured target + UCA binding provider must not call stage repository."""
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent())
    stage = _CountingStageRepo()
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, stage_repo=stage, catalog_invoker=catalog)
    handler.dispatch_once(
        bound_dispatch(uca_target_on_configured_path()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "2" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert stage.calls == 0
    assert catalog.calls == 0


def test_txp5r2p3r2_neg_corrupt_intent_fail_closed() -> None:
    repo = MagicMock()
    repo.get.side_effect = QualifiedMarketplaceToolExecutionIntentIntegrityError("corrupt")
    catalog = _CountingCatalogInvoker()
    handler = _handler(intent_repo=repo, catalog_invoker=catalog)
    result = handler.dispatch_once(
        bound_dispatch(configured_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=ExecutionId("execution_" + "3" * 24),
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "intent_corrupt"
    assert catalog.calls == 0
