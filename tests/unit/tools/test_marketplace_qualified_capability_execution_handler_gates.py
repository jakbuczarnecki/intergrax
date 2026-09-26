# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P3 handler integrity, store, and material failure gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
    QualifiedMarketplaceToolExecutionIntentWriteResult,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionService
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
    execution_target_reference_for_marketplace_qualified_tool,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver
from testing_support.canonical_me14_echo_tool import ME14_VERSION_V1
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import _release
from tests.unit.tools.test_marketplace_qualified_capability_execution_handler import (
    _Input,
    _Invoker,
)
from tests.unit.tools.test_qualified_marketplace_tool_activation_resolver import (
    _stage as me14_stage,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_00000000000000000000000000000001")
_RUN_ID = RunId("run_" + "a" * 32)
_ATTEMPT_ID = AttemptId("attempt_" + "b" * 28)
_EXECUTION_ID = ExecutionId("execution_" + "c" * 24)
_HOST = "host-gates-me14"


def _intent(**overrides) -> QualifiedMarketplaceToolExecutionIntent:
    base = QualifiedMarketplaceToolExecutionIntent(
        execution_request_id="exec-req-1",
        binding_operation_id="bind-1",
        resume_operation_id="resume-1",
        tenant_id="tenant-1",
        task_id=str(_TASK_ID),
        worker_need_id="worker-need-1",
        qualified_subject_reference="qualified-capability-subject:q:domain_handoff_reference:h",
        handoff_id="handoff-1",
        selected_operation="invoke",
    )
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _target(**overrides) -> QualifiedCapabilityExecutionTarget:
    base = QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-1",
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=_intent().qualified_subject_reference,
    )
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _dispatch(
    target: QualifiedCapabilityExecutionTarget,
    *,
    execution_request_id: str = "exec-req-1",
    tenant_id: str = "tenant-1",
) -> BoundCapabilityExecutionDispatchRequest:
    return BoundCapabilityExecutionDispatchRequest(
        execution_request_id=execution_request_id,
        execution_target=target,
        tenant_id=tenant_id,
        task_id=_TASK_ID,
    )


@dataclass
class _CountingActivation:
    calls: int = 0
    outcome: QualifiedMarketplaceToolActivationOutcome = (
        QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT
    )
    registry_tool_id: str = "tool-active"

    def ensure_exact_active(self, *, stage, execution_request_id: str):
        self.calls += 1
        from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
            QualifiedMarketplaceToolActivationResult,
        )

        return QualifiedMarketplaceToolActivationResult(
            outcome=self.outcome,
            registry_tool_id=self.registry_tool_id,
        )


@dataclass
class _ConfigurableMaterial:
    outcome: QualifiedToolInvocationMaterialOutcome = (
        QualifiedToolInvocationMaterialOutcome.AVAILABLE
    )
    material: BaseModel | None = field(default_factory=_Input)

    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=self.outcome,
            material=self.material,
            reason_detail="gate",
        )


@dataclass
class _RecordingInvocationResolver:
    calls: int = 0
    last_selected_operation: str = ""

    def resolve(self, **kwargs) -> ExecutionBoundCatalogToolInvokeRequest:
        self.calls += 1
        self.last_selected_operation = kwargs["selected_operation"]
        return DefaultQualifiedToolInvocationResolver().resolve(**kwargs)


def _handler_with(
    *,
    intent: QualifiedMarketplaceToolExecutionIntent | None,
    intent_repo=None,
    stage_repo=None,
    activation: _CountingActivation | QualifiedMarketplaceToolActivationResolver | None = None,
    material: _ConfigurableMaterial | None = None,
    invocation: _RecordingInvocationResolver | None = None,
    invoker: _Invoker | None = None,
) -> tuple[
    MarketplaceToolQualifiedCapabilityExecutionHandler,
    _Invoker,
    _CountingActivation | QualifiedMarketplaceToolActivationResolver,
]:
    store = InMemoryDocumentStore()
    intent_repository = intent_repo or DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(
        store,
    )
    stage_repository = stage_repo or DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    if intent is not None and intent_repo is None:
        intent_repository.record(intent)
        from intergrax.contracts.marketplace.handoff_traceability import (
            CapabilityHandoffConsumerTarget,
        )

        stage_repository.stage(
            MarketplaceQualifiedToolStage(
                handoff_id=intent.handoff_id,
                tenant_id=intent.tenant_id,
                selected_release=_release(),
                discovery_correlation_id="discovery-1",
                selection_id="selection-1",
                consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
                downstream_consumer_id="tool.qualification_staging.v1",
                recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=UTC),
            ),
        )
    activation_resolver = activation or _CountingActivation()
    inv = invoker or _Invoker()
    handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repository,
        stage_repository=stage_repository,
        activation_resolver=activation_resolver,
        material_provider=material or _ConfigurableMaterial(),
        invocation_resolver=invocation or DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=inv,
    )
    return handler, inv, activation_resolver


def _me14_activation_resolver() -> QualifiedMarketplaceToolActivationResolver:
    lifecycle = ToolHostLifecycleService(host_profile_id=_HOST)
    provider = Me14ToolCatalogProvider()
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    acquisition = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {provider.catalog_source_id: provider},
        ),
        activation=lifecycle,
        materializer=materializer,
    )
    return QualifiedMarketplaceToolActivationResolver(
        activation_read=lifecycle,
        acquisition=acquisition,
        host_profile_id=_HOST,
    )


def test_wrong_binding_provider_id() -> None:
    handler, invoker, activation = _handler_with(intent=_intent())
    target = _target(binding_provider_id="wrong.provider")
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_invalid_execution_target_reference() -> None:
    handler, invoker, activation = _handler_with(intent=_intent())
    target = _target(execution_target_reference="not-a-handoff-ref")
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_intent_tenant_mismatch() -> None:
    handler, invoker, activation = _handler_with(intent=_intent(tenant_id="tenant-1"))
    result = handler.dispatch_once(
        _dispatch(_target(), tenant_id="tenant-other"),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_intent_task_mismatch() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(task_id="task_00000000000000000000000000000099"),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_intent_subject_mismatch() -> None:
    handler, invoker, activation = _handler_with(intent=_intent())
    target = _target(qualified_subject_reference="qualified-capability-subject:other")
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_intent_handoff_mismatch() -> None:
    handler, invoker, activation = _handler_with(intent=_intent(handoff_id="handoff-1"))
    target = _target(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-other",
        ),
    )
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_stage_missing() -> None:
    handler, invoker, activation = _handler_with(intent=_intent(), stage_repo=_EmptyStageRepo())
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


@dataclass
class _EmptyStageRepo:
    def stage(self, record: MarketplaceQualifiedToolStage) -> None:
        pass

    def get(self, *, tenant_id: str, handoff_id: str) -> MarketplaceQualifiedToolStage | None:
        return None


@dataclass
class _UnavailableIntentRepo:
    def record(self, intent):
        return QualifiedMarketplaceToolExecutionIntentWriteResult(
            outcome=QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED,
        )

    def get(self, *, execution_request_id: str):
        raise QualifiedMarketplaceToolExecutionIntentUnavailableError("down")


@dataclass
class _CorruptIntentRepo:
    def record(self, intent):
        return QualifiedMarketplaceToolExecutionIntentWriteResult(
            outcome=QualifiedMarketplaceToolExecutionIntentWriteOutcome.CREATED,
        )

    def get(self, *, execution_request_id: str):
        raise QualifiedMarketplaceToolExecutionIntentIntegrityError("corrupt")


@dataclass
class _UnavailableStageRepo:
    def stage(self, record: MarketplaceQualifiedToolStage) -> None:
        pass

    def get(self, *, tenant_id: str, handoff_id: str):
        raise MarketplaceQualifiedToolStageUnavailableError("down")


@dataclass
class _CorruptStageRepo:
    def stage(self, record: MarketplaceQualifiedToolStage) -> None:
        pass

    def get(self, *, tenant_id: str, handoff_id: str):
        raise MarketplaceQualifiedToolStageIntegrityError("corrupt")


def test_intent_repository_unavailable() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        intent_repo=_UnavailableIntentRepo(),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_intent_repository_integrity() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        intent_repo=_CorruptIntentRepo(),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_stage_repository_unavailable() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        stage_repo=_UnavailableStageRepo(),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_stage_repository_integrity() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        stage_repo=_CorruptStageRepo(),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 0


def test_material_unavailable() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        material=_ConfigurableMaterial(
            outcome=QualifiedToolInvocationMaterialOutcome.UNAVAILABLE,
        ),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 1


def test_material_invalid() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        material=_ConfigurableMaterial(
            outcome=QualifiedToolInvocationMaterialOutcome.INVALID,
        ),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 1


def test_material_available_but_none_failed() -> None:
    handler, invoker, activation = _handler_with(
        intent=_intent(),
        material=_ConfigurableMaterial(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=None,
        ),
    )
    result = handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0
    assert isinstance(activation, _CountingActivation) and activation.calls == 1


def test_active_different_release_failed_no_tool_runtime() -> None:
    store = InMemoryDocumentStore()
    intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    handoff_id = "handoff-me14"
    subject_ref = "qualified-capability-subject:q:domain_handoff_reference:me14"
    intent = _intent(
        handoff_id=handoff_id,
        tenant_id="tenant-a",
        qualified_subject_reference=subject_ref,
    )
    intent_repo.record(intent)
    activation = _me14_activation_resolver()
    activation.ensure_exact_active(stage=me14_stage(), execution_request_id="warmup")
    conflicting_stage = me14_stage(version_label=ME14_VERSION_V1 + "-other")

    @dataclass
    class _ConflictingStageRepo:
        def stage(self, record: MarketplaceQualifiedToolStage) -> None:
            pass

        def get(self, *, tenant_id: str, handoff_id: str) -> MarketplaceQualifiedToolStage:
            return conflicting_stage

    invoker = _Invoker()
    handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repo,
        stage_repository=_ConflictingStageRepo(),
        activation_resolver=activation,
        material_provider=_ConfigurableMaterial(),
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=invoker,
    )
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            handoff_id,
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=subject_ref,
    )
    result = handler.dispatch_once(
        _dispatch(target, tenant_id="tenant-a"),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0


def test_invocation_resolver_receives_semantic_operation_and_fields() -> None:
    recording = _RecordingInvocationResolver()
    handler, invoker, _ = _handler_with(intent=_intent(selected_operation="invoke"), invocation=recording)
    handler.dispatch_once(
        _dispatch(_target()),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert recording.calls == 1
    assert recording.last_selected_operation == "invoke"
    assert invoker.calls == 1
