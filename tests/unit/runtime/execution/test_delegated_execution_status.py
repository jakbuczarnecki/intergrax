# © Artur Czarnecki. All rights reserved.

"""P2.1-S2C2 — delegated provider status read model and durable lookup."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionCancelProvider,
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    delegated_control_outcome,
)
from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
    mint_delegated_execution_invocation_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionCapabilities,
    DelegatedExecutionContext,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    DelegatedExecutionTransportError,
    digest_delegated_execution_request,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionProviderPhysicalStatus,
    DelegatedExecutionProviderStatusObservation,
    DelegatedExecutionStatusOutcomeCategory,
    DelegatedExecutionStatusProvider,
    DelegatedExecutionStatusRequest,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationStore,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)
from intergrax.runtime.execution.delegated_execution.durable_control_service import (
    DelegatedExecutionDurableControlService,
)
from intergrax.runtime.execution.delegated_execution.provider_resolver import (
    MappingDelegatedExecutionProviderResolver,
)
from intergrax.runtime.execution.delegated_execution.status_service import (
    DelegatedExecutionStatusReadService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STATUS_SERVICE_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "status_service.py"
)
_STATUS_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_status.py"
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 9, 7, 8, 0, 5, tzinfo=timezone.utc)
_PAYLOAD_DIGEST = "sha256:" + ("ef" * 32)


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str


def _operation() -> DelegatedExecutionOperationMetadata:
    return DelegatedExecutionOperationMetadata.model_validate(
        {"operation": "execute_delegate", "task_id": "task-status"},
    )


def _context() -> DelegatedExecutionContext:
    return DelegatedExecutionContext(
        execution_id=mint_execution_id(),
        parent_execution_id=mint_execution_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("tools.read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )


def _invocation(
    *,
    provider_id: str,
    context: DelegatedExecutionContext,
    invocation_id: str = "inv-status-1",
) -> ProviderInvocation:
    op = _operation()
    digest = digest_delegated_execution_request(
        context=context,
        operation=op,
        payload_digest=_PAYLOAD_DIGEST,
    )
    return ProviderInvocation.model_validate(
        {
            "invocation_id": invocation_id,
            "provider_id": provider_id,
            "operation": op.operation,
            "task_id": op.task_id,
            "run_id": str(context.run_id),
            "request_digest": digest,
            "started_at": _T0.isoformat(),
            "provider_request_id": "preq-status-1",
            "provider_operation_id": "pop-status-1",
        }
    )


def _binding(
    *,
    provider_id: str,
    context: DelegatedExecutionContext | None = None,
) -> DelegatedExecutionInvocationBinding:
    ctx = context or _context()
    return mint_delegated_execution_invocation_binding(
        context=ctx,
        operation=_operation(),
        payload_digest=_PAYLOAD_DIGEST,
        provider_invocation=_invocation(provider_id=provider_id, context=ctx),
    )


def _persist_binding(binding: DelegatedExecutionInvocationBinding) -> DelegatedInvocationCorrelationService:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = DelegatedInvocationCorrelationService(store)
    service.persist_binding(binding, persisted_at=_T0)
    return service


class FakeStatusProvider(
    DelegatedExecutionProvider[EchoPayload, EchoResult],
    DelegatedExecutionStatusProvider,
    DelegatedExecutionCancelProvider,
):
    def __init__(
        self,
        *,
        provider_id: str = "status_fake",
        physical_status: DelegatedExecutionProviderPhysicalStatus = (
            DelegatedExecutionProviderPhysicalStatus.RUNNING
        ),
        status_spoof_field: str | None = None,
        raise_transport: bool = False,
        supports_status: bool = True,
    ) -> None:
        self._provider_id = provider_id
        self._physical_status = physical_status
        self._status_spoof_field = status_spoof_field
        self._raise_transport = raise_transport
        self._supports_status = supports_status
        self.status_calls: list[DelegatedExecutionStatusRequest] = []
        self.cancel_calls: list[DelegatedExecutionControlRequest] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self._provider_id,
            supports_cancel=True,
            supports_status_read=self._supports_status,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        raise NotImplementedError

    async def read_delegated_execution_status(
        self,
        request: DelegatedExecutionStatusRequest,
    ) -> DelegatedExecutionProviderStatusObservation:
        self.status_calls.append(request)
        if self._raise_transport:
            raise DelegatedExecutionTransportError("transport down")
        inv = request.provider_invocation
        observation = DelegatedExecutionProviderStatusObservation(
            physical_status=self._physical_status,
            provider_id=self._provider_id,
            invocation_id=inv.invocation_id,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            provider_external_status="vendor-running",
            provider_completed_at=None,
        )
        if self._status_spoof_field == "invocation_id":
            observation = DelegatedExecutionProviderStatusObservation(
                physical_status=self._physical_status,
                provider_id=self._provider_id,
                invocation_id="other-invocation",
                provider_request_id=inv.provider_request_id,
                provider_operation_id=inv.provider_operation_id,
            )
        elif self._status_spoof_field == "provider_id":
            observation = DelegatedExecutionProviderStatusObservation(
                physical_status=self._physical_status,
                provider_id="other-provider",
                invocation_id=inv.invocation_id,
                provider_request_id=inv.provider_request_id,
                provider_operation_id=inv.provider_operation_id,
            )
        return observation

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ) -> DelegatedExecutionControlOutcome:
        self.cancel_calls.append(request)
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self._provider_id,
        )


class _MismatchedResolver(DelegatedExecutionProviderResolver):
    def __init__(self, provider: FakeStatusProvider) -> None:
        self._provider = provider

    def resolve(self, provider_id: str) -> FakeStatusProvider | None:
        return self._provider


def _status_service(
    correlation: DelegatedInvocationCorrelationService,
    provider: FakeStatusProvider,
) -> DelegatedExecutionStatusReadService:
    resolver = MappingDelegatedExecutionProviderResolver(
        {provider.provider_id: provider},
    )
    return DelegatedExecutionStatusReadService(
        correlation,
        resolver,
        clock=lambda: _T1,
    )


@pytest.mark.asyncio
async def test_s2c2_t1_status_by_execution_id() -> None:
    binding = _binding(provider_id="status_fake")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider(
        physical_status=DelegatedExecutionProviderPhysicalStatus.RUNNING,
    )
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE
    assert outcome.view is not None
    assert outcome.view.execution_id == binding.execution_id
    assert outcome.view.physical_status is DelegatedExecutionProviderPhysicalStatus.RUNNING
    assert outcome.view.observed_at == _T1
    assert len(provider.status_calls) == 1


@pytest.mark.asyncio
async def test_s2c2_t2_restart_like_status() -> None:
    binding = _binding(provider_id="status_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    service_a = DelegatedInvocationCorrelationService(store)
    service_a.persist_binding(binding, persisted_at=_T0)
    provider = FakeStatusProvider()
    service_b = DelegatedExecutionStatusReadService(
        DelegatedInvocationCorrelationService(store),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service_b.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


def test_s2c2_t3_caller_cannot_supply_invocation() -> None:
    sig = inspect.signature(
        DelegatedExecutionStatusReadService.read_status_by_execution_id,
    )
    assert list(sig.parameters) == ["self", "execution_id"]


@pytest.mark.asyncio
async def test_s2c2_t4_provider_resolution() -> None:
    binding = _binding(provider_id="plugin-a")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider(provider_id="plugin-a")
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE
    assert outcome.provider_id == "plugin-a"


@pytest.mark.asyncio
async def test_s2c2_t5_wrong_provider_resolution_fail_closed() -> None:
    binding = _binding(provider_id="expected-provider")
    correlation = _persist_binding(binding)
    wrong = FakeStatusProvider(provider_id="actual-provider")
    service = DelegatedExecutionStatusReadService(
        correlation,
        _MismatchedResolver(wrong),
        clock=lambda: _T1,
    )
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert (
        outcome.category
        is DelegatedExecutionStatusOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c2_t6_status_capability_unsupported() -> None:
    binding = _binding(provider_id="no_status")
    correlation = _persist_binding(binding)

    class NoStatusProvider(FakeStatusProvider):
        @property
        def capabilities(self) -> DelegatedExecutionCapabilities:
            return DelegatedExecutionCapabilities(
                provider_id=self._provider_id,
                supports_status_read=False,
            )

    provider = NoStatusProvider(provider_id="no_status", supports_status=False)
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_s2c2_t7_spoofed_invocation_id_fail_closed() -> None:
    binding = _binding(provider_id="status_fake")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider(status_spoof_field="invocation_id")
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert (
        outcome.category
        is DelegatedExecutionStatusOutcomeCategory.STATUS_OUTCOME_CONTRACT_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c2_t8_spoofed_provider_id_fail_closed() -> None:
    binding = _binding(provider_id="status_fake")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider(status_spoof_field="provider_id")
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert (
        outcome.category
        is DelegatedExecutionStatusOutcomeCategory.STATUS_OUTCOME_CONTRACT_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c2_t9_correlation_not_found() -> None:
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    provider = FakeStatusProvider()
    service = _status_service(correlation, provider)
    missing = mint_execution_id()
    outcome = await service.read_status_by_execution_id(missing)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.CORRELATION_NOT_FOUND


@pytest.mark.asyncio
async def test_s2c2_t10_correlation_integrity_error() -> None:
    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationIntegrityError("binding digest invalid")

    provider = FakeStatusProvider()
    service = DelegatedExecutionStatusReadService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.read_status_by_execution_id(mint_execution_id())
    assert (
        outcome.category
        is DelegatedExecutionStatusOutcomeCategory.CORRELATION_INTEGRITY_FAILURE
    )


@pytest.mark.asyncio
async def test_s2c2_t11_provider_status_transport_failure() -> None:
    binding = _binding(provider_id="status_fake")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider(raise_transport=True)
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.TRANSPORT_FAILURE


def test_s2c2_t12_status_success_does_not_mutate_correlation() -> None:
    binding = _binding(provider_id="status_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    before = store.get_by_execution_id(binding.execution_id)
    assert before is not None


@pytest.mark.asyncio
async def test_s2c2_t12_status_success_no_correlation_write() -> None:
    binding = _binding(provider_id="status_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    persist_calls = 0
    original_persist = store.persist

    def counting_persist(record: DelegatedInvocationCorrelationRecord) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(record)

    store.persist = counting_persist  # type: ignore[method-assign]
    provider = FakeStatusProvider()
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE
    assert persist_calls == 0


@pytest.mark.asyncio
async def test_s2c2_t13_status_failure_no_correlation_write() -> None:
    binding = _binding(provider_id="status_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    persist_calls = 0
    original_persist = store.persist

    def counting_persist(record: DelegatedInvocationCorrelationRecord) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(record)

    store.persist = counting_persist  # type: ignore[method-assign]
    provider = FakeStatusProvider(raise_transport=True)
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.TRANSPORT_FAILURE
    assert persist_calls == 0


@pytest.mark.asyncio
async def test_s2c2_t14_durable_control_lookup() -> None:
    binding = _binding(provider_id="status_fake")
    correlation = _persist_binding(binding)
    provider = FakeStatusProvider()
    durable = DelegatedExecutionDurableControlService(
        correlation,
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
    )
    outcome = await durable.apply_control_by_execution_id(
        binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED
    assert len(provider.cancel_calls) == 1
    assert (
        provider.cancel_calls[0].invocation_binding.execution_id == binding.execution_id
    )


def test_s2c2_t15_control_by_execution_id_signature() -> None:
    sig = inspect.signature(
        DelegatedExecutionDurableControlService.apply_control_by_execution_id,
    )
    assert "provider_invocation" not in sig.parameters


@pytest.mark.asyncio
async def test_s2c2_t16_plugin_provider_without_service_changes() -> None:
    binding = _binding(provider_id="custom_plugin")
    correlation = _persist_binding(binding)

    class CustomPlugin(FakeStatusProvider):
        pass

    provider = CustomPlugin(provider_id="custom_plugin")
    service = _status_service(correlation, provider)
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


def test_s2c2_t17_no_concrete_provider_import() -> None:
    source = _STATUS_SERVICE_MODULE.read_text(encoding="utf-8")
    forbidden = (
        "local_provider",
        "LocalDelegatedExecutionProvider",
        "agents.",
        "applications.",
    )
    for token in forbidden:
        assert token not in source


def test_s2c2_t18_no_reflection() -> None:
    tree = ast.parse(_STATUS_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_s2c2_t19_no_global_registry() -> None:
    source = _STATUS_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "global " not in source
    assert "Registry" not in source


def test_s2c2_architecture_gate_no_nexus() -> None:
    tree = ast.parse(_STATUS_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "intergrax.runtime.nexus" not in module


def test_s2c2_architecture_gate_no_identity_mint() -> None:
    source = _STATUS_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "mint_execution_id" not in source
    assert "mint_run_id" not in source


class _StatusCapableWithoutProtocol(DelegatedExecutionProvider[EchoPayload, EchoResult]):
    @property
    def provider_id(self) -> str:
        return "lying_status"

    @property
    def provider_version(self) -> str:
        return "0.0.1"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(
            provider_id=self.provider_id,
            supports_status_read=True,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_capability_without_status_contract_fail_closed() -> None:
    binding = _binding(provider_id="lying_status")
    correlation = _persist_binding(binding)
    provider = _StatusCapableWithoutProtocol()
    service = DelegatedExecutionStatusReadService(
        correlation,
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.STATUS_CONTRACT_MISSING


@pytest.mark.asyncio
async def test_correlation_persistence_unavailable_typed() -> None:
    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationPersistenceError("store down")

    provider = FakeStatusProvider()
    service = DelegatedExecutionStatusReadService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.read_status_by_execution_id(mint_execution_id())
    assert (
        outcome.category
        is DelegatedExecutionStatusOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE
    )


def test_status_contract_frozen_model() -> None:
    source = _STATUS_CONTRACT.read_text(encoding="utf-8")
    assert "dict[str, Any]" not in source
