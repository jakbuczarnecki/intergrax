# © Artur Czarnecki. All rights reserved.

"""P2.1-S2C4 — delegated provider reattachment and durable lookup."""

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
from intergrax.contracts.delegated_execution_continuation import (
    DelegatedExecutionContinuationOutcomeCategory,
    DelegatedExecutionContinuationOutcomeUnknownError,
    DelegatedExecutionContinuationRequest,
    DelegatedExecutionProviderReattachmentObservation,
    DelegatedExecutionReattachmentKind,
    DelegatedExecutionReattachmentProvider,
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
    DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_NOT_FOUND_MESSAGE,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationNotFoundError,
    DelegatedInvocationCorrelationPersistenceError,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.execution.delegated_execution.continuation_service import (
    DelegatedExecutionContinuationService,
)
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
_CONTINUATION_SERVICE_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "continuation_service.py"
)
_CONTINUATION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_continuation.py"
)
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
        {"operation": "execute_delegate", "task_id": "task-reattach"},
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
    invocation_id: str = "inv-reattach-1",
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
            "provider_request_id": "preq-reattach-1",
            "provider_operation_id": "pop-reattach-1",
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


def _persist_binding(
    binding: DelegatedExecutionInvocationBinding,
) -> DelegatedInvocationCorrelationService:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = DelegatedInvocationCorrelationService(store)
    service.persist_binding(binding, persisted_at=_T0)
    return service


class FakeReattachProvider(
    DelegatedExecutionProvider[EchoPayload, EchoResult],
    DelegatedExecutionReattachmentProvider,
    DelegatedExecutionStatusProvider,
    DelegatedExecutionCancelProvider,
):
    def __init__(
        self,
        *,
        provider_id: str = "reattach_fake",
        kind: DelegatedExecutionReattachmentKind = DelegatedExecutionReattachmentKind.REATTACHED,
        spoof_field: str | None = None,
        raise_transport: bool = False,
        raise_unknown: bool = False,
        supports_reattachment: bool = True,
        physical_ops_created: int = 0,
    ) -> None:
        self._provider_id = provider_id
        self._kind = kind
        self._spoof_field = spoof_field
        self._raise_transport = raise_transport
        self._raise_unknown = raise_unknown
        self._supports_reattachment = supports_reattachment
        self.reattach_calls: list[DelegatedExecutionContinuationRequest] = []
        self.status_calls: list[DelegatedExecutionStatusRequest] = []
        self.cancel_calls: list[DelegatedExecutionControlRequest] = []
        self._physical_ops_created = physical_ops_created

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
            supports_status_read=True,
            supports_reattachment=self._supports_reattachment,
        )

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        raise NotImplementedError

    async def reattach_delegated_execution(
        self,
        request: DelegatedExecutionContinuationRequest,
    ) -> DelegatedExecutionProviderReattachmentObservation:
        self.reattach_calls.append(request)
        if self._physical_ops_created == 0:
            self._physical_ops_created = 1
        if self._raise_transport:
            raise DelegatedExecutionTransportError("transport down")
        if self._raise_unknown:
            raise DelegatedExecutionContinuationOutcomeUnknownError("ambiguous")
        inv = request.provider_invocation
        observation = DelegatedExecutionProviderReattachmentObservation(
            kind=self._kind,
            provider_id=self._provider_id,
            invocation_id=inv.invocation_id,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            physical_status=DelegatedExecutionProviderPhysicalStatus.RUNNING,
            provider_external_status="vendor-attached",
        )
        if self._spoof_field == "invocation_id":
            return DelegatedExecutionProviderReattachmentObservation(
                kind=self._kind,
                provider_id=self._provider_id,
                invocation_id="spoof-inv",
                provider_request_id=inv.provider_request_id,
                provider_operation_id=inv.provider_operation_id,
            )
        if self._spoof_field == "provider_id":
            return DelegatedExecutionProviderReattachmentObservation(
                kind=self._kind,
                provider_id="spoof-provider",
                invocation_id=inv.invocation_id,
                provider_request_id=inv.provider_request_id,
                provider_operation_id=inv.provider_operation_id,
            )
        if self._spoof_field == "provider_request_id":
            return DelegatedExecutionProviderReattachmentObservation(
                kind=self._kind,
                provider_id=self._provider_id,
                invocation_id=inv.invocation_id,
                provider_request_id="spoof-req",
                provider_operation_id=inv.provider_operation_id,
            )
        if self._spoof_field == "provider_operation_id":
            return DelegatedExecutionProviderReattachmentObservation(
                kind=self._kind,
                provider_id=self._provider_id,
                invocation_id=inv.invocation_id,
                provider_request_id=inv.provider_request_id,
                provider_operation_id="spoof-op",
            )
        return observation

    @property
    def physical_operation_count(self) -> int:
        return self._physical_ops_created

    async def read_delegated_execution_status(
        self,
        request: DelegatedExecutionStatusRequest,
    ) -> DelegatedExecutionProviderStatusObservation:
        self.status_calls.append(request)
        inv = request.provider_invocation
        return DelegatedExecutionProviderStatusObservation(
            physical_status=DelegatedExecutionProviderPhysicalStatus.RUNNING,
            provider_id=self._provider_id,
            invocation_id=inv.invocation_id,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
        )

    async def cancel_delegated_execution(
        self,
        request: DelegatedExecutionControlRequest,
    ):
        self.cancel_calls.append(request)
        return delegated_control_outcome(
            category=DelegatedExecutionControlOutcomeCategory.ACCEPTED,
            request=request,
            provider_id=self._provider_id,
        )


def _continuation_service(
    correlation: DelegatedInvocationCorrelationService,
    provider: FakeReattachProvider,
) -> DelegatedExecutionContinuationService:
    resolver = MappingDelegatedExecutionProviderResolver(
        {provider.provider_id: provider},
    )
    return DelegatedExecutionContinuationService(
        correlation,
        resolver,
        clock=lambda: _T1,
    )


@pytest.mark.asyncio
async def test_s2c4_t1_reattach_by_execution_id() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider()
    service = _continuation_service(correlation, provider)
    outcome = await service.reattach_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.REATTACHED
    assert outcome.view is not None
    assert outcome.view.execution_id == binding.execution_id
    assert len(provider.reattach_calls) == 1


@pytest.mark.asyncio
async def test_s2c4_t2_restart_like_reattach() -> None:
    binding = _binding(provider_id="reattach_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    service_a = DelegatedInvocationCorrelationService(store)
    service_a.persist_binding(binding, persisted_at=_T0)
    provider = FakeReattachProvider()
    service_b = DelegatedExecutionContinuationService(
        DelegatedInvocationCorrelationService(store),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service_b.reattach_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.REATTACHED


def test_s2c4_t3_public_api_execution_id_only() -> None:
    sig = inspect.signature(DelegatedExecutionContinuationService.reattach_by_execution_id)
    assert list(sig.parameters) == ["self", "execution_id"]


@pytest.mark.asyncio
async def test_s2c4_t5_correlation_not_found_no_provider() -> None:
    provider = FakeReattachProvider()
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    service = _continuation_service(correlation, provider)
    outcome = await service.reattach_by_execution_id(mint_execution_id())
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.CORRELATION_NOT_FOUND
    assert len(provider.reattach_calls) == 0


@pytest.mark.asyncio
async def test_s2c4_t6_correlation_integrity_no_provider() -> None:
    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationIntegrityError("bad digest")

    provider = FakeReattachProvider()
    service = DelegatedExecutionContinuationService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(mint_execution_id())
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.CORRELATION_INTEGRITY_FAILURE
    )
    assert len(provider.reattach_calls) == 0


@pytest.mark.asyncio
async def test_s2c4_t7_correlation_persistence_no_provider() -> None:
    class BrokenLookup:
        def load_binding_by_execution_id(self, execution_id: object):
            raise DelegatedInvocationCorrelationPersistenceError("store down")

    provider = FakeReattachProvider()
    service = DelegatedExecutionContinuationService(
        BrokenLookup(),
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(mint_execution_id())
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.CORRELATION_PERSISTENCE_UNAVAILABLE
    )
    assert len(provider.reattach_calls) == 0


@pytest.mark.asyncio
async def test_s2c4_t8_provider_unavailable() -> None:
    binding = _binding(provider_id="missing_provider")
    correlation = _persist_binding(binding)
    service = DelegatedExecutionContinuationService(
        correlation,
        MappingDelegatedExecutionProviderResolver({}),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.PROVIDER_UNAVAILABLE


@pytest.mark.asyncio
async def test_s2c4_t9_provider_id_mismatch() -> None:
    binding = _binding(provider_id="expected-provider")
    correlation = _persist_binding(binding)
    wrong = FakeReattachProvider(provider_id="actual-provider")

    class MismatchedResolver(DelegatedExecutionProviderResolver):
        def resolve(self, provider_id: str) -> FakeReattachProvider | None:
            return wrong

    service = DelegatedExecutionContinuationService(
        correlation,
        MismatchedResolver(),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(binding.execution_id)
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.PROVIDER_BINDING_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c4_t10_unsupported_reattachment() -> None:
    binding = _binding(provider_id="no_reattach")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(
        provider_id="no_reattach",
        supports_reattachment=False,
    )
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_s2c4_t11_capability_without_contract() -> None:
    binding = _binding(provider_id="lying_reattach")
    correlation = _persist_binding(binding)

    class LyingProvider(DelegatedExecutionProvider[EchoPayload, EchoResult]):
        @property
        def provider_id(self) -> str:
            return "lying_reattach"

        @property
        def provider_version(self) -> str:
            return "0.0.1"

        @property
        def capabilities(self) -> DelegatedExecutionCapabilities:
            return DelegatedExecutionCapabilities(
                provider_id=self.provider_id,
                supports_reattachment=True,
            )

        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> DelegatedExecutionOutcome[EchoResult]:
            raise NotImplementedError

    provider = LyingProvider()
    service = DelegatedExecutionContinuationService(
        correlation,
        MappingDelegatedExecutionProviderResolver({provider.provider_id: provider}),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(binding.execution_id)
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_CONTRACT_MISSING
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoof_field",
    ["provider_id", "invocation_id", "provider_request_id", "provider_operation_id"],
)
async def test_s2c4_t12_t15_spoof_fail_closed(spoof_field: str) -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(spoof_field=spoof_field)
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_CONTRACT_MISMATCH
    )


@pytest.mark.asyncio
async def test_s2c4_t16_provider_operation_not_found() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(
        kind=DelegatedExecutionReattachmentKind.OPERATION_NOT_FOUND,
    )
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.PROVIDER_OPERATION_NOT_FOUND
    )


@pytest.mark.asyncio
async def test_s2c4_t17_reattach_twice_single_physical_op() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(
        kind=DelegatedExecutionReattachmentKind.ALREADY_ATTACHED,
    )
    service = _continuation_service(correlation, provider)
    first = await service.reattach_by_execution_id(binding.execution_id)
    second = await service.reattach_by_execution_id(binding.execution_id)
    assert first.category is DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED
    assert second.category is DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED
    assert provider.physical_operation_count == 1
    assert len(provider.reattach_calls) == 2


@pytest.mark.asyncio
async def test_s2c4_t18_completed_provider_observation() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(
        kind=DelegatedExecutionReattachmentKind.OBSERVED_TERMINAL,
    )
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.OBSERVED_TERMINAL


@pytest.mark.asyncio
async def test_s2c4_t19_transport_failure() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(raise_transport=True)
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.TRANSPORT_FAILURE


@pytest.mark.asyncio
async def test_s2c4_t20_unknown_outcome() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider(raise_unknown=True)
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert (
        outcome.category
        is DelegatedExecutionContinuationOutcomeCategory.CONTINUATION_OUTCOME_UNKNOWN
    )


@pytest.mark.asyncio
async def test_s2c4_t21_status_after_reattach() -> None:
    binding = _binding(provider_id="reattach_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    provider = FakeReattachProvider()
    resolver = MappingDelegatedExecutionProviderResolver({provider.provider_id: provider})
    continuation = DelegatedExecutionContinuationService(
        correlation,
        resolver,
        clock=lambda: _T1,
    )
    status = DelegatedExecutionStatusReadService(correlation, resolver, clock=lambda: _T1)
    reattach = await continuation.reattach_by_execution_id(binding.execution_id)
    assert reattach.category is DelegatedExecutionContinuationOutcomeCategory.REATTACHED
    read = await status.read_status_by_execution_id(binding.execution_id)
    assert read.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


@pytest.mark.asyncio
async def test_s2c4_t22_control_after_reattach() -> None:
    binding = _binding(provider_id="reattach_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    provider = FakeReattachProvider()
    resolver = MappingDelegatedExecutionProviderResolver({provider.provider_id: provider})
    continuation = DelegatedExecutionContinuationService(
        correlation,
        resolver,
        clock=lambda: _T1,
    )
    control = DelegatedExecutionDurableControlService(correlation, resolver)
    await continuation.reattach_by_execution_id(binding.execution_id)
    outcome = await control.apply_control_by_execution_id(
        binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert outcome.control_outcome is not None
    assert outcome.control_outcome.category is DelegatedExecutionControlOutcomeCategory.ACCEPTED


@pytest.mark.asyncio
async def test_s2c4_t27_correlation_immutable() -> None:
    binding = _binding(provider_id="reattach_fake")
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    correlation.persist_binding(binding, persisted_at=_T0)
    before = correlation.load_binding_by_execution_id(binding.execution_id)
    provider = FakeReattachProvider()
    await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    after = correlation.load_binding_by_execution_id(binding.execution_id)
    assert before.model_dump() == after.model_dump()


@pytest.mark.asyncio
async def test_s2c4_t29_custom_plugin() -> None:
    binding = _binding(provider_id="custom_reattach")
    correlation = _persist_binding(binding)

    class CustomPlugin(FakeReattachProvider):
        pass

    provider = CustomPlugin(provider_id="custom_reattach")
    outcome = await _continuation_service(correlation, provider).reattach_by_execution_id(
        binding.execution_id,
    )
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.REATTACHED


class _CustomResolver(DelegatedExecutionProviderResolver):
    def __init__(self, provider: FakeReattachProvider) -> None:
        self._provider = provider

    def resolve(self, provider_id: str) -> FakeReattachProvider | None:
        if provider_id == self._provider.provider_id:
            return self._provider
        return None


@pytest.mark.asyncio
async def test_s2c4_t30_custom_resolver() -> None:
    binding = _binding(provider_id="reattach_fake")
    correlation = _persist_binding(binding)
    provider = FakeReattachProvider()
    service = DelegatedExecutionContinuationService(
        correlation,
        _CustomResolver(provider),
        clock=lambda: _T1,
    )
    outcome = await service.reattach_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionContinuationOutcomeCategory.REATTACHED


def test_s2c4_t31_no_vendor_import() -> None:
    source = _CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8")
    for token in ("local_provider", "LocalDelegatedExecutionProvider", "agents.", "applications."):
        assert token not in source


def test_s2c4_t32_no_nexus() -> None:
    tree = ast.parse(_CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "intergrax.runtime.nexus" not in module


def test_s2c4_t33_no_reflection() -> None:
    tree = ast.parse(_CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_s2c4_t34_no_global_registry() -> None:
    source = _CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "global " not in source
    assert "Registry" not in source


def test_s2c4_t35_no_any_abi() -> None:
    source = _CONTINUATION_CONTRACT.read_text(encoding="utf-8")
    assert "dict[str, Any]" not in source
    assert ": Any" not in source


def test_s2c4_t39_no_identity_mint_in_service() -> None:
    source = _CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8")
    forbidden = (
        "mint_execution_id",
        "mint_run_id",
        "mint_attempt_id",
        "mint_delegated_execution_invocation_binding",
        "persist_binding",
        "ProviderInvocation(",
    )
    for token in forbidden:
        assert token not in source


def test_s2c4_t40_no_lifecycle_mutation_imports() -> None:
    tree = ast.parse(_CONTINUATION_SERVICE_MODULE.read_text(encoding="utf-8"))
    forbidden_modules = (
        "intergrax.runtime.execution.lifecycle",
        "intergrax.runtime.execution.boundary",
        "intergrax.runtime.execution.child",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for forbidden in forbidden_modules:
                assert forbidden not in module
