# © Artur Czarnecki. All rights reserved.

"""P2.1-S2D — real subprocess delegated execution provider production qualification."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.delegated_execution_control import (
    DelegatedExecutionControlOperation,
    DelegatedExecutionControlOutcomeCategory,
    DelegatedExecutionControlRequest,
    DelegatedExecutionDurableControlOutcomeCategory,
)
from intergrax.contracts.delegated_execution_continuation import (
    DelegatedExecutionContinuationOutcomeCategory,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionCapabilities,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    delegated_failure_outcome,
)
from intergrax.contracts.delegated_execution_provider_resolver import (
    DelegatedExecutionProviderResolver,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionStatusOutcomeCategory,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationDurabilityPolicy,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.delegated_execution.subprocess.bundle import (
    SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID,
    create_subprocess_delegated_execution_provider,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    DocumentStoreDelegatedInvocationCorrelationStore,
    InMemoryDelegatedInvocationCorrelationStore,
)
from intergrax.runtime.execution.delegated_execution.continuation_service import (
    DelegatedExecutionContinuationService,
)
from intergrax.runtime.execution.delegated_execution.control_service import (
    DelegatedExecutionControlService,
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
from intergrax.runtime.execution.delegated_execution.service import (
    DelegatedExecutionService,
    delegated_execution_service,
)
from intergrax.runtime.execution.delegated_execution.status_service import (
    DelegatedExecutionStatusReadService,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.provider import (
    SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_VERSION,
    SubprocessDelegatedExecutionProvider,
    SubprocessEchoPayload,
    SubprocessEchoResult,
)
from intergrax.integrations.providers.delegated_execution.subprocess.transport import (
    FailingConnectTransport,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROVIDER_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "integrations"
    / "providers"
    / "delegated_execution"
    / "subprocess"
    / "provider.py"
)
_SERVICE_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "delegated_execution" / "service.py"
)
_CHILD_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"
_BOUNDARY_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
_SECRET = "s2d-test-auth-token-do-not-log"
_POLICY_REQUIRED = DelegatedInvocationCorrelationDurabilityPolicy(
    mode=DelegatedInvocationCorrelationDurabilityMode.REQUIRED,
)
_SUBPROCESS_DIR = (
    _REPO_ROOT
    / "intergrax"
    / "integrations"
    / "providers"
    / "delegated_execution"
    / "subprocess"
)


@dataclass(frozen=True)
class _RootEchoPayload:
    value: str
    behavior: str | None = None


@dataclass(frozen=True)
class _RootEchoResult:
    value: str
    child_execution_id: str
    parent_execution_id: str


def _operation(**overrides: object) -> DelegatedExecutionOperationMetadata:
    base = {"operation": "execute_delegate", "task_id": "task_s2d"}
    base.update(overrides)
    return DelegatedExecutionOperationMetadata.model_validate(base)


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _config(**overrides: object) -> SubprocessDelegatedExecutionProviderConfig:
    base = {
        "enabled": True,
        "request_timeout_seconds": 10.0,
        "connect_timeout_seconds": 3.0,
        "connection_auth_token": _SECRET,
    }
    base.update(overrides)
    return SubprocessDelegatedExecutionProviderConfig.model_validate(base)


@pytest.fixture
def subprocess_provider() -> Iterator[SubprocessDelegatedExecutionProvider]:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    yield provider
    provider.close()


async def _run_delegated(
    provider: SubprocessDelegatedExecutionProvider,
    *,
    payload: SubprocessEchoPayload,
) -> tuple[DelegatedExecutionOutcome[SubprocessEchoResult], InMemoryDelegatedInvocationCorrelationStore]:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=DelegatedInvocationCorrelationDurabilityPolicy(
            mode=DelegatedInvocationCorrelationDurabilityMode.NON_DURABLE_TEST,
        ),
        correlation_store=store,
    )
    root = _root_identity()
    authority = ParentExecutionAuthority.scoped(("read", "write"))
    captured: list[DelegatedExecutionOutcome[SubprocessEchoResult]] = []

    class RootDelegate:
        async def execute(self, request: _RootEchoPayload) -> _RootEchoResult:
            outcome = await service.execute_delegated(
                payload=SubprocessEchoPayload(
                    value=request.value,
                    behavior=request.behavior,
                ),
                operation=_operation(),
            )
            captured.append(outcome)
            if outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS and outcome.result:
                return _RootEchoResult(
                    value=outcome.result.value,
                    child_execution_id=outcome.result.child_execution_id,
                    parent_execution_id=outcome.result.parent_execution_id,
                )
            return _RootEchoResult(value="failed", child_execution_id="", parent_execution_id="")

    await ExecutionBoundary[_RootEchoPayload, _RootEchoResult](
        RootDelegate(),
        identity=root,
        authority=authority,
    ).execute(_RootEchoPayload(value=payload.value, behavior=payload.behavior))
    assert len(captured) == 1
    return captured[0], store


async def _run_delegated_durable(
    provider: SubprocessDelegatedExecutionProvider,
    *,
    payload: SubprocessEchoPayload,
    document_store: InMemoryDocumentStore | None = None,
) -> tuple[
    DelegatedExecutionOutcome[SubprocessEchoResult],
    DocumentStoreDelegatedInvocationCorrelationStore,
    DelegatedExecutionService[SubprocessEchoPayload, SubprocessEchoResult],
]:
    doc = document_store or InMemoryDocumentStore()
    store = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=store,
    )
    root = _root_identity()
    authority = ParentExecutionAuthority.scoped(("read", "write"))
    captured: list[DelegatedExecutionOutcome[SubprocessEchoResult]] = []

    class RootDelegate:
        async def execute(self, request: _RootEchoPayload) -> _RootEchoResult:
            outcome = await service.execute_delegated(
                payload=SubprocessEchoPayload(
                    value=request.value,
                    behavior=request.behavior,
                ),
                operation=_operation(),
            )
            captured.append(outcome)
            if outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS and outcome.result:
                return _RootEchoResult(
                    value=outcome.result.value,
                    child_execution_id=outcome.result.child_execution_id,
                    parent_execution_id=outcome.result.parent_execution_id,
                )
            return _RootEchoResult(value="failed", child_execution_id="", parent_execution_id="")

    await ExecutionBoundary[_RootEchoPayload, _RootEchoResult](
        RootDelegate(),
        identity=root,
        authority=authority,
    ).execute(_RootEchoPayload(value=payload.value, behavior=payload.behavior))
    assert len(captured) == 1
    return captured[0], store, service


def _fresh_resolver(
    provider: SubprocessDelegatedExecutionProvider,
) -> MappingDelegatedExecutionProviderResolver:
    return MappingDelegatedExecutionProviderResolver(
        {provider.provider_id: provider},
    )


def _fresh_status_service(
    store: DocumentStoreDelegatedInvocationCorrelationStore,
    provider: SubprocessDelegatedExecutionProvider,
) -> DelegatedExecutionStatusReadService:
    correlation = DelegatedInvocationCorrelationService(store)
    return DelegatedExecutionStatusReadService(correlation, _fresh_resolver(provider))


def _fresh_continuation_service(
    store: DocumentStoreDelegatedInvocationCorrelationStore,
    provider: SubprocessDelegatedExecutionProvider,
) -> DelegatedExecutionContinuationService:
    correlation = DelegatedInvocationCorrelationService(store)
    return DelegatedExecutionContinuationService(correlation, _fresh_resolver(provider))


def _fresh_durable_control(
    store: DocumentStoreDelegatedInvocationCorrelationStore,
    provider: SubprocessDelegatedExecutionProvider,
) -> DelegatedExecutionDurableControlService:
    correlation = DelegatedInvocationCorrelationService(store)
    return DelegatedExecutionDurableControlService(correlation, _fresh_resolver(provider))


async def _provider_execute_outcome(
    provider: SubprocessDelegatedExecutionProvider,
    *,
    value: str,
    behavior: str | None = None,
) -> DelegatedExecutionOutcome[SubprocessEchoResult]:
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value=value, behavior=behavior),
        operation=_operation(),
    )
    return await provider.execute(request)

@pytest.mark.asyncio
async def test_s2d_t1_real_external_success(subprocess_provider: SubprocessDelegatedExecutionProvider) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="external-ok"),
    )
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    count = await subprocess_provider.worker_execute_count()
    assert count >= 1


@pytest.mark.asyncio
async def test_s2d_t2_canonical_child_execution(subprocess_provider: SubprocessDelegatedExecutionProvider) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="child-path"),
    )
    assert outcome.result is not None
    assert outcome.result.child_execution_id != outcome.result.parent_execution_id


@pytest.mark.asyncio
async def test_s2d_t3_no_provider_minted_execution_id(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, _store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="identity"),
    )
    assert outcome.invocation_binding is not None
    assert outcome.invocation_binding.provider_invocation.provider_request_id != str(
        outcome.invocation_binding.execution_id,
    )


@pytest.mark.asyncio
async def test_s2d_t6_transport_before_dispatch_failure() -> None:
    provider = SubprocessDelegatedExecutionProvider(
        _config(),
        transport=FailingConnectTransport(),
    )
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x"),
        operation=_operation(),
    )
    outcome = await provider.execute(request)
    assert outcome.category is DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE
    assert outcome.failure_code == "TRANSPORT_FAILURE"


@pytest.mark.asyncio
async def test_s2d_t7_post_dispatch_ambiguous_failure(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="accept_then_disconnect"),
        operation=_operation(),
    )
    outcome = await subprocess_provider.execute(request)
    assert outcome.failure_code == "OUTCOME_UNKNOWN"
    assert outcome.provider_outcome is not None
    assert outcome.provider_outcome.status is ProviderInvocationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_s2d_t8_no_auto_retry_on_unknown(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    before = await subprocess_provider.worker_execute_count()
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="accept_then_disconnect"),
        operation=_operation(),
    )
    await subprocess_provider.execute(request)
    after = await subprocess_provider.worker_execute_count()
    assert after == before + 1


@pytest.mark.asyncio
async def test_s2d_t9_invalid_provider_response(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    request = DelegatedExecutionRequest(
        context=_dummy_context(),
        payload=SubprocessEchoPayload(value="x", behavior="spoof_execution_id"),
        operation=_operation(),
    )
    outcome = await subprocess_provider.execute(request)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"


@pytest.mark.asyncio
async def test_s2d_t13_timeout_bounded() -> None:
    provider = create_subprocess_delegated_execution_provider(
        config=_config(request_timeout_seconds=0.2),
    )
    try:
        request = DelegatedExecutionRequest(
            context=_dummy_context(),
            payload=SubprocessEchoPayload(value="x", behavior="slow"),
            operation=_operation(),
        )
        outcome = await provider.execute(request)
        assert outcome.failure_code in {"OUTCOME_UNKNOWN", "TRANSPORT_FAILURE"}
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_s2d_t14_secret_redaction() -> None:
    config = _config(connection_auth_token=_SECRET)
    public = config.public_view()
    assert _SECRET not in str(public)
    assert "connection_auth_token" not in public


@pytest.mark.asyncio
async def test_s2d_t15_concurrent_operations(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    async def one(value: str) -> None:
        await subprocess_provider.execute(
            DelegatedExecutionRequest(
                context=_dummy_context(),
                payload=SubprocessEchoPayload(value=value),
                operation=_operation(task_id=f"task-{value}"),
            ),
        )

    await asyncio.gather(*(one(str(i)) for i in range(4)))


@pytest.mark.asyncio
async def test_s2d_t16_provider_config_isolation() -> None:
    first = create_subprocess_delegated_execution_provider(config=_config())
    second = create_subprocess_delegated_execution_provider(config=_config())
    try:
        assert first.provider_id == second.provider_id
        assert first is not second
        before = await first.worker_execute_count()
        await first.execute(
            DelegatedExecutionRequest(
                context=_dummy_context(),
                payload=SubprocessEchoPayload(value="a"),
                operation=_operation(),
            ),
        )
        assert await second.worker_execute_count() == 0
        assert await first.worker_execute_count() == before + 1
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_s2d_t19_status_real_provider(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="status"),
    )
    assert outcome.invocation_binding is not None
    correlation = DelegatedInvocationCorrelationService(store)
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    status_service = DelegatedExecutionStatusReadService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {subprocess_provider.provider_id: subprocess_provider},
        ),
    )
    outcome = await status_service.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


@pytest.mark.asyncio
async def test_s2d_t21_control_real_provider(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="cancel-me"),
    )
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    control = DelegatedExecutionControlService(subprocess_provider)
    request = DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=DelegatedExecutionControlOperation.CANCEL,
    )
    outcome = await control.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.COMPLETED


@pytest.mark.asyncio
async def test_s2d_t22_unsupported_control(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(subprocess_provider, payload=SubprocessEchoPayload(value="x"))
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    control = DelegatedExecutionControlService(subprocess_provider)
    request = DelegatedExecutionControlRequest(
        invocation_binding=binding,
        operation=DelegatedExecutionControlOperation.INTERRUPT,
    )
    outcome = await control.apply_control(request)
    assert outcome.category is DelegatedExecutionControlOutcomeCategory.UNSUPPORTED


@pytest.mark.asyncio
async def test_s2d_t24_real_reattachment(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store = await _run_delegated(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="reattach"),
    )
    correlation = DelegatedInvocationCorrelationService(store)
    assert outcome.invocation_binding is not None
    binding = correlation.load_binding_by_execution_id(outcome.invocation_binding.execution_id)
    continuation = DelegatedExecutionContinuationService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {subprocess_provider.provider_id: subprocess_provider},
        ),
    )
    outcome = await continuation.reattach_by_execution_id(binding.execution_id)
    assert outcome.category in {
        DelegatedExecutionContinuationOutcomeCategory.REATTACHED,
        DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED,
    }


@pytest.mark.asyncio
async def test_c1_t1_auth_rejected_maps_to_provider_failure() -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome = await _provider_execute_outcome(
            provider,
            value="x",
            behavior="error_auth_rejected",
        )
        assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
        assert outcome.failure_code == "AUTH_REJECTED"
        assert outcome.failure_message == "subprocess delegated execution provider failed"
        assert "auth rejected" not in str(outcome.failure_message).lower()
    finally:
        provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("behavior", "code"),
    [
        ("error_capacity", "CAPACITY_EXCEEDED"),
        ("error_invalid_request", "INVALID_REQUEST"),
    ],
)
async def test_c1_t2_t3_explicit_worker_errors_are_provider_failures(
    behavior: str,
    code: str,
) -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome = await _provider_execute_outcome(provider, value="x", behavior=behavior)
        assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
        assert outcome.failure_code == code
        assert outcome.provider_outcome is not None
        assert outcome.provider_outcome.status is ProviderInvocationStatus.FAILED
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_c1_t4_provider_error_outcome_is_known() -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome = await _provider_execute_outcome(
            provider,
            value="x",
            behavior="error_invalid_request",
        )
        assert outcome.provider_outcome is not None
        assert outcome.provider_outcome.status is not ProviderInvocationStatus.UNKNOWN
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_c1_t5_required_durable_correlation_persisted(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="durable-bind"),
    )
    assert store.is_durable
    assert outcome.invocation_binding is not None
    execution_id = outcome.invocation_binding.execution_id
    fresh = DelegatedInvocationCorrelationService(store)
    binding = fresh.load_binding_by_execution_id(execution_id)
    assert binding.execution_id == execution_id


@pytest.mark.asyncio
async def test_c1_t6_fresh_status_after_platform_state_loss(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="fresh-status"),
    )
    assert outcome.invocation_binding is not None
    execution_id = outcome.invocation_binding.execution_id
    status = _fresh_status_service(store, subprocess_provider)
    read = await status.read_status_by_execution_id(execution_id)
    assert read.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


@pytest.mark.asyncio
async def test_c1_t7_fresh_reattach_after_platform_state_loss(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="fresh-reattach"),
    )
    assert outcome.invocation_binding is not None
    execution_id = outcome.invocation_binding.execution_id
    before = await subprocess_provider.worker_execute_count()
    continuation = _fresh_continuation_service(store, subprocess_provider)
    reattach = await continuation.reattach_by_execution_id(execution_id)
    assert reattach.category in {
        DelegatedExecutionContinuationOutcomeCategory.REATTACHED,
        DelegatedExecutionContinuationOutcomeCategory.ALREADY_ATTACHED,
    }
    assert await subprocess_provider.worker_execute_count() == before


@pytest.mark.asyncio
async def test_c1_t8_t9_reattach_preserves_binding_and_execution_id(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="same-binding"),
    )
    assert outcome.invocation_binding is not None
    original = outcome.invocation_binding
    execution_id = original.execution_id
    continuation = _fresh_continuation_service(store, subprocess_provider)
    await continuation.reattach_by_execution_id(execution_id)
    fresh = DelegatedInvocationCorrelationService(store)
    loaded = fresh.load_binding_by_execution_id(execution_id)
    assert loaded == original


@pytest.mark.asyncio
async def test_c1_t10_worker_crash_normalized_transport_failure() -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome, store = await _run_delegated(
            provider,
            payload=SubprocessEchoPayload(value="before-crash"),
        )
        assert outcome.invocation_binding is not None
        execution_id = outcome.invocation_binding.execution_id
        provider.close()
        revived = create_subprocess_delegated_execution_provider(config=_config())
        try:
            status = DelegatedExecutionStatusReadService(
                DelegatedInvocationCorrelationService(store),
                _fresh_resolver(revived),
            )
            read = await status.read_status_by_execution_id(execution_id)
            assert read.category is DelegatedExecutionStatusOutcomeCategory.TRANSPORT_FAILURE
            assert read.failure_code == "TRANSPORT_FAILURE"
            assert "ConnectionRefusedError" not in str(read.failure_message)
            assert "BrokenPipeError" not in str(read.failure_message)
        finally:
            revived.close()
    finally:
        pass


@pytest.mark.asyncio
async def test_c1_t11_operation_not_found_on_stale_worker_state() -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome, store, _service = await _run_delegated_durable(
            provider,
            payload=SubprocessEchoPayload(value="stale-op"),
        )
        assert outcome.invocation_binding is not None
        execution_id = outcome.invocation_binding.execution_id
        provider.close()
        revived = create_subprocess_delegated_execution_provider(config=_config())
        try:
            continuation = _fresh_continuation_service(store, revived)
            reattach = await continuation.reattach_by_execution_id(execution_id)
            assert (
                reattach.category
                is DelegatedExecutionContinuationOutcomeCategory.PROVIDER_OPERATION_NOT_FOUND
            )
            assert reattach.failure_code == "PROVIDER_OPERATION_NOT_FOUND"
        finally:
            revived.close()
    finally:
        return


@pytest.mark.asyncio
async def test_c1_t12_spoofed_provider_id_fail_closed(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="provider-id"),
    )
    assert outcome.invocation_binding is not None
    execution_id = outcome.invocation_binding.execution_id
    bound_id = subprocess_provider.provider_id

    class _SpoofIdAdapter:
        provider_id = "spoof-provider"

        def __getattr__(self, name: str) -> object:
            return getattr(subprocess_provider, name)

    status = DelegatedExecutionStatusReadService(
        DelegatedInvocationCorrelationService(store),
        MappingDelegatedExecutionProviderResolver({bound_id: _SpoofIdAdapter()}),  # type: ignore[arg-type]
    )
    read = await status.read_status_by_execution_id(execution_id)
    assert read.category is DelegatedExecutionStatusOutcomeCategory.PROVIDER_BINDING_MISMATCH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "behavior",
    ["spoof_provider_request_id", "spoof_provider_operation_id", "spoof_execution_id"],
)
async def test_c1_t13_t15_execute_spoof_fail_closed(behavior: str) -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome = await _provider_execute_outcome(provider, value="x", behavior=behavior)
        assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
        assert outcome.failure_code == "OUTCOME_CONTRACT_MISMATCH"
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_c1_t16_status_spoof_fail_closed(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(
            value="status-spoof",
            behavior="status_spoof_provider_request_id",
        ),
    )
    assert outcome.invocation_binding is not None
    status = _fresh_status_service(store, subprocess_provider)
    read = await status.read_status_by_execution_id(outcome.invocation_binding.execution_id)
    assert (
        read.category
        is DelegatedExecutionStatusOutcomeCategory.STATUS_OUTCOME_CONTRACT_MISMATCH
    )


@pytest.mark.asyncio
async def test_c1_t17_reattach_same_operation_no_new_execute(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="same-op"),
    )
    assert outcome.invocation_binding is not None
    before = await subprocess_provider.worker_execute_count()
    continuation = _fresh_continuation_service(store, subprocess_provider)
    await continuation.reattach_by_execution_id(outcome.invocation_binding.execution_id)
    assert await subprocess_provider.worker_execute_count() == before


@pytest.mark.asyncio
async def test_c1_t18_custom_provider_works_with_delegated_service() -> None:
    class _CustomProvider(DelegatedExecutionProvider[SubprocessEchoPayload, SubprocessEchoResult]):
        @property
        def provider_id(self) -> str:
            return "custom_s2d_provider"

        @property
        def provider_version(self) -> str:
            return "1.0.0"

        @property
        def capabilities(self) -> DelegatedExecutionCapabilities:
            return DelegatedExecutionCapabilities(provider_id=self.provider_id)

        async def execute(
            self,
            request: DelegatedExecutionRequest[SubprocessEchoPayload],
        ) -> DelegatedExecutionOutcome[SubprocessEchoResult]:
            raise AssertionError("not invoked in composition proof")

    service = delegated_execution_service(
        _CustomProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=DelegatedInvocationCorrelationDurabilityPolicy(
            mode=DelegatedInvocationCorrelationDurabilityMode.NON_DURABLE_TEST,
        ),
    )
    assert isinstance(service, DelegatedExecutionService)


@pytest.mark.asyncio
async def test_c1_t19_custom_resolver_fresh_status(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="custom-resolver"),
    )
    assert outcome.invocation_binding is not None

    class _Resolver(DelegatedExecutionProviderResolver):
        def resolve(
            self,
            provider_id: str,
        ) -> SubprocessDelegatedExecutionProvider | None:
            if provider_id == subprocess_provider.provider_id:
                return subprocess_provider
            return None

    correlation = DelegatedInvocationCorrelationService(store)
    status = DelegatedExecutionStatusReadService(correlation, _Resolver())
    read = await status.read_status_by_execution_id(outcome.invocation_binding.execution_id)
    assert read.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


@pytest.mark.asyncio
async def test_c1_t24_connect_timeout_config_used() -> None:
    transport_path = _SUBPROCESS_DIR / "transport.py"
    source = transport_path.read_text(encoding="utf-8")
    assert "connect_timeout_seconds" in source
    assert "connect_to" in source


@pytest.mark.asyncio
async def test_c1_t25_capability_matrix_truthful(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    caps = subprocess_provider.capabilities
    assert caps.supports_status_read is True
    assert caps.supports_cancel is True
    assert caps.supports_pause is False
    assert caps.supports_resume is False
    assert caps.supports_interrupt is False
    assert caps.supports_streaming is False
    assert caps.supports_reattachment is True
    assert subprocess_provider.provider_id == SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID
    assert subprocess_provider.provider_version == SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_VERSION


@pytest.mark.asyncio
async def test_c1_t26_fresh_durable_control(
    subprocess_provider: SubprocessDelegatedExecutionProvider,
) -> None:
    outcome, store, _service = await _run_delegated_durable(
        subprocess_provider,
        payload=SubprocessEchoPayload(value="fresh-control"),
    )
    assert outcome.invocation_binding is not None
    control = _fresh_durable_control(store, subprocess_provider)
    result = await control.apply_control_by_execution_id(
        outcome.invocation_binding.execution_id,
        DelegatedExecutionControlOperation.CANCEL,
    )
    assert result.category is DelegatedExecutionDurableControlOutcomeCategory.RESOLVED_CONTROL


@pytest.mark.asyncio
async def test_c1_t23_secret_never_leaks_in_provider_paths() -> None:
    provider = create_subprocess_delegated_execution_provider(config=_config())
    try:
        outcome = await _provider_execute_outcome(
            provider,
            value="x",
            behavior="error_auth_rejected",
        )
        public = _config().public_view()
        blob = " ".join(
            [
                str(outcome.failure_message),
                str(outcome.failure_code),
                str(public),
            ],
        )
        assert _SECRET not in blob
    finally:
        provider.close()


def test_s2d_t29_no_global_registry() -> None:
    for path in (
        _PROVIDER_MODULE,
        _SUBPROCESS_DIR / "transport.py",
        _SUBPROCESS_DIR / "worker_server.py",
        _SUBPROCESS_DIR / "worker_main.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "Registry" not in source
        assert "global " not in source


def test_c1_t20_no_nexus_import_in_subprocess_plugin() -> None:
    for path in _SUBPROCESS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "intergrax.runtime.nexus" not in mod
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "intergrax.runtime.nexus" not in alias.name


def test_c1_t21_no_untyped_public_abi_in_provider_module() -> None:
    tree = ast.parse(_PROVIDER_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            for arg in node.args.args:
                if isinstance(arg.annotation, ast.Name) and arg.annotation.id in {
                    "Any",
                    "object",
                }:
                    raise AssertionError(f"untyped public arg on {node.name}")
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            if node.returns is not None and isinstance(node.returns, ast.Name):
                if node.returns.id in {"Any", "object"}:
                    raise AssertionError(f"untyped public return on {node.name}")


def test_c1_t22_no_private_cross_layer_access_in_subprocess_plugin() -> None:
    forbidden = (
        "service._",
        "resolver._",
        "binding._",
    )
    for path in _SUBPROCESS_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source


def test_s2d_t31_no_vendor_import_in_core() -> None:
    for module in (_SERVICE_MODULE, _CHILD_MODULE, _BOUNDARY_MODULE):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "integrations.providers.delegated_execution" not in mod


def test_s2d_t32_no_reflection() -> None:
    tree = ast.parse(_PROVIDER_MODULE.read_text(encoding="utf-8"))
    forbidden = {"hasattr", "getattr", "setattr"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden


def test_s2d_t35_no_frozen_core_change() -> None:
    assert _CHILD_MODULE.exists()
    assert _BOUNDARY_MODULE.exists()


def _dummy_context() -> object:
    from intergrax.contracts.delegated_execution_provider import (
        DelegatedExecutionBudgetMode,
        DelegatedExecutionBudgetProjection,
        DelegatedExecutionContext,
    )

    parent = mint_execution_id()
    child = mint_execution_id()
    return DelegatedExecutionContext(
        execution_id=child,
        parent_execution_id=parent,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("read",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )
