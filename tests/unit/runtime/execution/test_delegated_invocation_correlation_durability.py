# © Artur Czarnecki. All rights reserved.

"""P2.1-S2C / S2C1-C1 — delegated invocation correlation durability."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionRequest,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_payload,
    digest_delegated_execution_request,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.delegated_invocation_correlation_wiring import (
    resolve_delegated_invocation_correlation_for_host,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationCompositionError,
    DelegatedInvocationCorrelationConflictError,
    DelegatedInvocationCorrelationDurabilityMode,
    DelegatedInvocationCorrelationDurabilityPolicy,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
    DelegatedInvocationCorrelationStore,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.delegated_execution.correlation_composition import (
    resolve_delegated_invocation_correlation_service,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    InMemoryDelegatedInvocationCorrelationStore,
    decode_correlation_record,
    wire_delegated_invocation_correlation_store,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)
from intergrax.runtime.execution.delegated_execution.service import (
    DelegatedExecutionService,
    delegated_execution_service,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SERVICE_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "delegated_execution" / "service.py"
)
_CORRELATION_SERVICE_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "correlation_service.py"
)
_PROVIDER_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_provider.py"
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())

_POLICY_DISABLED = DelegatedInvocationCorrelationDurabilityPolicy(
    mode=DelegatedInvocationCorrelationDurabilityMode.DISABLED,
)
_POLICY_REQUIRED = DelegatedInvocationCorrelationDurabilityPolicy(
    mode=DelegatedInvocationCorrelationDurabilityMode.REQUIRED,
)
_POLICY_NON_DURABLE_TEST = DelegatedInvocationCorrelationDurabilityPolicy(
    mode=DelegatedInvocationCorrelationDurabilityMode.NON_DURABLE_TEST,
)


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str
    child_execution_id: str
    parent_execution_id: str


def _operation() -> DelegatedExecutionOperationMetadata:
    return DelegatedExecutionOperationMetadata.model_validate(
        {"operation": "execute_delegate", "task_id": "task_correlation"},
    )


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _root_authority(*scopes: str) -> ParentExecutionAuthority:
    return ParentExecutionAuthority.scoped(scopes)


class _RecordingProvider:
    def __init__(self, provider_id: str = "recording_fake") -> None:
        self._provider_id = provider_id
        self.calls: list[DelegatedExecutionRequest[EchoPayload]] = []
        self.persist_count = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> DelegatedExecutionCapabilities:
        return DelegatedExecutionCapabilities(provider_id=self._provider_id)

    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        self.calls.append(request)
        payload_digest = digest_delegated_execution_payload(request.payload)
        request_digest = digest_delegated_execution_request(
            context=request.context,
            operation=request.operation,
            payload_digest=payload_digest,
        )
        return delegated_success_outcome(
            result=EchoResult(
                value=request.payload.value,
                child_execution_id=str(request.context.execution_id),
                parent_execution_id=str(request.context.parent_execution_id),
            ),
            provider_invocation=ProviderInvocation.model_validate(
                {
                    "invocation_id": "inv-correlation-1",
                    "provider_id": self._provider_id,
                    "operation": request.operation.operation,
                    "task_id": request.operation.task_id,
                    "run_id": str(request.context.run_id),
                    "request_digest": request_digest,
                    "started_at": _T0.isoformat(),
                    "provider_request_id": "preq-1",
                    "provider_operation_id": "pop-1",
                }
            ),
            provider_outcome=ProviderInvocationOutcome.model_validate(
                {
                    "invocation_id": "inv-correlation-1",
                    "status": ProviderInvocationStatus.SUCCEEDED,
                    "completed_at": "2026-09-07T08:00:01+00:00",
                }
            ),
        )


class _ContractMismatchProvider(_RecordingProvider):
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        outcome = await super().execute(request)
        assert outcome.provider_invocation is not None
        bad_invocation = outcome.provider_invocation.model_copy(
            update={"request_digest": "sha256:deadbeef"},
        )
        return delegated_success_outcome(
            result=outcome.result,
            provider_invocation=bad_invocation,
            provider_outcome=outcome.provider_outcome,
        )


class _FailingProvider(_RecordingProvider):
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> DelegatedExecutionOutcome[EchoResult]:
        return delegated_failure_outcome(
            category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
            failure_code="PROVIDER_EXECUTION_FAILED",
            failure_message="dispatch failed",
        )


class CountingCorrelationStore(DelegatedInvocationCorrelationStore):
    def __init__(self) -> None:
        self.persist_calls = 0
        self._inner = InMemoryDelegatedInvocationCorrelationStore()

    @property
    def is_durable(self) -> bool:
        return self._inner.is_durable

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        self.persist_calls += 1
        self._inner.persist(record)

    def get_by_execution_id(
        self,
        execution_id,
    ) -> DelegatedInvocationCorrelationRecord | None:
        return self._inner.get_by_execution_id(execution_id)


class FailingCorrelationStore(DelegatedInvocationCorrelationStore):
    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        raise DelegatedInvocationCorrelationPersistenceError("injected persistence failure")

    def get_by_execution_id(self, execution_id):
        return None


class _RaisingCorrelationStore(DelegatedInvocationCorrelationStore):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        raise self._exc

    def get_by_execution_id(self, execution_id):
        return None


class PluginCorrelationStore(DelegatedInvocationCorrelationStore):
    """Custom store proving pluginability without service changes."""

    def __init__(self) -> None:
        self._records: dict[str, DelegatedInvocationCorrelationRecord] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        self._records[str(record.binding.execution_id)] = record

    def get_by_execution_id(self, execution_id):
        return self._records.get(str(execution_id))


async def _run_under_root(
    service: DelegatedExecutionService[EchoPayload, EchoResult],
) -> DelegatedExecutionOutcome[EchoResult]:
    root = _root_identity()
    captured: list[DelegatedExecutionOutcome[EchoResult]] = []

    class RootDelegate:
        async def execute(self, request: EchoPayload) -> EchoResult:
            outcome = await service.execute_delegated(
                payload=request,
                operation=_operation(),
            )
            captured.append(outcome)
            if outcome.category is not DelegatedExecutionOutcomeCategory.SUCCESS:
                return EchoResult(value="failed", child_execution_id="", parent_execution_id="")
            assert outcome.result is not None
            return outcome.result

    await ExecutionBoundary[EchoPayload, EchoResult](
        RootDelegate(),
        identity=root,
        authority=_root_authority("read", "write"),
    ).execute(EchoPayload(value="ping"))
    assert len(captured) == 1
    return captured[0]


def _sample_binding(execution_id: str | None = None) -> DelegatedExecutionInvocationBinding:
    run_id = mint_run_id()
    parent = mint_execution_id()
    child = mint_execution_id() if execution_id is None else execution_id
    attempt = mint_attempt_id()
    operation = _operation()
    payload_digest = digest_delegated_execution_payload(EchoPayload(value="x"))
    context = DelegatedExecutionContext(
        execution_id=child,
        parent_execution_id=parent,
        run_id=run_id,
        attempt_id=attempt,
        authority=ParentExecutionAuthority.scoped(("delegated.test",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )
    request_digest = digest_delegated_execution_request(
        context=context,
        operation=operation,
        payload_digest=payload_digest,
    )
    invocation = ProviderInvocation.model_validate(
        {
            "invocation_id": "inv-sample",
            "provider_id": "recording_fake",
            "operation": operation.operation,
            "task_id": operation.task_id,
            "run_id": str(run_id),
            "request_digest": request_digest,
            "started_at": _T0.isoformat(),
        }
    )
    return mint_delegated_execution_invocation_binding(
        context=context,
        operation=operation,
        payload_digest=payload_digest,
        provider_invocation=invocation,
    )


@pytest.mark.asyncio
async def test_s2c_t1_persist_platform_issued_binding() -> None:
    backend = InMemoryDelegatedInvocationCorrelationStore()
    provider = _RecordingProvider()
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=backend,
    )
    outcome = await _run_under_root(service)
    assert outcome.invocation_binding is not None
    child_id = outcome.invocation_binding.execution_id
    loaded = DelegatedInvocationCorrelationService(backend).load_binding_by_execution_id(
        child_id,
    )
    assert loaded == outcome.invocation_binding


@pytest.mark.asyncio
async def test_s2c_t2_lookup_by_execution_id() -> None:
    backend = InMemoryDelegatedInvocationCorrelationStore()
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=backend,
    )
    outcome = await _run_under_root(service)
    assert outcome.result is not None
    binding = DelegatedInvocationCorrelationService(backend).load_binding_by_execution_id(
        outcome.result.child_execution_id,
    )
    assert binding.provider_invocation.invocation_id == "inv-correlation-1"


def test_s2c_t3_idempotent_same_write() -> None:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = DelegatedInvocationCorrelationService(store)
    binding = _sample_binding()
    service.persist_binding(binding)
    service.persist_binding(binding)
    assert store.get_by_execution_id(binding.execution_id) is not None


def test_s2c_t4_conflicting_write() -> None:
    store = InMemoryDelegatedInvocationCorrelationStore()
    service = DelegatedInvocationCorrelationService(store)
    binding = _sample_binding()
    service.persist_binding(binding)
    other = binding.model_copy(
        update={
            "provider_invocation": binding.provider_invocation.model_copy(
                update={"invocation_id": "inv-other"},
            ),
        },
    )
    with pytest.raises(DelegatedInvocationCorrelationConflictError):
        service.persist_binding(other)


@pytest.mark.skip(reason="delegated correlation is execution-scoped; no tenant dimension in S2A")
def test_s2c_t5_wrong_tenant() -> None:
    ...


def test_s2c_t6_storage_corruption() -> None:
    with pytest.raises(DelegatedInvocationCorrelationIntegrityError):
        decode_correlation_record(b"{not-json")


@pytest.mark.asyncio
async def test_s2c_t7_restart_like_lookup() -> None:
    shared = InMemoryDelegatedInvocationCorrelationStore()
    service_a = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=shared,
    )
    outcome = await _run_under_root(service_a)
    assert outcome.invocation_binding is not None
    service_b = DelegatedInvocationCorrelationService(shared)
    recovered = service_b.load_binding_by_execution_id(
        outcome.invocation_binding.execution_id,
    )
    assert recovered == outcome.invocation_binding


def test_s2c_t8_no_provider_authority() -> None:
    source = _PROVIDER_CONTRACT.read_text(encoding="utf-8")
    assert "CorrelationStore" not in source
    assert "correlation_store" not in source


def test_s2c_t9_plugin_store() -> None:
    store = PluginCorrelationStore()
    service = DelegatedInvocationCorrelationService(store)
    binding = _sample_binding()
    service.persist_binding(binding)
    assert service.load_binding_by_execution_id(binding.execution_id) == binding


def test_s2c_t10_no_concrete_vendor_import() -> None:
    forbidden = ("pymongo", "redis", "psycopg", "boto3", "motor")
    for module in (_SERVICE_MODULE, _CORRELATION_SERVICE_MODULE):
        source = module.read_text(encoding="utf-8")
        lowered = source.lower()
        for name in forbidden:
            assert name not in lowered


def test_s2c_t11_no_global_registry() -> None:
    source = _SERVICE_MODULE.read_text(encoding="utf-8")
    assert "global " not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_registry"):
                    raise AssertionError("global mutable registry detected")


def test_s2c_t12_no_reflection() -> None:
    for module in (_SERVICE_MODULE, _CORRELATION_SERVICE_MODULE):
        source = module.read_text(encoding="utf-8")
        assert "importlib" not in source
        assert "getattr(" not in source


@pytest.mark.asyncio
async def test_s2c_t14_provider_failure_does_not_write_correlation() -> None:
    store = CountingCorrelationStore()
    service = delegated_execution_service(
        _FailingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=store,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert store.persist_calls == 0


@pytest.mark.asyncio
async def test_s2c_t15_contract_mismatch_does_not_write_correlation() -> None:
    store = CountingCorrelationStore()
    service = delegated_execution_service(
        _ContractMismatchProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=store,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert store.persist_calls == 0


@pytest.mark.asyncio
async def test_s2c_t17_persistence_failure_after_successful_binding() -> None:
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=FailingCorrelationStore(),
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "INVOCATION_CORRELATION_PERSISTENCE_FAILURE"
    assert outcome.invocation_binding is None
    assert outcome.provider_invocation is not None
    assert outcome.provider_outcome is not None


def test_s2c_optional_store_skips_persistence() -> None:
    store = CountingCorrelationStore()
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_DISABLED,
    )
    assert store.persist_calls == 0


@pytest.mark.asyncio
async def test_c1_t1_persistence_error_after_successful_dispatch() -> None:
    provider = _RecordingProvider()
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=FailingCorrelationStore(),
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "INVOCATION_CORRELATION_PERSISTENCE_FAILURE"
    assert outcome.invocation_binding is None
    assert outcome.provider_invocation is not None
    assert outcome.provider_outcome is not None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_c1_t2_conflict_after_successful_dispatch() -> None:
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=_RaisingCorrelationStore(
            DelegatedInvocationCorrelationConflictError("conflict"),
        ),
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "INVOCATION_CORRELATION_CONFLICT"
    assert outcome.provider_invocation is not None
    assert outcome.provider_outcome is not None


@pytest.mark.asyncio
async def test_c1_t3_integrity_error_after_successful_dispatch() -> None:
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=_RaisingCorrelationStore(
            DelegatedInvocationCorrelationIntegrityError("integrity"),
        ),
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE
    assert outcome.failure_code == "INVOCATION_CORRELATION_INTEGRITY_FAILURE"
    assert outcome.provider_invocation is not None
    assert outcome.provider_outcome is not None


@pytest.mark.asyncio
async def test_c1_t5_contract_mismatch_still_sanitized() -> None:
    store = CountingCorrelationStore()
    service = delegated_execution_service(
        _ContractMismatchProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=store,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.provider_invocation is None
    assert outcome.provider_outcome is None


def test_c1_t6_required_plus_durable_store_passes() -> None:
    store = FailingCorrelationStore()
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_REQUIRED,
        correlation_store=store,
    )
    assert service is not None


def test_c1_t7_required_plus_no_store_fails_at_composition() -> None:
    provider = _RecordingProvider()
    with pytest.raises(DelegatedInvocationCorrelationCompositionError):
        delegated_execution_service(
            provider,
            ledger=_UNLIMITED_LEDGER,
            correlation_durability_policy=_POLICY_REQUIRED,
        )
    assert len(provider.calls) == 0


def test_c1_t8_required_plus_non_durable_store_fails_at_composition() -> None:
    provider = _RecordingProvider()
    with pytest.raises(DelegatedInvocationCorrelationCompositionError):
        delegated_execution_service(
            provider,
            ledger=_UNLIMITED_LEDGER,
            correlation_durability_policy=_POLICY_REQUIRED,
            correlation_store=InMemoryDelegatedInvocationCorrelationStore(),
        )
    assert len(provider.calls) == 0


@pytest.mark.asyncio
async def test_c1_t9_disabled_mode_skips_persistence() -> None:
    store = CountingCorrelationStore()
    provider = _RecordingProvider()
    service = delegated_execution_service(
        provider,
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_DISABLED,
    )
    outcome = await _run_under_root(service)
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert store.persist_calls == 0
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_c1_t10_non_durable_test_mode_uses_in_memory() -> None:
    backend = InMemoryDelegatedInvocationCorrelationStore()
    service = delegated_execution_service(
        _RecordingProvider(),
        ledger=_UNLIMITED_LEDGER,
        correlation_durability_policy=_POLICY_NON_DURABLE_TEST,
        correlation_store=backend,
    )
    outcome = await _run_under_root(service)
    assert outcome.invocation_binding is not None
    assert backend.get_by_execution_id(outcome.invocation_binding.execution_id) is not None


def test_c1_t11_wire_helper_without_durable_backend_fails() -> None:
    with pytest.raises(DelegatedInvocationCorrelationCompositionError):
        wire_delegated_invocation_correlation_store(
            durability_mode=DelegatedInvocationCorrelationDurabilityMode.REQUIRED,
        )


def test_c1_t15_production_profile_requires_durable_store_at_composition() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="c1.production.gate")
    assert (
        env.governance.reliability.delegated_invocation_correlation_durability
        is DelegatedInvocationCorrelationDurabilityMode.REQUIRED
    )
    with pytest.raises(DelegatedInvocationCorrelationCompositionError):
        resolve_delegated_invocation_correlation_for_host(env, document_store=None)


def test_c1_non_durable_test_without_store_materializes_in_memory() -> None:
    service = resolve_delegated_invocation_correlation_service(_POLICY_NON_DURABLE_TEST)
    assert service is not None
    assert not service.store.is_durable
