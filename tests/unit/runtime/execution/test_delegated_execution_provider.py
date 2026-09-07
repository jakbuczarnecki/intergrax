# © Artur Czarnecki. All rights reserved.

"""P2.1-S1 delegated execution provider contract and local provider proofs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetBounds,
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionCapabilities,
    DelegatedExecutionCapabilityError,
    DelegatedExecutionContext,
    DelegatedExecutionContractError,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    DelegatedExecutionTransportError,
    assert_provider_native_ids_distinct_from_execution,
    delegated_failure_outcome,
    delegated_success_outcome,
    digest_delegated_execution_request,
    mint_delegated_provider_invocation,
    validate_provider_identity,
)
from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
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
from intergrax.runtime.execution.budget.models import (
    ExecutionBudgetAllocationMode,
    ExecutionBudgetReservationGrant,
)
from intergrax.runtime.execution.delegated_execution.context_projection import (
    project_delegated_execution_context_from_parts,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionProvider,
    _digest_payload,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_DIGEST = "sha256:" + ("ab" * 32)


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str
    authority_scopes: tuple[str, ...]
    budget_mode: DelegatedExecutionBudgetMode


def _budget_projection(
    *,
    mode: DelegatedExecutionBudgetMode = DelegatedExecutionBudgetMode.SHARED,
    allowance: DelegatedExecutionBudgetBounds | None = None,
) -> DelegatedExecutionBudgetProjection:
    return DelegatedExecutionBudgetProjection(
        allocation_mode=mode,
        reservation_allowance=allowance,
    )


def _context(
    *,
    authority: ParentExecutionAuthority | None = None,
    effective: EffectiveDelegationAuthority | None = None,
    budget: DelegatedExecutionBudgetProjection | None = None,
) -> DelegatedExecutionContext:
    parent = mint_execution_id()
    child = mint_execution_id()
    return DelegatedExecutionContext(
        execution_id=child,
        parent_execution_id=parent,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=authority or ParentExecutionAuthority.scoped(("tools.read",)),
        effective_delegation=effective,
        budget=budget or _budget_projection(),
    )


def _operation(**overrides: object) -> DelegatedExecutionOperationMetadata:
    base = {
        "operation": "execute_delegate",
        "task_id": "task_fixture",
    }
    base.update(overrides)
    return DelegatedExecutionOperationMetadata.model_validate(base)


def _request(
    *,
    payload: EchoPayload | None = None,
    context: DelegatedExecutionContext | None = None,
) -> DelegatedExecutionRequest[EchoPayload]:
    return DelegatedExecutionRequest(
        context=context or _context(),
        payload=payload or EchoPayload(value="ping"),
        operation=_operation(),
    )


def _invocation(**overrides: object) -> ProviderInvocation:
    ctx = _context()
    base = {
        "invocation_id": "inv-local-1",
        "provider_id": "local_delegated_execution",
        "operation": "execute_delegate",
        "task_id": "task_fixture",
        "run_id": str(ctx.run_id),
        "request_digest": _DIGEST,
        "started_at": _T0,
        "provider_request_id": "preq-1",
        "provider_operation_id": "pop-1",
    }
    base.update(overrides)
    return ProviderInvocation.model_validate(base)


def _provider_outcome(**overrides: object) -> ProviderInvocationOutcome:
    base = {
        "invocation_id": "inv-local-1",
        "status": ProviderInvocationStatus.SUCCEEDED,
        "completed_at": _T0,
    }
    base.update(overrides)
    return ProviderInvocationOutcome.model_validate(base)


class _EchoDelegate:
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        return EchoResult(
            value=request.payload.value,
            authority_scopes=request.context.authority.permission_scopes,
            budget_mode=request.context.budget.allocation_mode,
        )


class _RaisingDelegate:
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        raise RuntimeError("vendor-native boom")


class _TransportDelegate:
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        raise TimeoutError("connect timed out")


class _IOErrorDelegate:
    async def execute(
        self,
        request: DelegatedExecutionRequest[EchoPayload],
    ) -> EchoResult:
        raise OSError("connection reset by peer")


class _ArbitraryPayload:
    def __init__(self) -> None:
        self.secret = "must-not-digest"


class PydanticPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


def test_contract_models_are_immutable() -> None:
    ctx = _context()
    with pytest.raises(Exception):
        ctx.execution_id = mint_execution_id()  # type: ignore[misc]

    caps = DelegatedExecutionCapabilities(provider_id="local")
    with pytest.raises(Exception):
        caps.supports_cancel = True  # type: ignore[misc]


def test_empty_provider_id_rejected() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        DelegatedExecutionCapabilities(provider_id="   ")


def test_empty_provider_version_rejected() -> None:
    with pytest.raises(DelegatedExecutionContractError, match="provider_version"):
        validate_provider_identity(provider_id="local", provider_version="  ")


def test_malformed_execution_identity_rejected() -> None:
    parent = mint_execution_id()
    with pytest.raises(ValueError, match="ExecutionId"):
        DelegatedExecutionContext(
            execution_id="not-canonical",
            parent_execution_id=parent,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            authority=ParentExecutionAuthority.unknown(),
            budget=_budget_projection(),
        )


def test_impossible_outcome_state_rejected() -> None:
    with pytest.raises(DelegatedExecutionContractError, match="success outcome"):
        DelegatedExecutionOutcome(
            category=DelegatedExecutionOutcomeCategory.SUCCESS,
            result=EchoResult(value="x", authority_scopes=(), budget_mode=DelegatedExecutionBudgetMode.SHARED),
            failure_code="ERR",
        )

    with pytest.raises(DelegatedExecutionContractError, match="non-success outcome"):
        DelegatedExecutionOutcome(
            category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
            result=EchoResult(value="x", authority_scopes=(), budget_mode=DelegatedExecutionBudgetMode.SHARED),
            failure_code="ERR",
        )


def test_provider_native_ids_distinct_from_execution_id() -> None:
    ctx = _context()
    assert str(ctx.execution_id) != "preq-1"
    mint_delegated_provider_invocation(
        context=ctx,
        operation=_operation(),
        provider_id="local",
        request_digest=_DIGEST,
        started_at=_T0,
        invocation_id="inv-1",
        provider_request_id="preq-1",
    )
    with pytest.raises(DelegatedExecutionContractError, match="provider_request_id"):
        assert_provider_native_ids_distinct_from_execution(
            execution_id=ctx.execution_id,
            provider_request_id=str(ctx.execution_id),
        )


def test_provider_receives_admitted_execution_identity() -> None:
    ctx = _context()
    request = _request(context=ctx)
    assert request.context.execution_id == ctx.execution_id
    assert request.context.parent_execution_id == ctx.parent_execution_id
    assert "requested_permission_scopes" not in DelegatedExecutionRequest.__annotations__
    assert "requested_authority" not in DelegatedExecutionOperationMetadata.model_fields


def test_provider_request_has_no_authority_escalation_api() -> None:
    assert "requested_permission_scopes" not in DelegatedExecutionContext.model_fields
    assert "grant_authority" not in DelegatedExecutionContext.model_fields
    assert "requested_authority" not in DelegatedExecutionOperationMetadata.model_fields


@pytest.mark.asyncio
async def test_local_provider_preserves_effective_authority_context() -> None:
    authority = ParentExecutionAuthority.scoped(("tools.read", "memory.read"))
    effective = EffectiveDelegationAuthority(
        requested_permission_scopes=("tools.read",),
        parent_effective_scopes=("tools.read", "memory.read"),
        effective_permission_scopes=("tools.read",),
    )
    ctx = _context(authority=authority, effective=effective)
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    outcome = await provider.execute(_request(context=ctx))
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.result is not None
    assert outcome.result.authority_scopes == ("tools.read", "memory.read")


def test_budget_projection_is_readonly_reference() -> None:
    allowance = DelegatedExecutionBudgetBounds(max_total_tokens=100)
    projection = _budget_projection(
        mode=DelegatedExecutionBudgetMode.RESERVED,
        allowance=allowance,
    )
    assert "ledger" not in DelegatedExecutionBudgetProjection.model_fields
    assert "requested_budget" not in DelegatedExecutionBudgetProjection.model_fields
    assert projection.reservation_allowance == allowance


def test_capabilities_default_conservative() -> None:
    caps = DelegatedExecutionCapabilities(provider_id="conservative")
    assert caps.supports_cancel is False
    assert caps.supports_pause is False
    assert caps.supports_resume is False
    assert caps.supports_streaming is False
    assert caps.supports_interrupt is False


def test_local_provider_capabilities_match_real_support() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    caps = provider.capabilities
    assert caps.provider_id == provider.provider_id
    assert caps.supports_cancel is False
    assert caps.supports_pause is False
    assert caps.supports_resume is False
    assert caps.supports_streaming is False
    assert caps.supports_interrupt is False


@pytest.mark.asyncio
async def test_success_outcome_mapping() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    outcome = await provider.execute(_request(payload=EchoPayload(value="ok")))
    assert outcome.category is DelegatedExecutionOutcomeCategory.SUCCESS
    assert outcome.result is not None
    assert outcome.result.value == "ok"
    assert outcome.provider_invocation is not None
    assert outcome.provider_outcome is not None
    assert outcome.provider_outcome.status is ProviderInvocationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_provider_failure_mapping() -> None:
    provider = LocalDelegatedExecutionProvider(_RaisingDelegate())
    outcome = await provider.execute(_request())
    assert outcome.category is DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE
    assert outcome.result is None
    assert outcome.failure_code == "PROVIDER_EXECUTION_FAILED"
    assert outcome.failure_message == "delegated execution provider failed"
    assert outcome.provider_status is None
    assert isinstance(outcome, DelegatedExecutionOutcome)


@pytest.mark.asyncio
async def test_transport_failure_mapping() -> None:
    provider = LocalDelegatedExecutionProvider(_TransportDelegate())
    outcome = await provider.execute(_request())
    assert outcome.category is DelegatedExecutionOutcomeCategory.TRANSPORT_FAILURE
    assert outcome.failure_code == "TRANSPORT_TIMEOUT"
    assert outcome.failure_message == "delegated execution transport timed out"
    assert "connect timed out" not in (outcome.failure_message or "")


@pytest.mark.asyncio
async def test_vendor_exception_does_not_leak_as_public_abi() -> None:
    provider = LocalDelegatedExecutionProvider(_RaisingDelegate())
    outcome = await provider.execute(_request())
    assert not isinstance(outcome, RuntimeError)
    assert outcome.failure_code == "PROVIDER_EXECUTION_FAILED"
    assert outcome.failure_message == "delegated execution provider failed"
    assert "vendor-native boom" not in (outcome.failure_message or "")
    assert "RuntimeError" not in (outcome.provider_status or "")
    assert outcome.provider_status is None


@pytest.mark.asyncio
async def test_transport_timeout_does_not_leak_raw_message() -> None:
    provider = LocalDelegatedExecutionProvider(_TransportDelegate())
    outcome = await provider.execute(_request())
    assert outcome.failure_code == "TRANSPORT_TIMEOUT"
    assert outcome.failure_message == "delegated execution transport timed out"
    assert "connect timed out" not in (outcome.failure_message or "")
    assert outcome.provider_status is None


@pytest.mark.asyncio
async def test_transport_io_does_not_leak_raw_message() -> None:
    provider = LocalDelegatedExecutionProvider(_IOErrorDelegate())
    outcome = await provider.execute(_request())
    assert outcome.failure_code == "TRANSPORT_IO"
    assert outcome.failure_message == "delegated execution transport I/O failed"
    assert "connection reset by peer" not in (outcome.failure_message or "")
    assert outcome.provider_status is None


def test_pydantic_payload_digest_is_deterministic() -> None:
    first = _digest_payload(PydanticPayload(value="stable"))
    second = _digest_payload(PydanticPayload(value="stable"))
    assert first == second
    assert first.startswith("sha256:")


def test_dataclass_payload_digest_is_deterministic() -> None:
    first = _digest_payload(EchoPayload(value="stable"))
    second = _digest_payload(EchoPayload(value="stable"))
    assert first == second
    assert first.startswith("sha256:")


@pytest.mark.asyncio
async def test_unsupported_payload_fails_closed() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    request = DelegatedExecutionRequest(
        context=_context(),
        payload=_ArbitraryPayload(),  # type: ignore[arg-type]
        operation=_operation(),
    )
    with pytest.raises(DelegatedExecutionContractError, match="not supported"):
        await provider.execute(request)


@pytest.mark.asyncio
async def test_arbitrary_object_with_dict_does_not_use_reflection_fallback() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    request = DelegatedExecutionRequest(
        context=_context(),
        payload=_ArbitraryPayload(),  # type: ignore[arg-type]
        operation=_operation(),
    )
    with pytest.raises(DelegatedExecutionContractError):
        await provider.execute(request)


@pytest.mark.asyncio
async def test_local_delegate_receives_canonical_context() -> None:
    ctx = _context()
    captured: dict[str, Any] = {}

    class _CapturingDelegate:
        async def execute(
            self,
            request: DelegatedExecutionRequest[EchoPayload],
        ) -> EchoResult:
            captured["execution_id"] = request.context.execution_id
            captured["parent_execution_id"] = request.context.parent_execution_id
            captured["run_id"] = request.context.run_id
            return EchoResult(
                value=request.payload.value,
                authority_scopes=request.context.authority.permission_scopes,
                budget_mode=request.context.budget.allocation_mode,
            )

    provider = LocalDelegatedExecutionProvider(_CapturingDelegate())
    await provider.execute(_request(context=ctx, payload=EchoPayload(value="ctx")))
    assert captured["execution_id"] == ctx.execution_id
    assert captured["parent_execution_id"] == ctx.parent_execution_id
    assert captured["run_id"] == ctx.run_id


@pytest.mark.asyncio
async def test_payload_round_trip() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    outcome = await provider.execute(_request(payload=EchoPayload(value="round-trip")))
    assert outcome.result is not None
    assert outcome.result.value == "round-trip"


@pytest.mark.asyncio
async def test_provider_invocation_evidence_retained() -> None:
    ctx = _context()
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    outcome = await provider.execute(_request(context=ctx))
    assert outcome.provider_invocation is not None
    assert outcome.provider_invocation.provider_id == provider.provider_id
    assert outcome.provider_invocation.run_id == str(ctx.run_id)
    assert outcome.provider_invocation.invocation_id != str(ctx.execution_id)
    assert outcome.provider_invocation.provider_request_id != str(ctx.execution_id)
    assert outcome.provider_outcome is not None
    assert outcome.provider_outcome.invocation_id == outcome.provider_invocation.invocation_id


def test_context_projection_from_uer_grant() -> None:
    parent = mint_execution_id()
    child = mint_execution_id()
    grant = ExecutionBudgetReservationGrant(
        execution_id=child,
        parent_execution_id=parent,
        mode=ExecutionBudgetAllocationMode.RESERVED,
        reservation_allowance=RunBudget(max_total_tokens=50),
    )
    ctx = project_delegated_execution_context_from_parts(
        execution_id=child,
        parent_execution_id=parent,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("tools.read",)),
        effective_delegation=None,
        budget_grant=grant,
    )
    assert ctx.execution_id == child
    assert ctx.budget.allocation_mode is DelegatedExecutionBudgetMode.RESERVED
    assert ctx.budget.reservation_allowance is not None
    assert ctx.budget.reservation_allowance.max_total_tokens == 50


def test_delegated_execution_provider_protocol_surface() -> None:
    provider = LocalDelegatedExecutionProvider(_EchoDelegate())
    assert isinstance(provider, DelegatedExecutionProvider)


def test_helper_outcomes_validate() -> None:
    inv = _invocation()
    out = _provider_outcome()
    success = delegated_success_outcome(
        result=EchoResult(value="x", authority_scopes=(), budget_mode=DelegatedExecutionBudgetMode.SHARED),
        provider_invocation=inv,
        provider_outcome=out,
    )
    assert success.category is DelegatedExecutionOutcomeCategory.SUCCESS

    failure = delegated_failure_outcome(
        category=DelegatedExecutionOutcomeCategory.UNSUPPORTED,
        failure_code="UNSUPPORTED",
        failure_message="not supported",
    )
    assert failure.category is DelegatedExecutionOutcomeCategory.UNSUPPORTED


def test_digest_is_stable() -> None:
    ctx = _context()
    op = _operation()
    first = digest_delegated_execution_request(
        context=ctx,
        operation=op,
        payload_digest=_DIGEST,
    )
    second = digest_delegated_execution_request(
        context=ctx,
        operation=op,
        payload_digest=_DIGEST,
    )
    assert first == second


def test_error_taxonomy_types() -> None:
    from intergrax.contracts.delegated_execution_provider import (
        DelegatedExecutionProviderError,
    )

    assert issubclass(DelegatedExecutionTransportError, DelegatedExecutionProviderError)
    assert issubclass(DelegatedExecutionCapabilityError, DelegatedExecutionProviderError)
    assert issubclass(DelegatedExecutionContractError, DelegatedExecutionProviderError)
