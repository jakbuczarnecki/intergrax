# © Artur Czarnecki. All rights reserved.

"""GR-7-A6 — provider invocation reconciliation contract foundation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationReason,
    ProviderInvocationReconciliationRequest,
    ProviderInvocationReconciliationResult,
    ProviderInvocationReconciliationVerdict,
    prepare_provider_invocation_reconciliation,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 9, 17, 8, 0, 0, tzinfo=UTC)


def _inv(**kwargs: object) -> ProviderInvocation:
    base = {
        "invocation_id": "inv-1",
        "provider_id": "prov-a",
        "operation": "external_work.accept_quote",
        "task_id": "task-1",
        "run_id": "run-1",
        "external_task_id": "ext-1",
        "correlation_id": "corr-1",
        "idempotency_key": "idem-1",
        "request_digest": "digest",
        "started_at": _T0,
    }
    base.update(kwargs)
    return ProviderInvocation.model_validate(base)


def _outcome(**kwargs: object) -> ProviderInvocationOutcome:
    base = {
        "invocation_id": "inv-1",
        "status": ProviderInvocationStatus.UNKNOWN,
        "completed_at": _T0,
    }
    base.update(kwargs)
    return ProviderInvocationOutcome.model_validate(base)


def _accept_contract(*, reconciliation: ExternalEffectCapabilitySupport) -> ExternalEffectContract:
    probes = ("get_work",) if reconciliation is ExternalEffectCapabilitySupport.SUPPORTED else ()
    return ExternalEffectContract(
        contract_id="external_work.accept_quote.v1",
        operation_key="external_work.accept_quote",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=reconciliation,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=probes,
    )


def _request(**kwargs: object) -> ProviderInvocationReconciliationRequest:
    base = {
        "invocation": _inv(),
        "outcome": _outcome(),
        "effect_contract": _accept_contract(
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
        ),
        "tenant_id": "tenant-a",
        "plugin_id": "external_work.reconciliation.v1",
        "provider_id": "prov-a",
    }
    base.update(kwargs)
    return ProviderInvocationReconciliationRequest.model_validate(base)


def test_create_unknown_contract_reconciliation_not_available() -> None:
    contract = ExternalEffectContract(
        contract_id="external_work.create_work.v1",
        operation_key="external_work.create_work",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    prepared = prepare_provider_invocation_reconciliation(
        _request(
            invocation=_inv(operation="external_work.create_work"),
            effect_contract=contract,
        ),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_AVAILABLE
    assert prepared.reason is ProviderInvocationReconciliationReason.RECONCILIATION_UNSUPPORTED


def test_non_unknown_outcome_not_executed() -> None:
    prepared = prepare_provider_invocation_reconciliation(
        _request(outcome=_outcome(status=ProviderInvocationStatus.SUCCEEDED)),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_EXECUTED


def test_mismatch_invocation_id_not_executed() -> None:
    prepared = prepare_provider_invocation_reconciliation(
        _request(outcome=_outcome(invocation_id="inv-other")),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_EXECUTED


def test_operation_mismatch_not_executed() -> None:
    prepared = prepare_provider_invocation_reconciliation(
        _request(invocation=_inv(operation="external_work.cancel_work")),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_EXECUTED


def test_provider_binding_mismatch_not_executed() -> None:
    prepared = prepare_provider_invocation_reconciliation(
        _request(provider_id="other-provider"),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_EXECUTED


def test_supported_accept_with_correlation_prepares() -> None:
    prepared = prepare_provider_invocation_reconciliation(_request())
    assert not isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.correlation_id == "corr-1"


def test_missing_external_task_id_not_available() -> None:
    prepared = prepare_provider_invocation_reconciliation(
        _request(invocation=_inv(external_task_id=None)),
    )
    assert isinstance(prepared, ProviderInvocationReconciliationResult)
    assert prepared.verdict is ProviderInvocationReconciliationVerdict.NOT_AVAILABLE
    assert prepared.reason is ProviderInvocationReconciliationReason.DENIED_CORRELATION_INFEASIBLE


def test_runtime_module_has_no_external_work_integration_imports() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "runtime"
        / "enterprise_reliability"
        / "provider_invocation_reconciliation.py"
    ).read_text(encoding="utf-8")
    for token in (
        "ExternalWorkIntegration",
        "ExternalWorkAdapterResult",
        "ExternalWorkProviderCapabilities",
    ):
        assert token not in source
