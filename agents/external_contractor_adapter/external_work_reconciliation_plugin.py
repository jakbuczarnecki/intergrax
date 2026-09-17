# © Artur Czarnecki. All rights reserved.

"""External Work ERL reconciliation probe — read-only ``get_work`` observation (GR-7-A6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityStrategyContext,
    ReconciliationStrategyAdvice,
)
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)
from intergrax.contracts.external_work import ExternalTaskCorrelation, ExternalWorkStatus
from intergrax.integrations.contracts.external_work import ExternalWorkError, ExternalWorkIntegration

EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID = "external_work.reconciliation.v1"
EXTERNAL_WORK_RECONCILIATION_PLUGIN_VERSION = "1.0.0"
_PROBE_GET_WORK = "get_work"

_ACCEPT_CONTRACT_ID = "external_work.accept_quote.v1"
_CANCEL_CONTRACT_ID = "external_work.cancel_work.v1"


def _descriptor() -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id=EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID,
        version=EXTERNAL_WORK_RECONCILIATION_PLUGIN_VERSION,
        owner="external_contractor_adapter",
        capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
        capabilities=("external_work_status_probe",),
        tenant_scope=None,
        priority=0,
    )


def _insufficient(
    request: ReconciliationProbeRequest,
    *,
    rationale: str,
) -> ReconciliationProbeResult:
    return ReconciliationProbeResult(
        verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
        evidence_ref=(
            f"evidence://external_work/reconcile/"
            f"{request.correlation_id}/{request.attempt_index}/inconclusive"
        ),
        rationale=rationale,
    )


_ACCEPT_STATUS_EVIDENCE: dict[ExternalWorkStatus, ExternalEffectEvidenceVerdict] = {
    ExternalWorkStatus.ACCEPTED: ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
}

_CANCEL_STATUS_EVIDENCE: dict[ExternalWorkStatus, ExternalEffectEvidenceVerdict] = {
    ExternalWorkStatus.CANCELLED: ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
}


def _verdict_for_contract(
    *,
    contract_id: str,
    status: ExternalWorkStatus,
) -> ExternalEffectEvidenceVerdict:
    """Map aggregate ``get_work`` status to operation-specific evidence — conservative."""
    if contract_id == _ACCEPT_CONTRACT_ID:
        return _ACCEPT_STATUS_EVIDENCE.get(
            status,
            ExternalEffectEvidenceVerdict.INSUFFICIENT,
        )
    if contract_id == _CANCEL_CONTRACT_ID:
        return _CANCEL_STATUS_EVIDENCE.get(
            status,
            ExternalEffectEvidenceVerdict.INSUFFICIENT,
        )
    return ExternalEffectEvidenceVerdict.INSUFFICIENT


@dataclass
class ExternalWorkReconciliationCorrelationRegistry:
    """Bind durable invocation correlation for one in-flight reconcile probe."""

    _bindings: dict[tuple[str, str], ExternalTaskCorrelation] = field(
        default_factory=dict,
    )
    get_work_calls: int = 0

    def bind(
        self,
        *,
        correlation_id: str,
        contract_id: str,
        correlation: ExternalTaskCorrelation,
    ) -> None:
        key = (correlation_id.strip(), contract_id.strip())
        self._bindings[key] = correlation

    def resolve(self, request: ReconciliationProbeRequest) -> ExternalTaskCorrelation | None:
        return self._bindings.get(
            (request.correlation_id.strip(), request.contract_id.strip()),
        )


@dataclass(frozen=True, slots=True)
class ExternalWorkReconciliationPlugin:
    """``ReconciliationStrategy`` + ``ReconciliationProbeExecutor`` for External Work."""

    _integration: ExternalWorkIntegration
    _correlation_registry: ExternalWorkReconciliationCorrelationRegistry

    @property
    def plugin_id(self) -> str:
        return EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID

    @property
    def version(self) -> str:
        return EXTERNAL_WORK_RECONCILIATION_PLUGIN_VERSION

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return _descriptor()

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        return ReconciliationStrategyAdvice(
            probe_ref=_PROBE_GET_WORK,
            rationale="external_work_get_work_status_probe",
        )

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        if request.probe_ref.strip() != _PROBE_GET_WORK:
            return _insufficient(
                request,
                rationale=f"unsupported_probe_ref:{request.probe_ref}",
            )
        correlation = self._correlation_registry.resolve(request)
        if correlation is None:
            return _insufficient(
                request,
                rationale="correlation_binding_missing",
            )
        self._correlation_registry.get_work_calls += 1
        try:
            snapshot = self._integration.get_work(correlation)
        except ExternalWorkError as exc:
            return _insufficient(
                request,
                rationale=f"probe_read_failed:{exc}",
            )
        except Exception as exc:  # noqa: BLE001 — integration boundary
            return _insufficient(
                request,
                rationale=f"probe_transport_failed:{type(exc).__name__}",
            )
        verdict = _verdict_for_contract(
            contract_id=request.contract_id,
            status=snapshot.status,
        )
        return ReconciliationProbeResult(
            verdict=verdict,
            evidence_ref=(
                f"evidence://external_work/{correlation.external_task_id}/"
                f"{snapshot.status.value}/{request.attempt_index}"
            ),
            rationale=f"observed_status:{snapshot.status.value}",
        )


def external_task_correlation_from_invocation(
    *,
    task_id: str,
    run_id: str,
    provider_id: str,
    external_task_id: str,
    correlation_id: str | None,
    idempotency_key: str | None,
) -> ExternalTaskCorrelation:
    return ExternalTaskCorrelation(
        task_id=task_id,
        run_id=run_id,
        correlation_id=correlation_id or "",
        provider_id=provider_id,
        external_task_id=external_task_id,
        idempotency_key=idempotency_key,
    )


__all__ = [
    "EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID",
    "EXTERNAL_WORK_RECONCILIATION_PLUGIN_VERSION",
    "ExternalWorkReconciliationCorrelationRegistry",
    "ExternalWorkReconciliationPlugin",
    "external_task_correlation_from_invocation",
]
