# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Versioned persistence encoding for diagnostic ``Problem`` records (DIAG-STORAGE)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    DeterministicProblemReconciliationKey,
    ProblemReconciliationKeyKind,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
    LifecycleViolationTransition,
)
from intergrax.runtime.diagnostics.diagnostic_subject import (
    ApplicationDiagnosticSubjectRef,
    DiagnosticSubjectKind,
    ExecutionDiagnosticSubjectRef,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicFindingSignature,
    DeterministicLimitationSignature,
    DeterministicProblemSignature,
    DeterministicSignalFindingSignature,
    ProblemGroupingMethod,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubjectRef,
    problem_grouping_subject_ref_for_application_instance,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemLifecycleProvenance,
    ProblemOccurrence,
    ProblemReconciliationKey,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceIntegrityError
from intergrax.runtime.events.asof_projection import (
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType

_PERSISTENCE_SCHEMA_V1 = "intergrax.diagnostic_problem.persistence.v1"
_PERSISTENCE_SCHEMA_V2 = "intergrax.diagnostic_problem.persistence.v2"
_PAYLOAD_FIELD = "payload"


def encode_problem_record(problem: Problem) -> dict[str, Any]:
    """Serialize a bounded Problem aggregate for document/KV storage."""
    return {
        "schema_version": _PERSISTENCE_SCHEMA_V2,
        _PAYLOAD_FIELD: _encode_problem_payload_v2(problem),
    }


def decode_problem_record(data: object) -> Problem:
    """Reconstruct a typed bounded Problem from stored representation."""
    if not isinstance(data, dict):
        raise ProblemPersistenceIntegrityError("invalid diagnostic problem persistence record")
    schema_version = data.get("schema_version")
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ProblemPersistenceIntegrityError(
            "invalid diagnostic problem persistence payload",
        )
    try:
        if schema_version == _PERSISTENCE_SCHEMA_V2:
            return _decode_problem_payload_v2(payload)
        if schema_version == _PERSISTENCE_SCHEMA_V1:
            return _decode_problem_payload_v1(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProblemPersistenceIntegrityError(
            "malformed diagnostic problem persistence payload",
        ) from exc
    raise ProblemPersistenceIntegrityError(
        "unsupported diagnostic problem persistence schema",
    )


def decode_legacy_problem_record_with_occurrences(
    data: object,
) -> tuple[Problem, tuple[ProblemOccurrence, ...], tuple[ProblemGroupingSubjectRef, ...]]:
    """Decode legacy v1 records retaining inline occurrence history for migration."""
    if not isinstance(data, dict):
        raise ProblemPersistenceIntegrityError("invalid diagnostic problem persistence record")
    schema_version = data.get("schema_version")
    if schema_version != _PERSISTENCE_SCHEMA_V1:
        raise ProblemPersistenceIntegrityError(
            "legacy inline occurrence decode requires v1 schema",
        )
    payload = data.get(_PAYLOAD_FIELD)
    if not isinstance(payload, dict):
        raise ProblemPersistenceIntegrityError(
            "invalid diagnostic problem persistence payload",
        )
    try:
        inline_occurrences = tuple(
            _decode_occurrence(item) for item in _require_sequence(payload["occurrences"])
        )
        inline_subject_refs = tuple(
            _decode_subject_ref(item)
            for item in _require_sequence(payload["current_subject_refs"])
        )
        problem = _decode_bounded_problem_fields(payload)
        return problem, inline_occurrences, inline_subject_refs
    except (KeyError, TypeError, ValueError) as exc:
        raise ProblemPersistenceIntegrityError(
            "malformed diagnostic problem persistence payload",
        ) from exc


def _encode_problem_payload_v2(problem: Problem) -> dict[str, object]:
    return {
        "problem_id": str(problem.problem_id),
        "tenant_id": problem.tenant_id,
        "status": problem.status.value,
        "first_seen_at": _encode_datetime(problem.first_seen_at),
        "last_seen_at": _encode_datetime(problem.last_seen_at),
        "occurrence_count": problem.occurrence_count,
        "provenance": _encode_provenance(problem.provenance),
        "record_version": problem.record_version,
    }


def _decode_problem_payload_v2(payload: Mapping[str, object]) -> Problem:
    return Problem(
        problem_id=ProblemId(str(payload["problem_id"])),
        tenant_id=str(payload["tenant_id"]),
        status=ProblemStatus(str(payload["status"])),
        first_seen_at=_decode_datetime(payload["first_seen_at"]),
        last_seen_at=_decode_datetime(payload["last_seen_at"]),
        occurrence_count=int(payload["occurrence_count"]),  # type: ignore[arg-type]
        provenance=_decode_provenance(payload["provenance"]),
        record_version=int(payload["record_version"]),  # type: ignore[arg-type]
    )


def _decode_problem_payload_v1(payload: Mapping[str, object]) -> Problem:
    return _decode_bounded_problem_fields(payload)


def _decode_bounded_problem_fields(payload: Mapping[str, object]) -> Problem:
    return Problem(
        problem_id=ProblemId(str(payload["problem_id"])),
        tenant_id=str(payload["tenant_id"]),
        status=ProblemStatus(str(payload["status"])),
        first_seen_at=_decode_datetime(payload["first_seen_at"]),
        last_seen_at=_decode_datetime(payload["last_seen_at"]),
        occurrence_count=int(payload["occurrence_count"]),  # type: ignore[arg-type]
        provenance=_decode_provenance(payload["provenance"]),
        record_version=int(payload["record_version"]),  # type: ignore[arg-type]
    )


def _encode_legacy_problem_payload_v1(
    *,
    problem: Problem,
    current_subject_refs: tuple[ProblemGroupingSubjectRef, ...],
    occurrences: tuple[ProblemOccurrence, ...],
) -> dict[str, Any]:
    return {
        "problem_id": str(problem.problem_id),
        "tenant_id": problem.tenant_id,
        "status": problem.status.value,
        "first_seen_at": _encode_datetime(problem.first_seen_at),
        "last_seen_at": _encode_datetime(problem.last_seen_at),
        "occurrence_count": problem.occurrence_count,
        "current_subject_refs": [
            _encode_subject_ref(subject_ref) for subject_ref in current_subject_refs
        ],
        "occurrences": [_encode_occurrence(occurrence) for occurrence in occurrences],
        "provenance": _encode_provenance(problem.provenance),
        "record_version": problem.record_version,
    }


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return value.isoformat()


def _decode_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("datetime must be ISO string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return parsed


def _encode_subject_ref(subject_ref: ProblemGroupingSubjectRef) -> dict[str, str]:
    subject = subject_ref.subject
    if type(subject) is ExecutionDiagnosticSubjectRef:
        return {
            "kind": DiagnosticSubjectKind.EXECUTION.value,
            "tenant_id": subject.tenant_id,
            "task_id": str(subject.task_id),
            "run_id": str(subject.run_id),
        }
    if type(subject) is ApplicationDiagnosticSubjectRef:
        return {
            "kind": DiagnosticSubjectKind.APPLICATION_INSTANCE.value,
            "tenant_id": subject.tenant_id,
            "application_id": subject.application_id,
            "instance_id": subject.instance_id,
        }
    raise TypeError(f"unsupported diagnostic subject type: {type(subject).__name__}")


def _decode_subject_ref(value: object) -> ProblemGroupingSubjectRef:
    if not isinstance(value, dict):
        raise ValueError("invalid subject_ref")
    tenant_id = str(value["tenant_id"])
    kind = value.get("kind")
    if kind is None:
        return problem_grouping_subject_ref_for_execution(
            tenant_id=tenant_id,
            task_id=TaskId(str(value["task_id"])),
            run_id=RunId(str(value["run_id"])),
        )
    if kind == DiagnosticSubjectKind.EXECUTION.value:
        return problem_grouping_subject_ref_for_execution(
            tenant_id=tenant_id,
            task_id=TaskId(str(value["task_id"])),
            run_id=RunId(str(value["run_id"])),
        )
    if kind == DiagnosticSubjectKind.APPLICATION_INSTANCE.value:
        return problem_grouping_subject_ref_for_application_instance(
            tenant_id=tenant_id,
            application_id=str(value["application_id"]),
            instance_id=str(value["instance_id"]),
        )
    raise ValueError("unsupported diagnostic subject kind")


def encode_problem_occurrence_payload(occurrence: ProblemOccurrence) -> dict[str, object]:
    """Public occurrence payload encoder for occurrence persistence."""
    return _encode_occurrence(occurrence)


def decode_problem_occurrence_payload(payload: Mapping[str, object]) -> ProblemOccurrence:
    """Public occurrence payload decoder for occurrence persistence."""
    return _decode_occurrence(payload)


def _encode_occurrence(occurrence: ProblemOccurrence) -> dict[str, object]:
    return {
        "subject_ref": _encode_subject_ref(occurrence.subject_ref),
        "observed_at": _encode_datetime(occurrence.observed_at),
        "strategy_id": str(occurrence.strategy_id),
        "strategy_version": str(occurrence.strategy_version),
        "method": occurrence.method.value,
    }


def _decode_occurrence(value: object) -> ProblemOccurrence:
    if not isinstance(value, dict):
        raise ValueError("invalid occurrence")
    return ProblemOccurrence(
        subject_ref=_decode_subject_ref(value["subject_ref"]),
        observed_at=_decode_datetime(value["observed_at"]),
        strategy_id=ProblemGroupingStrategyId(str(value["strategy_id"])),
        strategy_version=ProblemGroupingStrategyVersion(str(value["strategy_version"])),
        method=ProblemGroupingMethod(str(value["method"])),
    )


def _encode_provenance(provenance: ProblemLifecycleProvenance) -> dict[str, object]:
    return {
        "strategy_id": str(provenance.strategy_id),
        "strategy_version": str(provenance.strategy_version),
        "method": provenance.method.value,
        "reconciliation_key": _encode_reconciliation_key(
            provenance.reconciliation_key,
        ),
    }


def _decode_provenance(value: object) -> ProblemLifecycleProvenance:
    if not isinstance(value, dict):
        raise ValueError("invalid provenance")
    return ProblemLifecycleProvenance(
        strategy_id=ProblemGroupingStrategyId(str(value["strategy_id"])),
        strategy_version=ProblemGroupingStrategyVersion(str(value["strategy_version"])),
        method=ProblemGroupingMethod(str(value["method"])),
        reconciliation_key=_decode_reconciliation_key(value["reconciliation_key"]),
    )


def _encode_reconciliation_key(
    reconciliation_key: ProblemReconciliationKey,
) -> dict[str, object]:
    if reconciliation_key.kind is ProblemReconciliationKeyKind.DETERMINISTIC:
        if not isinstance(reconciliation_key, DeterministicProblemReconciliationKey):
            raise TypeError("deterministic reconciliation key type mismatch")
        return {
            "kind": reconciliation_key.kind.value,
            "tenant_id": reconciliation_key.tenant_id,
            "strategy_id": str(reconciliation_key.strategy_id),
            "strategy_version": str(reconciliation_key.strategy_version),
            "signature": _encode_signature(reconciliation_key.signature),
        }
    raise TypeError(f"unsupported reconciliation key kind: {reconciliation_key.kind}")


def _decode_reconciliation_key(value: object) -> ProblemReconciliationKey:
    if not isinstance(value, dict):
        raise ValueError("invalid reconciliation key")
    kind = value.get("kind")
    if kind == ProblemReconciliationKeyKind.DETERMINISTIC.value:
        return DeterministicProblemReconciliationKey(
            tenant_id=str(value["tenant_id"]),
            strategy_id=ProblemGroupingStrategyId(str(value["strategy_id"])),
            strategy_version=ProblemGroupingStrategyVersion(
                str(value["strategy_version"]),
            ),
            signature=_decode_signature(value["signature"]),
        )
    raise ValueError("unsupported reconciliation key kind")


def _encode_signature(signature: DeterministicProblemSignature) -> dict[str, object]:
    encoded: dict[str, object] = {
        "findings": [_encode_finding(item) for item in signature.findings],
        "limitations": [_encode_limitation(item) for item in signature.limitations],
    }
    if signature.subject_domain is not None:
        encoded["subject_domain"] = signature.subject_domain.value
    return encoded


def _decode_signature(value: object) -> DeterministicProblemSignature:
    if not isinstance(value, dict):
        raise ValueError("invalid deterministic signature")
    subject_domain_raw = value.get("subject_domain")
    subject_domain = (
        DiagnosticSubjectKind(str(subject_domain_raw))
        if subject_domain_raw is not None
        else None
    )
    return DeterministicProblemSignature(
        findings=tuple(
            _decode_finding(item) for item in _require_sequence(value["findings"])
        ),
        limitations=tuple(
            _decode_limitation(item)
            for item in _require_sequence(value["limitations"])
        ),
        subject_domain=subject_domain,
    )


def _encode_finding(
    finding: DeterministicFindingSignature | DeterministicSignalFindingSignature,
) -> dict[str, object]:
    if type(finding) is DeterministicSignalFindingSignature:
        encoded: dict[str, object] = {
            "source": "platform_signal",
            "problem_kind": finding.problem_kind,
            "severity": finding.severity,
            "source_layer": finding.source_layer,
            "source_component": finding.source_component,
            "status": finding.status,
        }
        if finding.error_code is not None:
            encoded["error_code"] = finding.error_code
        if finding.exception_type is not None:
            encoded["exception_type"] = finding.exception_type
        return encoded
    encoded = {
        "source": "lifecycle",
        "kind": finding.kind.value,
        "scope": finding.scope.value,
        "source_anomaly_kind": finding.source_anomaly_kind.value,
    }
    if finding.lifecycle_transition is not None:
        encoded["lifecycle_transition"] = _encode_lifecycle_transition(
            finding.lifecycle_transition,
        )
    return encoded


def _decode_finding(
    value: object,
) -> DeterministicFindingSignature | DeterministicSignalFindingSignature:
    if not isinstance(value, dict):
        raise ValueError("invalid finding signature")
    source = value.get("source")
    if source == "platform_signal":
        error_code_raw = value.get("error_code")
        exception_type_raw = value.get("exception_type")
        return DeterministicSignalFindingSignature(
            problem_kind=str(value["problem_kind"]),
            severity=str(value["severity"]),
            source_layer=str(value["source_layer"]),
            source_component=str(value["source_component"]),
            status=str(value["status"]),
            error_code=str(error_code_raw) if error_code_raw is not None else None,
            exception_type=(
                str(exception_type_raw) if exception_type_raw is not None else None
            ),
        )
    transition_raw = value.get("lifecycle_transition")
    transition = (
        _decode_lifecycle_transition(transition_raw)
        if transition_raw is not None
        else None
    )
    return DeterministicFindingSignature(
        kind=DiagnosticFindingKind(str(value["kind"])),
        scope=LifecycleAnomalyScope(str(value["scope"])),
        source_anomaly_kind=LifecycleAnomalyKind(str(value["source_anomaly_kind"])),
        lifecycle_transition=transition,
    )


def _encode_limitation(
    limitation: DeterministicLimitationSignature,
) -> dict[str, str]:
    return {
        "kind": limitation.kind.value,
        "source_anomaly_kind": limitation.source_anomaly_kind.value,
    }


def _decode_limitation(value: object) -> DeterministicLimitationSignature:
    if not isinstance(value, dict):
        raise ValueError("invalid limitation signature")
    return DeterministicLimitationSignature(
        kind=DiagnosticLimitationKind(str(value["kind"])),
        source_anomaly_kind=LifecycleAnomalyKind(str(value["source_anomaly_kind"])),
    )


def _encode_lifecycle_transition(
    transition: LifecycleViolationTransition,
) -> dict[str, str]:
    return {
        "violation_kind": transition.violation_kind.value,
        "prior_status": transition.prior_status.value,
        "violating_event_type": transition.violating_event_type.value,
    }


def _decode_lifecycle_transition(value: object) -> LifecycleViolationTransition:
    if not isinstance(value, dict):
        raise ValueError("invalid lifecycle transition")
    return LifecycleViolationTransition(
        violation_kind=RunLifecycleViolationKind(str(value["violation_kind"])),
        prior_status=RunExecutionLifecycleStatus(str(value["prior_status"])),
        violating_event_type=RuntimeEventType(str(value["violating_event_type"])),
    )


def _require_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError("expected JSON array")
    return tuple(value)
