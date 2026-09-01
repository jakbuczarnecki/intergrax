# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded lookup for functional validation evidence (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.contracts.functional_diagnostic_bounds import MAX_FUNCTIONAL_DIAGNOSTIC_VALIDATIONS
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysisIntegrityError,
)
from intergrax.runtime.observability.functional_validation_evidence import (
    FunctionalValidationEvidence,
)


class _ValidationScopeResolution(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNASSIGNED_ATTEMPT = "unassigned_attempt"


@dataclass(frozen=True, slots=True)
class FunctionalValidationEvidenceLookup:
    """
    Bounded, analysis-scoped validation evidence index.

    Scope is enforced by DIAG — callers must not be trusted to supply only
  in-scope validations. Wrong-scope records fail closed at lookup construction
  or resolution time.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    validations: tuple[FunctionalValidationEvidence, ...]
    _index: dict[EventId, FunctionalValidationEvidence]

    @classmethod
    def for_scope(
        cls,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
        validations: tuple[FunctionalValidationEvidence, ...] = (),
    ) -> FunctionalValidationEvidenceLookup:
        if type(validations) is not tuple:
            raise FunctionalDiagnosticAnalysisIntegrityError("validations must be a tuple")
        if len(validations) > MAX_FUNCTIONAL_DIAGNOSTIC_VALIDATIONS:
            raise FunctionalDiagnosticAnalysisIntegrityError(
                f"validations must contain <= {MAX_FUNCTIONAL_DIAGNOSTIC_VALIDATIONS} items",
            )
        normalized_tenant = _require_tenant_id(tenant_id)
        index = _build_validation_index(
            validations=validations,
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        return cls(
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            validations=validations,
            _index=index,
        )

    def get(self, validation_id: EventId) -> FunctionalValidationEvidence | None:
        validation = self._index.get(validation_id)
        if validation is None:
            return None
        resolution = _resolve_validation_scope(
            validation=validation,
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )
        if resolution is _ValidationScopeResolution.MISMATCH:
            raise FunctionalDiagnosticAnalysisIntegrityError(
                "validation scope mismatch for requested validation_id",
            )
        if resolution is _ValidationScopeResolution.UNASSIGNED_ATTEMPT:
            return None
        return validation


def _build_validation_index(
    *,
    validations: tuple[FunctionalValidationEvidence, ...],
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
) -> dict[EventId, FunctionalValidationEvidence]:
    index: dict[EventId, FunctionalValidationEvidence] = {}
    for validation in validations:
        resolution = _resolve_validation_scope(
            validation=validation,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        if resolution is _ValidationScopeResolution.MISMATCH:
            raise FunctionalDiagnosticAnalysisIntegrityError(
                "validation scope mismatch in supplied validations",
            )
        existing = index.get(validation.validation_id)
        if existing is not None:
            if existing != validation:
                raise FunctionalDiagnosticAnalysisIntegrityError(
                    "conflicting duplicate validation_id in lookup",
                )
            continue
        index[validation.validation_id] = validation
    return index


def _resolve_validation_scope(
    *,
    validation: FunctionalValidationEvidence,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
) -> _ValidationScopeResolution:
    correlation = validation.correlation
    if correlation.tenant_id != tenant_id:
        return _ValidationScopeResolution.MISMATCH
    if correlation.task_id != task_id:
        return _ValidationScopeResolution.MISMATCH
    if correlation.run_id != run_id:
        return _ValidationScopeResolution.MISMATCH
    validation_attempt = correlation.attempt_id
    if attempt_id is not None:
        if validation_attempt is not None and validation_attempt != attempt_id:
            return _ValidationScopeResolution.MISMATCH
        if validation_attempt is None:
            return _ValidationScopeResolution.UNASSIGNED_ATTEMPT
    if attempt_id is None and validation_attempt is not None:
        return _ValidationScopeResolution.MISMATCH
    return _ValidationScopeResolution.MATCH


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise FunctionalDiagnosticAnalysisIntegrityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise FunctionalDiagnosticAnalysisIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized:
        raise FunctionalDiagnosticAnalysisIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    return normalized


__all__ = ["FunctionalValidationEvidenceLookup"]
