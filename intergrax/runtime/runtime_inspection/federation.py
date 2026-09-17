# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Federated read-only runtime inspection composition (INSPECT-01-A)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
    RuntimeInspectionNotFoundError,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.failures import (
    RuntimeInspectionSourceFailure,
    RuntimeInspectionSourceFailureCode,
)
from intergrax.contracts.runtime_inspection.query import RuntimeInspectionQuery
from intergrax.contracts.runtime_inspection.read_port import RuntimeInspectionReadPort
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionIdentitySection
from intergrax.contracts.runtime_inspection.snapshot import RuntimeInspectionSnapshot
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionContinuationReadPort,
    RuntimeInspectionDiagnosticReadPort,
    RuntimeInspectionEvidenceReadPort,
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScopeReader,
    RuntimeInspectionGovernanceReadPort,
    RuntimeInspectionScopeLookupOutcome,
    RuntimeInspectionToolReadPort,
)
from intergrax.runtime.runtime_inspection.adapters.execution_reconstruction import (
    execution_state_section,
)
from intergrax.runtime.runtime_inspection.timeline import build_timeline_section


def _merge_completeness(
    *values: RuntimeInspectionCompleteness,
) -> RuntimeInspectionCompleteness:
    if any(item is RuntimeInspectionCompleteness.UNAVAILABLE for item in values):
        if all(item is RuntimeInspectionCompleteness.UNAVAILABLE for item in values):
            return RuntimeInspectionCompleteness.UNAVAILABLE
        return RuntimeInspectionCompleteness.PARTIAL
    if any(item is RuntimeInspectionCompleteness.PARTIAL for item in values):
        return RuntimeInspectionCompleteness.PARTIAL
    if any(item is RuntimeInspectionCompleteness.REDACTED for item in values):
        return RuntimeInspectionCompleteness.REDACTED
    return RuntimeInspectionCompleteness.COMPLETE


def _identity_completeness(scope) -> RuntimeInspectionCompleteness:
    if scope.attempt_id is None:
        return RuntimeInspectionCompleteness.PARTIAL
    return RuntimeInspectionCompleteness.COMPLETE


class FederatedRuntimeInspectionReadService(RuntimeInspectionReadPort):
    """Explicitly injected read sources — no discovery, registry, or reflection."""

    def __init__(
        self,
        *,
        scope_reader: RuntimeInspectionExecutionScopeReader,
        execution_facts_reader: RuntimeInspectionExecutionFactsReader,
        diagnostic_reader: RuntimeInspectionDiagnosticReadPort | None = None,
        evidence_reader: RuntimeInspectionEvidenceReadPort | None = None,
        tool_reader: RuntimeInspectionToolReadPort | None = None,
        governance_reader: RuntimeInspectionGovernanceReadPort | None = None,
        continuation_reader: RuntimeInspectionContinuationReadPort | None = None,
    ) -> None:
        self._scope_reader = scope_reader
        self._execution_facts_reader = execution_facts_reader
        self._diagnostic_reader = diagnostic_reader
        self._evidence_reader = evidence_reader
        self._tool_reader = tool_reader
        self._governance_reader = governance_reader
        self._continuation_reader = continuation_reader
        _validate_unique_source_ids(
            scope_reader.source_id,
            execution_facts_reader.source_id,
            diagnostic_reader.source_id if diagnostic_reader is not None else None,
            evidence_reader.source_id if evidence_reader is not None else None,
            tool_reader.source_id if tool_reader is not None else None,
            governance_reader.source_id if governance_reader is not None else None,
            continuation_reader.source_id if continuation_reader is not None else None,
        )

    def inspect(self, query: RuntimeInspectionQuery) -> RuntimeInspectionSnapshot:
        lookup = self._scope_reader.resolve_scope(
            tenant_id=query.tenant_id,
            execution_id=query.execution_id,
        )
        if lookup.outcome is RuntimeInspectionScopeLookupOutcome.NOT_FOUND:
            raise RuntimeInspectionNotFoundError(execution_id=query.execution_id)
        if lookup.outcome is RuntimeInspectionScopeLookupOutcome.TENANT_DENIED:
            raise RuntimeInspectionTenantBoundaryError(execution_id=query.execution_id)
        if lookup.scope is None:
            raise RuntimeInspectionNotFoundError(execution_id=query.execution_id)
        scope = lookup.scope
        if scope.tenant_id != query.tenant_id:
            raise RuntimeInspectionTenantBoundaryError(execution_id=query.execution_id)

        source_failures: list[RuntimeInspectionSourceFailure] = []
        try:
            reconstruction = self._execution_facts_reader.read_execution_facts(scope)
        except RuntimeInspectionError as exc:
            if exc.code is RuntimeInspectionErrorCode.EXECUTION_FACTS_UNAVAILABLE:
                raise
            raise
        except Exception as exc:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.EXECUTION_FACTS_UNAVAILABLE,
                "execution facts reader failed",
                execution_id=query.execution_id,
                source_id=self._execution_facts_reader.source_id,
            ) from exc

        execution_section = execution_state_section(
            reconstruction,
            source_id=self._execution_facts_reader.source_id,
        )
        timeline_section = build_timeline_section(
            reconstruction,
            source_id=self._execution_facts_reader.source_id,
            limit=query.timeline_limit,
        )

        diagnostics_section = None
        evidence_section = None
        tools_section = None
        governance_section = None
        continuation_section = None
        optional_completeness: list[RuntimeInspectionCompleteness] = []

        if self._diagnostic_reader is not None:
            try:
                diagnostics_section = self._diagnostic_reader.read_diagnostics(scope)
                optional_completeness.append(diagnostics_section.completeness)
            except Exception:
                source_failures.append(
                    RuntimeInspectionSourceFailure(
                        source_id=self._diagnostic_reader.source_id,
                        domain="diagnostics",
                        code=RuntimeInspectionSourceFailureCode.UNAVAILABLE,
                        reason_code="diagnostic_reader_failed",
                    ),
                )
                optional_completeness.append(RuntimeInspectionCompleteness.UNAVAILABLE)

        if self._evidence_reader is not None:
            try:
                evidence_section = self._evidence_reader.read_evidence_references(scope)
                optional_completeness.append(evidence_section.completeness)
            except Exception:
                source_failures.append(
                    RuntimeInspectionSourceFailure(
                        source_id=self._evidence_reader.source_id,
                        domain="evidence",
                        code=RuntimeInspectionSourceFailureCode.UNAVAILABLE,
                        reason_code="evidence_reader_failed",
                    ),
                )
                optional_completeness.append(RuntimeInspectionCompleteness.UNAVAILABLE)

        if self._tool_reader is not None:
            tools_section, tool_failure = _read_optional_section(
                "tool_runtime",
                self._tool_reader.source_id,
                lambda: self._tool_reader.read_tool_invocations(scope),
            )
            if tool_failure is not None:
                source_failures.append(tool_failure)
                optional_completeness.append(RuntimeInspectionCompleteness.UNAVAILABLE)
            elif tools_section is not None:
                optional_completeness.append(tools_section.completeness)

        if self._governance_reader is not None:
            governance_section, governance_failure = _read_optional_section(
                "governance",
                self._governance_reader.source_id,
                lambda: self._governance_reader.read_governance_decisions(scope),
            )
            if governance_failure is not None:
                source_failures.append(governance_failure)
                optional_completeness.append(RuntimeInspectionCompleteness.UNAVAILABLE)
            elif governance_section is not None:
                optional_completeness.append(governance_section.completeness)

        if self._continuation_reader is not None:
            continuation_section, continuation_failure = _read_optional_section(
                "continuation",
                self._continuation_reader.source_id,
                lambda: self._continuation_reader.read_continuation_state(scope),
            )
            if continuation_failure is not None:
                source_failures.append(continuation_failure)
                optional_completeness.append(RuntimeInspectionCompleteness.UNAVAILABLE)
            elif continuation_section is not None:
                optional_completeness.append(continuation_section.completeness)

        identity = RuntimeInspectionIdentitySection(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            execution_id=scope.execution_id,
            identity_completeness=_identity_completeness(scope),
        )

        completeness = _merge_completeness(
            identity.identity_completeness,
            execution_section.completeness,
            timeline_section.completeness,
            *optional_completeness,
        )
        if source_failures and completeness is RuntimeInspectionCompleteness.COMPLETE:
            completeness = RuntimeInspectionCompleteness.PARTIAL

        return RuntimeInspectionSnapshot(
            observed_at=datetime.now(timezone.utc),
            identity=identity,
            execution=execution_section,
            timeline=timeline_section,
            diagnostics=diagnostics_section,
            evidence=evidence_section,
            tools=tools_section,
            governance=governance_section,
            continuation=continuation_section,
            completeness=completeness,
            source_failures=tuple(source_failures),
        )


def _read_optional_section(domain: str, source_id: str, read_callable):
    try:
        section = read_callable()
        return section, None
    except RuntimeInspectionError as exc:
        if exc.code is RuntimeInspectionErrorCode.SOURCE_INTEGRITY:
            return None, RuntimeInspectionSourceFailure(
                source_id=source_id,
                domain=domain,
                code=RuntimeInspectionSourceFailureCode.INTEGRITY,
                reason_code="source_integrity",
            )
        return None, RuntimeInspectionSourceFailure(
            source_id=source_id,
            domain=domain,
            code=RuntimeInspectionSourceFailureCode.UNAVAILABLE,
            reason_code="source_reader_failed",
        )
    except Exception:
        return None, RuntimeInspectionSourceFailure(
            source_id=source_id,
            domain=domain,
            code=RuntimeInspectionSourceFailureCode.UNAVAILABLE,
            reason_code="source_reader_failed",
        )


def _validate_unique_source_ids(*source_ids: str | None) -> None:
    seen: set[str] = set()
    for source_id in source_ids:
        if source_id is None:
            continue
        if source_id in seen:
            raise ValueError(f"duplicate runtime inspection source_id: {source_id!r}")
        seen.add(source_id)


__all__ = ["FederatedRuntimeInspectionReadService"]
