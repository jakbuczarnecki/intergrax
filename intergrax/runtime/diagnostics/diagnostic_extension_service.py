# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded extension collect + analyze orchestration (R5)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from intergrax.contracts.diagnostic_analyzer import (
    DiagnosticAnalyzer,
    DiagnosticExtensionCertainty,
    DiagnosticFindingCandidate,
)
from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticEvidenceContext,
    DiagnosticEvidenceScope,
    DiagnosticExecutionContext,
    DiagnosticExtensionEvidence,
    validate_extension_evidence_tenant_scope,
)
from intergrax.runtime.diagnostics.diagnostic_extension_evidence_store import (
    DiagnosticExtensionEvidenceStore,
)
from intergrax.runtime.diagnostics.diagnostic_extension_read_models import (
    DiagnosticExtensionEvidenceView,
    DiagnosticExtensionFindingView,
    DiagnosticExtensionOccurrenceEnrichment,
    DiagnosticExtensionPluginStatus,
    DiagnosticExtensionReadStatus,
)
from intergrax.runtime.diagnostics.diagnostic_extension_registry import (
    DiagnosticExtensionRegistry,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
)
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef

DEFAULT_EXTENSION_TIME_BUDGET_MS = 250


@dataclass(slots=True)
class DiagnosticExtensionService:
    """Collect typed evidence and run bounded analyzers — no Problem authority."""

    registry: DiagnosticExtensionRegistry
    evidence_store: DiagnosticExtensionEvidenceStore
    time_budget_ms: int = DEFAULT_EXTENSION_TIME_BUDGET_MS

    def enrich_for_occurrence(
        self,
        *,
        subject_ref: ProblemGroupingSubjectRef,
        reconstruction: ExecutionReconstruction,
    ) -> DiagnosticExtensionOccurrenceEnrichment | None:
        if not self.registry.evidence_contributors and not self.registry.analyzers:
            return None

        execution = subject_ref.execution()
        if execution is None:
            return DiagnosticExtensionOccurrenceEnrichment(
                read_status=DiagnosticExtensionReadStatus.UNAVAILABLE,
                contributed_evidence=(),
                extension_findings=(),
                limitations=("Extension enrichment requires execution diagnostic subject.",),
            )

        scope = DiagnosticEvidenceScope(
            tenant_id=subject_ref.tenant_id,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=None,
            execution_id=None,
        )
        if scope.tenant_id != reconstruction.tenant_id:
            raise ValueError("subject_ref tenant_id does not match reconstruction")

        exec_context = DiagnosticExecutionContext(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=None,
            execution_id=None,
            evidence_scope=scope,
            time_budget_ms=self.time_budget_ms,
        )

        limitations: list[str] = []

        collect_degraded = self._collect_evidence(exec_context, limitations=limitations)
        degraded = collect_degraded

        evidence = self.evidence_store.query_for_scope(scope)
        findings, analyze_degraded = self._run_analyzers(
            evidence,
            tenant_id=scope.tenant_id,
            time_budget_ms=exec_context.time_budget_ms,
        )
        degraded = degraded or analyze_degraded

        status = (
            DiagnosticExtensionReadStatus.DEGRADED
            if degraded
            else DiagnosticExtensionReadStatus.COMPLETE
        )
        return DiagnosticExtensionOccurrenceEnrichment(
            read_status=status,
            contributed_evidence=_evidence_views(evidence),
            extension_findings=findings,
            limitations=tuple(limitations),
        )

    def _collect_evidence(
        self,
        context: DiagnosticExecutionContext,
        *,
        limitations: list[str],
    ) -> bool:
        deadline = time.monotonic() + (context.time_budget_ms / 1000.0)
        evidence_context = DiagnosticEvidenceContext.from_execution_context(context)
        degraded = False
        for contributor in self.registry.evidence_contributors:
            if time.monotonic() > deadline:
                limitations.append("Extension evidence collection exceeded time budget.")
                return True
            try:
                collected = contributor.collect(evidence_context)
            except Exception:
                limitations.append(
                    f"Evidence contributor {contributor.contributor_id!r} is unavailable.",
                )
                degraded = True
                continue
            for item in collected:
                try:
                    validate_extension_evidence_tenant_scope(
                        item,
                        tenant_id=context.tenant_id,
                    )
                    self.evidence_store.append(item)
                except ValueError:
                    limitations.append(
                        f"Evidence contributor {contributor.contributor_id!r} returned out-of-scope evidence.",
                    )
                    degraded = True
        return degraded

    def _run_analyzers(
        self,
        evidence: tuple[DiagnosticExtensionEvidence, ...],
        *,
        tenant_id: str,
        time_budget_ms: int,
    ) -> tuple[tuple[DiagnosticExtensionFindingView, ...], bool]:
        deadline = time.monotonic() + (time_budget_ms / 1000.0)
        views: list[DiagnosticExtensionFindingView] = []
        degraded = False
        for analyzer in self.registry.analyzers:
            if time.monotonic() > deadline:
                degraded = True
                break
            scoped = tuple(
                item for item in evidence if item.scope.tenant_id == tenant_id
            )
            try:
                candidates = analyzer.analyze(scoped)
            except Exception:
                degraded = True
                views.append(
                    _unavailable_finding_view(analyzer),
                )
                continue
            views.extend(_candidate_views(analyzer, candidates, tenant_id=tenant_id))
        return tuple(views), degraded


def _evidence_views(
    evidence: tuple[DiagnosticExtensionEvidence, ...],
) -> tuple[DiagnosticExtensionEvidenceView, ...]:
    return tuple(
        DiagnosticExtensionEvidenceView(
            evidence_id=item.evidence_id,
            evidence_namespace=item.evidence_namespace,
            kind=item.kind,
            summary=item.summary,
        )
        for item in evidence
    )


def _candidate_views(
    analyzer: DiagnosticAnalyzer,
    candidates: tuple[DiagnosticFindingCandidate, ...],
    *,
    tenant_id: str,
) -> list[DiagnosticExtensionFindingView]:
    views: list[DiagnosticExtensionFindingView] = []
    for candidate in candidates:
        if candidate.scope.tenant_id != tenant_id:
            raise ValueError("analyzer emitted cross-tenant finding candidate")
        views.append(
            DiagnosticExtensionFindingView(
                analyzer_id=analyzer.analyzer_id,
                analyzer_namespace=analyzer.analyzer_namespace,
                kind=candidate.kind,
                certainty=candidate.confidence,
                summary=candidate.summary,
                evidence_refs=candidate.evidence_refs,
            ),
        )
    return views


def _unavailable_finding_view(analyzer: DiagnosticAnalyzer) -> DiagnosticExtensionFindingView:
    return DiagnosticExtensionFindingView(
        analyzer_id=analyzer.analyzer_id,
        analyzer_namespace=analyzer.analyzer_namespace,
        kind=f"{analyzer.analyzer_namespace}.plugin_unavailable",
        certainty=DiagnosticExtensionCertainty.UNSUPPORTED,
        summary="Extension analyzer failed; central diagnostics continue.",
        evidence_refs=(),
        plugin_status=DiagnosticExtensionPluginStatus.PLUGIN_UNAVAILABLE,
    )


__all__ = [
    "DEFAULT_EXTENSION_TIME_BUDGET_MS",
    "DiagnosticExtensionService",
]
