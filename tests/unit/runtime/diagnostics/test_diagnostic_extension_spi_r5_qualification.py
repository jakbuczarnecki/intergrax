# © Artur Czarnecki. All rights reserved.

"""DIAG R5 qualification matrix — extension SPI (A1–A7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.diagnostic_analyzer import (
    DiagnosticExtensionCertainty,
    DiagnosticFindingCandidate,
    DiagnosticFindingScope,
)
from intergrax.contracts.diagnostic_extension_evidence import DiagnosticExtensionEvidence
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from intergrax.runtime.diagnostics.diagnostic_extension_read_models import (
    DiagnosticExtensionPluginStatus,
    DiagnosticExtensionReadStatus,
)
from intergrax.runtime.diagnostics.diagnostic_extension_registry import (
    DiagnosticExtensionConfigurationError,
    DiagnosticExtensionRegistry,
)
from intergrax.runtime.diagnostics.diagnostic_extension_evidence_store import (
    InMemoryDiagnosticExtensionEvidenceStore,
)
from intergrax.runtime.diagnostics.diagnostic_extension_service import (
    DiagnosticExtensionService,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from testing_support.runtime.diagnostic_extension_spi_r5_harness import (
    _DEFAULT_NAMESPACE,
    build_diagnostic_extension_spi_r5_harness,
)
from testing_support.runtime.execution_failure_evidence_r2_closure_harness import (
    build_execution_failure_evidence_r2_closure_harness,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    read_service_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_SYMBOLS = (
    "ApplicationDiagnosticEngine",
    "ApplicationProblemStore",
    "OwnDiagnosticEngine",
)


class _EvidenceContributor:
    contributor_id = "r5-contributor"
    evidence_namespace = _DEFAULT_NAMESPACE
    priority = 0

    def collect(
        self,
        context: DiagnosticEvidenceContext,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        scope = context.to_evidence_scope()
        return (
            DiagnosticExtensionEvidence.mint(
                scope=scope,
                evidence_namespace=self.evidence_namespace,
                kind=f"{self.evidence_namespace}.sap.timeout",
                summary="timeout after 30s",
            ),
        )


class _TimeoutAnalyzer:
    analyzer_id = "r5-analyzer"
    analyzer_namespace = _DEFAULT_NAMESPACE
    priority = 0

    def analyze(
        self,
        evidence: tuple[DiagnosticExtensionEvidence, ...],
    ) -> tuple[DiagnosticFindingCandidate, ...]:
        for item in evidence:
            if item.kind.endswith("sap.timeout"):
                return (
                    DiagnosticFindingCandidate(
                        kind=f"{self.analyzer_namespace}.connector.timeout",
                        confidence=DiagnosticExtensionCertainty.SUPPORTED,
                        evidence_refs=(item.evidence_id,),
                        scope=DiagnosticFindingScope(
                            tenant_id=item.scope.tenant_id,
                            evidence_namespace=item.evidence_namespace,
                        ),
                        summary="SAP connector timeout pattern",
                    ),
                )
        return ()


class _CrashingAnalyzer:
    analyzer_id = "r5-crash"
    analyzer_namespace = _DEFAULT_NAMESPACE
    priority = 1

    def analyze(
        self,
        evidence: tuple[DiagnosticExtensionEvidence, ...],
    ) -> tuple[DiagnosticFindingCandidate, ...]:
        raise RuntimeError("analyzer crash")


class _CrossTenantContributor:
    contributor_id = "cross-tenant"
    evidence_namespace = _DEFAULT_NAMESPACE
    priority = 0

    def collect(
        self,
        context: DiagnosticEvidenceContext,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        scope = context.to_evidence_scope()
        bad_scope = type(scope)(
            tenant_id="tenant-other",
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            execution_id=scope.execution_id,
        )
        return (
            DiagnosticExtensionEvidence.mint(
                scope=bad_scope,
                evidence_namespace=self.evidence_namespace,
                kind=f"{self.evidence_namespace}.leak",
                summary="must not persist",
            ),
        )


@pytest.mark.asyncio
async def test_r5_a1_plugin_contributes_evidence_problem_unchanged() -> None:
    registry = DiagnosticExtensionRegistry(evidence_contributors=(_EvidenceContributor(),))
    harness = build_diagnostic_extension_spi_r5_harness(registry=registry)

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r5-a1")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    listed = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    assert len(listed.problems) == 1
    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=listed.problems[0].problem_id,
    )
    assert detail is not None
    enrichment = detail.occurrences[0].extension_enrichment
    assert enrichment is not None
    assert enrichment.contributed_evidence
    assert harness.read_service.list_problems(tenant_id=harness.tenant_id).total_count == 1


@pytest.mark.asyncio
async def test_r5_a2_analyzer_finding_candidate_problem_lifecycle_owns_state() -> None:
    registry = DiagnosticExtensionRegistry(
        evidence_contributors=(_EvidenceContributor(),),
        analyzers=(_TimeoutAnalyzer(),),
    )
    harness = build_diagnostic_extension_spi_r5_harness(registry=registry)

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r5-a2")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    listed = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    assert listed.problems
    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=listed.problems[0].problem_id,
    )
    assert detail is not None
    occurrence = detail.occurrences[0]
    assert occurrence.assessment is not None
    assert any(
        f.kind is DiagnosticFindingKind.EXECUTION_FAILED
        for f in occurrence.assessment.findings
    )
    enrichment = occurrence.extension_enrichment
    assert enrichment is not None
    assert enrichment.read_status is DiagnosticExtensionReadStatus.COMPLETE
    assert enrichment.extension_findings
    assert enrichment.extension_findings[0].kind.endswith("connector.timeout")


@pytest.mark.asyncio
async def test_r5_a3_analyzer_crash_diagnostics_continue_degraded() -> None:
    registry = DiagnosticExtensionRegistry(
        evidence_contributors=(_EvidenceContributor(),),
        analyzers=(_CrashingAnalyzer(),),
    )
    harness = build_diagnostic_extension_spi_r5_harness(registry=registry)

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r5-a3")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=harness.read_service.list_problems(tenant_id=harness.tenant_id).problems[0].problem_id,
    )
    assert detail is not None
    enrichment = detail.occurrences[0].extension_enrichment
    assert enrichment is not None
    assert enrichment.read_status is DiagnosticExtensionReadStatus.DEGRADED
    assert any(
        f.plugin_status is DiagnosticExtensionPluginStatus.PLUGIN_UNAVAILABLE
        for f in enrichment.extension_findings
    )
    assert detail.occurrences[0].assessment is not None


@pytest.mark.asyncio
async def test_r5_a4_tenant_isolation() -> None:
    registry = DiagnosticExtensionRegistry(evidence_contributors=(_CrossTenantContributor(),))
    harness = build_diagnostic_extension_spi_r5_harness(registry=registry)

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r5-a4")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=harness.read_service.list_problems(tenant_id=harness.tenant_id).problems[0].problem_id,
    )
    assert detail is not None
    enrichment = detail.occurrences[0].extension_enrichment
    assert enrichment is not None
    assert enrichment.contributed_evidence == ()


@pytest.mark.unit
@pytest.mark.gate
def test_r5_a5_duplicate_plugin_deterministic_conflict() -> None:
    class _One(_EvidenceContributor):
        contributor_id = "dup"

    class _Two(_EvidenceContributor):
        contributor_id = "dup"

    with pytest.raises(DiagnosticExtensionConfigurationError):
        DiagnosticExtensionRegistry(evidence_contributors=(_One(), _Two()))


@pytest.mark.unit
@pytest.mark.gate
def test_r5_a6_malicious_plugin_has_no_problem_persistence_port() -> None:
    service = DiagnosticExtensionService(
        registry=DiagnosticExtensionRegistry.empty(),
        evidence_store=InMemoryDiagnosticExtensionEvidenceStore(),
    )
    assert not hasattr(service, "problem_persistence")
    assert not isinstance(service.evidence_store, ProblemPersistence)


@pytest.mark.asyncio
async def test_r5_a7_no_extensions_legacy_behavior() -> None:
    execution = build_execution_failure_evidence_r2_closure_harness()
    execution.read_service = read_service_for_tests(
        execution.problem_persistence,
        execution.execution_reconstructor,
        occurrence_persistence=execution.occurrence_persistence,
    )

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r5-a7")

    class _Root:
        async def execute(self, request: object) -> None:
            await execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    execution.bind_root_delegate(_Root())
    context = execution.resolve_root_context()
    try:
        await execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    execution.seed_terminal_lifecycle_events(context, failed=True)
    execution.run_terminal_diagnostics(context)

    detail = execution.read_service.get_problem(
        tenant_id=execution.tenant_id,
        problem_id=execution.read_service.list_problems(tenant_id=execution.tenant_id).problems[0].problem_id,
    )
    assert detail is not None
    assert detail.occurrences[0].extension_enrichment is None


@pytest.mark.unit
@pytest.mark.gate
def test_r5_quality_gates_forbidden_symbols_and_single_engine() -> None:
    intergrax_root = _REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_SYMBOLS:
            if symbol in text:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{symbol}")
    assert hits == []
