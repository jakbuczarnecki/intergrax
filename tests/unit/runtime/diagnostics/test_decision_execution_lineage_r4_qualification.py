# © Artur Czarnecki. All rights reserved.

"""DIAG R4 qualification matrix — Decision↔Execution diagnostic lineage (A1–A7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationKind,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.runtime.diagnostics.decision_context_read_models import (
    DecisionContextFact,
    DecisionContextReadStatus,
    DecisionContextUnavailableReason,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from intergrax.runtime.diagnostics.diagnostic_precision import DiagnosticPrecision
from intergrax.contracts.execution_identity import mint_attempt_id
from intergrax.runtime.execution.runtime import RootExecutionContext
from testing_support.runtime.decision_execution_lineage_r4_harness import (
    build_decision_execution_lineage_r4_harness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_SYMBOLS = (
    "DecisionDiagnosticEngine",
    "DecisionProblemStore",
    "DecisionFailureCause",
)


def _identity_for_context(
    harness,
    context: RootExecutionContext,
    *,
    decision_id: str | None = None,
    attempt_id: object | None = None,
    tenant_id: str | None = None,
) -> DecisionIdentity:
    resolved_attempt = attempt_id or context.attempt_id
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=initial_decision_version(),
        scope=DecisionScope(namespace="qualification", subject="r4"),
        tenant_id=tenant_id or harness.tenant_id,
        execution=DecisionExecutionLineage(
            task_id=harness.execution.task_id,
            run_id=context.run_id,
            attempt_id=resolved_attempt,
            execution_id=context.execution_id,
        ),
    )


def _failure_findings(harness):
    listed = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    findings = []
    for summary in listed.problems:
        detail = harness.read_service.get_problem(
            tenant_id=harness.tenant_id,
            problem_id=summary.problem_id,
        )
        assert detail is not None
        for occurrence in detail.occurrences:
            if occurrence.assessment is None:
                continue
            findings.extend(occurrence.assessment.findings)
    return findings, listed


@pytest.mark.asyncio
async def test_r4_a1_decision_successful_execution_correlation_no_problem() -> None:
    harness = build_decision_execution_lineage_r4_harness()

    class _Leaf:
        async def execute(self, request: object) -> str:
            return "ok"

    class _Root:
        async def execute(self, request: object) -> str:
            return await harness.execution.child_runner.execute(
                request=object(),
                delegate=_Leaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    identity = _identity_for_context(harness, context)
    harness.append_correlation(identity)
    await harness.execution.runtime.execute(object(), context)
    harness.execution.seed_terminal_lifecycle_events(context, failed=False)

    records = harness.correlation_persistence.query_by_execution_scope(
        tenant_id=harness.tenant_id,
        task_id=harness.execution.task_id,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        execution_id=context.execution_id,
    )
    assert len(records) == 1
    listed = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    assert listed.problems == ()


@pytest.mark.asyncio
async def test_r4_a2_failed_execution_decision_context_not_cause() -> None:
    harness = build_decision_execution_lineage_r4_harness()
    decision_id = mint_decision_id()

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r4-a2")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    identity = _identity_for_context(harness, context, decision_id=decision_id)
    harness.append_correlation(identity)
    harness.bind_contextual_facts(
        str(decision_id),
        (DecisionContextFact(key="selected_provider", value="openai"),),
    )

    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    findings, listed = _failure_findings(harness)
    assert listed.problems
    exec_failed = [
        finding
        for finding in findings
        if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED
    ]
    assert exec_failed
    assert exec_failed[0].precision is DiagnosticPrecision.EXECUTION_LEVEL
    assert all("decision" not in finding.claim.lower() for finding in findings)

    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=listed.problems[0].problem_id,
    )
    assert detail is not None
    occurrence = detail.occurrences[0]
    assert occurrence.decision_context is not None
    assert (
        occurrence.decision_context.read_status is DecisionContextReadStatus.AVAILABLE
    )
    assert len(occurrence.decision_context.related_decisions) == 1
    related = occurrence.decision_context.related_decisions[0]
    assert related.decision_id == decision_id
    assert related.contextual_facts[0].value == "openai"


@pytest.mark.asyncio
async def test_r4_a3_multiple_decisions_no_causal_inference() -> None:
    harness = build_decision_execution_lineage_r4_harness()

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r4-a3")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    harness.append_correlation(_identity_for_context(harness, context))
    harness.append_correlation(_identity_for_context(harness, context))

    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=_failure_findings(harness)[1].problems[0].problem_id,
    )
    assert detail is not None
    ctx = detail.occurrences[0].decision_context
    assert ctx is not None
    assert len(ctx.related_decisions) == 2
    assert all(
        "not causal" in limitation.lower() or "contextual" in limitation.lower()
        for limitation in ctx.limitations
    )


@pytest.mark.asyncio
async def test_r4_a4_retry_same_decision_different_attempts() -> None:
    harness = build_decision_execution_lineage_r4_harness()
    decision_id = mint_decision_id()
    context = harness.execution.resolve_root_context()
    retry_attempt = mint_attempt_id()

    harness.append_correlation(
        _identity_for_context(
            harness,
            context,
            decision_id=decision_id,
            attempt_id=context.attempt_id,
        ),
        kind=DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION,
    )
    harness.append_correlation(
        _identity_for_context(
            harness,
            context,
            decision_id=decision_id,
            attempt_id=retry_attempt,
        ),
        kind=DecisionExecutionCorrelationKind.DECISION_RETRY_ATTEMPT,
    )

    records = harness.correlation_persistence.query_by_execution_scope(
        tenant_id=harness.tenant_id,
        task_id=harness.execution.task_id,
        run_id=context.run_id,
    )
    assert len(records) == 2
    assert {record.decision_id for record in records} == {decision_id}
    assert {record.decision_attempt_id for record in records} == {
        context.attempt_id,
        retry_attempt,
    }


def test_r4_a5_cross_tenant_correlation_isolated() -> None:
    harness = build_decision_execution_lineage_r4_harness()
    context = harness.execution.resolve_root_context()
    identity = _identity_for_context(harness, context, tenant_id="tenant-a")
    harness.append_correlation(identity)

    records = harness.correlation_persistence.query_by_execution_scope(
        tenant_id="tenant-b",
        task_id=harness.execution.task_id,
        run_id=context.run_id,
    )
    assert records == ()


@pytest.mark.asyncio
async def test_r4_a6_missing_decision_evidence_execution_diagnostic_works() -> None:
    harness = build_decision_execution_lineage_r4_harness()

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r4-a6")

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

    findings, listed = _failure_findings(harness)
    assert any(
        finding.kind is DiagnosticFindingKind.EXECUTION_FAILED for finding in findings
    )
    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=listed.problems[0].problem_id,
    )
    assert detail is not None
    ctx = detail.occurrences[0].decision_context
    assert ctx is not None
    assert ctx.read_status is DecisionContextReadStatus.UNAVAILABLE
    assert (
        ctx.unavailable_reason
        is DecisionContextUnavailableReason.NO_CORRELATION_EVIDENCE
    )


@pytest.mark.asyncio
async def test_r4_a7_decision_persistence_outage_degraded_enrichment() -> None:
    harness = build_decision_execution_lineage_r4_harness()
    harness.decision_context_provider.unavailable = True

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r4-a7")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    harness.append_correlation(_identity_for_context(harness, context))
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)

    detail = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=_failure_findings(harness)[1].problems[0].problem_id,
    )
    assert detail is not None
    ctx = detail.occurrences[0].decision_context
    assert ctx is not None
    assert ctx.read_status is DecisionContextReadStatus.DEGRADED
    assert any(
        finding.kind is DiagnosticFindingKind.EXECUTION_FAILED
        for finding in (detail.occurrences[0].assessment.findings if detail.occurrences[0].assessment else ())
    )


@pytest.mark.unit
@pytest.mark.gate
def test_r4_quality_gates_forbidden_symbols_and_single_engine() -> None:
    intergrax_root = _REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_SYMBOLS:
            if symbol in text:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{symbol}")
    assert hits == []

    for member in DiagnosticFindingKind:
        assert "decision" not in member.value
