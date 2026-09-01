# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH,
    MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS,
    MAX_FUNCTIONAL_OPERATOR_FAILURES,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticAssessmentBuilder,
    DiagnosticAssessmentIntegrityError,
    DiagnosticCertainty,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.diagnostic_assessment_composer import (
    DiagnosticAssessmentComposer,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticCheckResult,
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus,
    FunctionalOperatorProjector,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyAnalyzer,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_SPEC_ID = FunctionalDiagnosticSpecificationId("fdspec_a0000000000000000000000000000001")
_CHECK_SEARCH = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000001")
_CHECK_CANDIDATES = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000002")
_CHECK_SELECTION = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000003")
_CHECK_SYNTHESIS = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000004")
_CHECK_TOOL = FunctionalDiagnosticCheckId("fdcheck_b0000000000000000000000000000001")
_CHECK_WEB = FunctionalDiagnosticCheckId("fdcheck_c0000000000000000000000000000001")
_CHECK_MODEL = FunctionalDiagnosticCheckId("fdcheck_d0000000000000000000000000000001")
_CHECK_FAIL_B = FunctionalDiagnosticCheckId("fdcheck_e0000000000000000000000000000001")
_CHECK_FAIL_C = FunctionalDiagnosticCheckId("fdcheck_f0000000000000000000000000000001")

_RECONSTRUCTOR = ExecutionReconstructor(
    runtime_events=InMemoryRuntimeEventStore(),
    causal_evidence=InMemoryCausalEvidencePersistence(),
)
_LIFECYCLE_ANALYZER = LifecycleAnomalyAnalyzer()
_ASSESSMENT_BUILDER = DiagnosticAssessmentBuilder()
_COMPOSER = DiagnosticAssessmentComposer()
_PROJECTOR = FunctionalOperatorProjector()
def _check_result(
    check_id: FunctionalDiagnosticCheckId,
    status: FunctionalDiagnosticCheckStatus,
    *,
    claim: str = "bounded factual claim",
    refs: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
) -> FunctionalDiagnosticCheckResult:
    return FunctionalDiagnosticCheckResult(
        check_id=check_id,
        status=status,
        factual_claim=claim,
        supporting_evidence_refs=tuple(mint_event_id() for _ in refs),
        limitations=limitations,
    )


def _analysis(
    *,
    tenant_id: str = _TENANT,
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: AttemptId | None = None,
    specification_version: int = 1,
    check_results: tuple[FunctionalDiagnosticCheckResult, ...],
    first_proven_failure: FunctionalDiagnosticCheckId | None = None,
    limitations: tuple[str, ...] = (),
) -> FunctionalDiagnosticAnalysis:
    return FunctionalDiagnosticAnalysis(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id,
        specification_id=_SPEC_ID,
        specification_version=specification_version,
        check_results=check_results,
        first_proven_failure=first_proven_failure,
        limitations=limitations,
    )


def _empty_lifecycle_assessment(
    *,
    tenant_id: str = _TENANT,
    task_id: str | None = None,
    run_id: str | None = None,
) -> DiagnosticAssessment:
    task = task_id or mint_task_id()
    run = run_id or mint_run_id()
    reconstruction = _RECONSTRUCTOR.reconstruct_execution(tenant_id, task, run)
    lifecycle = _LIFECYCLE_ANALYZER.analyze(reconstruction)
    return _ASSESSMENT_BUILDER.assess(reconstruction, lifecycle)


def _append_event(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: str,
    run_id: str,
    attempt_id: str,
    event_type: RuntimeEventType,
) -> None:
    event = sample_runtime_event(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    ).model_copy(update={"event_type": event_type})
    store.append(event, tenant_id=tenant_id)


def _lifecycle_assessment_with_multiple_terminal_outcomes(
    *,
    tenant_id: str = _TENANT,
) -> DiagnosticAssessment:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.TASK_COMPLETED,
    ):
        _append_event(
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=event_type,
        )
    reconstructor = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reconstructor.reconstruct_execution(tenant_id, task_id, run_id)
    lifecycle = _LIFECYCLE_ANALYZER.analyze(reconstruction)
    return _ASSESSMENT_BUILDER.assess(reconstruction, lifecycle)


def test_o1_proven_functional_failure_projects_operator_failure() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search ok"),
            _check_result(
                _CHECK_SELECTION,
                FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
                claim="selected artifact did not match expected artifact",
                refs=("evidence-1",),
            ),
        ),
        first_proven_failure=_CHECK_SELECTION,
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE
    assert len(projection.failures) == 1
    assert projection.failures[0].check_id == _CHECK_SELECTION
    assert projection.failures[0].supporting_evidence_refs


def test_o2_all_checks_proven_pass_projects_proven_success() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search ok"),
            _check_result(_CHECK_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="candidates ok"),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS
    assert projection.summary.passed == 2
    assert not projection.failures


def test_o3_insufficient_evidence_is_inconclusive() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search ok"),
            _check_result(
                _CHECK_SELECTION,
                FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
                claim="not enough evidence to prove selection",
            ),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    assert not projection.failures
    assert projection.limitations[0].status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE


def test_o4_blocked_upstream_is_limitation_not_failure() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search ok"),
            _check_result(
                _CHECK_SYNTHESIS,
                FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM,
                claim="blocked by upstream dependency",
            ),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    assert not projection.failures
    assert projection.limitations[0].status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_o5_multiple_failures_retained() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_FAIL_B, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="check b failed"),
            _check_result(_CHECK_FAIL_C, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="check c failed"),
        ),
        first_proven_failure=_CHECK_FAIL_B,
    )
    projection = _PROJECTOR.project(analysis)
    assert len(projection.failures) == 2
    assert {failure.check_id for failure in projection.failures} == {_CHECK_FAIL_B, _CHECK_FAIL_C}


def test_o6_first_proven_failed_check_not_root_cause() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_FAIL_B, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="b failed"),
            _check_result(_CHECK_FAIL_C, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="c failed"),
        ),
        first_proven_failure=_CHECK_FAIL_B,
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.first_proven_failed_check == _CHECK_FAIL_B
    field_names = {field.name for field in projection.__dataclass_fields__.values()}
    assert "root_cause" not in field_names


def test_o7_spec_id_and_version_retained() -> None:
    analysis = _analysis(
        specification_version=3,
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="ok"),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.specification_id == _SPEC_ID
    assert projection.specification_version == 3


def test_o8_attempt_id_retained() -> None:
    attempt = mint_attempt_id()
    analysis = _analysis(
        attempt_id=attempt,
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="ok"),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.attempt_id == attempt


def test_o9_supporting_refs_bounded() -> None:
    refs = tuple(f"ref-{index}" for index in range(MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS + 4))
    analysis = _analysis(
        check_results=(
            _check_result(
                _CHECK_SELECTION,
                FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
                claim="failed",
                refs=refs,
            ),
        ),
        first_proven_failure=_CHECK_SELECTION,
    )
    projection = _PROJECTOR.project(analysis)
    assert len(projection.failures[0].supporting_evidence_refs) == MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS


def test_o10_lifecycle_findings_remain_unchanged() -> None:
    lifecycle_assessment = _lifecycle_assessment_with_multiple_terminal_outcomes()
    assert lifecycle_assessment.has_findings
    original_finding_kinds = tuple(finding.kind for finding in lifecycle_assessment.findings)

    composed = _COMPOSER.compose(
        lifecycle_assessment,
        _analysis(
            tenant_id=lifecycle_assessment.tenant_id,
            task_id=lifecycle_assessment.task_id,
            run_id=lifecycle_assessment.run_id,
            check_results=(
                _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="failed"),
            ),
            first_proven_failure=_CHECK_SEARCH,
        ),
    )
    assert tuple(finding.kind for finding in composed.lifecycle_assessment.findings) == original_finding_kinds
    assert composed.lifecycle_assessment.findings[0].kind is DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES


def test_o11_functional_and_lifecycle_findings_coexist() -> None:
    lifecycle_assessment = _lifecycle_assessment_with_multiple_terminal_outcomes()
    composed = _COMPOSER.compose(
        lifecycle_assessment,
        _analysis(
            tenant_id=lifecycle_assessment.tenant_id,
            task_id=lifecycle_assessment.task_id,
            run_id=lifecycle_assessment.run_id,
            check_results=(
                _check_result(_CHECK_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="selection fail"),
            ),
            first_proven_failure=_CHECK_SELECTION,
        ),
    )
    assert composed.has_lifecycle_findings
    assert composed.functional_projection is not None
    assert composed.functional_projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE


def test_o12_no_functional_analysis_preserves_lifecycle_only() -> None:
    lifecycle_assessment = _empty_lifecycle_assessment()
    composed = _COMPOSER.compose(lifecycle_assessment, None)
    assert composed.functional_projection is None
    assert composed.lifecycle_assessment == lifecycle_assessment


def test_o13_retrieval_like_generic_projection() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search completed"),
            _check_result(_CHECK_CANDIDATES, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="candidates available"),
            _check_result(_CHECK_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="wrong candidate"),
        ),
        first_proven_failure=_CHECK_SELECTION,
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE
    assert projection.first_proven_failed_check == _CHECK_SELECTION


def test_o14_tool_generic_projection() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_TOOL, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="tool invocation failed"),
        ),
        first_proven_failure=_CHECK_TOOL,
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.failures[0].check_id == _CHECK_TOOL


def test_o15_web_search_generic_projection() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_WEB, FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE, claim="no web evidence"),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    assert projection.limitations[0].check_id == _CHECK_WEB


def test_o16_model_routing_generic_projection() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_MODEL, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="model route ok"),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_SUCCESS


def test_completed_execution_with_functional_failure_are_independent() -> None:
    lifecycle_assessment = _empty_lifecycle_assessment()
    assert not lifecycle_assessment.has_findings

    composed = _COMPOSER.compose(
        lifecycle_assessment,
        _analysis(
            tenant_id=lifecycle_assessment.tenant_id,
            task_id=lifecycle_assessment.task_id,
            run_id=lifecycle_assessment.run_id,
            check_results=(
                _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="search ok"),
                _check_result(_CHECK_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim="selection fail"),
            ),
            first_proven_failure=_CHECK_SELECTION,
        ),
    )
    assert not composed.has_lifecycle_findings
    assert composed.functional_projection is not None
    assert (
        composed.functional_projection.outcome_status
        is FunctionalOperatorOutcomeStatus.PROVEN_FUNCTIONAL_FAILURE
    )


def test_scope_mismatch_fail_closed() -> None:
    lifecycle_assessment = _empty_lifecycle_assessment()
    mismatched_analysis = _analysis(
        tenant_id="other-tenant",
        check_results=(
            _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="ok"),
        ),
    )
    with pytest.raises(DiagnosticAssessmentIntegrityError):
        _COMPOSER.compose(lifecycle_assessment, mismatched_analysis)


def test_not_evaluated_safely_represented() -> None:
    analysis = _analysis(
        check_results=(
            _check_result(
                _CHECK_SYNTHESIS,
                FunctionalDiagnosticCheckStatus.NOT_EVALUATED,
                claim="check not reached",
            ),
        ),
    )
    projection = _PROJECTOR.project(analysis)
    assert projection.outcome_status is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
    assert projection.limitations[0].status is FunctionalDiagnosticCheckStatus.NOT_EVALUATED


def test_claim_length_bounded() -> None:
    long_claim = "x" * (MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH + 20)
    analysis = _analysis(
        check_results=(
            _check_result(_CHECK_SELECTION, FunctionalDiagnosticCheckStatus.PROVEN_FAIL, claim=long_claim),
        ),
        first_proven_failure=_CHECK_SELECTION,
    )
    projection = _PROJECTOR.project(analysis)
    assert len(projection.failures[0].factual_claim) == MAX_FUNCTIONAL_DIAGNOSTIC_CLAIM_LENGTH


def test_failure_projection_bounded() -> None:
    failures = tuple(
        _check_result(
            FunctionalDiagnosticCheckId(f"fdcheck_{index:032x}"),
            FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
            claim=f"fail-{index}",
        )
        for index in range(MAX_FUNCTIONAL_OPERATOR_FAILURES + 3)
    )
    analysis = _analysis(check_results=failures, first_proven_failure=failures[0].check_id)
    projection = _PROJECTOR.project(analysis)
    assert len(projection.failures) == MAX_FUNCTIONAL_OPERATOR_FAILURES


def test_lifecycle_finding_certainty_unchanged() -> None:
    lifecycle_assessment = _lifecycle_assessment_with_multiple_terminal_outcomes()
    composed = _COMPOSER.compose(
        lifecycle_assessment,
        _analysis(
            tenant_id=lifecycle_assessment.tenant_id,
            task_id=lifecycle_assessment.task_id,
            run_id=lifecycle_assessment.run_id,
            check_results=(
                _check_result(_CHECK_SEARCH, FunctionalDiagnosticCheckStatus.PROVEN_PASS, claim="ok"),
            ),
        ),
    )
    assert composed.lifecycle_assessment.findings[0].certainty is DiagnosticCertainty.PROVEN
