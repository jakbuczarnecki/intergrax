# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-04, F-D5-11): no green evidence or clean comparison without evidence."""

from __future__ import annotations

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalRun,
    EvalScenarioTelemetry,
    EvaluatorKind,
    EvaluatorOutcome,
    EvaluatorOutcomeStatus,
)
from intergrax.eval.eval_comparison import compare_eval_runs
from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceStatus,
    build_eval_run_evidence_report,
)
from pydantic import ValidationError
from tests.qualification.pdc_x.d5._support import build_run, provenance

pytestmark = pytest.mark.unit


def _run(
    outcomes: tuple[EvaluatorOutcome, ...],
    telemetry: tuple[EvalScenarioTelemetry, ...],
    *,
    code_sha: str | None = None,
    dataset_digest: str | None = None,
) -> EvalRun:
    base = provenance()
    return EvalRun(
        tenant_id=base.tenant_id,
        agent=base.agent,
        model=base.model,
        prompt=base.prompt,
        profile=base.profile,
        code_sha=code_sha or base.code_sha,
        dataset_id="d5r.sample",
        dataset_version="1.0.0",
        dataset_digest=dataset_digest or "sha256:" + "a" * 64,
        seed=base.seed,
        outcomes=outcomes,
        telemetry=telemetry,
    )


def _out(
    scenario: str,
    status: EvaluatorOutcomeStatus = EvaluatorOutcomeStatus.PASSED,
    kind: EvaluatorKind = EvaluatorKind.SAFETY,
    evaluator_id: str = "safety.scan",
) -> EvaluatorOutcome:
    return EvaluatorOutcome(evaluator_id=evaluator_id, kind=kind, scenario_id=scenario, status=status)


def _tel(scenario: str, error: str | None = None) -> EvalScenarioTelemetry:
    return EvalScenarioTelemetry(scenario_id=scenario, output_digest="sha256:" + "b" * 64, error=error)


def _status(run: EvalRun) -> EvalEvidenceStatus:
    return build_eval_run_evidence_report(run).results[0].status


def test_executor_failure_is_failed_evidence_not_passed() -> None:
    run = _run((_out("s1"),), (_tel("s1", error="executor failed: RuntimeError"),))
    assert _status(run) is EvalEvidenceStatus.FAILED


def test_complete_clean_run_is_still_passed() -> None:
    run = _run((_out("s1"), _out("s2")), (_tel("s1"), _tel("s2")))
    assert _status(run) is EvalEvidenceStatus.PASSED


def test_run_without_rule_outcomes_is_unavailable() -> None:
    assert _status(_run((), (_tel("s1"),))) is EvalEvidenceStatus.UNAVAILABLE
    assert _status(_run((), ())) is EvalEvidenceStatus.UNAVAILABLE


def test_scenario_without_rule_outcome_is_unavailable() -> None:
    run = _run((_out("s1"),), (_tel("s1"), _tel("s2")))
    assert _status(run) is EvalEvidenceStatus.UNAVAILABLE


def test_rule_failure_stays_failed() -> None:
    run = _run((_out("s1", EvaluatorOutcomeStatus.FAILED),), (_tel("s1"),))
    assert _status(run) is EvalEvidenceStatus.FAILED


def test_pending_human_review_is_never_passed_evidence() -> None:
    human = _out("s1", EvaluatorOutcomeStatus.PENDING, EvaluatorKind.HUMAN, "human.review")
    run = _run((_out("s1"), human), (_tel("s1"),))
    assert run.metrics().human_pending == 1
    assert _status(run) is EvalEvidenceStatus.UNAVAILABLE


def test_empty_runs_do_not_compare_clean() -> None:
    comparison = compare_eval_runs(_run((), ()), _run((), ()))
    assert not comparison.clean
    assert comparison.threshold_breached


def test_candidate_missing_a_baseline_scenario_is_not_clean() -> None:
    baseline = _run((_out("s1"), _out("s2")), (_tel("s1"), _tel("s2")))
    candidate = _run((_out("s1"),), (_tel("s1"),))
    comparison = compare_eval_runs(baseline, candidate)
    assert not comparison.clean
    assert any(item.metric == "coverage" for item in comparison.threshold_breaches)


def test_candidate_scenario_without_rule_outcome_is_not_clean() -> None:
    baseline = _run((_out("s1"),), (_tel("s1"),))
    candidate = _run((), (_tel("s1"),))
    assert not compare_eval_runs(baseline, candidate).clean


def test_candidate_with_pending_human_review_is_not_clean() -> None:
    human = _out("s1", EvaluatorOutcomeStatus.PENDING, EvaluatorKind.HUMAN, "human.review")
    baseline = _run((_out("s1"),), (_tel("s1"),))
    candidate = _run((_out("s1"), human), (_tel("s1"),))
    comparison = compare_eval_runs(baseline, candidate)
    assert not comparison.clean
    assert any(item.metric == "human_review_pending" for item in comparison.threshold_breaches)


def test_identical_complete_runs_are_still_clean() -> None:
    assert compare_eval_runs(build_run(), build_run()).clean
    run = _run((_out("s1"),), (_tel("s1"),))
    assert compare_eval_runs(run, run).clean


@pytest.mark.parametrize("suffix", ["\n", "\r\n"])
def test_code_sha_with_trailing_newline_is_rejected(suffix: str) -> None:
    with pytest.raises(ValidationError):
        _run((), (), code_sha="abcdef1" + suffix)
    assert _run((), (), code_sha="abcdef1").code_sha == "abcdef1"


def test_dataset_digest_with_trailing_newline_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _run((), (), dataset_digest="sha256:" + "a" * 64 + "\n")
