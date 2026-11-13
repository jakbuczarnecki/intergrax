# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R005..R007: one evaluator contract, safety via guardrail, isolated judge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalContractError,
    EvalRevisionRef,
    EvalScenario,
    EvalSubject,
    EvalSubjectOutput,
    EvaluatorKind,
    EvaluatorOutcome,
    EvaluatorOutcomeStatus,
)
from intergrax.eval.evaluators import (
    Evaluator,
    EvaluatorRegistry,
    EvaluatorRegistryError,
    ExactMatchEvaluator,
    GoldenRetrievalEvaluator,
    GuardrailSafetyEvaluator,
    register_existing_evaluators,
)
from intergrax.eval.typed_run import execute_eval_run
from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    GuardrailScanResult,
)
from intergrax.runtime.architecture.evaluation_automation import (
    EvaluationSignal,
    evaluate_automated_results,
)
from intergrax.runtime.architecture.evaluation_modes import EvaluationMode, EvaluationModeResult
from intergrax.runtime.evaluation.human_review_sample_queue import HumanReviewSampleQueue
from tests.qualification.pdc_x.d5._support import (
    ANSWERS,
    build_run,
    executor_for,
    judge_fixed,
    make_dataset,
    provenance,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
JUDGE_MODEL = EvalRevisionRef(ref_id="model:judge", revision="j-1")


class FakeGuardrailBackend:
    """External guardrail provider double (the only fake; the evaluator path is real)."""

    slug = "fake-guardrail"

    def __init__(self, blocked_text: str = "") -> None:
        self._blocked = blocked_text
        self.contexts: list[GuardrailContext | None] = []

    def scan_input(self, text, *, context=None):  # pragma: no cover - unused
        return GuardrailScanResult()

    def scan_output(self, text, *, context=None, prompt=None):
        self.contexts.append(context)
        if self._blocked and self._blocked in text:
            return GuardrailScanResult(allowed=False, categories=("harm",), detail="harmful output")
        return GuardrailScanResult()

    def scan_tool_call(self, tool_name, arguments, *, context=None):  # pragma: no cover
        return GuardrailScanResult()

    def health_check(self) -> bool:
        return True


def _full_registry(queue: HumanReviewSampleQueue | None = None) -> EvaluatorRegistry:
    return register_existing_evaluators(
        EvaluatorRegistry(),
        guardrail_backend=FakeGuardrailBackend(),
        judge=("judge.groundedness", judge_fixed(0.9), JUDGE_MODEL),
        review_queue=queue or HumanReviewSampleQueue(),
        review_agent_id="agent:research",
    )


def test_r005_registry_covers_all_four_kinds() -> None:
    registry = _full_registry()
    assert registry.kinds_covered() == frozenset(EvaluatorKind)
    for evaluator in registry.evaluators():
        assert isinstance(evaluator, Evaluator)


def test_r005_default_registration_registers_existing_deterministic_evaluators() -> None:
    registry = register_existing_evaluators(EvaluatorRegistry())
    ids = {item.evaluator_id for item in registry.by_kind(EvaluatorKind.DETERMINISTIC)}
    assert ids == {
        "task_success.exact_match",
        "tool_use.expected_tools",
        "citation.expected_citations",
        "retrieval.golden_harness",
    }
    assert registry.by_kind(EvaluatorKind.SAFETY) == ()


def test_r005_duplicate_and_non_contract_registration_rejected() -> None:
    registry = EvaluatorRegistry()
    registry.register(ExactMatchEvaluator())
    with pytest.raises(EvaluatorRegistryError):
        registry.register(ExactMatchEvaluator())
    with pytest.raises(EvaluatorRegistryError):
        registry.register(object())  # type: ignore[arg-type]


def test_r005_existing_golden_harness_runs_through_the_contract() -> None:
    cases = json.loads((REPO_ROOT / "tests/fixtures/rag_golden/retrieval_cases.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    case = cases[0]
    scenario = EvalScenario(scenario_id=f"golden:{case['name']}", input_payload=case)
    subject = EvalSubject(
        scenario=scenario,
        run_ref="evalrun:test",
        output=EvalSubjectOutput(),
        seed=1,
    )
    outcome = GoldenRetrievalEvaluator().evaluate(subject)
    assert outcome is not None
    assert outcome.kind is EvaluatorKind.DETERMINISTIC
    assert outcome.status is EvaluatorOutcomeStatus.PASSED


def test_r006_safety_kind_via_guardrail_provider() -> None:
    backend = FakeGuardrailBackend(blocked_text="BADWORD")
    evaluator = GuardrailSafetyEvaluator(backend)
    assert evaluator.kind is EvaluatorKind.SAFETY
    scenario = EvalScenario(scenario_id="s-safe")

    def subject(text: str) -> EvalSubject:
        return EvalSubject(
            scenario=scenario,
            run_ref="evalrun:t",
            output=EvalSubjectOutput(output_text=text),
            seed=1,
        )

    safe = evaluator.evaluate(subject("hello"))
    unsafe = evaluator.evaluate(subject("contains BADWORD here"))
    assert safe is not None and safe.passed
    assert unsafe is not None and not unsafe.passed
    assert "harm" in unsafe.reasons
    assert backend.contexts[0] is not None and backend.contexts[0].run_id == "evalrun:t"


def test_r006_safety_failure_lands_in_the_typed_run() -> None:
    backend = FakeGuardrailBackend(blocked_text="Paris")
    registry = register_existing_evaluators(EvaluatorRegistry(), guardrail_backend=backend)
    run = build_run(registry=registry)
    safety = [o for o in run.outcomes if o.kind is EvaluatorKind.SAFETY]
    assert {o.scenario_id: o.passed for o in safety} == {
        "s-math": True,
        "s-capital": False,
        "s-tool": True,
    }
    assert run.metrics().safety_pass_rate == pytest.approx(2 / 3)


def test_r007_judge_results_are_isolated_from_rule_results() -> None:
    high = build_run(
        registry=register_existing_evaluators(
            EvaluatorRegistry(),
            judge=("judge.groundedness", judge_fixed(0.95), JUDGE_MODEL),
        )
    )
    low = build_run(
        registry=register_existing_evaluators(
            EvaluatorRegistry(),
            judge=("judge.groundedness", judge_fixed(0.10), JUDGE_MODEL),
        )
    )
    assert high.metrics().judge_pass_rate == 1.0
    assert low.metrics().judge_pass_rate == 0.0
    # The judge verdict never changes any rule metric or rule evidence.
    assert high.metrics().deterministic_pass_rate == low.metrics().deterministic_pass_rate
    assert high.rule_outcomes == low.rule_outcomes
    assert high.scenario_successes() == low.scenario_successes()
    assert all(o.kind is EvaluatorKind.MODEL_ASSISTED for o in low.judge_outcomes)
    assert all(o.judge_model == JUDGE_MODEL for o in low.judge_outcomes)


def test_r007_judge_receives_only_the_subject() -> None:
    seen: list[EvalSubject] = []

    def spy(subject: EvalSubject) -> tuple[float, str]:
        seen.append(subject)
        return 1.0, ""

    registry = register_existing_evaluators(EvaluatorRegistry(), judge=("judge.spy", spy, JUDGE_MODEL))
    execute_eval_run(
        dataset=make_dataset(),
        executor=executor_for(ANSWERS),
        registry=registry,
        provenance=provenance(),
    )
    assert len(seen) == 3
    assert set(EvalSubject.model_fields) == {"scenario", "run_ref", "output", "seed"}


def test_r007_outcome_must_match_its_evaluator_kind() -> None:
    class LyingEvaluator:
        evaluator_id = "rule.liar"
        kind = EvaluatorKind.DETERMINISTIC

        def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome:
            return EvaluatorOutcome(
                evaluator_id=self.evaluator_id,
                kind=EvaluatorKind.MODEL_ASSISTED,
                scenario_id=subject.scenario.scenario_id,
                status=EvaluatorOutcomeStatus.PASSED,
                judge_model=JUDGE_MODEL,
            )

    registry = EvaluatorRegistry()
    registry.register(LyingEvaluator())
    with pytest.raises(EvalContractError, match="does not match"):
        execute_eval_run(
            dataset=make_dataset(),
            executor=executor_for(ANSWERS),
            registry=registry,
            provenance=provenance(),
        )


def test_r007_model_assisted_outcome_requires_judge_model() -> None:
    with pytest.raises(ValueError, match="judge_model"):
        EvaluatorOutcome(
            evaluator_id="j",
            kind=EvaluatorKind.MODEL_ASSISTED,
            scenario_id="s",
            status=EvaluatorOutcomeStatus.PASSED,
        )
    with pytest.raises(ValueError, match="judge_model"):
        EvaluatorOutcome(
            evaluator_id="r",
            kind=EvaluatorKind.DETERMINISTIC,
            scenario_id="s",
            status=EvaluatorOutcomeStatus.PASSED,
            judge_model=JUDGE_MODEL,
        )


def test_r007_human_pending_goes_to_existing_queue_and_never_counts_as_pass() -> None:
    queue = HumanReviewSampleQueue()
    run = build_run(registry=_full_registry(queue))
    assert run.metrics().human_pending == 3
    assert len(queue.list_pending()) == 3
    assert run.metrics().human_pass_rate is None


def test_r007_automated_records_expose_rule_and_judge_apart() -> None:
    report = evaluate_automated_results(
        mode_results=[
            EvaluationModeResult(
                run_id="r1",
                target_id="agent:x",
                mode=EvaluationMode.OFFLINE,
                success=True,
                score=0.9,
            )
        ],
        rule_signals_by_run_id={"r1": [EvaluationSignal(signal_id="p", value=1.0, threshold=1.0)]},
        llm_judge_scores_by_run_id={"r1": 0.2},
    )
    record = report.records[0]
    assert record.rule_passed is True
    assert record.judge_passed is False
