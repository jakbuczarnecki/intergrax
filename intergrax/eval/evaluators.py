# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""One evaluator contract with explicit kinds, plus adapters for existing evaluators.

Kinds: ``DETERMINISTIC``, ``MODEL_ASSISTED`` (isolated judge), ``HUMAN``, ``SAFETY``.

Isolation rules enforced here and in ``typed_run.execute_eval_run``:

- an evaluator only ever receives an ``EvalSubject``; no evaluator sees another
  evaluator's outcome, so a judge cannot be conditioned on rule results;
- an outcome must carry the kind and id of the evaluator that produced it;
- ``MODEL_ASSISTED`` outcomes name their judge model and are aggregated apart from rule
  results (see ``EvalRun.rule_outcomes`` / ``judge_outcomes``).

Existing evaluators are registered through adapters, not rewritten.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from intergrax.contracts.evaluation_contracts import (
    EvalContractError,
    EvalRevisionRef,
    EvalSubject,
    EvaluatorKind,
    EvaluatorOutcome,
    EvaluatorOutcomeStatus,
)
from intergrax.contracts.guardrail_verification import assess_guardrail_scan
from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailContext,
    LlmGuardrailBackend,
)
from intergrax.runtime.evaluation.human_review_sample_queue import HumanReviewSampleQueue


class EvaluatorRegistryError(EvalContractError):
    """Invalid evaluator registration or outcome."""


@runtime_checkable
class Evaluator(Protocol):
    """Single evaluator contract. ``evaluate`` returns ``None`` when not applicable."""

    @property
    def evaluator_id(self) -> str: ...

    @property
    def kind(self) -> EvaluatorKind: ...

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None: ...


class EvaluatorRegistry:
    """Registry of evaluators behind the single contract (in-process, deterministic order)."""

    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        if not isinstance(evaluator, Evaluator):
            raise EvaluatorRegistryError("object does not satisfy the Evaluator contract")
        if not isinstance(evaluator.kind, EvaluatorKind):
            raise EvaluatorRegistryError("evaluator kind must be an EvaluatorKind")
        evaluator_id = evaluator.evaluator_id.strip()
        if not evaluator_id:
            raise EvaluatorRegistryError("evaluator_id must not be blank")
        if evaluator_id in self._evaluators:
            raise EvaluatorRegistryError(f"evaluator already registered: {evaluator_id}")
        self._evaluators[evaluator_id] = evaluator

    def get(self, evaluator_id: str) -> Evaluator:
        return self._evaluators[evaluator_id]

    def evaluators(self) -> tuple[Evaluator, ...]:
        return tuple(self._evaluators[key] for key in sorted(self._evaluators))

    def by_kind(self, kind: EvaluatorKind) -> tuple[Evaluator, ...]:
        return tuple(item for item in self.evaluators() if item.kind is kind)

    def kinds_covered(self) -> frozenset[EvaluatorKind]:
        return frozenset(item.kind for item in self._evaluators.values())


def _outcome(
    evaluator: Evaluator,
    subject: EvalSubject,
    *,
    passed: bool,
    score: float | None = None,
    reasons: tuple[str, ...] = (),
    judge_model: EvalRevisionRef | None = None,
) -> EvaluatorOutcome:
    return EvaluatorOutcome(
        evaluator_id=evaluator.evaluator_id,
        kind=evaluator.kind,
        scenario_id=subject.scenario.scenario_id,
        status=EvaluatorOutcomeStatus.PASSED if passed else EvaluatorOutcomeStatus.FAILED,
        score=score if score is not None else (1.0 if passed else 0.0),
        reasons=reasons,
        judge_model=judge_model,
    )


class ExactMatchEvaluator:
    """Task success: output equals ``expected['output']`` (same rule as ``EvalRunner``)."""

    kind = EvaluatorKind.DETERMINISTIC

    def __init__(self, evaluator_id: str = "task_success.exact_match") -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        expected = subject.scenario.expected.get("output")
        if expected is None:
            return None
        ok = subject.output.error is None and subject.output.output_text == expected
        return _outcome(self, subject, passed=ok, reasons=() if ok else ("output differs from expected",))


class ToolUseEvaluator:
    """Tool use: every tool in ``expected['tools']`` was called."""

    kind = EvaluatorKind.DETERMINISTIC

    def __init__(self, evaluator_id: str = "tool_use.expected_tools") -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        expected = subject.scenario.expected.get("tools")
        if expected is None:
            return None
        missing = [name for name in expected if name not in subject.output.tool_calls]
        return _outcome(
            self,
            subject,
            passed=not missing,
            reasons=tuple(f"tool not called: {name}" for name in missing),
        )


class CitationEvaluator:
    """Citations: every id in ``expected['citations']`` is cited by the output."""

    kind = EvaluatorKind.DETERMINISTIC

    def __init__(self, evaluator_id: str = "citation.expected_citations") -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        expected = subject.scenario.expected.get("citations")
        if expected is None:
            return None
        missing = [cid for cid in expected if cid not in subject.output.cited_ids]
        return _outcome(
            self,
            subject,
            passed=not missing,
            reasons=tuple(f"citation missing: {cid}" for cid in missing),
        )


class GoldenRetrievalEvaluator:
    """Adapter registering the existing ``rag.evaluation.golden_harness`` as DETERMINISTIC.

    The scenario input payload is one golden retrieval case. The harness is not
    rewritten; it is imported lazily and called as is.
    """

    kind = EvaluatorKind.DETERMINISTIC

    def __init__(self, evaluator_id: str = "retrieval.golden_harness") -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        payload = subject.scenario.input_payload
        if "documents" not in payload or "query" not in payload:
            return None
        from intergrax.rag.evaluation.golden_harness import run_golden_retrieval

        report = run_golden_retrieval([payload])
        case = report.results[0]
        return _outcome(
            self,
            subject,
            passed=case.passed,
            score=min(1.0, max(0.0, case.recall)),
            reasons=() if case.passed else (f"recall {case.recall} < {case.min_recall}",),
        )


class GuardrailSafetyEvaluator:
    """SAFETY kind implemented through an existing guardrail integration backend."""

    kind = EvaluatorKind.SAFETY

    def __init__(
        self,
        backend: LlmGuardrailBackend,
        evaluator_id: str | None = None,
    ) -> None:
        self._backend = backend
        self.evaluator_id = evaluator_id or f"safety.guardrail.{backend.slug}"

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        scan = self._backend.scan_output(
            subject.output.output_text,
            context=GuardrailContext(run_id=subject.run_ref),
        )
        assessment = assess_guardrail_scan(scan)
        reasons = () if assessment.passed else (assessment.detail, *assessment.categories)
        return _outcome(self, subject, passed=assessment.passed, reasons=tuple(reasons))


JudgeCallable = Callable[[EvalSubject], tuple[float, str]]


class ModelAssistedJudgeEvaluator:
    """Isolated model-assisted judge. The judge callable sees the subject only."""

    kind = EvaluatorKind.MODEL_ASSISTED

    def __init__(
        self,
        evaluator_id: str,
        judge: JudgeCallable,
        judge_model: EvalRevisionRef,
        *,
        pass_threshold: float = 0.75,
    ) -> None:
        self.evaluator_id = evaluator_id
        self._judge = judge
        self._judge_model = judge_model
        self._threshold = pass_threshold

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        score, rationale = self._judge(subject)
        score = min(1.0, max(0.0, float(score)))
        passed = score >= self._threshold
        return _outcome(
            self,
            subject,
            passed=passed,
            score=score,
            reasons=(rationale,) if rationale else (),
            judge_model=self._judge_model,
        )


class HumanReviewEvaluator:
    """HUMAN kind on top of the existing human review sample queue.

    A scenario without a recorded decision is enqueued and reported ``PENDING``.
    """

    kind = EvaluatorKind.HUMAN

    def __init__(
        self,
        queue: HumanReviewSampleQueue,
        *,
        agent_id: str,
        decisions: Mapping[str, bool] | None = None,
        evaluator_id: str = "human.review_queue",
    ) -> None:
        self.evaluator_id = evaluator_id
        self._queue = queue
        self._agent_id = agent_id
        self._decisions = dict(decisions or {})

    def evaluate(self, subject: EvalSubject) -> EvaluatorOutcome | None:
        scenario_id = subject.scenario.scenario_id
        if scenario_id in self._decisions:
            return _outcome(self, subject, passed=self._decisions[scenario_id])
        self._queue.enqueue(
            run_id=subject.run_ref,
            agent_id=self._agent_id,
            scenario_id=scenario_id,
            reason="awaiting human review",
        )
        return EvaluatorOutcome(
            evaluator_id=self.evaluator_id,
            kind=self.kind,
            scenario_id=scenario_id,
            status=EvaluatorOutcomeStatus.PENDING,
            reasons=("awaiting human review",),
        )


def register_existing_evaluators(
    registry: EvaluatorRegistry,
    *,
    guardrail_backend: LlmGuardrailBackend | None = None,
    judge: tuple[str, JudgeCallable, EvalRevisionRef] | None = None,
    review_queue: HumanReviewSampleQueue | None = None,
    review_agent_id: str = "",
    review_decisions: Mapping[str, bool] | None = None,
) -> EvaluatorRegistry:
    """Register the existing evaluators behind the single contract.

    Deterministic evaluators are always registered. SAFETY, MODEL_ASSISTED and HUMAN
    evaluators are registered when their provider (guardrail backend, judge, review
    queue) is supplied; there is no default provider for them.
    """
    registry.register(ExactMatchEvaluator())
    registry.register(ToolUseEvaluator())
    registry.register(CitationEvaluator())
    registry.register(GoldenRetrievalEvaluator())
    if guardrail_backend is not None:
        registry.register(GuardrailSafetyEvaluator(guardrail_backend))
    if judge is not None:
        evaluator_id, callable_, model_ref = judge
        registry.register(ModelAssistedJudgeEvaluator(evaluator_id, callable_, model_ref))
    if review_queue is not None:
        registry.register(HumanReviewEvaluator(review_queue, agent_id=review_agent_id, decisions=review_decisions))
    return registry
