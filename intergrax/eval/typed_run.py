# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Assemble a typed ``EvalRun`` from a dataset, an executor and registered evaluators.

This is a composition function, not an engine: the system under evaluation is injected
as an executor (``EvalRunner`` results map in through ``eval_result_to_subject_output``)
and verdicts come from the evaluators registered in ``EvaluatorRegistry``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intergrax.contracts.evaluation_contracts import (
    EvalContractError,
    EvalDataset,
    EvalRevisionRef,
    EvalRun,
    EvalScenario,
    EvalScenarioTelemetry,
    EvalSubject,
    EvalSubjectOutput,
    EvaluatorOutcome,
    text_digest,
)
from intergrax.eval.eval_result import EvalResult
from intergrax.eval.evaluators import EvaluatorRegistry, EvaluatorRegistryError

EvalExecutor = Callable[[EvalScenario, int], EvalSubjectOutput]


@dataclass(frozen=True, slots=True)
class EvalRunProvenance:
    """Pinned inputs of a run. Revisions are neutral typed references."""

    tenant_id: str
    agent: EvalRevisionRef
    model: EvalRevisionRef
    prompt: EvalRevisionRef
    profile: EvalRevisionRef
    code_sha: str
    seed: int


def eval_result_to_subject_output(
    result: EvalResult,
    *,
    latency_ms: float = 0.0,
    tool_calls: tuple[str, ...] = (),
    cited_ids: tuple[str, ...] = (),
) -> EvalSubjectOutput:
    """Map an existing ``EvalResult`` (kept unchanged) into a typed subject output."""
    return EvalSubjectOutput(
        output_text=result.final_answer,
        tool_calls=tool_calls,
        cited_ids=cited_ids,
        cost=result.total_cost,
        latency_ms=latency_ms,
        tokens=result.total_tokens,
        error=result.error,
    )


def scenario_from_eval_case(case: Any) -> EvalScenario:
    """Map an existing ``EvalCase`` into a typed scenario (stable id = ``case_id``)."""
    return EvalScenario(
        scenario_id=case.case_id,
        description=case.description or "",
        expected={"output": case.expected_output},
        risk_tags=tuple(case.tags or ()),
    )


def _run_executor(executor: EvalExecutor, scenario: EvalScenario, seed: int) -> EvalSubjectOutput:
    try:
        return executor(scenario, seed)
    except Exception as exc:  # noqa: BLE001 - executor failure is recorded as a failed case
        return EvalSubjectOutput(error=f"executor failed: {type(exc).__name__}")


def execute_eval_run(
    *,
    dataset: EvalDataset,
    executor: EvalExecutor,
    registry: EvaluatorRegistry,
    provenance: EvalRunProvenance,
) -> EvalRun:
    """Run every scenario through ``executor`` and every applicable evaluator.

    Deterministic by construction: scenarios keep dataset order, evaluators run in id
    order, and the seed is handed to the executor and to every subject.
    """
    if not registry.evaluators():
        raise EvaluatorRegistryError("no evaluators registered")
    run_ref = EvalRun(
        tenant_id=provenance.tenant_id,
        agent=provenance.agent,
        model=provenance.model,
        prompt=provenance.prompt,
        profile=provenance.profile,
        code_sha=provenance.code_sha,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_digest=dataset.content_digest,
        seed=provenance.seed,
    ).run_id

    outcomes: list[EvaluatorOutcome] = []
    telemetry: list[EvalScenarioTelemetry] = []
    for scenario in dataset.scenarios:
        output = _run_executor(executor, scenario, provenance.seed)
        telemetry.append(
            EvalScenarioTelemetry(
                scenario_id=scenario.scenario_id,
                output_digest=text_digest(output.output_text),
                cost=output.cost,
                latency_ms=output.latency_ms,
                tokens=output.tokens,
                error=output.error,
            )
        )
        subject = EvalSubject(scenario=scenario, run_ref=run_ref, output=output, seed=provenance.seed)
        for evaluator in registry.evaluators():
            outcome = evaluator.evaluate(subject)
            if outcome is None:
                continue
            if (
                outcome.evaluator_id != evaluator.evaluator_id
                or outcome.kind is not evaluator.kind
                or outcome.scenario_id != scenario.scenario_id
            ):
                raise EvalContractError(
                    f"outcome of {evaluator.evaluator_id} does not match its evaluator id, kind or scenario"
                )
            outcomes.append(outcome)

    return EvalRun(
        tenant_id=provenance.tenant_id,
        agent=provenance.agent,
        model=provenance.model,
        prompt=provenance.prompt,
        profile=provenance.profile,
        code_sha=provenance.code_sha,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_digest=dataset.content_digest,
        seed=provenance.seed,
        outcomes=tuple(outcomes),
        telemetry=tuple(telemetry),
    )
