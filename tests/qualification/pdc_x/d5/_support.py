# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for PDC-D5 qualification tests (real contracts; fake executor/judge only)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.contracts.evaluation_contracts import (
    EvalDataset,
    EvalRevisionRef,
    EvalRun,
    EvalScenario,
    EvalSubject,
    EvalSubjectOutput,
)
from intergrax.eval.evaluators import EvaluatorRegistry, register_existing_evaluators
from intergrax.eval.typed_run import EvalRunProvenance, execute_eval_run

CODE_SHA = "094f825f9c0ffee0000000000000000000000000"
JUDGE_MODEL_FOR_TESTS = EvalRevisionRef(ref_id="model:judge", revision="j-1")

ANSWERS: dict[str, EvalSubjectOutput] = {
    "s-math": EvalSubjectOutput(output_text="4", cost=1.0, latency_ms=10.0, tokens=10),
    "s-capital": EvalSubjectOutput(
        output_text="Paris",
        tool_calls=("search",),
        cited_ids=("doc-1",),
        cost=3.0,
        latency_ms=30.0,
        tokens=30,
    ),
    "s-tool": EvalSubjectOutput(
        output_text="done", tool_calls=("search", "fetch"), cost=2.0, latency_ms=20.0, tokens=20
    ),
}


def make_dataset(version: str = "1.0.0") -> EvalDataset:
    return EvalDataset(
        dataset_id="d5.sample",
        version=version,
        scenarios=(
            EvalScenario(
                scenario_id="s-math",
                input_payload={"q": "2+2"},
                expected={"output": "4"},
                risk_tags=("math",),
            ),
            EvalScenario(
                scenario_id="s-capital",
                input_payload={"q": "capital of France"},
                expected={"output": "Paris", "tools": ["search"], "citations": ["doc-1"]},
            ),
            EvalScenario(
                scenario_id="s-tool",
                input_payload={"q": "tools"},
                expected={"output": "done", "tools": ["search", "fetch"]},
            ),
        ),
    )


def provenance(*, tenant: str = "tenant-a", seed: int = 7, model_revision: str = "m-1") -> EvalRunProvenance:
    return EvalRunProvenance(
        tenant_id=tenant,
        agent=EvalRevisionRef(ref_id="agent:research", revision="a-1"),
        model=EvalRevisionRef(ref_id="model:fake", revision=model_revision),
        prompt=EvalRevisionRef(ref_id="prompt:research", revision="p-1"),
        profile=EvalRevisionRef(ref_id="profile:default", revision="pr-1"),
        code_sha=CODE_SHA,
        seed=seed,
    )


def executor_for(answers: Mapping[str, EvalSubjectOutput]):
    def _execute(scenario: EvalScenario, seed: int) -> EvalSubjectOutput:
        return answers[scenario.scenario_id]

    return _execute


def judge_fixed(score: float):
    def _judge(subject: EvalSubject) -> tuple[float, str]:
        return score, "fixed judge rationale"

    return _judge


def build_run(
    *,
    answers: Mapping[str, EvalSubjectOutput] | None = None,
    tenant: str = "tenant-a",
    seed: int = 7,
    registry: EvaluatorRegistry | None = None,
    dataset: EvalDataset | None = None,
    model_revision: str = "m-1",
) -> EvalRun:
    return execute_eval_run(
        dataset=dataset or make_dataset(),
        executor=executor_for(answers or ANSWERS),
        registry=registry or register_existing_evaluators(EvaluatorRegistry()),
        provenance=provenance(tenant=tenant, seed=seed, model_revision=model_revision),
    )
