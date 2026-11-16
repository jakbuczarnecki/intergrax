# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R009 (xd_08): a rerun with pinned inputs is digest-identical."""

from __future__ import annotations

import pytest
from intergrax.contracts.evaluation_contracts import EvalSubjectOutput
from intergrax.eval.evaluators import EvaluatorRegistry, register_existing_evaluators
from tests.qualification.pdc_x.d5._support import (
    ANSWERS,
    JUDGE_MODEL_FOR_TESTS,
    build_run,
    make_dataset,
)

pytestmark = pytest.mark.unit


def test_xd08_rerun_with_pinned_inputs_is_digest_identical() -> None:
    first = build_run()
    second = build_run()
    assert first.run_id == second.run_id
    assert first.inputs_digest == second.inputs_digest
    assert first.result_digest == second.result_digest
    assert first.model_dump_json() == second.model_dump_json()


def test_xd08_latency_noise_does_not_change_the_digest() -> None:
    noisy = {key: value.model_copy(update={"latency_ms": value.latency_ms + 123.4}) for key, value in ANSWERS.items()}
    assert build_run(answers=noisy).result_digest == build_run().result_digest


def test_xd08_judge_included_rerun_is_digest_identical() -> None:
    from tests.qualification.pdc_x.d5._support import judge_fixed

    def registry() -> EvaluatorRegistry:
        return register_existing_evaluators(
            EvaluatorRegistry(),
            judge=("judge.groundedness", judge_fixed(0.9), JUDGE_MODEL_FOR_TESTS),
        )

    assert build_run(registry=registry()).result_digest == build_run(registry=registry()).result_digest


@pytest.mark.parametrize(
    "change",
    ["seed", "model_revision", "dataset_version", "output"],
)
def test_xd08_changed_pinned_input_changes_the_digest(change: str) -> None:
    base = build_run()
    if change == "seed":
        other = build_run(seed=8)
    elif change == "model_revision":
        other = build_run(model_revision="m-2")
    elif change == "dataset_version":
        other = build_run(dataset=make_dataset(version="1.0.1"))
    else:
        answers = dict(ANSWERS)
        answers["s-math"] = EvalSubjectOutput(output_text="5", cost=1.0, latency_ms=10.0, tokens=10)
        other = build_run(answers=answers)
    assert other.result_digest != base.result_digest
