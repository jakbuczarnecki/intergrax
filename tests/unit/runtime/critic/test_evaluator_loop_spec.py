# © Artur Czarnecki. All rights reserved.

"""Evaluator-loop spec tests (Phase CRIT-V-1.3)."""

from __future__ import annotations

import pytest

from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_evaluator_loop_spec_allows_single_iteration_without_revise_node() -> None:
    spec = EvaluatorLoopSpec(max_iterations=1, min_score=0.75)
    assert spec.revise_node_id is None


def test_evaluator_loop_spec_requires_revise_node_for_multi_iteration() -> None:
    with pytest.raises(ValueError, match="revise_node_id"):
        EvaluatorLoopSpec(max_iterations=3)


def test_evaluator_loop_spec_accepts_revise_target() -> None:
    spec = EvaluatorLoopSpec(
        max_iterations=3,
        min_score=0.8,
        revise_node_id="revise_worker",
        escalate_on_exhaustion=False,
    )
    assert spec.revise_node_id == "revise_worker"
    assert spec.escalate_on_exhaustion is False
