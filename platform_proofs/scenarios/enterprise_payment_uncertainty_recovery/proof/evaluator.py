"""Proof-owned evaluation seam — falsification assertions live here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    passed: bool
    failures: tuple[str, ...] = ()


def evaluate_scenario_run(domain_result: object) -> ScenarioEvaluation:
    """Evaluate application output against the Scenario Specification contract."""
    raise NotImplementedError("Implement proof evaluator contract.")
