# © Artur Czarnecki. All rights reserved.

"""Canonical evaluator-loop model attempt diagnostic (DS-E2E-15J-O1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True, slots=True)
class EvaluatorModelAttemptDiagV1(DiagnosticPayload):
    """
    One committed evaluator-loop model invocation.

    ``attempt_index`` is zero-based and unique per ``(run_id, node_id, attempt_index)``.
    ``max_iterations`` is the effective evaluator-loop bound for this node.
    """

    run_id: str
    node_id: str
    attempt_index: int
    max_iterations: int

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.evaluator_loop.model_attempt.v1"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node_id": self.node_id,
            "attempt_index": self.attempt_index,
            "max_iterations": self.max_iterations,
        }

    def redact(self) -> EvaluatorModelAttemptDiagV1:
        return self
