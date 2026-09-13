"""Typed diagnostic payloads for ERL-QUAL-004 lifecycle trace steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True, slots=True)
class ErlQual004LifecycleStepDiagV1(DiagnosticPayload):
    """Business context for one qualification lifecycle step (scenario-owned semantics)."""

    trace_id: str
    correlation_id: str
    execution_id: str
    scenario_id: str
    step_id: str
    outcome: str
    variant_id: str
    component_identity: str
    business_detail: dict[str, Any]

    @classmethod
    def schema_id(cls) -> str:
        return "platform_proofs.erl_qual_004.lifecycle_step.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "execution_id": self.execution_id,
            "scenario_id": self.scenario_id,
            "step_id": self.step_id,
            "outcome": self.outcome,
            "variant_id": self.variant_id,
            "component_identity": self.component_identity,
            "business_detail": self.business_detail,
        }

    def redact(self) -> ErlQual004LifecycleStepDiagV1:
        return self
