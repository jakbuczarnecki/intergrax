# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerPlanSourceContractViolationDiagV1(DiagnosticPayload):
    """
    Emitted when PlanSource returns a value that violates the contract
    (e.g. raw plan is not a string).
    """

    plan_source_type: str
    raw_type: str

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.plan_source_contract_violation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_source_type": self.plan_source_type,
            "raw_type": self.raw_type,
        }
