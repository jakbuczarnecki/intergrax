# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerEnginePlanProducedDiagV1(DiagnosticPayload):
    """
    Summary payload emitted when EnginePlanner produced an EnginePlan.
    """

    intent: str
    next_step: Optional[str]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.engine_plan_produced"

    def to_dict(self) -> Dict[str, Any]:
        # Keep serialization stable and JSON-safe.
        return {
            "intent": self.intent,
            "next_step": self.next_step,
        }
