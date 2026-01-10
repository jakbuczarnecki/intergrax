# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerReplanContextInjectedDiagV1(DiagnosticPayload):
    has_replan_ctx: bool
    replan_reason: Optional[str]
    replan_hash: str
    replan_json_len: int

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.replan_context_injected"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_replan_ctx": self.has_replan_ctx,
            "replan_reason": self.replan_reason,
            "replan_hash": self.replan_hash,
            "replan_json_len": self.replan_json_len,
        }
