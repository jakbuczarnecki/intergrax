# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerBuildDebugDiagV1(DiagnosticPayload):
    planner_forced_plan_used: bool
    planner_source_kind: Optional[str]
    planner_source_detail: Optional[str]

    planner_replan_ctx_present: bool
    planner_replan_ctx_hash: Optional[str]

    planner_raw_len: int
    planner_raw_hash: str
    planner_raw_preview: str
    planner_raw_tail_preview: str

    planner_forced_plan_json_len: Optional[int]
    planner_forced_plan_hash: Optional[str]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.build_debug"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner_forced_plan_used": self.planner_forced_plan_used,
            "planner_source_kind": self.planner_source_kind,
            "planner_source_detail": self.planner_source_detail,
            "planner_replan_ctx_present": self.planner_replan_ctx_present,
            "planner_replan_ctx_hash": self.planner_replan_ctx_hash,
            "planner_raw_len": self.planner_raw_len,
            "planner_raw_hash": self.planner_raw_hash,
            "planner_raw_preview": self.planner_raw_preview,
            "planner_raw_tail_preview": self.planner_raw_tail_preview,
            "planner_forced_plan_json_len": self.planner_forced_plan_json_len,
            "planner_forced_plan_hash": self.planner_forced_plan_hash,
        }
